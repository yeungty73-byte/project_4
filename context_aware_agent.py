"""context_aware_agent.py - Context-aware PPO agent with BSTS intermediary metrics.

Extends the base ContextAwarePPOAgent with:
  1. IntermediaryMetricsHead: predicts per-step physics (curvature, perp_v, etc.)
  2. RaceLineAwareness: auxiliary loss for v_perp -> 0 at barrier
  3. BSTS-informed reward shaping via decomposed trend/seasonal signals

The agent learns WHY things work by understanding the causal chain:
  intermediary metrics (curvature, perp_v, brake_zone) -> success metrics (completion, reward)
"""
# import math  # removed: unused
# import numpy as np  # removed: unused
import torch
import torch.nn as nn
from torch.distributions import Normal
from agents import Agent, PPOAgent

# Number of intermediary physics features appended to observation
N_INTERMEDIARY_FEATURES = 6  # curvature, perp_v, brake_zone, race_line_dev, lateral_g, angular_rate


class ContextHead(nn.Module):
    """Classifies LiDAR context.
    
    Context classes:
      0 = clear (open track)
      1 = curb (near track boundary -- v_perp must -> 0 here)
      2 = obstacle (bot or static object)
      3 = corner_approach (upcoming high-curvature turn)
      4 = high_speed_straight (long straight, can push speed)
    """

    def __init__(self, lidar_latent_dim=32, hidden_dim=64, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(lidar_latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, lidar_latent):
        return self.mlp(lidar_latent)


class IntermediaryMetricsHead(nn.Module):
    """Predicts intermediary physics metrics from observation.
    
    These are the causal drivers of success: the agent learns to predict
    and control these metrics, understanding WHY actions lead to outcomes.
    
    Outputs (all scalar per step):
      - curvature: current path curvature (1/R)
      - perp_velocity_at_barrier: v dot n_barrier (must -> 0 at curb/obstacle)
      - brake_zone_score: 0 = no braking needed, 1 = heavy braking
      - race_line_deviation: distance from optimal line
      - lateral_g: centripetal acceleration proxy
      - angular_rate: heading change rate
    """

    def __init__(self, latent_dim=32, hidden_dim=32, n_metrics=N_INTERMEDIARY_FEATURES):
        super().__init__()
        self.n_metrics = n_metrics
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_metrics),
        )
        nn.init.orthogonal_(self.mlp[0].weight, gain=0.5)
        nn.init.orthogonal_(self.mlp[2].weight, gain=0.1)

    def forward(self, latent):
        return self.mlp(latent)


def compute_perp_velocity_loss(perp_v_pred, context_class):
    """Auxiliary loss: penalize perpendicular velocity at barriers.
    
    v_perp should be zero when context is curb (1) or obstacle (2).
    This implements the calculus constraint: at any barrier surface,
    the velocity component perpendicular to that surface must vanish.
    
    minimize: sum_{t where context in {curb, obstacle}} |v_perp_t|^2
    """
    # context_class shape: (batch,) with values 0-4
    barrier_mask = ((context_class == 1) | (context_class == 2)).float()
    # perp_v_pred shape: (batch,) - predicted perpendicular velocity
    perp_v_sq = perp_v_pred ** 2
    loss = (perp_v_sq * barrier_mask).mean()
    return loss


def compute_brake_zone_loss(brake_pred, curvature_pred):
    """Auxiliary loss: minimize brake zones while respecting curvature.
    
    The agent should brake only where curvature demands it.
    Penalize braking on straights; allow braking proportional to curvature.
    
    minimize: sum_t max(0, brake_t - alpha * curvature_t)
    """
    alpha = 2.0  # braking tolerance per unit curvature
    excess_brake = torch.relu(brake_pred - alpha * curvature_pred.detach())
    return excess_brake.mean()


class ContextAwarePPOAgent(PPOAgent):
# REF: Hettiarachchi, R. et al. (2024). U-Transformer with skip connections for autonomous racing. arXiv preprint.
# REF: Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
    """
    PPO agent extended with:
    1. ContextHead: classifies what LiDAR sees (clear/curb/obstacle/corner/straight)
    2. IntermediaryMetricsHead: predicts physics metrics (causal understanding)
    3. Race-line calculus: v_perp -> 0 at barrier, minimize brake zones
    4. BSTS decomposition awareness: trend/seasonal signals inform exploration
    
    The agent understands WHY things work through the causal chain:
      observation -> intermediary metrics -> success metrics
    
    Auxiliary losses:
      - Context classification (supervised from environment labels)
      - Perpendicular velocity at barrier -> 0 (physics constraint)
      - Brake zone minimization (efficiency constraint)
      - Intermediary metrics prediction (self-supervised from trajectory)
    """

    def __init__(self, obs_dim=64, act_dim=2, name='context_aware_bsts',
                 lidar_latent_dim=32, n_context_classes=5):
        Agent.__init__(self, name=name)
        nn.Module.__init__(self)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lidar_latent_dim = lidar_latent_dim

        # Shared encoder: obs -> lidar_latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, lidar_latent_dim),
        )

        # Context head: classifies driving situation
        self.context_head = ContextHead(
            lidar_latent_dim=lidar_latent_dim,
            hidden_dim=64,
            num_classes=n_context_classes
        )

        # Intermediary metrics head: predicts physics
        self.intermediary_head = IntermediaryMetricsHead(
            latent_dim=lidar_latent_dim,
            hidden_dim=32,
            n_metrics=N_INTERMEDIARY_FEATURES
        )

        # Conditioned trunk: latent + context_embedding + intermediary_embedding
        ctx_embed_dim = 8
        intermed_embed_dim = 8
        self.context_embed = nn.Embedding(n_context_classes, ctx_embed_dim)
        self.intermed_proj = nn.Linear(N_INTERMEDIARY_FEATURES, intermed_embed_dim)

        trunk_input_dim = lidar_latent_dim + ctx_embed_dim + intermed_embed_dim
        self.shared_trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )

        # Actor (policy) head - continuous: outputs mean
        self.actor_mean = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, act_dim),
            nn.Tanh(),  # bound mean to [-1, 1]
        )
        # Learnable log std
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

        # Init weights
        for module in [self.encoder, self.shared_trunk, self.actor_mean, self.critic]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)
        # Actor last layer: small init for exploration
        nn.init.orthogonal_(self.actor_mean[-2].weight, gain=0.01)  # Linear before Tanh
        # Critic last layer: standard init
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

        # BSTS state: track trend direction for exploration annealing
        self.bsts_trend = 0.0  # updated externally from analyze_logs
        self.bsts_seasonal = 0.0

    def encode(self, x):
        """Encode observation to lidar latent."""
        return self.encoder(x)

    def forward(self, x):
        """Full forward pass with context + intermediary metrics conditioning."""
        lidar_latent = self.encode(x)

        # Context classification
        ctx_logits = self.context_head(lidar_latent)
        ctx_class = ctx_logits.argmax(dim=-1)
        ctx_emb = self.context_embed(ctx_class)

        # Intermediary metrics prediction
        intermed_pred = self.intermediary_head(lidar_latent)
        intermed_emb = self.intermed_proj(intermed_pred.detach())  # stop gradient for stability

        # Conditioned trunk
        shared_input = torch.cat([lidar_latent, ctx_emb, intermed_emb], dim=-1)
        shared_out = self.shared_trunk(shared_input)

        mean = self.actor_mean(shared_out)
        std = self.actor_log_std.exp().expand_as(mean)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        value = self.critic(shared_out)

        return mean, std, value, ctx_logits, intermed_pred

    def get_action(self, x):
        """Get action for inference."""
        with torch.no_grad():
            mean, std, _, _, _ = self.forward(x)
            dist = Normal(mean, std)
            action = dist.sample()
            return action.squeeze(0).cpu().numpy()

    def get_action_and_value(self, x, action=None):
        """PPO training: returns action, log_prob, entropy, value, ctx_logits, intermed_pred."""
        mean, std, value, ctx_logits, intermed_pred = self.forward(x)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        # sum over action dims
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1), ctx_logits, intermed_pred

    def get_value(self, x):
        """Value estimate only."""
        with torch.no_grad():
            _, _, value, _, _ = self.forward(x)
            return value.squeeze(-1)

    def get_context(self, x):
        """Return predicted context class."""
        with torch.no_grad():
            _, lidar_latent = x, self.encode(x)
            ctx_logits = self.context_head(lidar_latent)
        return ctx_logits.argmax(dim=-1).cpu().numpy()

    def get_intermediary_metrics(self, x):
        """Return predicted intermediary physics metrics."""
        with torch.no_grad():
            lidar_latent = self.encode(x)
            pred = self.intermediary_head(lidar_latent)
        return pred.cpu().numpy()

    def compute_auxiliary_losses(self, ctx_logits, ctx_labels,
                                  intermed_pred, intermed_targets,
                                  context_classes):
        """Compute all auxiliary losses for richer learning signal.

        Args:
            ctx_logits: (batch, n_classes) context predictions
            ctx_labels: (batch,) ground truth context labels
            intermed_pred: (batch, N_INTERMEDIARY_FEATURES) predicted metrics
            intermed_targets: (batch, N_INTERMEDIARY_FEATURES) actual metrics from trajectory
            context_classes: (batch,) predicted context class indices

        Returns:
            dict of losses:
              - ctx_loss: context classification cross-entropy
              - perp_v_loss: perpendicular velocity at barrier penalty
              - brake_zone_loss: excess braking penalty
              - intermed_loss: intermediary metrics prediction MSE
        """
        # Context classification loss
        ctx_loss = nn.functional.cross_entropy(ctx_logits, ctx_labels)

        # Perpendicular velocity at barrier -> 0
        # intermed_pred[:, 1] = perp_velocity_at_barrier
        perp_v_pred = intermed_pred[:, 1]
        perp_v_loss = compute_perp_velocity_loss(perp_v_pred, context_classes)

        # Brake zone minimization
        # intermed_pred[:, 2] = brake_zone_score, [:, 0] = curvature
        brake_pred = intermed_pred[:, 2]
        curv_pred = intermed_pred[:, 0]
        brake_loss = compute_brake_zone_loss(brake_pred, curv_pred)

        # Intermediary metrics prediction loss (self-supervised)
        intermed_loss = nn.functional.mse_loss(intermed_pred, intermed_targets)

        return {
            'ctx_loss': ctx_loss,
            'perp_v_loss': perp_v_loss,
            'brake_zone_loss': brake_loss,
            'intermed_loss': intermed_loss,
        }

    def update_bsts_state(self, trend: float, seasonal: float):
        """Update BSTS decomposition state from analyze_logs.

        This allows the agent to adapt its exploration based on
        the structural time series analysis of training progress.
        """
        self.bsts_trend = trend
        self.bsts_seasonal = seasonal

    def get_exploration_bonus(self):
        """BSTS-informed exploration bonus.

        If trend is degrading, increase exploration (higher entropy bonus).
        If seasonal component is large, reduce exploration to avoid oscillation.
        """
        base_bonus = 0.01
        if self.bsts_trend < -0.01:  # degrading
            return base_bonus * 2.0  # explore more
        elif self.bsts_trend > 0.01:  # improving
            return base_bonus * 0.5  # exploit more
        return base_bonus
