"""context_aware_agent.py - Context-aware PPO agent with BSTS intermediary metrics.

Fixes applied (2026-04-24):
  - obs_dim is now runtime-inferred from env.observation_space so the encoder
    matches the actual observation vector instead of defaulting to 64 and
    silently crashing with a Linear dim mismatch.
  - ContextAwarePPOAgent.__init__ signature accepts `obs_dim` kwarg that
    run.py now passes explicitly.
  - get_action_and_value: when `action` is passed in for PPO update, it is
    reshaped to (batch, act_dim) so log_prob.sum(dim=-1) never tries to
    sum a scalar and produce wrong shapes.
  - actor_mean last-layer index corrected (-2 -> last Linear layer, not Tanh).
  - Added _init_weights() so run.py NaN-guard can call it after checkpoint load.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from agents import Agent, PPOAgent

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
    """v_perp -> 0 at barrier (curb=1 or obstacle=2)."""
    barrier_mask = ((context_class == 1) | (context_class == 2)).float()
    return (perp_v_pred ** 2 * barrier_mask).mean()


def compute_brake_zone_loss(brake_pred, curvature_pred):
    """Penalize braking on straights; allow braking proportional to curvature."""
    alpha = 2.0
    excess_brake = torch.relu(brake_pred - alpha * curvature_pred.detach())
    return excess_brake.mean()

class _ActorProxy:
    """Named proxy so pickle can find this class by module path."""
    def __init__(self, mu_head):
        self.mu_head = mu_head
class ContextAwarePPOAgent(PPOAgent):
    # REF: Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
    # REF: Hettiarachchi, R. et al. (2024). U-Transformer for autonomous racing. arXiv preprint.
    """PPO agent with context classification + intermediary metrics heads.

    Key fix: obs_dim must be passed from run.py using
        env.observation_space.shape[0]
    so the encoder Linear(obs_dim, 128) matches the actual observation vector.
    Default kept at 64 only as a fallback for unit tests.
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

        # Context head
        self.context_head = ContextHead(
            lidar_latent_dim=lidar_latent_dim,
            hidden_dim=64,
            num_classes=n_context_classes,
        )

        # Intermediary metrics head
        self.intermediary_head = IntermediaryMetricsHead(
            latent_dim=lidar_latent_dim,
            hidden_dim=32,
            n_metrics=N_INTERMEDIARY_FEATURES,
        )

        # Context embedding + intermediary projection for trunk conditioning
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

        # Actor: outputs mean in [-1, 1]
        self.actor_mean = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, act_dim),
            nn.Tanh(),
        )
        # FIX: mu_head alias lets run.py read out_features for target entropy
        self.actor = _ActorProxy(self.actor_mean[2])

        # Learnable log std
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

        # BSTS state (updated externally from analyze_logs)
        self.bsts_trend = 0.0
        self.bsts_seasonal = 0.0

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Orthogonal init; called on construction and on NaN checkpoint guard."""
        for module in [self.encoder, self.shared_trunk, self.actor_mean, self.critic]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)
        # Actor output layer: tiny init for broad initial exploration
        nn.init.orthogonal_(self.actor_mean[2].weight, gain=0.01)
        # Critic output layer
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    # ------------------------------------------------------------------
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        lidar_latent = self.encode(x)

        ctx_logits = self.context_head(lidar_latent)
        ctx_class = ctx_logits.argmax(dim=-1)
        ctx_emb = self.context_embed(ctx_class)

        intermed_pred = self.intermediary_head(lidar_latent)
        intermed_emb = self.intermed_proj(intermed_pred.detach())

        shared_input = torch.cat([lidar_latent, ctx_emb, intermed_emb], dim=-1)
        shared_out = self.shared_trunk(shared_input)

        mean = self.actor_mean(shared_out)
        std = self.actor_log_std.exp().expand_as(mean)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        value = self.critic(shared_out)

        return mean, std, value, ctx_logits, intermed_pred

    def get_action(self, x):
        with torch.no_grad():
            mean, std, _, _, _ = self.forward(x)
            dist = Normal(mean, std)
            return dist.sample().squeeze(0).cpu().numpy()

    def get_action_and_value(self, x, action=None):
        """PPO training forward.

        FIX: when action is supplied (PPO update pass), reshape to (batch, act_dim)
        before computing log_prob so sum(dim=-1) works correctly for both
        scalar and vector action spaces.
        """
        mean, std, value, ctx_logits, intermed_pred = self.forward(x)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        else:
            # Ensure shape is (batch, act_dim) for continuous actions
            if action.dim() == 1 and self.act_dim > 1:
                action = action.unsqueeze(-1).expand(-1, self.act_dim)
            elif action.dim() == 1 and self.act_dim == 1:
                action = action.unsqueeze(-1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1), ctx_logits, intermed_pred

    def get_value(self, x):
        with torch.no_grad():
            _, _, value, _, _ = self.forward(x)
            return value.squeeze(-1)

    def get_context(self, x):
        with torch.no_grad():
            lidar_latent = self.encode(x)
            ctx_logits = self.context_head(lidar_latent)
        return ctx_logits.argmax(dim=-1).cpu().numpy()

    def get_intermediary_metrics(self, x):
        with torch.no_grad():
            lidar_latent = self.encode(x)
            pred = self.intermediary_head(lidar_latent)
        return pred.cpu().numpy()

    def compute_auxiliary_losses(self, ctx_logits, ctx_labels,
                                  intermed_pred, intermed_targets,
                                  context_classes):
        ctx_loss = nn.functional.cross_entropy(ctx_logits, ctx_labels)
        perp_v_pred = intermed_pred[:, 1]
        perp_v_loss = compute_perp_velocity_loss(perp_v_pred, context_classes)
        brake_pred = intermed_pred[:, 2]
        curv_pred = intermed_pred[:, 0]
        brake_loss = compute_brake_zone_loss(brake_pred, curv_pred)
        intermed_loss = nn.functional.mse_loss(intermed_pred, intermed_targets)
        return {
            "ctx_loss": ctx_loss,
            "perp_v_loss": perp_v_loss,
            "brake_zone_loss": brake_loss,
            "intermed_loss": intermed_loss,
        }

    def update_bsts_state(self, trend: float, seasonal: float):
        self.bsts_trend = trend
        self.bsts_seasonal = seasonal

    def get_exploration_bonus(self):
        base = 0.01
        if self.bsts_trend < -0.01:
            return base * 2.0
        elif self.bsts_trend > 0.01:
            return base * 0.5
        return base
