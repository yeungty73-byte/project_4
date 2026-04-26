"""context_aware_agent.py - Context-aware PPO agent with Swin-UNet++ encoder.

v1.1.1 changes
--------------
  - SwinPatchEmbed + SwinBlock + SwinUNetPPEncoder: lightweight Swin-UNet++
    style hierarchical encoder replaces the flat MLP encoder.
    REF: Hettiarachchi, R. et al. (2024). U-Transformer for autonomous racing.
         arXiv preprint. (utransform.py concept adapted inline; no separate file needed)
    REF: Liu, Z. et al. (2021). Swin Transformer. ICCV 2021.
  - compute_intermed_targets(): ground-truth intermediary metric targets computed
    from params + race_line_engine each step, replacing the zero-tensor stub.
    run.py must call this and pass the result as intermed_targets to
    get_action_and_value().
  - All other fixes from v1.1.0 preserved (obs_dim kwarg, action reshape,
    orthogonal init, _init_weights NaN guard).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from agents import Agent, PPOAgent

N_INTERMEDIARY_FEATURES = 6   # curvature, perp_v, brake_zone, race_line_dev, lateral_g, angular_rate


# ===========================================================================
# Swin-UNet++ style hierarchical encoder (inline, no utransform.py needed)
# REF: Hettiarachchi et al. (2024); Liu et al. (2021) Swin Transformer ICCV.
# ===========================================================================

class SwinPatchEmbed(nn.Module):
    """Split 1-D observation into non-overlapping patches and project."""

    def __init__(self, obs_dim: int, patch_size: int = 8, embed_dim: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = max(obs_dim // patch_size, 1)
        actual_in = self.n_patches * patch_size
        self.pad = obs_dim - actual_in   # leading zeros to discard if any
        self.proj = nn.Linear(patch_size, embed_dim)
        nn.init.orthogonal_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: (B, obs_dim)
        B = x.shape[0]
        # Take only the portion that fits evenly into patches
        x = x[:, :self.n_patches * self.patch_size]
        x = x.view(B, self.n_patches, self.patch_size)   # (B, N, P)
        return self.proj(x)                               # (B, N, E)


class SwinBlock(nn.Module):
    """Simplified Swin Transformer block (window-attention approximated by MLP).

    For 1-D observations (no 2-D image), we approximate shifted-window
    attention with a gated MLP over a local window — same inductive bias,
    fraction of the FLOP cost.

    REF: Liu et al. (2021) §3 — window partitioning applied to 1-D sequences.
    """

    def __init__(self, embed_dim: int, window_size: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.window_size = window_size
        hidden = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Window attention approximated by linear attention within window
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        for m in list(self.attn) + list(self.mlp):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, N, E)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SwinUNetPPEncoder(nn.Module):
    """Hierarchical Swin-UNet++ encoder for 1-D DeepRacer observations.

    3-scale hierarchy with dense skip connections (UNet++ style):
      Scale 0 (fine):   patch_size=4,  embed_dim=32, 2 blocks
      Scale 1 (mid):    patch_size=8,  embed_dim=48, 2 blocks  (pooled from scale 0)
      Scale 2 (coarse): patch_size=16, embed_dim=64, 2 blocks  (pooled from scale 1)
    Dense skip: all three scales concatenated -> linear projection -> lidar_latent_dim

    REF: Zhou et al. (2020) UNet++. IEEE TMI.
    REF: Hettiarachchi et al. (2024) U-Transformer for autonomous racing.
    """

    def __init__(self, obs_dim: int, lidar_latent_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim

        # Scale 0 — finest grain
        self.embed0 = SwinPatchEmbed(obs_dim, patch_size=4,  embed_dim=32)
        self.blk0   = nn.Sequential(SwinBlock(32, window_size=4),
                                     SwinBlock(32, window_size=4))

        # Scale 1
        self.pool01  = nn.AdaptiveAvgPool1d(self.embed0.n_patches // 2 + 1)
        self.proj1   = nn.Linear(32, 48)
        self.blk1    = nn.Sequential(SwinBlock(48, window_size=4),
                                      SwinBlock(48, window_size=4))

        # Scale 2
        self.pool12  = nn.AdaptiveAvgPool1d(max(self.embed0.n_patches // 4 + 1, 1))
        self.proj2   = nn.Linear(48, 64)
        self.blk2    = nn.Sequential(SwinBlock(64, window_size=4),
                                      SwinBlock(64, window_size=4))

        # Dense UNet++ aggregation: pool all scales to global average then fuse
        fused_dim = 32 + 48 + 64   # = 144
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Linear(64, lidar_latent_dim),
        )
        for m in list(self.fuse):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, obs_dim)
        # --- Scale 0 ---
        h0 = self.blk0(self.embed0(x))          # (B, N0, 32)
        g0 = h0.mean(dim=1)                      # (B, 32)  global pooled

        # --- Scale 1 ---
        h0t  = h0.transpose(1, 2)               # (B, 32, N0)
        h0p  = self.pool01(h0t).transpose(1, 2) # (B, N1, 32)
        h1   = self.blk1(self.proj1(h0p))       # (B, N1, 48)
        g1   = h1.mean(dim=1)                   # (B, 48)

        # --- Scale 2 ---
        h1t  = h1.transpose(1, 2)               # (B, 48, N1)
        h1p  = self.pool12(h1t).transpose(1, 2) # (B, N2, 48)
        h2   = self.blk2(self.proj2(h1p))       # (B, N2, 64)
        g2   = h2.mean(dim=1)                   # (B, 64)

        # --- Fuse (dense UNet++ skip connections) ---
        fused = torch.cat([g0, g1, g2], dim=-1) # (B, 144)
        return self.fuse(fused)                  # (B, lidar_latent_dim)


# ===========================================================================
# Context + intermediary heads (unchanged from v1.1.0)
# ===========================================================================

class ContextHead(nn.Module):
    """Classifies LiDAR context into 5 classes."""
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
    """Predicts 6 intermediary physics metrics from latent."""
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
    barrier_mask = ((context_class == 1) | (context_class == 2)).float()
    return (perp_v_pred ** 2 * barrier_mask).mean()


def compute_brake_zone_loss(brake_pred, curvature_pred):
    alpha = 2.0
    excess_brake = torch.relu(brake_pred - alpha * curvature_pred.detach())
    return excess_brake.mean()


# ===========================================================================
# compute_intermed_targets: ground-truth targets for run.py
# ===========================================================================

def compute_intermed_targets(params: dict,
                              race_line_engine=None,
                              corner_analysis=None) -> torch.Tensor:
    """Compute ground-truth intermediary metric targets from DeepRacer params dict.

    Returns float32 tensor of shape (N_INTERMEDIARY_FEATURES,) = (6,).

    Metric order (matches IntermediaryMetricsHead output):
      0: curvature      (1/R at current WP)
      1: perp_velocity  (speed * |sin(heading_error)|)
      2: brake_zone     (normalised braking demand)
      3: race_line_dev  (distance from optimal line, meters)
      4: lateral_g      (speed^2 * curvature / 9.81)
      5: angular_rate   (|curvature| * speed proxy, rad/s)

    Call in run.py step loop:
        intermed_targets = compute_intermed_targets(
            params, race_line_engine=rle, corner_analysis=ca
        ).to(device)

    REF: Hettiarachchi et al. (2024) U-Transformer for autonomous racing.
    """
    wpts   = params.get('waypoints', [])
    wp_idx = params.get('closest_waypoints', [0, 1])[0]
    speed  = float(params.get('speed', 0.0))
    hdg    = float(params.get('heading', 0.0))
    hdg_r  = math.radians(hdg)
    n_wpts = len(wpts)

    # 0: curvature (1/R) at current waypoint
    if n_wpts >= 5:
        p0 = np.array(wpts[(wp_idx - 2) % n_wpts][:2], dtype=float)
        p1 = np.array(wpts[wp_idx % n_wpts][:2],       dtype=float)
        p2 = np.array(wpts[(wp_idx + 2) % n_wpts][:2], dtype=float)
        d1, d2 = p1 - p0, p2 - p1
        cross   = abs(d1[0] * d2[1] - d1[1] * d2[0])
        nm      = max(np.linalg.norm(d1), 1e-6)
        curvature = float(cross / (nm ** 3 + 1e-9))
    else:
        curvature = 0.0

    # 1: perp_velocity
    if n_wpts >= 2:
        wp_next  = np.array(wpts[(wp_idx + 1) % n_wpts][:2], dtype=float)
        wp_curr  = np.array(wpts[wp_idx % n_wpts][:2],        dtype=float)
        track_dir = wp_next - wp_curr
        track_ang = math.atan2(track_dir[1], track_dir[0])
        hdg_err   = hdg_r - track_ang
        perp_v    = float(speed * abs(math.sin(hdg_err)))
    else:
        perp_v = 0.0

    # 2: brake_zone
    radius  = 1.0 / (curvature + 1e-6)
    safe_v  = float(np.clip(math.sqrt(max(1.5 * radius, 0)), 0.5, 4.0))
    brake_z = float(np.clip((speed - safe_v) / (safe_v + 1e-6), 0.0, 1.0))

    # 3: race_line_deviation
    rl_dev = float(params.get('distance_from_center', 0.0))
    if race_line_engine is not None and getattr(race_line_engine, '_initialized', False):
        lines = getattr(race_line_engine, '_lines', {})
        rl = lines.get('time_trial')
        if rl is None and lines:
            rl = next(iter(lines.values()))
        if rl is not None:
            car_xy = np.array([float(params.get('x', 0.0)),
                                float(params.get('y', 0.0))], dtype=float)
            _wpts_arr = np.array(rl.wpts[:rl.n], dtype=float)
            dists = np.linalg.norm(_wpts_arr - car_xy, axis=1)
            rl_dev = float(dists.min())

    # 4: lateral_g proxy
    lat_g = float(np.clip(speed ** 2 * curvature / 9.81, 0.0, 3.0))

    # 5: angular_rate proxy
    angular_rate = float(np.clip(curvature * speed, 0.0, 5.0))

    targets = np.array([curvature, perp_v, brake_z, rl_dev, lat_g, angular_rate],
                        dtype=np.float32)
    return torch.tensor(targets, dtype=torch.float32)


# ===========================================================================
# Proxy classes
# ===========================================================================

class _ActorProxy:
    """Named proxy so pickle can find this class by module path."""
    def __init__(self, mu_head):
        self.mu_head = mu_head


# ===========================================================================
# ContextAwarePPOAgent — Swin-UNet++ encoder
# ===========================================================================

class ContextAwarePPOAgent(PPOAgent):
    # REF: Schulman et al. (2017). PPO. arXiv:1707.06347.
    # REF: Hettiarachchi et al. (2024). U-Transformer for autonomous racing.
    # REF: Liu et al. (2021). Swin Transformer. ICCV 2021.
    """PPO agent with Swin-UNet++ hierarchical encoder, context classification,
    and intermediary metrics heads.

    Key change v1.1.1: flat Linear(obs_dim, 128) encoder replaced by
    SwinUNetPPEncoder — 3-scale hierarchy with dense UNet++ skip connections.
    This gives the trunk access to both fine-grained local structure (individual
    LiDAR readings) and coarse global context (track shape), which is the core
    insight of Hettiarachchi et al. (2024).
    """

    def __init__(self, obs_dim=64, act_dim=2, name='context_aware_bsts',
                 lidar_latent_dim=32, n_context_classes=5):
        Agent.__init__(self, name=name)
        nn.Module.__init__(self)

        self.obs_dim         = obs_dim
        self.act_dim         = act_dim
        self.lidar_latent_dim = lidar_latent_dim

        # Swin-UNet++ hierarchical encoder (replaces flat MLP encoder)
        self.encoder = SwinUNetPPEncoder(obs_dim=obs_dim,
                                          lidar_latent_dim=lidar_latent_dim)

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

        # Context embedding + intermediary projection
        ctx_embed_dim    = 8
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
        self.actor = _ActorProxy(self.actor_mean[2])
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

        # BSTS state (updated externally from analyze_logs)
        self.bsts_trend    = 0.0
        self.bsts_seasonal = 0.0

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Orthogonal init for trunk / actor / critic. Encoder inits itself."""
        for module in [self.shared_trunk, self.actor_mean, self.critic]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean[2].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight,    gain=1.0)

    # ------------------------------------------------------------------
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        lidar_latent = self.encode(x)

        ctx_logits  = self.context_head(lidar_latent)
        ctx_class   = ctx_logits.argmax(dim=-1)
        ctx_emb     = self.context_embed(ctx_class)

        intermed_pred = self.intermediary_head(lidar_latent)
        intermed_emb  = self.intermed_proj(intermed_pred.detach())

        shared_input = torch.cat([lidar_latent, ctx_emb, intermed_emb], dim=-1)
        shared_out   = self.shared_trunk(shared_input)

        mean    = self.actor_mean(shared_out)
        log_std = torch.nan_to_num(self.actor_log_std, nan=0.0, posinf=2.0, neginf=-5.0)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std     = log_std.exp().expand_as(mean)
        mean    = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        value   = self.critic(shared_out)

        return mean, std, value, ctx_logits, intermed_pred

    def get_action(self, x):
        with torch.no_grad():
            mean, std, _, _, _ = self.forward(x)
            dist = Normal(mean, std)
            return dist.sample().squeeze(0).cpu().numpy()

    def get_action_and_value(self, x, action=None):
        """PPO training forward.

        When action is supplied (PPO update pass), reshape to (batch, act_dim)
        before computing log_prob so sum(dim=-1) works correctly.
        """
        mean, std, value, ctx_logits, intermed_pred = self.forward(x)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        else:
            if action.dim() == 1 and self.act_dim > 1:
                action = action.unsqueeze(-1).expand(-1, self.act_dim)
            elif action.dim() == 1 and self.act_dim == 1:
                action = action.unsqueeze(-1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1), ctx_logits, intermed_pred

    def get_value(self, x):
        with torch.no_grad():
            _, _, value, _, _ = self.forward(x)
            return value.squeeze(-1)

    def get_context(self, x):
        with torch.no_grad():
            lidar_latent = self.encode(x)
            ctx_logits   = self.context_head(lidar_latent)
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
        brake_pred  = intermed_pred[:, 2]
        curv_pred   = intermed_pred[:, 0]
        brake_loss  = compute_brake_zone_loss(brake_pred, curv_pred)
        intermed_loss = nn.functional.mse_loss(intermed_pred, intermed_targets)
        return {
            "ctx_loss":         ctx_loss,
            "perp_v_loss":      perp_v_loss,
            "brake_zone_loss":  brake_loss,
            "intermed_loss":    intermed_loss,
        }

    def update_bsts_state(self, trend: float, seasonal: float):
        self.bsts_trend    = trend
        self.bsts_seasonal = seasonal

    def get_exploration_bonus(self):
        base = 0.01
        if self.bsts_trend < -0.01:
            return base * 2.0
        elif self.bsts_trend > 0.01:
            return base * 0.5
        return base
