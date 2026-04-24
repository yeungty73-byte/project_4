"""
Intrinsic Curiosity Module (ICM) — Pathak et al. (2017)
========================================================
REF: Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
     Curiosity-driven exploration by self-supervised prediction.
     Proceedings of the 34th ICML, PMLR 70:2778-2787.

Architecture
------------
  encoder       : obs  -> phi  (shared latent; shallow so it stays action-relevant)
  inverse_model : (phi_t, phi_t+1) -> a_hat  (trains encoder to ignore distractors)
  forward_model : (phi_t, a_t)     -> phi_t+1_hat
  r_i = eta * 0.5 * ||phi_t+1 - phi_t+1_hat||^2  *  hotspot_weight(segment)

Hotspot weighting (failure-aware exploration)
---------------------------------------------
  The agent gets MORE curiosity bonus exactly where crashes happen most:
    failure_sampler.get_failure_hotspots() -> [(seg_id, crash_count), ...]
    update_hotspot_weights() normalises crash counts to [1.0, hotspot_scale]
    so the worst crash segment gets a 3x bonus by default.

run.py integration (search ICM_HOOK in run.py)
----------------------------------------------
  # --- ICM_HOOK: after agent creation ---
  from icm import ICM
  icm_module = ICM(
      obs_dim        = env.observation_space.shape[0],
      act_dim        = 2,
      phi_dim        = 64,
      eta            = 0.01,
      hotspot_scale  = 3.0,
  ).to(DEVICE)
  icm_optimizer = torch.optim.Adam(icm_module.parameters(), lr=3e-4)

  # --- ICM_HOOK: inside step loop, after shaped_reward is computed ---
  if _prev_obs is not None:
      r_i = icm_module.intrinsic_reward(
          _prev_obs.to(DEVICE), obs_tensor.to(DEVICE),
          action_tensor.float().to(DEVICE),
          progress_pct = _prog,
      ).item()
      shaped_reward += r_i
      ep_icm_sum = ep_icm_sum + r_i   # for logging

  # --- ICM_HOOK: after PPO update, inside training loop ---
  icm_module.update_hotspot_weights(failure_sampler.get_failure_hotspots())
  # obs_batch / next_obs_batch / actions_batch come from the PPO rollout buffer
  icm_loss, fwd_l, inv_l = icm_module.loss(obs_batch, next_obs_batch, actions_batch)
  icm_optimizer.zero_grad()
  icm_loss.backward()
  torch.nn.utils.clip_grad_norm_(icm_module.parameters(), 0.5)
  icm_optimizer.step()
  writer.add_scalar('icm/total_loss',   icm_loss.item(), global_step)
  writer.add_scalar('icm/forward_loss', fwd_l.item(),    global_step)
  writer.add_scalar('icm/inverse_loss', inv_l.item(),    global_step)
  writer.add_scalar('icm/ep_bonus_sum', ep_icm_sum,      global_step)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class ICMEncoder(nn.Module):
    """Observation -> latent phi.

    Deliberately shallow: the inverse-model gradient strips out
    distractors (track appearance, lighting) so only action-relevant
    dynamics survive in phi.
    REF: Pathak et al. (2017) Section 3 — feature space phi.
    """

    def __init__(self, obs_dim: int, phi_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, phi_dim),
            nn.ELU(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ICMInverseModel(nn.Module):
    """(phi_t, phi_t+1) -> predicted a_t.

    Gradient through this loss trains the encoder to represent
    ONLY the information needed to infer what action caused the
    transition — filtering visual distractors.
    REF: Pathak et al. (2017) Equation 2.
    """

    def __init__(self, phi_dim: int = 64, act_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phi_dim * 2, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, phi_t: torch.Tensor, phi_t1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi_t, phi_t1], dim=-1))


class ICMForwardModel(nn.Module):
    """(phi_t, a_t) -> predicted phi_t+1.

    Prediction error = intrinsic reward r_i.
    High r_i means the transition was NOVEL / SURPRISING to the model.
    This naturally spikes at the first-quarter crash zone because the
    model has never seen those state transitions successfully resolved.
    REF: Pathak et al. (2017) Equation 3.
    """

    def __init__(self, phi_dim: int = 64, act_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phi_dim + act_dim, 128),
            nn.ELU(),
            nn.Linear(128, phi_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, phi_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi_t, action], dim=-1))


# ---------------------------------------------------------------------------
# Full ICM module
# ---------------------------------------------------------------------------

class ICM(nn.Module):
    """
    Intrinsic Curiosity Module with failure-hotspot weighted bonuses.

    Parameters
    ----------
    obs_dim        : env.observation_space.shape[0]
    act_dim        : 2 for DeepRacer (steer + throttle)
    phi_dim        : ICM latent dim, default 64
    beta           : forward/inverse loss split, 0.2 from Pathak et al.
    eta            : intrinsic reward scale (0.005 – 0.02; tune vs extrinsic)
    hotspot_scale  : max bonus multiplier at worst crash segment (default 3.0)
    num_segments   : must match FailurePointSampler.num_segments (default 10)
    """

    def __init__(
        self,
        obs_dim:        int,
        act_dim:        int   = 2,
        phi_dim:        int   = 64,
        beta:           float = 0.2,
        eta:            float = 0.01,
        hotspot_scale:  float = 3.0,
        num_segments:   int   = 10,
    ):
        super().__init__()
        self.phi_dim        = phi_dim
        self.act_dim        = act_dim
        self.beta           = beta
        self.eta            = eta
        self.hotspot_scale  = hotspot_scale
        self._num_segments  = num_segments

        self.encoder       = ICMEncoder(obs_dim, phi_dim)
        self.inverse_model = ICMInverseModel(phi_dim, act_dim)
        self.forward_model = ICMForwardModel(phi_dim, act_dim)

        # segment_id -> multiplier in [1.0, hotspot_scale]
        self._hotspot_weights: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Hotspot management
    # ------------------------------------------------------------------

    def update_hotspot_weights(self, hotspot_list: list):
        """
        Ingest failure_sampler.get_failure_hotspots() each episode end.
        hotspot_list : [(seg_id, crash_count), ...]

        Normalises counts to [1.0, hotspot_scale] so worst segment
        gets the full multiplier.  Safe to call with empty list.
        """
        if not hotspot_list:
            return
        counts  = {int(seg): int(cnt) for seg, cnt in hotspot_list}
        max_c   = max(counts.values(), default=1)
        self._hotspot_weights = {
            seg: 1.0 + (self.hotspot_scale - 1.0) * (cnt / max(max_c, 1))
            for seg, cnt in counts.items()
        }

    def _segment(self, progress_pct: float) -> int:
        """Map 0–100% progress -> segment id [0, num_segments-1]."""
        return min(
            int(progress_pct / (100.0 / self._num_segments)),
            self._num_segments - 1,
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    # ------------------------------------------------------------------
    # Intrinsic reward  (per-step, no graph retained)
    # ------------------------------------------------------------------

    def intrinsic_reward(
        self,
        obs_t:        torch.Tensor,
        obs_t1:       torch.Tensor,
        action:       torch.Tensor,
        progress_pct: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute scalar intrinsic reward r_i for a single transition.
        Returns a detached scalar tensor — safe to .item() and add to reward.

        r_i = eta * 0.5 * ||phi_t+1 - phi_hat_t+1||^2  * hotspot_weight
        REF: Pathak et al. (2017) Eq. 3.

        Parameters
        ----------
        obs_t        : shape (obs_dim,) or (1, obs_dim)
        obs_t1       : shape (obs_dim,) or (1, obs_dim)
        action       : shape (act_dim,) or (1, act_dim)
        progress_pct : current track progress 0–100 for hotspot scaling
        """
        with torch.no_grad():
            obs_t   = obs_t.float().reshape(1, -1)
            obs_t1  = obs_t1.float().reshape(1, -1)
            action  = action.float().reshape(1, -1)

            phi_t      = self._encode(obs_t)
            phi_t1     = self._encode(obs_t1)
            phi_t1_hat = self.forward_model(phi_t, action)

            r_i = self.eta * 0.5 * (phi_t1 - phi_t1_hat).pow(2).mean()

            if progress_pct is not None:
                seg = self._segment(float(progress_pct))
                w   = self._hotspot_weights.get(seg, 1.0)
                r_i = r_i * w

        return r_i.detach()

    # ------------------------------------------------------------------
    # Training loss  (after PPO update, on rollout minibatch)
    # ------------------------------------------------------------------

    def loss(
        self,
        obs_t:   torch.Tensor,
        obs_t1:  torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple:
        """
        Compute ICM losses over a training minibatch.

        total = beta * forward_loss + (1 - beta) * inverse_loss
        REF: Pathak et al. (2017) Eq. 4.

        Parameters
        ----------
        obs_t   : (batch, obs_dim)
        obs_t1  : (batch, obs_dim)   next-obs from rollout buffer
        actions : (batch, act_dim)

        Returns
        -------
        (total_loss, forward_loss, inverse_loss)  — all differentiable.
        """
        phi_t   = self._encode(obs_t.float())
        phi_t1  = self._encode(obs_t1.float())

        # Forward loss: model's ability to predict next latent state
        phi_t1_hat   = self.forward_model(phi_t, actions.float())
        forward_loss = 0.5 * (phi_t1.detach() - phi_t1_hat).pow(2).mean()

        # Inverse loss: recover action from state pair
        action_hat   = self.inverse_model(phi_t, phi_t1)
        inverse_loss = nn.functional.mse_loss(action_hat, actions.float())

        total = self.beta * forward_loss + (1.0 - self.beta) * inverse_loss
        return total, forward_loss, inverse_loss

    # ------------------------------------------------------------------
    # Entropy boost passthrough (wires StuckTracker.entropy_boost)
    # ------------------------------------------------------------------

    def apply_entropy_boost(
        self,
        log_std_param: nn.Parameter,
        entropy_boost: float,
        max_boost:     float = 0.5,
    ):
        """
        Apply StuckTracker entropy_boost to actor log_std directly.

        Previously entropy_boost was computed in StuckTracker but
        silently discarded.  Call this after get_annealing_params():

            params = stuck_tracker.get_annealing_params(wp_idx)
            icm_module.apply_entropy_boost(
                agent.actor_log_std, params['entropy_boost']
            )

        Clamps to max_boost to prevent runaway exploration.
        REF: Haarnoja et al. (2018) SAC — entropy regularisation.
        """
        boost = float(min(entropy_boost, max_boost))
        if boost > 0.0:
            with torch.no_grad():
                log_std_param.data.add_(boost * 0.1)   # gentle nudge
                log_std_param.data.clamp_(-3.0, 1.0)   # stay sane
