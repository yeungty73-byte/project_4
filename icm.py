"""Intrinsic Curiosity Module (ICM).

REF: Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
     Curiosity-driven exploration by self-supervised prediction.
     Proceedings of the 34th ICML, PMLR 70:2778-2787.

Architecture
------------
  encoder       : obs  -> phi  (shared feature space)
  inverse_model : (phi_t, phi_t+1) -> a_hat  (keeps phi action-relevant)
  forward_model : (phi_t, a_t)     -> phi_t+1_hat
  r_i = eta * 0.5 * ||phi_t+1 - phi_t+1_hat||^2  * hotspot_weight

Hotspot weighting
-----------------
  failure_sampler.get_failure_hotspots() -> [(seg_id, crash_count), ...]
  Each step's progress% is mapped to a segment; the ICM bonus is scaled
  proportionally so the agent is more curious exactly where crashes happen.

Wiring into run.py (minimal diff)
----------------------------------
  # after agent creation:
  from icm import ICM
  icm_module = ICM(obs_dim=env.observation_space.shape[0], act_dim=2, device=str(DEVICE))
  icm_module.to(DEVICE)
  icm_optimizer = torch.optim.Adam(icm_module.parameters(), lr=3e-4)

  # inside step loop (after shaped_reward is computed):
  if _prev_obs is not None:
      r_i = icm_module.intrinsic_reward(
          _prev_obs.to(DEVICE), obs_tensor.to(DEVICE),
          action_tensor.to(DEVICE), progress_pct=_prog
      ).item()
      shaped_reward += r_i

  # after PPO update:
  icm_module.update_hotspot_weights(failure_sampler.get_failure_hotspots())
  icm_loss, fwd_l, inv_l = icm_module.loss(obs_batch, next_obs_batch, actions_batch)
  icm_optimizer.zero_grad(); icm_loss.backward(); icm_optimizer.step()
  writer.add_scalar('icm/loss', icm_loss.item(), global_step)
  writer.add_scalar('icm/forward_loss', fwd_l.item(), global_step)
  writer.add_scalar('icm/inverse_loss', inv_l.item(), global_step)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class ICMEncoder(nn.Module):
    """Maps observation -> feature vector phi.

    Kept deliberately shallow so it learns only the action-relevant
    structure of the observation (via inverse-model gradient).
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
    """Predicts a_t from (phi_t, phi_t+1).

    Gradient through this loss trains the encoder to represent
    only action-relevant state features, filtering out distractors.
    REF: Pathak et al. (2017) Section 3.
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
    """Predicts phi_t+1 from (phi_t, a_t).

    Prediction error = intrinsic reward: high where transitions are novel.
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
    """Intrinsic Curiosity Module with hotspot-weighted bonus.

    Parameters
    ----------
    obs_dim       : observation space dimensionality (from env.observation_space.shape[0])
    act_dim       : action dimensionality (2 for DeepRacer: steer + throttle)
    phi_dim       : ICM feature space size (default 64)
    beta          : weight between forward (beta) and inverse (1-beta) loss
                    REF: Pathak et al. (2017) Eq. 4 -- beta=0.2 recommended
    eta           : intrinsic reward scale (keep small: 0.005-0.02)
    hotspot_scale : max multiplier applied at worst crash segment (default 3.0)
    device        : torch device string
    """

    def __init__(
        self,
        obs_dim:       int,
        act_dim:       int   = 2,
        phi_dim:       int   = 64,
        beta:          float = 0.2,
        eta:           float = 0.01,
        hotspot_scale: float = 3.0,
        device:        str   = 'cpu',
    ):
        super().__init__()
        self.phi_dim       = phi_dim
        self.beta          = beta
        self.eta           = eta
        self.hotspot_scale = hotspot_scale
        self.device_str    = device

        self.encoder       = ICMEncoder(obs_dim, phi_dim)
        self.inverse_model = ICMInverseModel(phi_dim, act_dim)
        self.forward_model = ICMForwardModel(phi_dim, act_dim)

        # segment_id -> multiplicative weight in [1.0, hotspot_scale]
        self._hotspot_weights: dict[int, float] = {}
        self._num_segments = 10  # must match FailurePointSampler.num_segments

    # ------------------------------------------------------------------
    # Hotspot weight management
    # ------------------------------------------------------------------

    def update_hotspot_weights(self, hotspot_list: list):
        """Ingest failure_sampler.get_failure_hotspots() each episode.

        hotspot_list : [(seg_id: int, crash_count: int), ...]
        Normalises crash counts to [1.0, hotspot_scale] so the segment
        with the most crashes gets the full scale multiplier.
        """
        if not hotspot_list:
            return
        counts = {int(seg): int(cnt) for seg, cnt in hotspot_list}
        max_c = max(counts.values(), default=1)
        self._hotspot_weights = {
            seg: 1.0 + (self.hotspot_scale - 1.0) * (cnt / max(max_c, 1))
            for seg, cnt in counts.items()
        }

    def _progress_to_segment(self, progress_pct: float) -> int:
        """Map 0-100% progress to a segment index [0, num_segments-1]."""
        return min(
            int(progress_pct / (100.0 / self._num_segments)),
            self._num_segments - 1,
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    # ------------------------------------------------------------------
    # Intrinsic reward (used at step time -- detached from graph)
    # ------------------------------------------------------------------

    def intrinsic_reward(
        self,
        obs_t:        torch.Tensor,
        obs_t1:       torch.Tensor,
        action:       torch.Tensor,
        progress_pct: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute per-step scalar intrinsic reward (no gradient).

        Parameters
        ----------
        obs_t        : observation at time t,   shape (obs_dim,) or (1, obs_dim)
        obs_t1       : observation at time t+1, same shape
        action       : action taken at t,       shape (act_dim,) or (1, act_dim)
        progress_pct : current progress 0-100 for hotspot weighting

        Returns
        -------
        Scalar tensor (detached).  Add directly to shaped_reward.
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
                seg = self._progress_to_segment(float(progress_pct))
                w   = self._hotspot_weights.get(seg, 1.0)
                r_i = r_i * w

        return r_i.detach()

    # ------------------------------------------------------------------
    # Training loss (used after PPO update)
    # ------------------------------------------------------------------

    def loss(
        self,
        obs_t:   torch.Tensor,
        obs_t1:  torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ICM training losses over a minibatch.

        Parameters
        ----------
        obs_t   : (batch, obs_dim)
        obs_t1  : (batch, obs_dim)  -- next observations from rollout buffer
        actions : (batch, act_dim)

        Returns
        -------
        (total_loss, forward_loss, inverse_loss)
        Minimise total_loss with icm_optimizer.
        """
        phi_t  = self._encode(obs_t.float())
        phi_t1 = self._encode(obs_t1.float())

        # Forward loss -- prediction error IS the intrinsic reward signal
        phi_t1_hat   = self.forward_model(phi_t, actions.float())
        forward_loss = 0.5 * (phi_t1.detach() - phi_t1_hat).pow(2).mean()

        # Inverse loss -- trains encoder to be action-relevant
        action_hat   = self.inverse_model(phi_t, phi_t1)
        inverse_loss = nn.functional.mse_loss(action_hat, actions.float())

        total = self.beta * forward_loss + (1.0 - self.beta) * inverse_loss
        return total, forward_loss, inverse_loss
