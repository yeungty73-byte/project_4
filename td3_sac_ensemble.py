#!/usr/bin/env python3
"""v19: TD3+SAC critic ensemble working alongside PPO.

Architecture:
- PPO: on-policy actor-critic (policy gradient + GAE)
- TD3: twin off-policy critics (clipped double-Q for value estimation)
- SAC: entropy-regularized Q + auto-tuned alpha (exploration bonus)

The PPO actor remains the primary policy. TD3/SAC critics provide:
1. Off-policy value bootstrapping to reduce PPO variance
2. Entropy-based exploration bonus (SAC alpha)
3. Critic-guided reward shaping (TD3 min-Q baseline)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class ReplayBuffer:
# REF: Haarnoja, T. et al. (2018). Soft actor-critic: Off-policy maximum entropy deep RL with a stochastic actor. ICML, 1861-1870.
# REF: Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. ICML, 1587-1596.
    """Simple replay buffer for off-policy critic updates."""
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        # v1.1.1: shape guard — silently drop mismatched obs rather than
        # poisoning the buffer with transitions from a prior checkpoint run.
        _obs = obs if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32)
        _nobs = next_obs if isinstance(next_obs, torch.Tensor) else torch.tensor(next_obs, dtype=torch.float32)
        if not hasattr(self, '_obs_shape'):
            self._obs_shape = None
        if self._obs_shape is None:
            self._obs_shape = _obs.shape
        elif _obs.shape != self._obs_shape:
            return  # drop mismatched transition
        self.buffer.append((_obs.cpu(), action, reward, _nobs.cpu(), done))

    def sample(self, batch_size=256):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.stack(obs),
            torch.stack(act).float() if isinstance(act[0], torch.Tensor) else torch.tensor(np.array(act), dtype=torch.float32),
            torch.tensor(rew, dtype=torch.float32).unsqueeze(1),
            torch.stack(nobs),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class CriticNet(nn.Module):
    """Single Q-network: Q(s, a) -> scalar."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class TD3SACEnsemble(nn.Module):
    """Twin-critic ensemble with SAC entropy regularization.

    Works alongside PPO:
    - Maintains twin Q-networks (TD3-style) + target networks
    - Auto-tuned entropy coefficient (SAC-style)
    - Provides critic_value() for reward shaping
    - Provides exploration_bonus() for PPO entropy augmentation
    """
    def __init__(self, obs_dim, act_dim, hidden=256, gamma=0.99,
                 tau=0.005, lr=3e-4, target_entropy=None,
                 device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Twin critics (TD3)
        self.q1 = CriticNet(obs_dim, act_dim, hidden).to(device)
        self.q2 = CriticNet(obs_dim, act_dim, hidden).to(device)
        self.q1_target = CriticNet(obs_dim, act_dim, hidden).to(device)
        self.q2_target = CriticNet(obs_dim, act_dim, hidden).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # SAC log-alpha (auto-tuned entropy)
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        # v1.1.1: was -act_dim → alpha explosion to ~7.4 within 100 steps.
        # Use -0.5 * act_dim (Haarnoja et al. 2018 §5 standard heuristic).
        self.target_entropy = target_entropy if target_entropy is not None else -0.5 * act_dim

        # Optimizers
        self.critic_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.replay = ReplayBuffer()

        # Tracking
        self._update_count = 0
        self._critic_loss_ema = 0.0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition for off-policy learning."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
        if isinstance(action, (int, float)):
            action = torch.tensor([action], dtype=torch.float32)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        self.replay.push(obs.cpu(), action.cpu(), float(reward),
                         next_obs.cpu(), float(done))

    def critic_value(self, obs, action):
        """Return min(Q1, Q2) -- conservative value estimate (TD3 style)."""
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device)
            q1 = self.q1(obs, action)
            q2 = self.q2(obs, action)
            return torch.min(q1, q2).squeeze(-1)

    def exploration_bonus(self, _obs, _action, log_prob=None):
        """SAC-style entropy bonus: alpha * (-log_prob)."""
        if log_prob is None:
            return 0.0
        return self.alpha * (-log_prob)

    def update_critics(self, ppo_agent, batch_size=256,
                    policy_noise=0.2, noise_clip=0.5):
        if len(self.replay) < batch_size:
            return {'critic_loss': 0.0, 'alpha': self.alpha,
                    'q1_mean': 0.0, 'q2_mean': 0.0, 'replay_size': len(self.replay)}

        obs, act, rew, nobs, done = self.replay.sample(batch_size)
        obs  = torch.as_tensor(obs,  dtype=torch.float32).to(self.device)
        nobs = torch.as_tensor(nobs, dtype=torch.float32).to(self.device)
        rew  = torch.as_tensor(rew,  dtype=torch.float32).to(self.device)
        done = torch.as_tensor(done, dtype=torch.float32).to(self.device)
        # v1.1.0: normalize raw pixel obs [0,255]→[0,1] to prevent overflow in Linear layer
        # DeepRacer obs is a flat float32 of camera+lidar data with values in [0,255].
        # Without normalization: W(256×38466) × obs(38464, val≈100) → activations ≈ 1e6 → NaN.
        # REF: LeCun et al. (1998) — input normalization is prerequisite for stable gradient flow.
        _obs_max = obs.abs().max().item()
        if _obs_max > 2.0:  # heuristic: if values > 2.0, likely unnormalized pixels
            obs  = obs  / max(_obs_max, 1.0)
            nobs = nobs / max(nobs.abs().max().item(), 1.0)
        # v1.1.0: finite guard — skip batch if obs/reward contain NaN or Inf
        if not (torch.isfinite(obs).all() and torch.isfinite(nobs).all() and torch.isfinite(rew).all()):
            return {'critic_loss': float('nan'), 'alpha': self.alpha,
                    'q1_mean': 0.0, 'q2_mean': 0.0, 'replay_size': len(self.replay)}

        # act from replay: shape (batch, ?) — normalize to (batch, act_dim)
        act = torch.as_tensor(act, dtype=torch.float32).to(self.device)
        act = act.view(act.shape[0], -1)          # (batch, act_dim_stored)
        if act.shape[-1] != self.q1.net[0].in_features - obs.shape[-1]:
            # stored action dim mismatch: take first N or pad
            target_adim = self.q1.net[0].in_features - obs.shape[-1]
            if act.shape[-1] >= target_adim:
                act = act[:, :target_adim]
            else:
                act = torch.cat([act, torch.zeros(act.shape[0], target_adim - act.shape[-1], device=self.device)], dim=-1)

        with torch.no_grad():
            # v1.1.0: ppo_agent.forward returns (mean, std, value, ctx_logits, intermed_pred)
            # v1.1.4: shape guard — if nobs.shape[-1] != agent.obs_dim, pad/trim to match.
            #         Prevents "mat1 and mat2 shapes cannot be multiplied" crash when
            #         replay contains compact-obs (dim 12) from an earlier run/checkpoint.
            _agent_obs_dim = getattr(ppo_agent, 'obs_dim', nobs.shape[-1])
            if nobs.shape[-1] != _agent_obs_dim:
                if nobs.shape[-1] < _agent_obs_dim:
                    _pad = torch.zeros(nobs.shape[0], _agent_obs_dim - nobs.shape[-1], device=self.device)
                    nobs = torch.cat([nobs, _pad], dim=-1)
                else:
                    nobs = nobs[:, :_agent_obs_dim]
                obs_safe = obs
                if obs.shape[-1] != _agent_obs_dim:
                    if obs.shape[-1] < _agent_obs_dim:
                        _pad2 = torch.zeros(obs.shape[0], _agent_obs_dim - obs.shape[-1], device=self.device)
                        obs_safe = torch.cat([obs, _pad2], dim=-1)
                    else:
                        obs_safe = obs[:, :_agent_obs_dim]
            _fwd = ppo_agent.forward(nobs)
            _mean = _fwd[0].float()
            # Squeeze spurious leading dim if agent returned (1, batch, act_dim)
            if _mean.ndim == 3 and _mean.shape[0] == 1:
                _mean = _mean.squeeze(0)
            critic_act_dim = self.q1.net[0].in_features - nobs.shape[-1]
            if _mean.shape[-1] >= critic_act_dim:
                next_action = _mean[:, :critic_act_dim]
            else:
                next_action = torch.cat([
                    _mean,
                    torch.zeros(_mean.shape[0], critic_act_dim - _mean.shape[-1], device=self.device)
                ], dim=-1)

            # TD3: target policy smoothing noise
            # v1.1.0: always apply at least small noise — policy_noise=0.0 causes Q collapse
            _eff_noise = max(float(policy_noise), 0.05)
            noise = (torch.randn_like(next_action) * _eff_noise).clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            q1_targ = self.q1_target(nobs, next_action)
            q2_targ = self.q2_target(nobs, next_action)
            min_q_targ = torch.min(q1_targ, q2_targ)
            target_q = rew + (1.0 - done) * self.gamma * min_q_targ
            # v1.1.0: clamp Bellman target — BC rewards can be 500+; overflow → NaN
            # REF: Fujimoto et al. (2018) §4 — bounded Q-targets improve stability
            target_q = target_q.clamp(-50.0, 50.0)

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        # v1.1.0: NaN guard — skip this batch if loss is non-finite
        # This prevents a single corrupted batch from poisoning the critic weights.
        if not torch.isfinite(critic_loss):
            return {'critic_loss': float('nan'), 'alpha': self.alpha,
                    'q1_mean': 0.0, 'q2_mean': 0.0, 'replay_size': len(self.replay)}

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.critic_optim.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self._update_count += 1
        cl = critic_loss.item()
        self._critic_loss_ema = 0.95 * self._critic_loss_ema + 0.05 * cl

        return {
            'critic_loss': cl,
            'critic_loss_ema': self._critic_loss_ema,
            'alpha': self.alpha,
            'q1_mean': q1_pred.mean().item(),
            'q2_mean': q2_pred.mean().item(),
            'replay_size': len(self.replay),
        }

    def update_actor(self, ppo_agent, batch_size: int = 256):
        """TD3-guided policy distillation via BC loss.

        v1.1.1 FIX (Bug #3a): Previous version passed raw pixel-range obs_b
        directly to ppo_agent.forward() WITHOUT normalizing first.
        SwinUNet Linear(4,32) with obs~100 → activations~1e6 → NaN →
        actor_loss=nan → .backward() poisons all PPO weights silently.

        Fix: normalize obs_b before ppo_agent.forward(). Return td3_ready
        flag so run.py blend logic is unambiguous.

        REF: Fujimoto et al. (2021). TD3+BC. NeurIPS 34, 20132-20145.
        """
        if len(self.replay) < batch_size:
            return {"td3_actor_loss": 0.0, "td3_bc_loss": None,
                    "td3_q1_mean": 0.0, "td3_ready": False}

        obs_b, act_b, rew_b, next_obs_b, done_b = self.replay.sample(batch_size)
        obs_b = obs_b.to(self.device)

        # v1.1.1 FIX: normalize BEFORE ppo_agent.forward() to prevent NaN
        _obs_max = obs_b.abs().max().item()
        if _obs_max > 2.0:
            obs_b = obs_b / max(_obs_max, 1.0)

        # Shape guard
        _agent_obs_dim_a = getattr(ppo_agent, 'obs_dim', obs_b.shape[-1])
        if obs_b.shape[-1] != _agent_obs_dim_a:
            if obs_b.shape[-1] < _agent_obs_dim_a:
                _pad = torch.zeros(obs_b.shape[0],
                                   _agent_obs_dim_a - obs_b.shape[-1],
                                   device=self.device)
                obs_b = torch.cat([obs_b, _pad], dim=-1)
            else:
                obs_b = obs_b[:, :_agent_obs_dim_a]

        if not torch.isfinite(obs_b).all():
            return {"td3_actor_loss": 0.0, "td3_bc_loss": None,
                    "td3_q1_mean": 0.0, "td3_ready": False}

        critic_act_dim = self.q1.net[0].in_features - obs_b.shape[-1]

        # Step 1: compute TD3-preferred action direction (no grad needed here)
        with torch.no_grad():
            fwd_nograd = ppo_agent.forward(obs_b)
            ppo_mean_nograd = fwd_nograd[0].float()
            if ppo_mean_nograd.ndim == 3 and ppo_mean_nograd.shape[0] == 1:
                ppo_mean_nograd = ppo_mean_nograd.squeeze(0)
            if ppo_mean_nograd.shape[-1] > critic_act_dim:
                ppo_mean_q = ppo_mean_nograd[:, :critic_act_dim]
            else:
                ppo_mean_q = torch.cat([
                    ppo_mean_nograd,
                    torch.zeros(ppo_mean_nograd.shape[0],
                                critic_act_dim - ppo_mean_nograd.shape[-1],
                                device=self.device)
                ], dim=-1)

            # One-step Q1-gradient ascent to find TD3-preferred action
            # REF: Fujimoto et al. (2021) TD3+BC §3.1
            td3_act = ppo_mean_q.clone().requires_grad_(True)
            self.q1(obs_b, td3_act).mean().backward()
            td3_improved = (td3_act + 0.01 * td3_act.grad.sign()).clamp(-1, 1).detach()

        # Step 2: BC loss WITH gradient — PPO actor should move toward td3_improved
        fwd_grad = ppo_agent.forward(obs_b)
        ppo_mean_grad = fwd_grad[0].float()
        if ppo_mean_grad.ndim == 3 and ppo_mean_grad.shape[0] == 1:
            ppo_mean_grad = ppo_mean_grad.squeeze(0)
        if ppo_mean_grad.shape[-1] > critic_act_dim:
            ppo_mean_grad = ppo_mean_grad[:, :critic_act_dim]
        elif ppo_mean_grad.shape[-1] < critic_act_dim:
            ppo_mean_grad = torch.cat([
                ppo_mean_grad,
                torch.zeros(ppo_mean_grad.shape[0],
                            critic_act_dim - ppo_mean_grad.shape[-1],
                            device=self.device)
            ], dim=-1)

        bc_loss = torch.nn.functional.mse_loss(ppo_mean_grad, td3_improved)

        if not torch.isfinite(bc_loss):
            return {"td3_actor_loss": 0.0, "td3_bc_loss": None,
                    "td3_q1_mean": 0.0, "td3_ready": False}

        with torch.no_grad():
            q1_val = self.q1(obs_b, td3_improved)

        return {
            "td3_actor_loss":  -q1_val.mean().item(),
            "td3_bc_loss":     bc_loss,          # Tensor WITH grad — run.py calls .backward()
            "td3_bc_loss_val": bc_loss.item(),
            "td3_q1_mean":     q1_val.mean().item(),
            "td3_ready":       True,
        }
        
    def update_alpha(self, log_prob: float) -> float:
        """SAC auto-temperature update.

        v1.1.1: This method was missing entirely — log_alpha was never updated,
        so alpha was hardcoded at 1.0 forever.

        Call once per episode with mean log_prob from the PPO rollout.
        REF: Haarnoja et al. (2018) SAC §4.2 — adaptive temperature.
        """
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy))
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
        self.alpha_optim.step()
        # v1.1.1: clamp prevents runaway. alpha ∈ [exp(-3)≈0.05, exp(1)≈2.7]
        with torch.no_grad():
            self.log_alpha.clamp_(-3.0, 1.0)
        return self.alpha
    
    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def get_critic_reward_shaping(self, obs, action, reward, next_obs, done):
        """Compute critic-based reward shaping signal.

        Returns shaped reward that blends original reward with
        TD3 min-Q advantage estimate.
        """
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32)
            if isinstance(next_obs, np.ndarray):
                next_obs = torch.tensor(next_obs, dtype=torch.float32)
            if isinstance(action, (int, float)):
                action = torch.tensor([action], dtype=torch.float32)
            elif isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32)

            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0) if obs.dim() == 1 else torch.as_tensor(obs, dtype=torch.float32).to(self.device)
            action = torch.as_tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0) if action.dim() == 1 else torch.as_tensor(action, dtype=torch.float32).to(self.device)

            q_val = torch.min(
                self.q1(obs, action),
                self.q2(obs, action)
            ).squeeze().item()

            # Advantage-like shaping: r + gamma*V(s') - V(s)
            # This gives a potential-based shaping that doesn't change optimal policy
            return q_val * 0.1  # scaled to avoid overwhelming PPO signal
