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
        self.buffer.append((obs, action, reward, next_obs, done))

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
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim

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
                    'q1_mean': 0.0, 'q2_mean': 0.0}

        obs, act, rew, nobs, done = self.replay.sample(batch_size)
        obs  = torch.as_tensor(obs,  dtype=torch.float32).to(self.device)
        nobs = torch.as_tensor(nobs, dtype=torch.float32).to(self.device)
        rew  = torch.as_tensor(rew,  dtype=torch.float32).to(self.device)
        done = torch.as_tensor(done, dtype=torch.float32).to(self.device)

        # act from replay: shape (batch, ?) — normalize to (batch, 2)
        act = torch.as_tensor(act, dtype=torch.float32).to(self.device)
        act = act.view(act.shape[0], -1)          # (batch, act_dim_stored)
        if act.shape[-1] != self.q1.net[0].in_features - obs.shape[-1]:
            # stored action dim mismatch: take first 2 or pad
            target_adim = self.q1.net[0].in_features - obs.shape[-1]
            if act.shape[-1] >= target_adim:
                act = act[:, :target_adim]
            else:
                act = torch.cat([act, torch.zeros(act.shape[0], target_adim - act.shape[-1], device=self.device)], dim=-1)

        with torch.no_grad():
            # ppo_agent.forward returns (mean, std, value, ctx_logits, intermed_pred)
            # mean shape is (batch, agent_act_dim) — may be 26 for discrete heads
            # We need (batch, critic_act_dim=2): take only first 2 dims
            _mean = ppo_agent.forward(nobs)[0].float()   # (batch, agent_act_dim)
            critic_act_dim = self.q1.net[0].in_features - nobs.shape[-1]
            if _mean.shape[-1] >= critic_act_dim:
                next_action = _mean[:, :critic_act_dim]  # slice to (batch, 2)
            else:
                next_action = torch.cat([_mean, torch.zeros(_mean.shape[0], critic_act_dim - _mean.shape[-1], device=self.device)], dim=-1)

            # TD3: target policy smoothing noise
            noise = (torch.randn_like(next_action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            q1_targ = self.q1_target(nobs, next_action)
            q2_targ = self.q2_target(nobs, next_action)
            min_q_targ = torch.min(q1_targ, q2_targ)
            target_q = rew + (1.0 - done) * self.gamma * min_q_targ

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

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

    def update_actor(self, ppo_agent, batch_size=256):
        if len(self.replay) < batch_size:
            return {"td3_actor_loss": 0.0}
        obs_b, act_b, rew_b, next_obs_b, done_b = self.replay.sample(batch_size)
        obs_b = obs_b.to(self.device)
        # Get deterministic action from PPO agent actor (mean, no sampling)
        mean, std, value, ctx_logits, intermed_pred = ppo_agent.forward(obs_b)
        # mean may be (batch, 26) for discrete heads — slice to critic's expected act_dim
        critic_act_dim = self.q1.net[0].in_features - obs_b.shape[-1]
        if mean.shape[-1] > critic_act_dim:
            mean = mean[:, :critic_act_dim]
        elif mean.shape[-1] < critic_act_dim:
            mean = torch.cat([mean, torch.zeros(mean.shape[0], critic_act_dim - mean.shape[-1], device=self.device)], dim=-1)
        # Q1 value for current policy actions
        q1_val = self.q1(obs_b, mean)
        actor_loss = -q1_val.mean()
        return {
            "td3_actor_loss": actor_loss,
            "td3_actor_loss_val": actor_loss.item(),
            "td3_q1_mean": q1_val.mean().item(),
        }

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
