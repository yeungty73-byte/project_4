"""TD3-Only Agent for DeepRacer v200.

Pure Twin Delayed DDPG (TD3) implementation. Ablates SAC and PPO.
REF: Fujimoto et al., "Addressing Function Approximation Error in
     Actor-Critic Methods", ICML 2018. \cite{fujimoto2018addressing}
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # actions in [-1, 1]


class Critic(nn.Module):
    """Twin Q-networks for TD3."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.q1_fc2 = nn.Linear(hidden, hidden)
        self.q1_out = nn.Linear(hidden, 1)
        # Q2
        self.q2_fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.q2_fc2 = nn.Linear(hidden, hidden)
        self.q2_out = nn.Linear(hidden, 1)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2

    def q1(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_out(q1)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=500_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action if hasattr(action, '__len__') else [action]
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(DEVICE),
            torch.FloatTensor(self.actions[idx]).to(DEVICE),
            torch.FloatTensor(self.rewards[idx]).to(DEVICE),
            torch.FloatTensor(self.next_obs[idx]).to(DEVICE),
            torch.FloatTensor(self.dones[idx]).to(DEVICE),
        )


class TD3Agent:
    """Pure TD3 agent with delayed policy updates and target smoothing.

    REF: \cite{fujimoto2018addressing}
    """
    def __init__(self, obs_dim, act_dim, hidden=256, gamma=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, lr_actor=3e-4, lr_critic=3e-4):
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_updates = 0

        self.actor = Actor(obs_dim, act_dim, hidden).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(obs_dim, act_dim)

    def select_action(self, obs, noise_scale=0.1):
        """Select action with exploration noise."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        if noise_scale > 0:
            action += np.random.normal(0, noise_scale, size=self.act_dim)
        return np.clip(action, -1.0, 1.0)

    def update(self, batch_size=256):
        """Single TD3 update step. Returns dict of losses."""
        if self.replay_buffer.size < batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)
        self.total_updates += 1

        # --- Critic update ---
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = rewards + (1.0 - dones) * self.gamma * torch.min(q1_target, q2_target)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        info = {'critic_loss': critic_loss.item()}

        # --- Delayed policy update ---
        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft target updates
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1.0 - self.tau) * pt.data)
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1.0 - self.tau) * pt.data)

            info['actor_loss'] = actor_loss.item()

        return info

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'total_updates': self.total_updates,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.total_updates = ckpt.get('total_updates', 0)
