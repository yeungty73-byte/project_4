import abc
import torch
import torch.nn as nn

try:
    from transforms import EncodeObservation
except (ImportError, ModuleNotFoundError):
    EncodeObservation = None  # archived; not used in base Agent


class Agent(abc.ABC):
# REF: Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
    '''Abstract base class for all DeepRacer agents.'''

    def __init__(self, name='agent'):
        self.name = name

    @abc.abstractmethod
    def get_action(self, x):
        pass


class RandomAgent(Agent):
    '''A random agent for demonstrating usage of the environment.'''

    def __init__(self, environment, name='random'):
        super().__init__(name=name)
        self.action_space = environment.action_space

    def get_action(self, x):
        return self.action_space.sample()


class PPOAgent(Agent, nn.Module):
    '''Baseline PPO agent stub - kept for import compatibility with run.py.
    The real PPO agent is ContextAwarePPOAgent in context_aware_agent.py.
    '''

    def __init__(self, obs_dim=64, act_dim=1, name='ppo_baseline'):
        Agent.__init__(self, name=name)
        nn.Module.__init__(self)
        self.fc = nn.Linear(obs_dim, act_dim)

    def get_action(self, x):
        return self.fc(x)
