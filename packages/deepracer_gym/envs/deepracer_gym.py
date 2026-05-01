"""
packages/deepracer_gym/envs/deepracer_gym.py
v1.1.7a: thread demo_mode + arbiter_blend into send_action so RaceLineArbiter
         fires at blend=1.0 during demo() calls.

FIX-STEP-ARBITER:
  DeepracerGymEnv.step() now extracts the last reward_params from the adapter
  response and passes (arbiter_blend=self._arbiter_blend, rp=rp) to send_action.
  self._arbiter_blend defaults to 0.0 (pure RL during training).
  utils.demo() sets env.deepracer_gym_adapter ... but the cleanest interface is
  to expose set_demo_mode(blend) on the env, which demo() calls once after reset().

  REF: Heilmeier et al. (2020) Vehicle System Dynamics 58(10) -- race-line following.
       Brayshaw & Harrison (2005) AIAA/ISSMO -- braking arbiter.
"""
import os
import numpy as np
import gymnasium as gym
from loguru import logger
from gymnasium import spaces
from typing import TypeAlias, Callable
import matplotlib.pyplot as plt

from deepracer_gym.gym_adapter import DeepracerGymAdapter
from deepracer_gym.envs.utils import (
    make_action_space,
    make_observation_space,
    num_channels,
    string_to_port,
    get_host_name
)
from configs.reward_function import (
    reward_function as DEFAULT_REWARD_FUNCTION
)

ActionType: TypeAlias = (int | np.ndarray | list[float])
HOST: str = '127.0.0.1'
DEFAULT_PORT: int = 8888
PACE_DOMAIN: str = '.pace.gatech.edu'
try:
    if get_host_name().endswith(PACE_DOMAIN):
        port = string_to_port(os.environ['USER'])
    else:
        port = DEFAULT_PORT
except:
    port = DEFAULT_PORT


class DeepracerGymEnv(gym.Env):
    metadata = {
        'render_modes': ['rgb_array', 'human'],
        'render_fps': 30
    }

    def __init__(
        self,
        host: str = HOST,
        port: int = port,
        render_mode: str = 'rgb_array',
        reward_function: Callable = DEFAULT_REWARD_FUNCTION,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f'Using port {port} for deepracer server.')
        self.render_mode = render_mode
        self.action_space, self._action_metadata = make_action_space()
        self.observation_space, self._observation_metadata = make_observation_space()
        self.reward_function = reward_function

        if isinstance(self.action_space, spaces.Discrete):
            action_space_type = 'discrete'
        elif isinstance(self.action_space, spaces.Box):
            action_space_type = 'continuous'
        else:
            action_space_type = 'continuous'

        self.deepracer_gym_adapter = DeepracerGymAdapter(
            action_space_type, host=host, port=port
        )
        # v1.1.7a: arbiter blend -- 0.0 = pure RL (training), 1.0 = pure arbiter (demo)
        self._arbiter_blend: float = 0.0
        self._last_rp: dict = {}

    # -- v1.1.7a: demo-mode toggle --------------------------------------------
    def set_demo_mode(self, blend: float = 1.0) -> None:
        """Call once after reset() in demo() to enable race-line arbiter.
        blend=1.0 -> full race-line authority.  blend=0.0 -> pure RL (default).
        """
        self._arbiter_blend = float(blend)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._arbiter_blend = 0.0   # always reset to RL mode; demo() calls set_demo_mode after
        self._last_rp = {}
        observation, info = self.deepracer_gym_adapter.env_reset()
        # cache rp for first step
        if isinstance(info, dict):
            self._last_rp = info.get('reward_params', {}) or {}
        return observation, info

    def step(self, action: ActionType):
        assert self.action_space.contains(action), \
            f'Infeasible action. Action space does not contain {action}.'

        # v1.1.7a FIX-STEP-ARBITER: pass arbiter_blend + last rp to send_action
        observation, terminated, truncated, info = (
            self.deepracer_gym_adapter.send_action(
                action,
                arbiter_blend=self._arbiter_blend,
                rp=self._last_rp if self._arbiter_blend > 0.0 else None,
            )
        )

        # Update cached rp for next step
        if isinstance(info, dict):
            self._last_rp = info.get('reward_params', {}) or {}

        # PATCHED: safely handle missing reward_params
        reward_params = info.get('reward_params', {})
        if reward_params:
            reward = self.reward_function(reward_params)
        else:
            reward = 0.0

        return observation, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        observation, _, _, _ = self.deepracer_gym_adapter._parse_response(
            self.deepracer_gym_adapter.response
        )

        measurement = None
        for sensor in observation:
            if 'CAMERA' in sensor:
                measurement = observation[sensor]

        if measurement is None:
            raise ValueError(
                f'Cannot render output of sensors {list(observation.keys())}.'
            )

        channels = num_channels(measurement)
        if channels == 2:
            measurement = np.hstack((
                measurement[0, :, :], measurement[1, :, :]
            ))

        channels = num_channels(measurement)
        if channels == 1:
            measurement = np.stack(3 * (measurement,), axis=-1)
        elif channels == 3:
            measurement = measurement.transpose(1, 2, 0)

        if mode == 'human':
            plt.imshow(np.asarray(measurement))
            plt.axis('off')
        elif mode == 'rgb_array':
            return np.asarray(measurement)
