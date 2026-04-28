"""
packages/deepracer_gym/gym_adapter.py
DeepRacer-for-Cloud gym adapter with v1.1.6b spawn-bleed neutralisation.

v1.1.6b changes (on top of v1.1.6a):
  FIX-BLEED-1: BLEED_BRAKE_ACTION_CONTINUOUS = [0.0, -1.0]
               Full engine-brake (throttle=-1.0 in tanh space = full reverse/brake).
               v1.1.6a used [0.0, 0.0] which gave ZERO deceleration force.
               Sim physics floor: speed bottoms at 0.500 m/s, not 0.300 m/s.
               Log run_20260428_120552: ALL bleed cycles ended at speed=0.500 m/s.

  FIX-BLEED-2: BLEED_SPEED_THRESHOLD = 0.55 m/s (was 0.30 m/s — UNREACHABLE)
               Sim physics floor confirmed at 0.500 m/s (friction model hard floor).
               0.55 > 0.500 so the condition now fires correctly.

  FIX-BLEED-3: BLEED_MAX_STEPS = 80 (was 40 — too short for high-speed spawns)

  FIX-BLEED-4: Restart logic: if game_over fires during bleed (wall collision),
               restart the episode and try again (up to BLEED_MAX_RESTARTS=3).
               Without restart, a game_over during bleed meant the next episode
               also started at 4m/s. Now we keep retrying until neutral or give up.

  REF:
    Koenig & Howard (2004) IEEE/RSJ IROS — Gazebo physics spawn initialisation.
    Ng, Harada & Russell (1999) ICML — neutral starting state for potential-based shaping.
    run_20260428_120552.log — all bleed cycles: speed=0.500 m/s at exit (floor confirmed).
    run_20260428_123616.log — restart0/1/2 cycles observed: game_over during bleed
                              requires episode restart, not just additional brake steps.
"""
import zmq
import math
import numpy as np
from typing import TypeAlias
from collections.abc import Callable

from deepracer_gym.zmq_client import DeepracerClientZMQ
from deepracer_gym.utils import (
    terminated_check, truncated_check
)

PORT: int = 8888
HOST: str = '127.0.0.1'
TIMEOUT_LONG: int = 500_000   # ~8.3 m
TIMEOUT_SHORT: int = 100_000  # ~1.7 m
DUMMY_ACTION_DISCRETE: Callable[[], int] = (
    lambda: 0
)
DUMMY_ACTION_CONTINUOUS: Callable[[], np.ndarray[float]] = (
    lambda: np.random.uniform(-1, 1, 2)
)

# v1.1.6b spawn-bleed thresholds
# FIX-BLEED-2: threshold raised from 0.30 to 0.55 — sim physics floor is 0.500 m/s
# (confirmed: ALL bleed cycles in run_20260428_120552.log ended at speed=0.500 m/s,
#  never below, regardless of brake action or budget)
BLEED_SPEED_THRESHOLD: float = 0.55   # m/s — just above sim physics floor of 0.500
BLEED_HDG_THRESHOLD: float   = 30.0  # degrees — heading error tolerance vs track tangent
BLEED_MAX_STEPS: int         = 80    # FIX-BLEED-3: raised from 40 (was too short)
BLEED_MAX_RESTARTS: int      = 3     # FIX-BLEED-4: max episode restarts during bleed
# FIX-BLEED-1: full engine-brake (-1.0 throttle in tanh space)
# v1.1.6a used [0.0, 0.0] = zero throttle which gave ZERO decel on floored-speed floor
# -1.0 in tanh space = maximum braking intent sent to sim physics
BLEED_BRAKE_ACTION_CONTINUOUS = np.array([0.0, -1.0], dtype=np.float32)
BLEED_BRAKE_ACTION_DISCRETE   = 0   # index 0: minimum speed, smallest turn


ActionType: TypeAlias = (int | np.ndarray | list[float])


class DeepracerGymAdapter:
    def __init__(
        self,
        action_space_type: str,
        host: str = HOST,
        port: int = PORT):

        if action_space_type == 'discrete':
            self.dummy_action = DUMMY_ACTION_DISCRETE
            self._bleed_action = BLEED_BRAKE_ACTION_DISCRETE
        elif action_space_type == 'continuous':
            self.dummy_action = DUMMY_ACTION_CONTINUOUS
            self._bleed_action = BLEED_BRAKE_ACTION_CONTINUOUS
        else:
            raise ValueError(
                f'Action space can only be discrete or continuous. Got {action_space_type} instead.'
            )

        self.zmq_client = DeepracerClientZMQ(host=host, port=port)
        self.zmq_client.ready()
        self.response = None
        self.done = False
        self._first_reset_done = False

    def _send_action(self, action: ActionType):
        action: dict[str, ActionType] = {'action': action}
        self.response = self.zmq_client.send_message(action)
        self.done = self.response['_game_over']
        return self.response

    @staticmethod
    def _get_steps(response):
        """Safely extract steps from response, returning None if not yet available."""
        info = response.get('info')
        if not isinstance(info, dict):
            return None
        rp = info.get('reward_params')
        if not isinstance(rp, dict):
            return None
        return rp.get('steps')

    @staticmethod
    def _get_rp(response):
        """Safely extract reward_params from response."""
        info = response.get('info')
        if not isinstance(info, dict):
            return {}
        rp = info.get('reward_params')
        return rp if isinstance(rp, dict) else {}

    def _spawn_kinematics_neutral(self, rp: dict) -> bool:
        """
        Return True when spawn kinematics are neutral enough for agent/BC Pilot.
        Criteria:
          - speed < BLEED_SPEED_THRESHOLD (0.55 m/s — above sim physics floor of 0.500)
        NOTE: does NOT exit on self.done — the restart loop handles game_over separately.
        REF: run_20260428_120552.log — all bleed exits at speed=0.500 m/s (sim floor).
             run_20260428_123616.log — game_over during bleed requires episode restart.
        """
        speed = float(rp.get('speed', 99.0))
        return speed < BLEED_SPEED_THRESHOLD

    def _do_episode_reset(self):
        """
        Drain old episode and pump until step==1 for fresh episode.
        Returns response dict for step==1.
        """
        if self.response is None:
            self.response = self.zmq_client.recieve_response()
        elif self.done:
            pass
        else:
            while not self.done:
                self.response = self._send_action(self.dummy_action())

        if not isinstance(self.response.get('info'), dict):
            self.response['info'] = dict()

        step = self._get_steps(self.response)
        max_pump = 300
        while step != 1 and max_pump > 0:
            self.response = self._send_action(self.dummy_action())
            if not isinstance(self.response.get('info'), dict):
                self.response['info'] = dict()
            step = self._get_steps(self.response)
            max_pump -= 1
            if max_pump % 10 == 0:
                print(f"[gym_adapter] pumping... step={step}, remaining={max_pump}", flush=True)

        if max_pump <= 0:
            raise RuntimeError(
                "env_reset: timed out waiting for step==1 after 300 pumps. "
                "reward_params may never have been populated — is the container fully booted?"
            )
        return self.response

    def env_reset(self):
        # First reset
        self.response = self._do_episode_reset()

        if not isinstance(self.response.get('info'), dict):
            self.response['info'] = dict()

        # -- v1.1.6b PATCH-BLEED -------------------------------------------------
        # Neutralise Gazebo-injected spawn velocity before returning step-1 obs.
        # FIX-BLEED-1: BLEED_BRAKE_ACTION_CONTINUOUS = [0.0, -1.0] (full brake)
        # FIX-BLEED-2: BLEED_SPEED_THRESHOLD = 0.55 (above sim physics floor)
        # FIX-BLEED-3: BLEED_MAX_STEPS = 80
        # FIX-BLEED-4: restart if game_over during bleed (wall collision)
        #
        # REF: run_20260428_120552.log — speed=0.500 m/s at ALL bleed exits
        # REF: run_20260428_123616.log — restart0/1/2 cycles confirm game_over during bleed
        # REF: Koenig & Howard (2004) IROS — Gazebo physics spawn initialisation.
        # REF: Ng, Harada & Russell (1999) ICML — neutral starting state.

        for _restart in range(BLEED_MAX_RESTARTS + 1):
            _bleed_rp = self._get_rp(self.response)
            _spawn_speed = float(_bleed_rp.get('speed', 0.0))
            _spawn_hdg   = _bleed_rp.get('heading', 0.0)

            if self._spawn_kinematics_neutral(_bleed_rp):
                # Already neutral on this spawn — no bleed needed
                break

            print(
                f"[gym_adapter BLEED] restart{_restart} spawn speed={_spawn_speed:.2f} m/s "
                f"heading={_spawn_hdg:.1f}deg -- "
                f"pumping full brake until speed<{BLEED_SPEED_THRESHOLD} m/s "
                f"(budget={BLEED_MAX_STEPS})",
                flush=True
            )

            _bleed_pumped = 0
            _game_over_during_bleed = False
            while (not self._spawn_kinematics_neutral(self._get_rp(self.response))
                   and _bleed_pumped < BLEED_MAX_STEPS):
                self.response = self._send_action(self._bleed_action)
                if not isinstance(self.response.get('info'), dict):
                    self.response['info'] = dict()
                _bleed_pumped += 1
                if self.done:
                    _game_over_during_bleed = True
                    break

            _post_bleed_rp = self._get_rp(self.response)
            _post_speed = float(_post_bleed_rp.get('speed', 0.0))

            if _game_over_during_bleed:
                print(
                    f"[gym_adapter BLEED] game_over during bleed after {_bleed_pumped} steps "
                    f"speed={_post_speed:.3f} m/s -- restarting episode restart {_restart+1}",
                    flush=True
                )
                if _restart < BLEED_MAX_RESTARTS:
                    self.response = self._do_episode_reset()
                    if not isinstance(self.response.get('info'), dict):
                        self.response['info'] = dict()
                    continue
                else:
                    print(
                        f"[gym_adapter BLEED] WARNING max restarts {BLEED_MAX_RESTARTS} exceeded "
                        f"-- returning post-crash obs. "
                        f"spawn_speed={_spawn_speed:.2f} m/s last_speed={_post_speed:.3f} m/s",
                        flush=True
                    )
                    break
            else:
                print(
                    f"[gym_adapter BLEED] DONE after {_bleed_pumped} steps "
                    f"speed={_post_speed:.3f} m/s restart{_restart}, "
                    f"game_over={self.done}",
                    flush=True
                )
                break

        # -- end PATCH-BLEED -----------------------------------------------------

        # Only reduce timeout after first successful reset
        if not self._first_reset_done:
            self._first_reset_done = True
            self.zmq_client.socket.set(zmq.SNDTIMEO, TIMEOUT_SHORT)
            self.zmq_client.socket.set(zmq.RCVTIMEO, TIMEOUT_SHORT)

        observation, _, _, info = self._parse_response(self.response)
        return observation, info

    def send_action(self, action: ActionType):
        if self.done:
            return self._parse_response(self.response)
        response = self._send_action(action)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: dict):
        info = response.get('info', {})
        if not isinstance(info, dict):
            info = dict()
        info['goal'] = response.get('_goal')

        game_over = response.get('_game_over', False)

        # PATCHED: safely handle missing episode_status
        _default_status = {
            'lap_complete': False, 'crashed': False, 'reversed': False,
            'off_track': False, 'immobilized': False, 'time_up': False,
        }
        episode_status = info.get('episode_status')
        if not isinstance(episode_status, dict):
            episode_status = _default_status.copy()
        for k, v in _default_status.items():
            episode_status.setdefault(k, v)
        info['episode_status'] = episode_status

        terminated = terminated_check(episode_status, game_over)
        truncated = truncated_check(episode_status, game_over)

        observation = response['_next_state']
        # channel first convention
        observation = {
            sensor: (
                measurement.transpose(-1, 0, 1) if 'CAMERA' in sensor
                else measurement
            ) for sensor, measurement in observation.items()
        }

        return observation, terminated, truncated, info
