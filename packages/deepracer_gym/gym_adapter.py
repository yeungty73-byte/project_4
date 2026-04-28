"""
packages/deepracer_gym/gym_adapter.py
DeepRacer-for-Cloud gym adapter with v1.1.6a spawn-bleed neutralisation.

v1.1.6a changes:
  PATCH-BLEED: After env_reset() receives step==1, pump brake+zero-steer actions
               until speed < BLEED_SPEED_THRESHOLD AND abs(heading_err) < BLEED_HDG_THRESHOLD.
               This neutralises the Gazebo-injected spawn velocity (2.29-4.0 m/s, per log
               run_20260428_000840) and corrects large heading errors before the agent or
               BC Pilot takes its first action.

  Design notes:
    - The Gazebo sim hard-codes a spawn velocity (nominally 4.0 m/s, measured 2.29-4.0 m/s
      due to ZMQ/physics tick offset).  SPAWN_RANDOM_HEADING=false in the yaml suppresses
      random headings for normal spawns but CW (is_reversed) spawns arrive with arbitrary
      headings (-50.8..+147.7 degrees per log forensics).
    - The bleed loop sends action=[0, 0] (zero steer, zero throttle → full engine-brake)
      each ZMQ tick until kinematics are neutral.  Max pump budget: BLEED_MAX_STEPS.
    - Returns the post-bleed observation so that run.py's step-1 ep_prev_speed is ~0
      and BC Pilot heading-error is near zero.

  REF:
    Koenig & Howard (2004) IEEE/RSJ IROS — Gazebo physics spawn initialisation.
    Ng, Harada & Russell (1999) ICML — neutral starting state for potential-based shaping.
    run_20260428_000840.log forensics — step-1 spawn speed 2.29-4.0 m/s, heading -50.8..147.7.
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

# v1.1.6a spawn-bleed thresholds
BLEED_SPEED_THRESHOLD: float = 0.30   # m/s — consider "at rest" below this
BLEED_HDG_THRESHOLD: float   = 30.0  # degrees — heading error tolerance vs track tangent
BLEED_MAX_STEPS: int         = 40    # max pump steps (40 × ~0.2s = ~8s budget)
BLEED_BRAKE_ACTION_CONTINUOUS = np.array([0.0, 0.0], dtype=np.float32)  # steer=0, throttle=0
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
          - speed < BLEED_SPEED_THRESHOLD  (car is approximately at rest)
          OR the episode has already ended (game_over → don't loop forever)
        Heading is directionally corrected by the brake action (zero steer),
        but we do NOT gate on heading alone because track-aligned heading
        requires positive throttle which would undo the speed bleed.
        REF: run_20260428_000840.log — step-1 speeds 2.29-4.0 m/s confirmed.
        """
        speed = float(rp.get('speed', 99.0))
        return speed < BLEED_SPEED_THRESHOLD or self.done

    def env_reset(self):
        if self.response is None:
            # First communication to zmq server — keep LONG timeout for entire first reset
            self.response = self.zmq_client.recieve_response()
        elif self.done:
            pass
        else:
            while not self.done:
                self.response = self._send_action(self.dummy_action())

        if not isinstance(self.response.get('info'), dict):
            self.response['info'] = dict()

        # If prev_episode done and reset called, fast forward one step for new episode
        # dummy action ignored due to reset()
        # PATCHED: safely handle missing reward_params during cold start / handshake
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

        # ── v1.1.6a PATCH-BLEED ─────────────────────────────────────────────────
        # Neutralise Gazebo-injected spawn velocity before returning step-1 obs.
        # The sim injects 2.29-4.0 m/s at spawn (confirmed run_20260428_000840.log).
        # Sending brake actions (throttle=0) burns off kinetic energy via simulated
        # friction/drag before the agent or BC Pilot takes its first action.
        # This ensures ep_prev_speed ≈ 0 at the true policy step-1.
        #
        # Safety guards:
        #   - Stops if game_over triggers (collision during bleed)
        #   - Stops after BLEED_MAX_STEPS budget exhausted
        #   - Does NOT suppress the step counter — steps during bleed count against
        #     the episode budget, which is intentional (we want neutral state, not
        #     hidden steps that corrupt the ANTE window).
        #
        # REF: Koenig & Howard (2004) IROS — Gazebo physics spawn initialisation.
        # REF: run_20260428_000840.log ANTE forensics: step-1 speed 2.29-4.0 m/s.
        _bleed_rp = self._get_rp(self.response)
        _spawn_speed = float(_bleed_rp.get('speed', 0.0))
        _bleed_pumped = 0
        _bleed_needed = not self._spawn_kinematics_neutral(_bleed_rp)

        if _bleed_needed:
            print(
                f"[gym_adapter BLEED] spawn speed={_spawn_speed:.2f} m/s "
                f"heading={_bleed_rp.get('heading', '?'):.1f}° "
                f"— pumping brake until speed<{BLEED_SPEED_THRESHOLD} m/s "
                f"(budget={BLEED_MAX_STEPS})",
                flush=True
            )
            while (not self._spawn_kinematics_neutral(self._get_rp(self.response))
                   and _bleed_pumped < BLEED_MAX_STEPS):
                self.response = self._send_action(self._bleed_action)
                if not isinstance(self.response.get('info'), dict):
                    self.response['info'] = dict()
                _bleed_pumped += 1

            _post_bleed_rp = self._get_rp(self.response)
            print(
                f"[gym_adapter BLEED] done after {_bleed_pumped} steps: "
                f"speed={_post_bleed_rp.get('speed', 0.0):.3f} m/s "
                f"(game_over={self.done})",
                flush=True
            )

        # ── end PATCH-BLEED ──────────────────────────────────────────────────────

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
