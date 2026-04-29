"""
packages/deepracer_gym/gym_adapter.py
DeepRacer-for-Cloud gym adapter with v1.1.6c spawn-bleed + heading alignment.

v1.1.6b changes (on top of v1.1.6a):
  FIX-BLEED-1: BLEED_BRAKE_ACTION_CONTINUOUS = [0.0, -1.0]
               Full engine-brake (throttle=-1.0 in tanh space = full reverse/brake).
               v1.1.6a used [0.0, 0.0] which gave ZERO deceleration force.
               Sim physics floor: speed bottoms at 0.500 m/s, not 0.300 m/s.
               Log run_20260428_120552: ALL bleed cycles ended at speed=0.500 m/s.

  FIX-BLEED-2: BLEED_SPEED_THRESHOLD = 0.55 m/s (was 0.30 m/s -- UNREACHABLE)
               Sim physics floor confirmed at 0.500 m/s (friction model hard floor).
               0.55 > 0.500 so the condition now fires correctly.

  FIX-BLEED-3: BLEED_MAX_STEPS = 80 (was 40 -- too short for high-speed spawns)

  FIX-BLEED-4: Restart logic: if game_over fires during bleed (wall collision),
               restart the episode and try again (up to BLEED_MAX_RESTARTS=3).

v1.1.6c changes (on top of v1.1.6b):
  FIX-HDGALIGN: After speed bleed, run a heading-alignment phase.
                Computes track tangent from waypoints[closest_waypoints].
                Applies proportional steering at crawl throttle to reduce
                heading error toward track direction before returning obs.
                Confirmed need: run_20260429_172738.log -- ALL post-bleed
                headings remain arbitrary (-50.8 deg to +180 deg) causing immediate
                wall crashes (ep_len 15-20 steps, vperp=3.7 m/s at crash).
                After HDGALIGN, does a brief re-bleed to return speed to floor.

  BUG-C1 FIX:  _bc_spawn_neutral in run.py used threshold 0.30 < physics floor
               0.500 -> always False -> 0 BC episodes. Fixed via apply_v116c_patches.py
               (sets _bc_spawn_neutral = True).

REF:
  Koenig & Howard (2004) IEEE/RSJ IROS -- Gazebo physics spawn initialisation.
  Ng, Harada & Russell (1999) ICML -- neutral starting state for potential-based shaping.
  run_20260428_000840.log -- step-1 speed 2.29-4.0 m/s, heading -50.8 to +147.7 deg.
  run_20260428_120552.log -- ALL bleed cycles ended speed=0.500 m/s (floor confirmed).
  run_20260429_172738.log -- post-bleed heading still arbitrary: BC harvest 0 eps.
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
BLEED_SPEED_THRESHOLD: float = 0.55   # m/s -- just above sim physics floor of 0.500
BLEED_HDG_THRESHOLD: float   = 25.0   # degrees -- heading error tolerance vs track tangent
BLEED_MAX_STEPS: int         = 80     # FIX-BLEED-3: raised from 40
BLEED_MAX_RESTARTS: int      = 3      # FIX-BLEED-4: max episode restarts during bleed
HDGALIGN_MAX_STEPS: int      = 60     # v1.1.6c: heading alignment phase max steps
HDGALIGN_THROTTLE: float     = 0.25   # v1.1.6c: crawl throttle during heading phase
HDGALIGN_STEER_GAIN: float   = 0.70   # v1.1.6c: proportional steer gain on heading error

# FIX-BLEED-1: full engine-brake (-1.0 throttle in tanh space)
BLEED_BRAKE_ACTION_CONTINUOUS = np.array([0.0, -1.0], dtype=np.float32)
BLEED_BRAKE_ACTION_DISCRETE   = 0   # index 0: minimum speed, smallest turn


ActionType: TypeAlias = (int | np.ndarray | list[float])


def _compute_heading_error(heading_deg: float, waypoints, closest_waypoints) -> float:
    """
    Compute signed heading error (degrees) relative to track tangent.
    Positive = car pointing CCW of tangent (steer CW/right to correct).
    Negative = car pointing CW of tangent (steer CCW/left to correct).
    Returns 0.0 if waypoints are invalid.

    REF: run_20260429_172738.log -- post-bleed headings -50.8 to +180 deg
         causing immediate crashes (heading ~50-90 deg off track tangent).
    """
    try:
        wps = waypoints
        ci = closest_waypoints
        if wps is None or ci is None or len(wps) < 2 or len(ci) < 2:
            return 0.0
        idx_a = int(ci[0])
        idx_b = int(ci[1])
        if idx_a >= len(wps) or idx_b >= len(wps):
            return 0.0
        wp_a = wps[idx_a]
        wp_b = wps[idx_b]
        if isinstance(wp_a, (list, tuple)):
            ax, ay = float(wp_a[0]), float(wp_a[1])
            bx, by = float(wp_b[0]), float(wp_b[1])
        else:
            ax, ay = float(wps[2 * idx_a]), float(wps[2 * idx_a + 1])
            bx, by = float(wps[2 * idx_b]), float(wps[2 * idx_b + 1])
        tangent_deg = math.degrees(math.atan2(by - ay, bx - ax))
        err = heading_deg - tangent_deg
        while err > 180.0:
            err -= 360.0
        while err < -180.0:
            err += 360.0
        return err
    except Exception:
        return 0.0


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
        Return True when spawn speed is at/below physics floor.
        REF: run_20260428_120552.log -- all bleed exits at speed=0.500 m/s (sim floor).
        """
        speed = float(rp.get('speed', 99.0))
        return speed < BLEED_SPEED_THRESHOLD

    def _heading_is_aligned(self, rp: dict) -> bool:
        """
        Return True when heading is within BLEED_HDG_THRESHOLD of track tangent.
        v1.1.6c: Used to gate HDGALIGN phase.
        REF: run_20260429_172738.log -- post-bleed headings -50.8 to +180 deg (all wrong).
        """
        heading = float(rp.get('heading', 0.0))
        wps = rp.get('waypoints')
        ci  = rp.get('closest_waypoints')
        err = _compute_heading_error(heading, wps, ci)
        return abs(err) <= BLEED_HDG_THRESHOLD

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
                "reward_params may never have been populated -- is the container fully booted?"
            )
        return self.response

    def env_reset(self):
        self.response = self._do_episode_reset()

        if not isinstance(self.response.get('info'), dict):
            self.response['info'] = dict()

        # -- v1.1.6b PATCH-BLEED / v1.1.6c PATCH-HDGALIGN ----------------------
        # Phase 1: Speed Bleed -- neutralise Gazebo-injected spawn velocity.
        # Phase 2: Heading Align -- steer car to track tangent direction.
        # Phase 3: Re-bleed speed after heading correction.
        #
        # REF: run_20260428_120552.log -- speed=0.500 m/s at ALL bleed exits
        # REF: run_20260429_172738.log -- post-bleed heading -50.8 to +180 deg
        # REF: Koenig & Howard (2004) IROS -- Gazebo physics spawn.
        # REF: Ng et al. (1999) ICML -- neutral starting state.

        for _restart in range(BLEED_MAX_RESTARTS + 1):
            _bleed_rp = self._get_rp(self.response)
            _spawn_speed = float(_bleed_rp.get('speed', 0.0))
            _spawn_hdg   = _bleed_rp.get('heading', 0.0)

            # Phase 1: Speed Bleed
            if not self._spawn_kinematics_neutral(_bleed_rp):
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
                            f"spawn_speed={_spawn_speed:.2f} last_speed={_post_speed:.3f}",
                            flush=True
                        )
                        break
                else:
                    print(
                        f"[gym_adapter BLEED] DONE after {_bleed_pumped} steps "
                        f"speed={_post_speed:.3f} m/s restart{_restart}, game_over={self.done}",
                        flush=True
                    )
            else:
                print(
                    f"[gym_adapter BLEED] already neutral speed={_spawn_speed:.2f}<{BLEED_SPEED_THRESHOLD}",
                    flush=True
                )

            # Phase 2: Heading Alignment (v1.1.6c FIX-HDGALIGN)
            # REF: run_20260429_172738.log -- EVERY post-bleed heading wrong;
            #      BC harvest: 0 episodes (all discarded), crashes at step 15-20.
            _hdg_rp = self._get_rp(self.response)
            _heading_now = float(_hdg_rp.get('heading', 0.0))
            _wps  = _hdg_rp.get('waypoints')
            _ci   = _hdg_rp.get('closest_waypoints')
            _hdg_err = _compute_heading_error(_heading_now, _wps, _ci)

            if abs(_hdg_err) > BLEED_HDG_THRESHOLD and not self.done:
                print(
                    f"[gym_adapter HDGALIGN] heading_error={_hdg_err:.1f}deg > {BLEED_HDG_THRESHOLD}deg "
                    f"-- entering heading alignment (budget={HDGALIGN_MAX_STEPS})",
                    flush=True
                )
                _hdg_steps = 0
                _hdg_gameover = False
                while abs(_hdg_err) > BLEED_HDG_THRESHOLD and _hdg_steps < HDGALIGN_MAX_STEPS:
                    # Proportional steer: -sign(err)*gain, clamped [-1,1]
                    # Positive err = car CCW of tangent -> steer CW (negative steer)
                    _steer_cmd = float(np.clip(-_hdg_err / 45.0 * HDGALIGN_STEER_GAIN, -1.0, 1.0))
                    _hdg_action = np.array([_steer_cmd, HDGALIGN_THROTTLE], dtype=np.float32)
                    self.response = self._send_action(_hdg_action)
                    if not isinstance(self.response.get('info'), dict):
                        self.response['info'] = dict()
                    _hdg_steps += 1
                    if self.done:
                        _hdg_gameover = True
                        break
                    _hdg_rp = self._get_rp(self.response)
                    _heading_now = float(_hdg_rp.get('heading', _heading_now))
                    _ci_upd = _hdg_rp.get('closest_waypoints', _ci)
                    if _ci_upd:
                        _ci = _ci_upd
                    _hdg_err = _compute_heading_error(_heading_now, _wps, _ci)

                print(
                    f"[gym_adapter HDGALIGN] DONE after {_hdg_steps} steps "
                    f"final_heading_error={_hdg_err:.1f}deg game_over={_hdg_gameover}",
                    flush=True
                )

                if _hdg_gameover:
                    if _restart < BLEED_MAX_RESTARTS:
                        print(f"[gym_adapter HDGALIGN] game_over -- restarting ep (restart {_restart+1})", flush=True)
                        self.response = self._do_episode_reset()
                        if not isinstance(self.response.get('info'), dict):
                            self.response['info'] = dict()
                        continue
                    else:
                        print("[gym_adapter HDGALIGN] WARNING: max restarts exceeded after HDGALIGN crash", flush=True)
                        break

                # Phase 3: Re-bleed speed after heading correction
                _rbleed_rp = self._get_rp(self.response)
                if not self._spawn_kinematics_neutral(_rbleed_rp):
                    _rbleed_spd = float(_rbleed_rp.get('speed', 0.0))
                    print(f"[gym_adapter REBLEED] speed={_rbleed_spd:.2f} m/s after HDGALIGN -- re-bleeding", flush=True)
                    for _ in range(BLEED_MAX_STEPS):
                        self.response = self._send_action(self._bleed_action)
                        if not isinstance(self.response.get('info'), dict):
                            self.response['info'] = dict()
                        if self._spawn_kinematics_neutral(self._get_rp(self.response)):
                            break
                        if self.done:
                            break
                    _final_spd = float(self._get_rp(self.response).get('speed', 0.0))
                    print(f"[gym_adapter REBLEED] DONE speed={_final_spd:.3f} m/s", flush=True)
            else:
                print(
                    f"[gym_adapter HDGALIGN] heading already aligned error={_hdg_err:.1f}deg <= {BLEED_HDG_THRESHOLD}deg",
                    flush=True
                )

            break  # exit restart loop on successful completion

        # -- end PATCH-BLEED / PATCH-HDGALIGN -----------------------------------

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
        observation = {
            sensor: (
                measurement.transpose(-1, 0, 1) if 'CAMERA' in sensor
                else measurement
            ) for sensor, measurement in observation.items()
        }

        return observation, terminated, truncated, info
