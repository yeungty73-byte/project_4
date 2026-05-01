"""
packages/deepracer_gym/gym_adapter.py
DeepRacer-for-Cloud gym adapter — v1.1.7a RACE-LINE-FOLLOWING + BRAKE-FIELD ARBITER
======================================================================================
SUMMARY OF ALL CHANGES vs v1.1.6c
──────────────────────────────────
FIX-RL-1 : RaceLineArbiter added.
  Reads waypoints from env_reset()'s first rp response.
  Computes MultiRaceLineEngine + CombinedBrakeField INSIDE the adapter so
  every action sent to Gazebo via _send_action() is arbitrated BEFORE dispatch.
  send_action() now accepts an optional arbiter_state dict (populated by run.py
  from its per-step rp) to let the arbiter override/blend raw RL actions.

FIX-RL-2 : arbiter_action() method — pure proportional controller blended with
  RL action.  blend = 0.0 (pure RL) early in training → 1.0 (pure race-line)
  in demo mode.  Blend schedule controlled by arbiter_blend param in send_action().

FIX-RL-3 : _track_tangent_and_error() replaces _compute_heading_error().
  Now returns (tangent_deg, error_deg, tangent_rad) — heading-align phase uses
  tangent_rad directly so arbiter can steer toward the track tangent vector.

FIX-RL-4 : heading_align phase calls arbiter.arbiter_action(blend=1.0) when
  race-line is warm, so the car follows the actual track tangent during alignment
  instead of a raw proportional bang-bang.

FIX-NONE-ATTR : process_action is NEVER called inside gym_adapter.  The adapter
  only ever sends raw np.ndarray([steer, throttle]) or int actions via _send_action().
  The 'NoneType has no attribute process_action_scale' error lived ENTIRELY in run.py,
  not here.  This file has zero process_action references — that's deliberate.

REF:
  Heilmeier et al. (2020) Min-curvature trajectory planning.  Vehicle Sys. Dynamics 58(10).
  Koenig & Howard (2004) Gazebo physics spawn.  IEEE/RSJ IROS.
  Ng, Harada & Russell (1999) Potential-based reward shaping.  ICML.
  Brayshaw & Harrison (2005) Quasi-steady braking point.  AIAA/ISSMO conf.
  Heinzmann & Zelinsky (2003) Quantified safety design.  Auton. Robots 15(2).
"""
import zmq
import math
import numpy as np
from typing import TypeAlias, Optional, Dict, Any
from collections.abc import Callable

from deepracer_gym.zmq_client import DeepracerClientZMQ
from deepracer_gym.utils import (
    terminated_check, truncated_check
)

PORT: int = 8888
HOST: str = '127.0.0.1'
TIMEOUT_LONG: int  = 500_000   # ~8.3 m
TIMEOUT_SHORT: int = 100_000  # ~1.7 m
DUMMY_ACTION_DISCRETE: Callable[[], int] = (
    lambda: 0
)
DUMMY_ACTION_CONTINUOUS: Callable[[], np.ndarray] = (
    lambda: np.random.uniform(-1, 1, 2)
)

# ── v1.1.6b/c spawn-bleed / heading-align thresholds ──────────────────────────
BLEED_SPEED_THRESHOLD: float = 0.55   # m/s — just above sim physics floor 0.500
BLEED_HDG_THRESHOLD: float   = 25.0   # degrees
BLEED_MAX_STEPS: int         = 80
BLEED_MAX_RESTARTS: int      = 3
HDGALIGN_MAX_STEPS: int      = 60
HDGALIGN_THROTTLE: float     = 0.25
HDGALIGN_STEER_GAIN: float   = 0.70
BLEED_BRAKE_ACTION_CONTINUOUS = np.array([0.0, -1.0], dtype=np.float32)
BLEED_BRAKE_ACTION_DISCRETE   = 0

# ── v1.1.7a Race-Line arbiter blend schedule ──────────────────────────────────
# During training run.py passes blend=0.0 (pure RL).
# demo() in utils.py passes blend=1.0 (full arbiter authority).
_ARBITER_BLEND_DEFAULT: float = 0.0   # run.py overrides via send_action(arbiter_blend=…)
_ARBITER_STEER_GAIN: float    = 0.80  # proportional steer gain to target heading
_ARBITER_THROTTLE_GAIN: float = 0.60  # proportional throttle gain to target speed


ActionType: TypeAlias = (int | np.ndarray | list)


# ─────────────────────────────────────────────────────────────────────────────
# HEADING / TANGENT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _track_tangent_and_error(heading_deg: float, waypoints, closest_waypoints):
    """Return (tangent_deg, heading_error_deg, tangent_rad).
    Positive error  = car CCW of tangent (steer CW / right to correct).
    Negative error  = car CW  of tangent (steer CCW / left).
    Returns (0.0, 0.0, 0.0) on bad inputs.
    """
    try:
        wps = waypoints
        ci  = closest_waypoints
        if wps is None or ci is None or len(wps) < 2 or len(ci) < 2:
            return 0.0, 0.0, 0.0
        idx_a = int(ci[0]); idx_b = int(ci[1])
        if idx_a >= len(wps) or idx_b >= len(wps):
            return 0.0, 0.0, 0.0
        wa = wps[idx_a]; wb = wps[idx_b]
        ax, ay = (float(wa[0]), float(wa[1])) if isinstance(wa, (list, tuple)) \
            else (float(wps[2*idx_a]), float(wps[2*idx_a+1]))
        bx, by = (float(wb[0]), float(wb[1])) if isinstance(wb, (list, tuple)) \
            else (float(wps[2*idx_b]), float(wps[2*idx_b+1]))
        tangent_rad = math.atan2(by - ay, bx - ax)
        tangent_deg = math.degrees(tangent_rad)
        err = heading_deg - tangent_deg
        while err >  180.0: err -= 360.0
        while err < -180.0: err += 360.0
        return tangent_deg, err, tangent_rad
    except Exception:
        return 0.0, 0.0, 0.0


def _compute_heading_error(heading_deg, waypoints, closest_waypoints) -> float:
    """Compatibility shim — returns heading error only (v1.1.6c API preserved)."""
    _, err, _ = _track_tangent_and_error(heading_deg, waypoints, closest_waypoints)
    return err


# ─────────────────────────────────────────────────────────────────────────────
# RACE-LINE ARBITER
# ─────────────────────────────────────────────────────────────────────────────

class RaceLineArbiter:
    """Wraps MultiRaceLineEngine + CombinedBrakeField.
    Provides arbiter_action() to blend RL action with race-line following.

    Initialised lazily on first rp with waypoints (inside env_reset).
    Called each step by send_action() with current rp dict.

    REF: Heilmeier et al. (2020) §4 — speed profile as constraint on race line.
         Brayshaw & Harrison (2005) — braking point arbiter.
    """

    def __init__(self):
        self._engine    = None   # MultiRaceLineEngine
        self._bf        = None   # CombinedBrakeField
        self._waypoints = None
        self._half_w    = 0.30
        self._ready     = False

    def _try_init(self, rp: dict) -> bool:
        if self._ready:
            return True
        wps = rp.get('waypoints')
        if not wps or len(wps) < 5:
            return False
        try:
            from race_line_engine import MultiRaceLineEngine
            from brake_field import CombinedBrakeField
            tw = float(rp.get('track_width', 0.60))
            self._engine    = MultiRaceLineEngine(wps, track_width=tw)
            self._engine.initialize()
            self._bf        = CombinedBrakeField(wps)
            self._waypoints = wps
            self._half_w    = tw / 2.0
            self._ready     = True
            print("[gym_adapter ARBITER] MultiRaceLineEngine + CombinedBrakeField initialised"
                  f" wps={len(wps)} track_width={tw:.2f}", flush=True)
        except Exception as _e:
            print(f"[gym_adapter ARBITER] init failed (non-fatal): {_e}", flush=True)
        return self._ready

    def arbiter_action(
        self,
        raw_action: np.ndarray,
        rp: dict,
        blend: float = 0.0,
        is_discrete: bool = False,
    ) -> np.ndarray:
        """Blend RL raw_action with a proportional race-line following signal.

        blend = 0.0 → pure RL (during training).
        blend = 1.0 → pure arbiter (during demo).
        Intermediate values produce a linear mix.

        For continuous action space:
          raw_action = [steer, throttle] both in [-1, 1] (tanh space).
          arbiter_steer    = clip(-hdg_err / 45° × GAIN, -1, 1)
          arbiter_throttle = clip((tgt_spd / 4.0) × 2 - 1, -1, 1)

        Returns blended np.ndarray([steer, throttle], float32).
        """
        if is_discrete or not isinstance(raw_action, np.ndarray) or raw_action.ndim == 0:
            return raw_action   # arbiter does not modify discrete actions

        if blend <= 0.0 or not self._try_init(rp):
            return raw_action

        try:
            hdg    = float(rp.get('heading',  0.0))
            spd    = float(rp.get('speed',    0.0))
            x      = float(rp.get('x',        0.0))
            y      = float(rp.get('y',        0.0))
            wps    = rp.get('waypoints', self._waypoints)
            cwps   = rp.get('closest_waypoints', [0, 1])
            dist_c = float(rp.get('distance_from_center', 0.0))
            tw     = float(rp.get('track_width', self._half_w * 2.0))
            wp_idx = int(cwps[0]) if cwps else 0

            # ── 1. Heading error → steer signal ──────────────────────────────
            _, hdg_err, _ = _track_tangent_and_error(hdg, wps, cwps)
            arb_steer = float(np.clip(-hdg_err / 45.0 * _ARBITER_STEER_GAIN, -1.0, 1.0))

            # ── 2. Brake-field safe speed → throttle signal ───────────────────
            heading_rad   = math.radians(hdg)
            barrier_dist  = float(rp.get('distance_from_center', 0.30))
            lat_offset_m  = dist_c * (-1 if rp.get('is_left_of_center', True) else 1)

            bf_out = self._bf.step(
                wp_idx=wp_idx,
                speed=spd,
                heading_rad=heading_rad,
                car_x=x,
                car_y=y,
                barrier_dist=barrier_dist,
                curb_dist=barrier_dist,
                car_lat_offset=lat_offset_m,
                track_half_w=tw / 2.0,
            )
            safe_spd = float(bf_out.get('race_line_safe_speed', spd))

            # ── 3. Race-line target speed & context ───────────────────────────
            context_code = 0   # 0=clear, 1=curb, 2=obstacle, 3=corner
            if self._engine._initialized:
                tgt_spd = self._engine.get_target_speed(
                    wp_idx=wp_idx, context=context_code,
                    brake_safe_speed=safe_spd)
                # Normalise to [-1,1] throttle: 0 m/s → -1, 4 m/s → 1
                max_spd = 4.0
                arb_throttle = float(np.clip((tgt_spd / max_spd) * 2.0 - 1.0, -1.0, 1.0))
            else:
                arb_throttle = float(raw_action[1]) if len(raw_action) > 1 else 0.0

            arb_action = np.array([arb_steer, arb_throttle], dtype=np.float32)

            # ── 4. Blend ──────────────────────────────────────────────────────
            blended = (1.0 - blend) * raw_action[:2].astype(np.float32) + blend * arb_action
            return np.clip(blended, -1.0, 1.0).astype(np.float32)

        except Exception as _e:
            print(f"[gym_adapter ARBITER] arbiter_action error (non-fatal): {_e}", flush=True)
            return raw_action

    def reset(self):
        """Call at episode boundary to reset BrakeField accumulators."""
        if self._bf is not None:
            try:
                self._bf.reset()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

class DeepracerGymAdapter:
    def __init__(
        self,
        action_space_type: str,
        host: str = HOST,
        port: int = PORT):

        if action_space_type == 'discrete':
            self.dummy_action     = DUMMY_ACTION_DISCRETE
            self._bleed_action    = BLEED_BRAKE_ACTION_DISCRETE
            self._is_discrete     = True
        elif action_space_type == 'continuous':
            self.dummy_action     = DUMMY_ACTION_CONTINUOUS
            self._bleed_action    = BLEED_BRAKE_ACTION_CONTINUOUS
            self._is_discrete     = False
        else:
            raise ValueError(
                f'Action space can only be discrete or continuous. Got {action_space_type} instead.'
            )

        self.zmq_client        = DeepracerClientZMQ(host=host, port=port)
        self.zmq_client.ready()
        self.response          = None
        self.done              = False
        self._first_reset_done = False
        self.arbiter           = RaceLineArbiter()   # FIX-RL-1

    # ── ZMQ I/O ───────────────────────────────────────────────────────────────

    def _send_action(self, action: ActionType):
        action_msg: dict = {'action': action}
        self.response = self.zmq_client.send_message(action_msg)
        self.done = self.response['_game_over']
        return self.response

    @staticmethod
    def _get_steps(response):
        info = response.get('info')
        if not isinstance(info, dict):
            return None
        rp = info.get('reward_params')
        if not isinstance(rp, dict):
            return None
        return rp.get('steps')

    @staticmethod
    def _get_rp(response):
        info = response.get('info')
        if not isinstance(info, dict):
            return {}
        rp = info.get('reward_params')
        return rp if isinstance(rp, dict) else {}

    # ── Spawn-state checks ────────────────────────────────────────────────────

    def _spawn_kinematics_neutral(self, rp: dict) -> bool:
        return float(rp.get('speed', 99.0)) < BLEED_SPEED_THRESHOLD

    def _heading_is_aligned(self, rp: dict) -> bool:
        heading = float(rp.get('heading', 0.0))
        wps = rp.get('waypoints')
        ci  = rp.get('closest_waypoints')
        err = _compute_heading_error(heading, wps, ci)
        return abs(err) <= BLEED_HDG_THRESHOLD

    # ── Episode drain / pump ──────────────────────────────────────────────────

    def _do_episode_reset(self):
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
                "Is the container fully booted?"
            )
        return self.response

    # ── env_reset ─────────────────────────────────────────────────────────────

    def env_reset(self):
        self.response = self._do_episode_reset()
        if not isinstance(self.response.get('info'), dict):
            self.response['info'] = dict()

        # v1.1.7a: reset arbiter BrakeField accumulators at episode boundary
        self.arbiter.reset()

        # ── Phase 1 + 2 + 3: Speed-Bleed / HDGALIGN / Re-Bleed (v1.1.6b/c) ──
        for _restart in range(BLEED_MAX_RESTARTS + 1):
            _bleed_rp    = self._get_rp(self.response)
            _spawn_speed = float(_bleed_rp.get('speed', 0.0))
            _spawn_hdg   = _bleed_rp.get('heading', 0.0)

            # ── Phase 1: Speed Bleed ──────────────────────────────────────────
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
                _post_speed    = float(_post_bleed_rp.get('speed', 0.0))

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

            # ── Phase 2: Heading Alignment (v1.1.6c + v1.1.7a FIX-RL-4) ─────
            _hdg_rp      = self._get_rp(self.response)
            _heading_now = float(_hdg_rp.get('heading', 0.0))
            _wps         = _hdg_rp.get('waypoints')
            _ci          = _hdg_rp.get('closest_waypoints')
            _, _hdg_err, _ = _track_tangent_and_error(_heading_now, _wps, _ci)

            if abs(_hdg_err) > BLEED_HDG_THRESHOLD and not self.done:
                print(
                    f"[gym_adapter HDGALIGN] heading_error={_hdg_err:.1f}deg > {BLEED_HDG_THRESHOLD}deg "
                    f"-- entering heading alignment (budget={HDGALIGN_MAX_STEPS})",
                    flush=True
                )
                _hdg_steps  = 0
                _hdg_gameover = False
                while abs(_hdg_err) > BLEED_HDG_THRESHOLD and _hdg_steps < HDGALIGN_MAX_STEPS:
                    # v1.1.7a FIX-RL-4: use arbiter steer if race-line ready, else proportional
                    _arb_out = self.arbiter.arbiter_action(
                        np.array([0.0, HDGALIGN_THROTTLE], dtype=np.float32),
                        _hdg_rp,
                        blend=1.0,
                        is_discrete=self._is_discrete,
                    )
                    if isinstance(_arb_out, np.ndarray) and len(_arb_out) >= 2:
                        _steer_cmd = float(_arb_out[0])
                        _thr_cmd   = float(_arb_out[1])
                    else:
                        _steer_cmd = float(np.clip(-_hdg_err / 45.0 * HDGALIGN_STEER_GAIN, -1.0, 1.0))
                        _thr_cmd   = HDGALIGN_THROTTLE
                    _hdg_action = np.array([_steer_cmd, _thr_cmd], dtype=np.float32)
                    self.response = self._send_action(_hdg_action)
                    if not isinstance(self.response.get('info'), dict):
                        self.response['info'] = dict()
                    _hdg_steps += 1
                    if self.done:
                        _hdg_gameover = True
                        break
                    _hdg_rp      = self._get_rp(self.response)
                    _heading_now = float(_hdg_rp.get('heading', _heading_now))
                    _ci_upd = _hdg_rp.get('closest_waypoints', _ci)
                    if _ci_upd:
                        _ci = _ci_upd
                    _, _hdg_err, _ = _track_tangent_and_error(_heading_now, _wps, _ci)

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

            # v1.1.7a: try to init arbiter from first rp with waypoints
            _final_rp = self._get_rp(self.response)
            if not self.arbiter._ready:
                self.arbiter._try_init(_final_rp)

            break  # exit restart loop

        if not self._first_reset_done:
            self._first_reset_done = True
            self.zmq_client.socket.set(zmq.SNDTIMEO, TIMEOUT_SHORT)
            self.zmq_client.socket.set(zmq.RCVTIMEO, TIMEOUT_SHORT)

        observation, _, _, info = self._parse_response(self.response)
        return observation, info

    # ── send_action (v1.1.7a: arbiter blending) ───────────────────────────────

    def send_action(
        self,
        action: ActionType,
        arbiter_blend: float = _ARBITER_BLEND_DEFAULT,
        rp: Optional[Dict[str, Any]] = None,
    ):
        """Send action to Gazebo, optionally blending with race-line arbiter.

        Parameters
        ----------
        action        : raw RL action (int for discrete, np.ndarray for continuous)
        arbiter_blend : 0.0 = pure RL, 1.0 = pure race-line arbiter.
                        Passed by run.py (0.0 during training) or utils.demo (1.0).
        rp            : reward_params dict from previous step (for arbiter state).
                        If None, arbiter blend is skipped gracefully.
        """
        if self.done:
            return self._parse_response(self.response)

        # FIX-RL-2: arbiter blending applied BEFORE Gazebo dispatch
        if (arbiter_blend > 0.0
                and rp is not None
                and not self._is_discrete
                and isinstance(action, np.ndarray)):
            action = self.arbiter.arbiter_action(
                action, rp, blend=arbiter_blend, is_discrete=False)

        response = self._send_action(action)
        return self._parse_response(response)

    # ── response parsing ──────────────────────────────────────────────────────

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
        truncated  = truncated_check(episode_status, game_over)

        observation = response['_next_state']
        observation = {
            sensor: (
                measurement.transpose(-1, 0, 1) if 'CAMERA' in sensor
                else measurement
            ) for sensor, measurement in observation.items()
        }

        return observation, terminated, truncated, info
