"""adaptive_reward_shaper.py  — v1.4.0
Standalone module. The same classes are also inlined into run.py (v1.4.0)
for zero-import-dependency deployment on the DeepRacer server.

Usage:
    from adaptive_reward_shaper import AdaptiveRewardShaper
    _shaper = AdaptiveRewardShaper(n_waypoints=len(waypoints))
    reward += _shaper.episode_start(is_reversed)
    reward, info = _shaper.shape(reward, rp, wp_idx, speed, v_perp, steer, prog_m, offtrack, dist_to_border)
    speed_budget = _shaper.speed_budget(wp_idx)
    curb_mul     = _shaper.curb_urgency_mul(wp_idx)
    bc_r         = _shaper.bc_pilot_reward(rp, prev_m, curr_m)
    _shaper.episode_end(wp_idx, is_offtrack)

References
----------
Khatib (1986) IJRR: APF repulsive gradient -- field magnitude scales with danger.
Heinzmann & Zelinsky (2003): per-step v_perp safety signal; body-width clearance.
Heilmeier et al. (2020) VSD: minimum curvature speed profile (per-WP budget).
Ng, Harada & Russell (1999) ICML: potential-based shaping, r=alpha*delta_progress_m.
Schulman et al. (2017) arXiv: PPO prospective penalty in short episodes.
AWS DeepRacer docs (2020): is_reversed=True => CW driving; spawn speed 4.0 m/s.
"""

from __future__ import annotations
import math
import numpy as np
from collections import deque
from typing import Dict, Tuple

# ── Module-level constants (mirrors run.py) ──────────────────────────────────
_BETA            = 0.05    # per-crash budget tightening rate (Heilmeier 2020)
_BC_ALPHA        = 0.30    # BC pilot reward scale (Ng et al. 1999)
_VPERP_THRESHOLD = 0.30    # m/s -- v_perp below this is safe (Heinzmann 2003)
_VPERP_PENALTY_W = 0.50    # per-step v_perp penalty weight
_BORDER_DIST_GATE= 1.50    # m -- penalty only within this dist to any boundary


class PerWaypointVPerpTracker:
    """EMA of perpendicular velocity experienced near each waypoint.

    Feeds curb_urgency_mul to CombinedBrakeField.step(), so corners the agent
    consistently botches get progressively higher curb urgency.

    REF: Khatib (1986) -- field magnitude proportional to observed danger gradient.
    """
    def __init__(self, n_waypoints: int, alpha: float = 0.15):
        self.n = max(int(n_waypoints), 1)
        self.alpha = float(alpha)
        self._ema = np.zeros(self.n, dtype=np.float32)
        self._k   = 1.0

    def update(self, wp_idx: int, v_perp: float):
        i = int(wp_idx) % self.n
        self._ema[i] = (1 - self.alpha) * self._ema[i] + self.alpha * max(0.0, float(v_perp))

    def curb_urgency_mul(self, wp_idx: int) -> float:
        """Return urgency multiplier [1.0, 3.0] for curb field at this WP.
        alpha_i = 1 + k * mean_v_perp[i]  (Khatib 1986)
        """
        i = int(wp_idx) % self.n
        return float(np.clip(1.0 + self._k * self._ema[i], 1.0, 3.0))

    def mean_vperp_at(self, wp_idx: int) -> float:
        return float(self._ema[int(wp_idx) % self.n])


class PerWaypointSpeedBudget:
    """Adaptive per-waypoint speed budget.

    v*_i <- v*_i * (1 - beta * 1[crash near WP_i])

    REF: Heilmeier et al. (2020) Eq. 10 -- minimum curvature speed profile.
         beta=0.05 gives ~10 crashes to halve budget from 4.0 m/s to 2.4 m/s.
    """
    _MIN_BUDGET = 0.80

    def __init__(self, n_waypoints: int, init_speed: float = 4.0):
        self.n = max(int(n_waypoints), 1)
        self._budget = np.full(self.n, float(init_speed), dtype=np.float32)

    def crash_at(self, wp_idx: int):
        """Tighten budget at WP and neighbors (+/-2 wps, triangular weight)."""
        i = int(wp_idx) % self.n
        for offset, w in [(0, 1.0), (1, 0.6), (-1, 0.6), (2, 0.3), (-2, 0.3)]:
            j = (i + offset) % self.n
            self._budget[j] = max(self._MIN_BUDGET,
                                  self._budget[j] * (1.0 - _BETA * w))

    def get(self, wp_idx: int) -> float:
        return float(self._budget[int(wp_idx) % self.n])

    def relax(self, wp_idx: int, amount: float = 0.01):
        i = int(wp_idx) % self.n
        self._budget[i] = min(4.0, self._budget[i] + amount)


class KalmanSignatureDetector:
    """Detects {progress_up, speed_change, steer_up} triple-signature 2-4 steps
    before an off-track crash and emits a prospective -0.30 penalty.

    REF: Schulman et al. (2017) -- prospective penalty gives PPO gradient signal
         in short (13-29 step) episodes where terminal signal arrives too late.
    """
    WINDOW  = 5
    PENALTY = -0.30

    def __init__(self):
        self._prog_buf  = deque(maxlen=self.WINDOW)
        self._spd_buf   = deque(maxlen=self.WINDOW)
        self._steer_buf = deque(maxlen=self.WINDOW)
        self._triggered = False

    def step(self, progress: float, speed: float, steer: float) -> float:
        self._prog_buf.append(float(progress))
        self._spd_buf.append(float(speed))
        self._steer_buf.append(abs(float(steer)))
        if len(self._prog_buf) < self.WINDOW:
            return 0.0
        prog_trend = float(self._prog_buf[-1]) - float(self._prog_buf[0])
        spd_change = abs(float(self._spd_buf[-1]) - float(self._spd_buf[0]))
        steer_high = float(np.mean(list(self._steer_buf))) > 0.35
        if prog_trend > 0.001 and spd_change > 0.3 and steer_high:
            if not self._triggered:
                self._triggered = True
                return self.PENALTY
        else:
            self._triggered = False
        return 0.0

    def reset(self):
        self._prog_buf.clear(); self._spd_buf.clear(); self._steer_buf.clear()
        self._triggered = False


class AdaptiveRewardShaper:
    """Coordinates all per-episode and per-step adaptive reward shaping.

    API (all methods are side-effect-safe when called out of order):
      shaper.episode_start(is_reversed)           -> float  immediate spawn penalty
      shaper.shape(reward, rp, wp_idx, ...)       -> (float, dict)
      shaper.curb_urgency_mul(wp_idx)             -> float  for CombinedBrakeField
      shaper.speed_budget(wp_idx)                 -> float  per-WP speed budget
      shaper.bc_pilot_reward(rp, prev_m, curr_m)  -> float  Ng et al. BC reward
      shaper.episode_end(wp_idx, is_offtrack)

    REF: Ng, Harada & Russell (1999) ICML -- potential-based shaping.
    REF: AWS DeepRacer docs (2020) -- is_reversed=True: CW driving; spawn=4.0 m/s.
    """
    def __init__(self, n_waypoints: int):
        self.n = max(int(n_waypoints), 1)
        self._vperp_tracker = PerWaypointVPerpTracker(self.n)
        self._speed_budget  = PerWaypointSpeedBudget(self.n)
        self._sig_detector  = KalmanSignatureDetector()
        self._ep_count      = 0
        self._reversed_ep   = False
        self._border_gate   = _BORDER_DIST_GATE

    def episode_start(self, is_reversed: bool) -> float:
        """Reset per-episode state. Return spawn penalty (-5.0 if reversed).

        is_reversed=True => driving CW on CCW track; structurally unrecoverable
        at 4.0 m/s in 13-16 steps.

        REF: AWS DeepRacer docs -- is_reversed=True iff car is driving CW on CCW track.
        REF: Schulman et al. (2017) -- large early penalty for clearly-bad spawns.
        """
        self._ep_count += 1
        self._reversed_ep = bool(is_reversed)
        self._sig_detector.reset()
        if is_reversed:
            return -5.0
        return 0.0

    def episode_end(self, wp_idx: int, is_offtrack: bool = False):
        """Update per-WP budgets based on episode outcome."""
        if is_offtrack:
            self._speed_budget.crash_at(int(wp_idx))
        else:
            self._speed_budget.relax(int(wp_idx), amount=0.02)

    def shape(
        self,
        reward:         float,
        rp:             dict,
        wp_idx:         int,
        speed:          float,
        v_perp_barrier: float,
        steer:          float,
        ep_progress_m:  float,
        is_offtrack:    bool,
        dist_to_border: float = 5.0,
    ) -> Tuple[float, Dict]:
        """Apply all per-step shaping rules. Returns (shaped_reward, sinfo_dict).

        Rules (all additive, documented):
          a) Reversed gating: reward -> 0.0 when is_reversed (CW driver)
          b) v_perp penalty outside brake field: -0.5 * max(0, v_perp - 0.3) within 1.5m
          c) Prospective Kalman-signature penalty: -0.30 pre-crash
          d) v_perp tracker update (telemetry)
          e) Speed budget relaxation for clean on-track steps

        REF: Ng et al. (1999) -- gradient-compatible dense signal.
        REF: Heinzmann & Zelinsky (2003) -- per-step v_perp safety signal.
        """
        shaped  = float(reward)
        _is_rev = bool(rp.get("is_reversed", self._reversed_ep))

        if _is_rev:
            shaped = 0.0
            return shaped, {"ars_reversed": True, "ars_vperp_pen": 0.0, "ars_sig_pen": 0.0}

        vperp_pen = 0.0
        _d  = float(dist_to_border) if dist_to_border is not None else 5.0
        _vp = float(v_perp_barrier) if v_perp_barrier is not None else 0.0
        if _d < self._border_gate and _vp > _VPERP_THRESHOLD:
            vperp_pen = _VPERP_PENALTY_W * (_vp - _VPERP_THRESHOLD)
            shaped -= vperp_pen

        prog    = float(rp.get("progress", ep_progress_m or 0.0))
        sig_pen = self._sig_detector.step(prog, float(speed), float(steer))
        shaped += sig_pen

        self._vperp_tracker.update(int(wp_idx), _vp)

        if not is_offtrack and float(speed) > 0.3:
            self._speed_budget.relax(int(wp_idx), amount=0.005)

        return shaped, {
            "ars_reversed":  _is_rev,
            "ars_vperp_pen": round(vperp_pen, 4),
            "ars_sig_pen":   round(sig_pen, 4),
            "ars_wp":        int(wp_idx),
        }

    def speed_budget(self, wp_idx: int) -> float:
        """Per-WP speed budget for race_line_engine.get_target_speed(wp_speed_budget=)."""
        return self._speed_budget.get(int(wp_idx))

    def curb_urgency_mul(self, wp_idx: int) -> float:
        """Per-WP curb urgency multiplier for CombinedBrakeField.step()."""
        return self._vperp_tracker.curb_urgency_mul(int(wp_idx))

    def bc_pilot_reward(
        self,
        rp:              dict,
        prev_progress_m: float,
        curr_progress_m: float,
    ) -> float:
        """Potential-based BC pilot reward: r = alpha * delta_progress_m.

        Decouples lap time from traversed distance: slow-but-far > fast-but-crash.
        phi(s) = alpha * arc_progress_m (Ng et al. 1999 -- potential shaping function).

        REF: Ng, Harada & Russell (1999) ICML -- phi(s) potential, reward = phi(s')-phi(s).
        REF: AWS DeepRacer docs -- progress key is percentage 0-100; we use arc_m for scale.
        """
        if rp.get("is_reversed", False):
            return 0.0
        delta_m = float(curr_progress_m) - float(prev_progress_m)
        return float(_BC_ALPHA * max(0.0, delta_m))
