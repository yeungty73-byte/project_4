"""adaptive_reward_shaper.py  -- v1.4.2
Standalone module. The same classes are also inlined into run.py
for zero-import-dependency deployment on the DeepRacer server.

v1.4.2 changes:
  - shape(): v_perp penalty uses DIVISION (attenuation), not SUBTRACTION (deduction).
    Division: reward / (1 + w * excess) -- reward never goes below 0 from this penalty.
    This eliminates the freeze trap: agent cannot earn more by doing nothing than
    by moving forward slowly with some v_perp.
  - TelemetryFeedbackAnnealer embedded in AdaptiveRewardShaper:
    update_phase(bsts_metrics) -- advances phase on actual mastery, not fixed %:
      Phase 0->1: any lap completion observed
      Phase 1->2: completion_ema >= 80% AND race_line_adherence_ema >= 0.45
    get_phase_weights(base_rw) -- returns closed-loop phase-appropriate weights.
  - bc_pilot_reward: phase-scaled alpha (0=0.15 slow-drive, 1=0.30, 2=0.60 fast-lap)

Usage:
    from adaptive_reward_shaper import AdaptiveRewardShaper
    _shaper = AdaptiveRewardShaper(n_waypoints=len(waypoints))
    reward += _shaper.episode_start(is_reversed)          # -5.0 if reversed
    reward, info = _shaper.shape(reward, rp, wp_idx, ...) # per-step
    rw = _shaper.get_phase_weights(rw)                    # phase-aware weights
    speed_budget = _shaper.speed_budget(wp_idx)
    curb_mul     = _shaper.curb_urgency_mul(wp_idx)
    bc_r         = _shaper.bc_pilot_reward(rp, prev_m, curr_m)
    _shaper.episode_end(wp_idx, is_offtrack)
    _shaper.update_phase(bsts_metrics)                    # closed-loop at episode end

References
----------
Khatib (1986) IJRR: APF repulsive gradient.
Heinzmann & Zelinsky (2003): per-step v_perp safety signal; body-width clearance.
Heilmeier et al. (2020) VSD: minimum curvature speed profile (per-WP budget).
Ng, Harada & Russell (1999) ICML: potential-based shaping, r=alpha*delta_progress_m.
Schulman et al. (2017) arXiv: PPO prospective penalty in short episodes.
AWS DeepRacer docs (2020): is_reversed=True => CW driving; spawn speed 4.0 m/s.
Almakhayita et al. (2025) PLoS ONE: reward design for generalizable deep RL agents.
"""

from __future__ import annotations
import math
import numpy as np
from collections import deque
from typing import Dict, Tuple

# Module-level constants
_BETA             = 0.05    # per-crash budget tightening (Heilmeier 2020)
_BC_ALPHA         = 0.30    # BC pilot reward scale (Ng et al. 1999)
_VPERP_THRESHOLD  = 0.30    # m/s -- v_perp below this is safe (Heinzmann 2003)
# v1.4.2: DIVISION weight, not subtraction.
# Factor = 1 + w * excess. At v_perp excess=0.3: factor=1.15 -> 13% attenuation.
_VPERP_ATTN_W     = 0.50    # per-step v_perp attenuation weight
_BORDER_DIST_GATE = 1.50    # m -- attenuation only within this dist to any boundary


class PerWaypointVPerpTracker:
    """EMA of perpendicular velocity experienced near each waypoint.
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
        i = int(wp_idx) % self.n
        return float(np.clip(1.0 + self._k * self._ema[i], 1.0, 3.0))

    def mean_vperp_at(self, wp_idx: int) -> float:
        return float(self._ema[int(wp_idx) % self.n])


class PerWaypointSpeedBudget:
    """Adaptive per-waypoint speed budget.
    REF: Heilmeier et al. (2020) Eq. 10 -- minimum curvature speed profile.
    """
    _MIN_BUDGET = 0.80

    def __init__(self, n_waypoints: int, init_speed: float = 4.0):
        self.n = max(int(n_waypoints), 1)
        self._budget = np.full(self.n, float(init_speed), dtype=np.float32)

    def crash_at(self, wp_idx: int):
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
    """Detects {progress_up, speed_change, steer_up} pre-crash signature.
    REF: Schulman et al. (2017) -- prospective penalty in short episodes.
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
    """All-in-one per-step reward shaping for AWS DeepRacer training.

    v1.4.2 PENALTY CHANGE: all penalties use DIVISION (attenuation), not SUBTRACTION.
    Division form: reward = reward / (1 + w * excess)
    This guarantees reward >= 0 for any non-negative base reward.

    v1.4.2 CLOSED-LOOP PHASE: TelemetryFeedbackAnnealer advances training phases
    based on actual completion/adherence metrics, not fixed % of total_timesteps.
    Call update_phase(bsts_metrics) at episode end to advance.
    Call get_phase_weights(rw) at rollout start to get phase-appropriate weights.

    Training philosophy (per Tim's spec):
      Phase 0: survive and reach finish line (any speed, v > 0)
      Phase 1: finish line reliably -> follow the racing line
      Phase 2: follows race line -> optimize speed profile
    """

    def __init__(self, n_waypoints: int = 120,
                 border_dist_gate: float = _BORDER_DIST_GATE):
        self.n = max(int(n_waypoints), 1)
        self._vperp_tracker = PerWaypointVPerpTracker(self.n)
        self._speed_budget  = PerWaypointSpeedBudget(self.n)
        self._sig_detector  = KalmanSignatureDetector()
        self._border_gate   = float(border_dist_gate)
        self._reversed_ep   = False
        self._ep_count      = 0
        # v1.4.2: telemetry-driven phase (0=survival, 1=raceline, 2=speed)
        self._current_phase        = 0
        self._tfa_completion_ema   = 0.0
        self._tfa_adherence_ema    = 0.5   # neutral start
        self._tfa_any_completion   = False
        self._tfa_alpha            = 0.10  # EMA alpha for phase metrics

    @property
    def current_phase(self) -> int:
        return self._current_phase

    def episode_start(self, is_reversed: bool) -> float:
        """Call at episode reset. Returns immediate reward delta.
        Returns -5.0 for reversed spawn (applied to step 1 rewards[] ONLY).
        REF: AWS DeepRacer docs -- is_reversed=True iff car is driving CW on CCW track.
        """
        self._reversed_ep = bool(is_reversed)
        self._sig_detector.reset()
        self._ep_count += 1
        if is_reversed:
            return -5.0
        return 0.0

    def episode_end(self, wp_idx: int, is_offtrack: bool = False):
        """Call at episode end to update crash credit at last waypoint."""
        if is_offtrack:
            self._speed_budget.crash_at(int(wp_idx))
            self._vperp_tracker._k = min(3.0, self._vperp_tracker._k * 1.05)
        else:
            self._speed_budget.relax(int(wp_idx), amount=0.02)
            self._vperp_tracker._k = max(1.0, self._vperp_tracker._k * 0.98)

    def update_phase(self, bsts_metrics: dict) -> int:
        """Feed episode bsts_metrics into TelemetryFeedbackAnnealer.
        Returns current training phase (0, 1, or 2).
        Call this at episode end AFTER bsts_metrics is assembled.

        Phase 0->1: any lap completion observed
        Phase 1->2: completion_ema >= 80% AND race_line_adherence_ema >= 0.45

        REF: Almakhayita et al. (2025) PLoS ONE -- adaptive reward design.
        """
        a  = self._tfa_alpha
        _c = float(bsts_metrics.get("completion_pct", 0.0) or 0.0)
        _a = float(bsts_metrics.get("race_line_adherence", 0.5) or 0.5)
        _l = float(bsts_metrics.get("lap_completed", 0.0) or 0.0)
        self._tfa_completion_ema = (1-a)*self._tfa_completion_ema + a*_c
        self._tfa_adherence_ema  = (1-a)*self._tfa_adherence_ema  + a*_a
        if _l > 0.5:
            self._tfa_any_completion = True
        prev = self._current_phase
        if self._current_phase == 0 and self._tfa_any_completion:
            self._current_phase = 1
        elif (self._current_phase == 1
              and self._tfa_completion_ema >= 0.80
              and self._tfa_adherence_ema >= 0.45):
            self._current_phase = 2
        return self._current_phase

    def get_phase_weights(self, base_rw: dict) -> dict:
        """Return reward weights for current telemetry-driven training phase.
        Phase 0 (survival): progress=0.62 dominant
        Phase 1 (race line): blend progress + race_line, graduated by completion_ema
        Phase 2 (speed): full curv_speed rewards
        REF: Almakhayita et al. (2025) PLoS ONE.
        """
        if self._current_phase == 0:
            w = {"center":0.05, "heading":0.18, "racing_line":0.02, "braking":0.01,
                 "progress":0.62, "corner":0.02, "speed_steering":0.02, "curv_speed":0.00,
                 "min_speed":0.06, "completion":0.02, "decel":0.00, "obstacle":0.00, "steering":0.00}
        elif self._current_phase == 1:
            t1 = float(np.clip(self._tfa_completion_ema / 0.80, 0.0, 1.0))
            def bl(a, b): return a*(1-t1)+b*t1
            w = {"center":bl(0.22,0.14), "heading":bl(0.20,0.12),
                 "racing_line":bl(0.18,0.14), "braking":bl(0.15,0.10),
                 "progress":bl(0.14,0.16), "corner":0.06,
                 "speed_steering":bl(0.03,0.10), "curv_speed":bl(0.01,0.08),
                 "min_speed":bl(0.01,0.06), "completion":bl(0.00,0.05),
                 "decel":bl(0.00,0.03), "obstacle":bl(0.00,0.03), "steering":bl(0.00,0.02)}
        else:
            w = {"center":0.07, "heading":0.06, "racing_line":0.05, "braking":0.07,
                 "progress":0.18, "corner":0.04, "speed_steering":0.10, "curv_speed":0.18,
                 "min_speed":0.10, "completion":0.10, "decel":0.02, "obstacle":0.02, "steering":0.01}
        total = sum(w.values())
        return {k: v/total for k, v in w.items()} if total > 0 else w

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
        """Apply all per-step shaping. Returns (shaped_reward, sinfo_dict).

        v1.4.2 PENALTY CHANGE -- division not deduction:
          BEFORE: shaped -= w * max(0, v_perp - threshold)  -> can go negative
          AFTER:  shaped /= (1 + w * excess)                -> always >= 0

        REF: Ng et al. (1999) -- potential-based shaping.
        REF: Heinzmann & Zelinsky (2003) -- v_perp safety via attenuation.
        """
        shaped  = float(reward)
        _is_rev = bool(rp.get("is_reversed", self._reversed_ep))

        if _is_rev:
            shaped = 0.0
            return shaped, {"ars_reversed": True, "ars_vperp_attn": 1.0, "ars_sig_pen": 0.0}

        vperp_attn = 1.0
        _d  = float(dist_to_border) if dist_to_border is not None else 5.0
        _vp = float(v_perp_barrier) if v_perp_barrier is not None else 0.0
        if _d < self._border_gate and _vp > _VPERP_THRESHOLD:
            excess = _vp - _VPERP_THRESHOLD
            # v1.4.2: DIVISION -- reward / (1 + w * excess)
            # At v_perp=0.6 (excess=0.3): factor=1.15 -> 13% cut
            # At v_perp=2.0 (excess=1.7): factor=1.85 -> 46% cut
            # NEVER negative. Agent always earns more by moving than by freezing.
            vperp_attn = 1.0 + _VPERP_ATTN_W * excess
            shaped = shaped / vperp_attn

        prog    = float(rp.get("progress", ep_progress_m or 0.0))
        sig_pen = self._sig_detector.step(prog, float(speed), float(steer))
        # sig_pen is one-shot -0.30 prospective penalty; kept additive (single-step signal)
        shaped += sig_pen

        self._vperp_tracker.update(int(wp_idx), _vp)

        if not is_offtrack and float(speed) > 0.3:
            self._speed_budget.relax(int(wp_idx), amount=0.005)

        return shaped, {
            "ars_reversed":   _is_rev,
            "ars_vperp_attn": round(vperp_attn, 4),
            "ars_sig_pen":    round(sig_pen, 4),
            "ars_wp":         int(wp_idx),
            "ars_phase":      self._current_phase,
        }

    def speed_budget(self, wp_idx: int) -> float:
        return self._speed_budget.get(int(wp_idx))

    def curb_urgency_mul(self, wp_idx: int) -> float:
        return self._vperp_tracker.curb_urgency_mul(int(wp_idx))

    def bc_pilot_reward(
        self,
        rp:              dict,
        prev_progress_m: float,
        curr_progress_m: float,
    ) -> float:
        """Potential-based BC pilot reward: r = phase_alpha * delta_progress_m.

        Phase 0: alpha=0.15 (teach: advance, any speed, v > 0)
        Phase 1: alpha=0.30 (teach: advance efficiently)
        Phase 2: alpha=0.60 (teach: fast completion)

        REF: Ng, Harada & Russell (1999) ICML -- potential phi(s)=alpha*arc_progress_m.
        """
        delta_m = float(curr_progress_m) - float(prev_progress_m)
        if rp.get("is_reversed", False):
            return 0.0
        _phase_alpha = [0.15, 0.30, 0.60][min(self._current_phase, 2)]
        return float(np.clip(_phase_alpha * max(0.0, delta_m), -10.0, 10.0))
