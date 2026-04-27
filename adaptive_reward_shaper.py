"""adaptive_reward_shaper.py — v1.4.1
Standalone module. Drop-in replacement.

CRITICAL DESIGN CHANGE v1.4.1:
  All penalties are now MULTIPLICATIVE (scale-down) not ADDITIVE (deduct).
  reward = base_reward * compliance_multiplier, compliance_multiplier in [0.1, 1.0].
  Ensures any forward progress ALWAYS yields positive reward.
  REF: Ng, Harada & Russell (1999) ICML -- reward shaping preserves policy invariance.
  REF: Amodei et al. (2016) Concrete Problems in AI Safety -- avoid negative net reward.

THREE-PHASE BSTS-DRIVEN CURRICULUM v1.4.1:
  Phase 0 "finish first": maximize track_progress_m; speed weight=0.01.
  Phase 1 "find the line": maintain track_progress; optimize race_line.
  Phase 2 "go fast": maximize speed given race line + progress maintained.
  Transitions driven by BSTS Kalman trends, not time-based thresholds.
  Reversion: track_progress trend below REVERT_THRESH for REVERT_WINDOW eps.

NOTE: This file adds BSTSPhaseController and get_phase_weights() to the v1.4.2
      base (which already has TelemetryFeedbackAnnealer + division penalties).
      The two phase-management approaches coexist: update_phase() for closed-loop
      episode-level transitions; BSTSPhaseController for Kalman-trend-level.

REF: Khatib (1986) IJRR: APF repulsive gradient.
REF: Heinzmann & Zelinsky (2003): per-step v_perp safety signal.
REF: Heilmeier et al. (2020) VSD: minimum curvature speed profile.
REF: Ng, Harada & Russell (1999) ICML: potential-based shaping.
REF: Schulman et al. (2017) arXiv:1707.06347: PPO prospective penalty.
REF: AWS DeepRacer docs (2020): is_reversed=True => CW; 4.0 m/s spawn.
REF: Amodei et al. (2016). Concrete problems in AI safety. arXiv:1606.06565.
REF: Almakhayita et al. (2025) PLoS ONE: reward design for generalizable deep RL.
"""

from __future__ import annotations
import math
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional

_BETA             = 0.05
_BC_ALPHA         = 0.30
_VPERP_THRESHOLD  = 0.30
_VPERP_ATTN_W     = 0.50    # v1.4.2: DIVISION weight
_BORDER_DIST_GATE = 1.50

# BSTS-Kalman-driven phase thresholds (v1.4.1)
_PHASE0_PROGRESS_THRESH = 0.008
_PHASE1_LINE_THRESH     = 0.005
_PHASE0_MIN_EPS         = 30
_REVERT_THRESH          = -0.003
_REVERT_WINDOW          = 10

# Phase reward weight profiles (v1.4.1 BSTSPhaseController)
_PHASE_WEIGHTS = {
    0: {"progress": 0.55, "racing_line": 0.05, "curv_speed": 0.01, "min_speed": 0.01,
        "heading": 0.25, "braking": 0.08, "corner": 0.02, "center": 0.03},
    1: {"progress": 0.35, "racing_line": 0.30, "curv_speed": 0.05, "min_speed": 0.02,
        "heading": 0.15, "braking": 0.08, "corner": 0.03, "center": 0.02},
    2: {"progress": 0.15, "racing_line": 0.25, "curv_speed": 0.20, "min_speed": 0.15,
        "heading": 0.10, "braking": 0.08, "corner": 0.05, "center": 0.02},
}


class PerWaypointVPerpTracker:
    """EMA of v_perp experienced near each waypoint.
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
    """Adaptive per-WP speed budget. REF: Heilmeier et al. (2020) Eq. 10."""
    _MIN_BUDGET = 0.80

    def __init__(self, n_waypoints: int, init_speed: float = 4.0):
        self.n = max(int(n_waypoints), 1)
        self._budget = np.full(self.n, float(init_speed), dtype=np.float32)

    def crash_at(self, wp_idx: int):
        i = int(wp_idx) % self.n
        for offset, w in [(0, 1.0), (1, 0.6), (-1, 0.6), (2, 0.3), (-2, 0.3)]:
            j = (i + offset) % self.n
            self._budget[j] = max(self._MIN_BUDGET, self._budget[j] * (1.0 - _BETA * w))

    def get(self, wp_idx: int) -> float:
        return float(self._budget[int(wp_idx) % self.n])

    def relax(self, wp_idx: int, amount: float = 0.01):
        i = int(wp_idx) % self.n
        self._budget[i] = min(4.0, self._budget[i] + amount)


class KalmanSignatureDetector:
    """Pre-crash triple-signature detector. v1.4.1: returns multiplier [0.7,1.0].
    REF: Schulman et al. (2017) -- prospective penalty in short episodes.
    """
    WINDOW   = 5
    MULT_MIN = 0.70

    def __init__(self):
        self._prog_buf  = deque(maxlen=self.WINDOW)
        self._spd_buf   = deque(maxlen=self.WINDOW)
        self._steer_buf = deque(maxlen=self.WINDOW)
        self._triggered = False

    def step(self, progress: float, speed: float, steer: float) -> float:
        """Returns compliance multiplier [MULT_MIN, 1.0]."""
        self._prog_buf.append(float(progress))
        self._spd_buf.append(float(speed))
        self._steer_buf.append(abs(float(steer)))
        if len(self._prog_buf) < self.WINDOW:
            return 1.0
        prog_trend = float(self._prog_buf[-1]) - float(self._prog_buf[0])
        spd_change = abs(float(self._spd_buf[-1]) - float(self._spd_buf[0]))
        steer_high = float(np.mean(list(self._steer_buf))) > 0.35
        if prog_trend > 0.001 and spd_change > 0.3 and steer_high:
            if not self._triggered:
                self._triggered = True
                return self.MULT_MIN
        else:
            self._triggered = False
        return 1.0

    def reset(self):
        self._prog_buf.clear(); self._spd_buf.clear(); self._steer_buf.clear()
        self._triggered = False


class BSTSPhaseController:
    """BSTS-Kalman-trend-driven phase transitions (v1.4.1).
    Complements TelemetryFeedbackAnnealer (v1.4.2) which operates at episode level.

    Phase 0 -> 1: track_progress_trend > PHASE0_PROGRESS_THRESH for 5 eps.
    Phase 1 -> 2: race_line_compliance_gradient_trend > PHASE1_LINE_THRESH for 5 eps.
    Any -> 0: track_progress_trend < REVERT_THRESH for REVERT_WINDOW eps.
    """
    def __init__(self):
        self._phase = 0
        self._ep_above_p0 = 0
        self._ep_above_p1 = 0
        self._recent = deque(maxlen=_REVERT_WINDOW)

    @property
    def current_phase(self) -> int:
        return self._phase

    def update_telemetry(self, progress_trend: float, line_trend: float, ep: int):
        self._recent.append(float(progress_trend))
        if (len(self._recent) >= _REVERT_WINDOW and
                all(t < _REVERT_THRESH for t in self._recent)):
            if self._phase > 0:
                self._phase = 0
                self._ep_above_p0 = 0
                self._ep_above_p1 = 0
            return
        if self._phase == 0:
            if float(progress_trend) > _PHASE0_PROGRESS_THRESH and ep >= _PHASE0_MIN_EPS:
                self._ep_above_p0 += 1
                if self._ep_above_p0 >= 5:
                    self._phase = 1
                    self._ep_above_p0 = 0
            else:
                self._ep_above_p0 = 0
        elif self._phase == 1:
            if float(line_trend) > _PHASE1_LINE_THRESH:
                self._ep_above_p1 += 1
                if self._ep_above_p1 >= 5:
                    self._phase = 2
                    self._ep_above_p1 = 0
            else:
                self._ep_above_p1 = 0

    def get_weights(self) -> Dict[str, float]:
        return dict(_PHASE_WEIGHTS[self._phase])

    def phase_label(self) -> str:
        return ["phase0_finish_first", "phase1_find_line", "phase2_go_fast"][self._phase]


class AdaptiveRewardShaper:
    """All-in-one reward shaping. v1.4.1+v1.4.2 merged.

    v1.4.2 DIVISION penalty: reward / (1 + w * excess) -- never negative.
    v1.4.1 BSTS-Kalman phase: BSTSPhaseController for Kalman-trend-level transitions.
    v1.4.2 TelemetryFeedback: update_phase(bsts_metrics) for episode-level transitions.
    Both phase systems coexist; get_phase_weights() uses whichever is further advanced.

    API (backward compatible with v1.4.2):
      episode_start(is_reversed) -> float  [-5.0 or 0.0]
      shape(reward, rp, ...) -> (float, dict)
      get_phase_weights(base_rw={}) -> dict
      update_phase(bsts_metrics) -> int
      speed_budget(wp_idx) -> float
      curb_urgency_mul(wp_idx) -> float
      bc_pilot_reward(rp, prev_m, curr_m) -> float
      episode_end(wp_idx, is_offtrack)
    """

    def __init__(self, n_waypoints: int = 120, border_dist_gate: float = _BORDER_DIST_GATE):
        self.n = max(int(n_waypoints), 1)
        self._vperp_tracker = PerWaypointVPerpTracker(self.n)
        self._speed_budget  = PerWaypointSpeedBudget(self.n)
        self._sig_detector  = KalmanSignatureDetector()
        self._phase_ctrl    = BSTSPhaseController()   # v1.4.1 Kalman-trend
        self._border_gate   = float(border_dist_gate)
        self._reversed_ep   = False
        self._ep_count      = 0
        # v1.4.2: TelemetryFeedbackAnnealer state
        self._current_phase        = 0
        self._tfa_completion_ema   = 0.0
        self._tfa_adherence_ema    = 0.5
        self._tfa_any_completion   = False
        self._tfa_alpha            = 0.10

    @property
    def current_phase(self) -> int:
        # Use maximum of both phase controllers (more advanced wins)
        return max(self._current_phase, self._phase_ctrl.current_phase)

    def episode_start(self, is_reversed: bool,
                      bsts_trends: Optional[Dict] = None,
                      episode_count: int = 0) -> float:
        """Returns -5.0 for reversed (additive, step 1 only), 0.0 otherwise.
        If bsts_trends provided, updates BSTSPhaseController.
        REF: AWS DeepRacer docs (2020); Schulman et al. (2017).
        """
        self._ep_count += 1
        self._reversed_ep = bool(is_reversed)
        self._sig_detector.reset()
        if bsts_trends is not None:
            prog_trend = float(bsts_trends.get("track_progress", 0.0))
            line_trend = float(bsts_trends.get("race_line_compliance_gradient", 0.0))
            self._phase_ctrl.update_telemetry(prog_trend, line_trend, episode_count)
        return -5.0 if is_reversed else 0.0

    def episode_end(self, wp_idx: int, is_offtrack: bool = False):
        if is_offtrack:
            self._speed_budget.crash_at(int(wp_idx))
            self._vperp_tracker._k = min(3.0, self._vperp_tracker._k * 1.05)
        else:
            self._speed_budget.relax(int(wp_idx), amount=0.02)
            self._vperp_tracker._k = max(1.0, self._vperp_tracker._k * 0.98)

    def update_phase(self, bsts_metrics: dict) -> int:
        """v1.4.2 TelemetryFeedbackAnnealer. REF: Almakhayita et al. (2025) PLoS ONE."""
        a  = self._tfa_alpha
        _c = float(bsts_metrics.get("completion_pct", 0.0) or 0.0)
        _a = float(bsts_metrics.get("race_line_adherence", 0.5) or 0.5)
        _l = float(bsts_metrics.get("lap_completed", 0.0) or 0.0)
        self._tfa_completion_ema = (1-a)*self._tfa_completion_ema + a*_c
        self._tfa_adherence_ema  = (1-a)*self._tfa_adherence_ema  + a*_a
        if _l > 0.5:
            self._tfa_any_completion = True
        if self._current_phase == 0 and self._tfa_any_completion:
            self._current_phase = 1
        elif (self._current_phase == 1
              and self._tfa_completion_ema >= 0.80
              and self._tfa_adherence_ema >= 0.45):
            self._current_phase = 2
        return self._current_phase

    def get_phase_weights(self, base_rw: Optional[Dict] = None) -> dict:
        """Returns phase-appropriate reward weights. Accepts optional base_rw for v1.4.2 compat.
        Uses maximum-advanced phase from both controllers.
        Phase 0: progress-dominant (0.55). Phase 1: racing_line+progress. Phase 2: speed.
        """
        phase = self.current_phase
        if phase == 0:
            w = {"center":0.05, "heading":0.18, "racing_line":0.02, "braking":0.01,
                 "progress":0.62, "corner":0.02, "speed_steering":0.02, "curv_speed":0.00,
                 "min_speed":0.06, "completion":0.02, "decel":0.00, "obstacle":0.00, "steering":0.00}
        elif phase == 1:
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

    def phase_label(self) -> str:
        labels = ["phase0_finish_first", "phase1_find_line", "phase2_go_fast"]
        return labels[min(self.current_phase, 2)]

    def shape(self, reward: float, rp: dict, wp_idx: int, speed: float,
              v_perp_barrier: float, steer: float, ep_progress_m: float,
              is_offtrack: bool, dist_to_border: float = 5.0) -> Tuple[float, Dict]:
        """v1.4.2 DIVISION penalty: shaped = base / (1 + w*excess). Always >= 0.
        v1.4.1 sig_mul: KalmanSignatureDetector returns multiplier [0.7, 1.0].
        Combined: shaped = (base / vperp_factor) * sig_mul.
        REF: Ng et al. (1999); Heinzmann & Zelinsky (2003); Amodei et al. (2016).
        """
        shaped  = float(reward)
        _is_rev = bool(rp.get("is_reversed", self._reversed_ep))

        if _is_rev:
            return 0.0, {"ars_reversed": True, "ars_vperp_attn": 1.0,
                         "ars_sig_mul": 1.0, "ars_compliance_mul": 0.0, "ars_phase": self.current_phase}

        _d  = float(dist_to_border) if dist_to_border is not None else 5.0
        _vp = float(v_perp_barrier) if v_perp_barrier is not None else 0.0
        vperp_attn = 1.0
        if _d < self._border_gate and _vp > _VPERP_THRESHOLD:
            excess = _vp - _VPERP_THRESHOLD
            vperp_attn = 1.0 + _VPERP_ATTN_W * excess
            shaped = shaped / vperp_attn  # division: never negative

        prog    = float(rp.get("progress", ep_progress_m or 0.0))
        sig_mul = self._sig_detector.step(prog, float(speed), float(steer))
        shaped  = shaped * sig_mul  # multiplier [0.7, 1.0]

        self._vperp_tracker.update(int(wp_idx), _vp)
        if not is_offtrack and float(speed) > 0.3:
            self._speed_budget.relax(int(wp_idx), amount=0.005)

        return shaped, {
            "ars_reversed":       _is_rev,
            "ars_vperp_attn":     round(vperp_attn, 4),
            "ars_vperp_mul":      round(1.0 / vperp_attn, 4),   # compat field
            "ars_sig_mul":        round(sig_mul, 4),
            "ars_compliance_mul": round(sig_mul / vperp_attn, 4),
            "ars_wp":             int(wp_idx),
            "ars_phase":          self.current_phase,
        }

    def speed_budget(self, wp_idx: int) -> float:
        return self._speed_budget.get(int(wp_idx))

    def curb_urgency_mul(self, wp_idx: int) -> float:
        return self._vperp_tracker.curb_urgency_mul(int(wp_idx))

    def bc_pilot_reward(self, rp: dict, prev_progress_m: float, curr_progress_m: float) -> float:
        """r = phase_alpha * delta_progress_m. REF: Ng et al. (1999)."""
        if rp.get("is_reversed", False):
            return 0.0
        delta_m = float(curr_progress_m) - float(prev_progress_m)
        _phase_alpha = [0.15, 0.30, 0.60][min(self.current_phase, 2)]
        return float(np.clip(_phase_alpha * max(0.0, delta_m), -10.0, 10.0))
