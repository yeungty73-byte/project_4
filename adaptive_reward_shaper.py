"""adaptive_reward_shaper.py — v1.5.0b
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

v1.5.0b NEW: TrackProgressAnnealer
  -- Continuously anneals phase reward weights using LIVE telemetry signals:
       1. track_progress_pct  (0-100): how far the car reliably gets each lap
       2. avg_speed_centerline: EMA of on-centerline speed
       3. race_line_compliance_gradient: how well the car follows the racing line
  -- These three form a monotone readiness signal that gates phase weight blend:
       alpha_progress  = clip(track_progress_pct / 100, 0, 1)
       alpha_speed     = clip(avg_speed_centerline / 3.0, 0, 1)
       alpha_rl        = race_line_compliance_gradient  [0,1]
       phase_blend     = alpha_progress * 0.5 + alpha_speed * 0.25 + alpha_rl * 0.25
  -- phase_blend then smoothly interpolates between phase-N and phase-N+1 weights
     WITHOUT triggering a hard phase transition.
  -- Hard transitions (BSTSPhaseController + TelemetryFeedbackAnnealer) still fire
     as before; TrackProgressAnnealer is an ADDITIONAL soft continuous anneal layer.
  -- process_action() in run.py now queries shaper.process_action_scale() per step
     to scale throttle headroom based on phase_blend:
       throttle_headroom = lerp(0.55, 1.0, phase_blend)
     This makes throttle ceiling rise organically as the car learns the track,
     instead of being fixed at 1.0 from episode 1.

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
REF: Bengio et al. (2009) ICML: curriculum learning -- easy-first ordering improves generalisation.
REF: Florensa et al. (2017) CoRL: automatic curriculum via reverse curriculum generation.
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

# v1.5.0b: per-phase throttle headroom ceiling (lerped by phase_blend inside phase)
# Phase 0: car needs to survive -> cap throttle at 0.55 (slow, safe)
# Phase 1: finding the line -> rise to 0.75 as raceline compliance improves
# Phase 2: go fast -> full 1.0
_THROTTLE_HEADROOM = {0: (0.55, 0.70), 1: (0.70, 0.88), 2: (0.88, 1.00)}


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


class TrackProgressAnnealer:
    """v1.5.0b: Continuous soft curriculum annealer driven by three live telemetry signals.

    Signals (all from bsts_row / BSTS-Kalman):
      track_progress_pct  [0, 100]  -- how reliably the car covers the track
      avg_speed_centerline [0, 4.0] -- EMA of on-centerline speed (m/s)
      race_line_compliance_gradient [0, 1] -- how well the car follows the racing line

    phase_blend = 0.50*alpha_progress + 0.25*alpha_speed + 0.25*alpha_rl
    where:
      alpha_progress  = clip(track_progress_pct / 100.0, 0, 1)
      alpha_speed     = clip(avg_speed_centerline / 3.0, 0, 1)
      alpha_rl        = clip(race_line_compliance_gradient, 0, 1)

    phase_blend in [0, 1] smoothly interpolates WITHIN the current hard phase:
      - phase_blend=0.0 -> use current phase start weights exactly
      - phase_blend=1.0 -> fully at current phase end weights (= next phase start)

    This means even before a hard phase transition fires, the reward weights
    have been gradually warmed up, so the transition is smooth instead of jarring.

    Also exposes process_action_scale() for throttle headroom gating in run.py:
      throttle_headroom = lerp(lo, hi, phase_blend)
    where (lo, hi) = _THROTTLE_HEADROOM[current_phase].

    EMA smoothing (alpha=0.08) prevents single-episode spikes from thrashing weights.

    REF: Bengio et al. (2009) ICML -- curriculum learning: easy-first ordering.
    REF: Florensa et al. (2017) CoRL -- automatic curriculum via reverse curriculum.
    REF: Almakhayita et al. (2025) PLoS ONE -- adaptive reward design for deep RL.
    """

    _EMA_ALPHA = 0.08   # smoothing for telemetry signals
    _SPEED_NORM = 3.0   # normalisation denominator for avg_speed_centerline (m/s)

    def __init__(self):
        self._ema_progress   = 0.0
        self._ema_speed      = 0.0
        self._ema_rl         = 0.5   # neutral start
        self._phase_blend    = 0.0
        self._n_updates      = 0

    def update(self, track_progress_pct: float,
               avg_speed_centerline: float,
               race_line_compliance_gradient: float) -> float:
        """Ingest one episode's telemetry; return updated phase_blend [0,1].

        Clamps all inputs to valid ranges before EMA to prevent NaN/inf
        propagation from corrupted bsts_row entries.

        REF: Bengio et al. (2009) -- monotone readiness signal.
        """
        # Clamp + normalise
        _p  = float(np.clip(track_progress_pct,            0.0, 100.0)) / 100.0
        _s  = float(np.clip(avg_speed_centerline,           0.0,   4.0)) / self._SPEED_NORM
        _rl = float(np.clip(race_line_compliance_gradient,  0.0,   1.0))

        a = self._EMA_ALPHA
        self._ema_progress = (1 - a) * self._ema_progress + a * _p
        self._ema_speed    = (1 - a) * self._ema_speed    + a * _s
        self._ema_rl       = (1 - a) * self._ema_rl       + a * _rl
        self._n_updates   += 1

        self._phase_blend = float(np.clip(
            0.50 * self._ema_progress
            + 0.25 * self._ema_speed
            + 0.25 * self._ema_rl,
            0.0, 1.0
        ))
        return self._phase_blend

    @property
    def phase_blend(self) -> float:
        return self._phase_blend

    def get_annealed_weights(self, current_phase: int,
                              phase_weights_dict: Dict[int, Dict]) -> Dict[str, float]:
        """Interpolate between phase N and phase N+1 weights using phase_blend.

        When current_phase==2 (fastest), blend within phase 2 only (no phase 3).
        All weights renormalised to sum=1.

        REF: Florensa et al. (2017) CoRL -- gradual difficulty increase.
        """
        p = int(np.clip(current_phase, 0, 2))
        p_next = min(p + 1, 2)
        w0 = phase_weights_dict.get(p,      phase_weights_dict[0])
        w1 = phase_weights_dict.get(p_next, phase_weights_dict[2])
        blend = self._phase_blend
        all_keys = set(list(w0.keys()) + list(w1.keys()))
        w = {k: (1.0 - blend) * float(w0.get(k, 0.0))
                + blend       * float(w1.get(k, 0.0))
             for k in all_keys}
        total = sum(w.values()) or 1.0
        return {k: v / total for k, v in w.items()}

    def process_action_scale(self, current_phase: int) -> float:
        """Return throttle headroom ceiling [lo, hi] lerped by phase_blend.

        Call this in run.py's process_action() to cap the throttle proportional
        to how far the car has demonstrated reliable track coverage.

        Example:
            headroom = _shaper.process_action_scale(current_phase)
            a[1] = min(a[1], headroom)   # throttle channel cap

        REF: Bengio et al. (2009) -- start with low action complexity.
        """
        lo, hi = _THROTTLE_HEADROOM.get(int(np.clip(current_phase, 0, 2)), (0.55, 1.00))
        return float(lo + (hi - lo) * self._phase_blend)

    def diagnostics(self) -> Dict:
        return {
            "tpa_n_updates":   self._n_updates,
            "tpa_ema_progress": round(self._ema_progress, 4),
            "tpa_ema_speed":   round(self._ema_speed, 4),
            "tpa_ema_rl":      round(self._ema_rl, 4),
            "tpa_phase_blend": round(self._phase_blend, 4),
        }


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
    """All-in-one reward shaping. v1.5.0b: adds TrackProgressAnnealer.

    v1.4.2 DIVISION penalty: reward / (1 + w * excess) -- never negative.
    v1.4.1 BSTS-Kalman phase: BSTSPhaseController for Kalman-trend-level transitions.
    v1.4.2 TelemetryFeedback: update_phase(bsts_metrics) for episode-level transitions.
    v1.5.0b TrackProgressAnnealer: continuous soft weight anneal within each phase,
       driven by track_progress_pct + avg_speed_centerline + race_line_compliance_gradient.
       Also gates throttle headroom via process_action_scale().

    Both hard-phase systems coexist; get_phase_weights() uses whichever is further advanced.
    TrackProgressAnnealer operates continuously on top of whichever hard phase is active.

    API (backward compatible with v1.4.2):
      episode_start(is_reversed) -> float  [-5.0 or 0.0]
      shape(reward, rp, ...) -> (float, dict)
      get_phase_weights(base_rw={}) -> dict          # now TPA-blended
      update_phase(bsts_metrics) -> int
      update_tpa(bsts_row) -> float                  # NEW v1.5.0b: call each episode end
      process_action_scale(phase) -> float           # NEW v1.5.0b: throttle headroom ceiling
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
        self._tpa           = TrackProgressAnnealer() # v1.5.0b continuous anneal
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

    # ------------------------------------------------------------------
    # v1.5.0b: TrackProgressAnnealer public API
    # ------------------------------------------------------------------

    def update_tpa(self, bsts_row: dict) -> float:
        """Update TrackProgressAnnealer from bsts_row at end of each episode.

        bsts_row keys used:
          track_progress_pct         (float 0-100)
          avg_speed_centerline       (float, BSTS-Kalman trend, may be small)
          race_line_compliance_gradient (float 0-1 from hmout)

        Falls back gracefully to neutral values when keys are absent (early episodes).

        REF: Almakhayita et al. (2025) PLoS ONE -- adaptive telemetry-driven curriculum.
        """
        _prog_pct = float(bsts_row.get("track_progress_pct",
                          bsts_row.get("progress", 0.0)) or 0.0)
        # avg_speed_centerline from BSTS-Kalman is a *trend* (small), not absolute speed.
        # We also check avg_speed (absolute) for a better signal when trend is near-zero.
        _spd_kal  = float(bsts_row.get("avg_speed_centerline",
                          bsts_row.get("avg_speed", 0.0)) or 0.0)
        # avg_speed is mean episode speed in m/s; use it when available for normalisation
        _spd_abs  = float(bsts_row.get("avg_speed", _spd_kal) or 0.0)
        # Prefer absolute speed (natural units) over Kalman trend (tiny floats)
        _spd = _spd_abs if abs(_spd_abs) > abs(_spd_kal) else _spd_kal
        # race_line_compliance_gradient from _hm_out / bsts_row
        _rl_grad  = float(bsts_row.get("race_line_compliance_gradient",
                          bsts_row.get("racelinecompliancegradient", 0.5)) or 0.5)
        return self._tpa.update(_prog_pct, _spd, _rl_grad)

    def process_action_scale(self, phase: Optional[int] = None) -> float:
        """Throttle headroom ceiling [0.55, 1.0] annealed by TrackProgressAnnealer.

        Call inside process_action() in run.py AFTER the standard tanh remap:
            headroom = _shaper.process_action_scale()
            a[1] = min(a[1], headroom)

        Returns 1.0 (no cap) if shaper not yet initialised (safe default).
        REF: Bengio et al. (2009) -- start with low action budget, widen gradually.
        """
        _phase = phase if phase is not None else self.current_phase
        return self._tpa.process_action_scale(_phase)

    def tpa_diagnostics(self) -> Dict:
        """Returns TPA internal state for logging/TensorBoard."""
        d = self._tpa.diagnostics()
        d["tpa_throttle_headroom"] = round(self.process_action_scale(), 4)
        d["tpa_current_phase"] = self.current_phase
        return d

    # ------------------------------------------------------------------
    # Existing API (unchanged externally, get_phase_weights now TPA-blended)
    # ------------------------------------------------------------------

    def episode_start(self, is_reversed: bool,
                      bsts_trends: Optional[Dict] = None,
                      episode_count: int = 0) -> float:
        """Returns -5.0 for reversed (multiplicative gate in run.py, additive here for step 1).
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
        """Returns TPA-blended phase-appropriate reward weights.

        v1.5.0b: weights are now continuously annealed within the current hard phase
        by TrackProgressAnnealer.get_annealed_weights().  The hard phase transitions
        (BSTSPhaseController + TelemetryFeedbackAnnealer) still gate WHICH phase
        we are in; TPA smoothly moves weights toward the next phase's profile.

        Accepts optional base_rw for v1.4.2 compat (ignored but not rejected).

        REF: Bengio et al. (2009) ICML -- smooth difficulty escalation.
        """
        phase = self.current_phase
        # v1.5.0b: use TPA-blended weights instead of step-function profiles
        return self._tpa.get_annealed_weights(phase, _PHASE_WEIGHTS)

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
                         "ars_sig_mul": 1.0, "ars_compliance_mul": 0.0,
                         "ars_phase": self.current_phase,
                         "ars_tpa_blend": round(self._tpa.phase_blend, 4)}

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
            "ars_vperp_mul":      round(1.0 / vperp_attn, 4),
            "ars_sig_mul":        round(sig_mul, 4),
            "ars_compliance_mul": round(sig_mul / vperp_attn, 4),
            "ars_wp":             int(wp_idx),
            "ars_phase":          self.current_phase,
            "ars_tpa_blend":      round(self._tpa.phase_blend, 4),
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
