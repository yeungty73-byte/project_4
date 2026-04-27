"""BSTS module: canonical home for BSTSFeedback + BSTSSeasonal.

Classes
-------
BSTSKalmanFilter -- Local-linear-trend 2-state Kalman filter (NEW v1.1.1)
BSTSFeedback     -- EMA-smoothed, Kalman-informed, race_type-aware reward-weight adjuster.
BSTSSeasonal     -- Batch BSTS decomposition + live per-step buffer for run.py integration.

v1.3.1 changes (2026-04-27)
----------------------------
  - BSTSKalmanFilter.update(): NaN/Inf GUARD.
    Skip non-finite observations instead of poisoning state.
    After posterior update, recover with np.nan_to_num if state leaked nan.
    REF: measurement-gating in Thrun, Burgard & Fox (2005) Probabilistic Robotics.
  - BSTSFeedback.update(): Skip non-finite metric values in EMA loop.
    Prevents nan EMAs from propagating into _run_kalmans().

v1.1.1 changes
--------------
  - BSTSKalmanFilter: proper 2-state (level+slope) Kalman with diffuse P prior,
    spike-and-slab beta shrinkage (_apply_spike_slab), and adaptive sigma_obs
    (_adapt_sigma_obs). REF: Welch & Bishop (2006); Scott & Varian (2014);
    George & McCulloch (1993).
  - BSTSFeedback.__init__: self._kf dict now lazily spawns BSTSKalmanFilter
    instances per metric. update() calls _run_kalmans() at the end of every
    call so kf_trends / kf_betas are always populated after the first update().
  - adjust_weights() corner-crash guard (v1.1.1): only fires after >10% avg
    progress to avoid false positives from start-box barrier hits.

REF: Scott, S. L. & Varian, H. R. (2014). Predicting the present with Bayesian
     structural time series. Int. J. Math. Model. Numer. Optim., 5(1-2), 4-23.
REF: Brodersen, K. H. et al. (2015). Inferring causal impact using Bayesian
     structural time-series models. Ann. Appl. Stat., 9(1), 247-274.
REF: Welch, G. & Bishop, G. (2006). An Introduction to the Kalman Filter.
     UNC-Chapel Hill Tech Report TR 95-041.
REF: George, E. I. & McCulloch, R. E. (1993). Variable selection via Gibbs
     sampling. JASA, 88(423), 881-889.
"""
import json, os, math, collections
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Denim theme palette (shared by BSTSSeasonal plots)
# ---------------------------------------------------------------------------
DENIM_DARK   = '#1B3A5C'
DENIM_MID    = '#3A6B8C'
DENIM_BRIGHT = '#5B9BD5'
AMBER        = '#D4A03C'
TERRA_COTTA  = '#C75B39'
GOLD_WARM    = '#E8C167'
SAGE_GREEN   = '#6BB38A'
MUTED_PURPLE = '#9B6EB7'
BG_DARK      = '#2E2E2E'
BG_LIGHT     = '#F0EDE6'
MUTED_GRAY   = '#8C8C8C'
WHITE_SMOKE  = '#F5F5F5'

COMPLEMENTARY_PAIRS = [
    (DENIM_BRIGHT, AMBER),
    (SAGE_GREEN, TERRA_COTTA),
    (MUTED_PURPLE, GOLD_WARM),
    (DENIM_MID, TERRA_COTTA),
]

from analyze_logs import SUCCESS_METRICS as SUCCESS_KEYS, INTERMEDIARY_METRICS as REGRESSOR_KEYS

# ---------------------------------------------------------------------------
# Normalised race_type token
# ---------------------------------------------------------------------------
_RACE_TYPE_MAP = {
    "time_trial":        "TIME_TRIAL",
    "timetrial":         "TIME_TRIAL",
    "tt":                "TIME_TRIAL",
    "object_avoidance":  "OBJECT_AVOIDANCE",
    "objectavoidance":   "OBJECT_AVOIDANCE",
    "oa":                "OBJECT_AVOIDANCE",
    "head_to_bot":       "HEAD_TO_BOT",
    "headtobot":         "HEAD_TO_BOT",
    "h2b":               "HEAD_TO_BOT",
    "h2h":               "HEAD_TO_BOT",
}

def _norm_race_type(raw: str) -> str:
    return _RACE_TYPE_MAP.get(str(raw).lower().replace(" ", "_"), "TIME_TRIAL")

class _NullModel:
    def get_season(self): return {'worst_segments': [], 'segments': {}}
    def get_seasonal(self): return {}
    def get_trend(self):
        # v1.1.1: never return {} — run.py calls .get('kf_level') on this
        return {'phase': 'plateau', 'return_slope': 0.0, 'trend_last': 0.0,
                'trend_mean': 0.0, 'kf_level': 0.0, 'kf_trend': 0.0,
                'kf_season': 0.0, 'kf_innov': 0.0}
        
# ===========================================================================
# BSTSKalmanFilter  (local-linear-trend 2-state Kalman) — NEW v1.1.1
# ===========================================================================
class BSTSKalmanFilter:
    """2-state local-linear-trend Kalman filter for a scalar metric time series.

    State vector x = [level, slope]^T
    Transition:  x_t = F x_{t-1} + w,  w ~ N(0, Q)
    Observation: y_t = H x_t    + v,  v ~ N(0, R)

    Additional components (preserved from project spec):
      _apply_spike_slab()  -- George & McCulloch (1993) soft beta shrinkage
      _adapt_sigma_obs()   -- Scott & Varian (2014) rolling residual variance

    After update(), self.slope is the posterior slope -> kf_trends[metric].

    Parameters
    ----------
    obs_noise    : float  Initial observation noise variance R (adapted online)
    level_noise  : float  Process noise on level state (Q[0,0])
    slope_noise  : float  Process noise on slope state (Q[1,1])
    p            : int    Number of regression coefficients for spike-and-slab
    S            : int    Number of seasonal states (for state index calculation)

    REF: Welch, G. & Bishop, G. (2006). An Introduction to the Kalman Filter.
         UNC TR 95-041.
    REF: Scott & Varian (2014) BSTS — Inv-Gamma prior on sigma_obs.
    REF: George & McCulloch (1993) Variable selection via Gibbs sampling.
    """

    def __init__(self, obs_noise: float = 1.0,
                 level_noise: float = 0.05,
                 slope_noise: float = 0.005,
                 p: int = 0,
                 S: int = 0):
        # Transition matrix: [level, slope] random walk
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]], dtype=float)
        # Observation model: observe level only
        self.H = np.array([[1.0, 0.0]], dtype=float)
        # Process noise covariance
        self.Q = np.diag([level_noise, slope_noise]).astype(float)
        # Observation noise (scalar variance, adapted by _adapt_sigma_obs)
        self.R = float(obs_noise)
        # Posterior state (level=0, slope=0)
        self.x = np.array([0.0, 0.0], dtype=float)
        # Posterior covariance — DIFFUSE PRIOR (large diagonal = high initial uncertainty)
        # REF: Thrun, Burgard & Fox (2005) Probabilistic Robotics — P must not be zeros.
        # P = zeros -> K = 0 on every step -> filter never moves. Use 1e4 * I.
        self.P = np.eye(2, dtype=float) * 1e4

        # Spike-and-slab support (George & McCulloch 1993)
        self.p = int(p)          # number of regression betas
        self.S = int(S)          # number of seasonal states
        # Extend state vector if regression betas are tracked
        if self.p > 0:
            beta_dim = 2 + self.S + self.p
            self.state = np.zeros(beta_dim, dtype=float)
        else:
            self.state = self.x   # alias; only level+slope tracked

        # Adaptive sigma internals
        self._residuals = []

        self._initialized = False
        self._n_obs = 0

    # ------------------------------------------------------------------
    def update(self, observation: float) -> None:
        """Ingest one scalar observation and update posterior.

        v1.3.1 NaN/Inf guard:
          - Non-finite observations (nan, ±inf) are SKIPPED entirely.
            A single nan would poison the entire state vector forever:
            innov = nan → K@[[nan]] = nan → x = [nan, nan] → slope = nan.
            Skipping is equivalent to measurement-gating (Thrun et al. 2005).
          - After the posterior update, any nan/inf that leaked through
            numerical edge cases is recovered with np.nan_to_num(0.0).
        REF: Thrun, Burgard & Fox (2005) Probabilistic Robotics §3.4
             measurement gating — reject outlier/corrupt observations.
        """
        z = float(observation)

        # v1.3.1: skip non-finite observations — do NOT poison Kalman state
        if not math.isfinite(z):
            return

        if not self._initialized:
            self.x[0] = z          # seed level with first obs
            if self.p > 0:
                self.state[0] = z
            self._initialized = True
            self._n_obs = 1
            return

        # --- Predict ---
        x_pred = self.F @ self.x
        P_pred  = self.F @ self.P @ self.F.T + self.Q

        # --- Innovation ---
        # H is shape (1,2), x_pred is shape (2,) → H@x_pred is shape (1,)
        # float() requires a 0-d array, NOT a 1-d array of length 1.
        # Fix: index [0] to extract the scalar element first.
        z_pred = float((self.H @ x_pred)[0])
        innov = z - z_pred
        self._residuals.append(innov)

        # --- Kalman gain ---
        R_mat = np.array([[self.R]])
        S_mat = self.H @ P_pred @ self.H.T + R_mat
        K     = P_pred @ self.H.T @ np.linalg.inv(S_mat)   # shape (2,1)

        # --- Posterior update ---
        self.x = x_pred + (K @ np.array([[innov]])).squeeze()
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        # v1.3.1: recover nan/inf that may have leaked through numerical edge cases
        if not np.all(np.isfinite(self.x)):
            self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(self.P)):
            self.P = np.nan_to_num(self.P, nan=1e4, posinf=1e4, neginf=0.0)

        if self.p > 0:
            self.state[:2] = self.x

        self._n_obs += 1

        # Adaptive components (every 10 steps)
        if self._n_obs % 10 == 0:
            self._adapt_sigma_obs()
        if self.p > 0 and self._n_obs % 5 == 0:
            self._apply_spike_slab()

    # ------------------------------------------------------------------
    def _apply_spike_slab(self):
        """Soft spike-and-slab beta shrinkage via pseudo-inclusion probability.

        REF: George & McCulloch (1993) Variable selection via Gibbs sampling.
             JASA, 88(423), 881-889.

        Approximation: betas with |beta| < inclusion_threshold are shrunk by 50%
        each step. This mimics the Gibbs spike-and-slab prior without full MCMC.
        Betas that earn their keep (|beta| > threshold) are left alone.
        """
        _beta_start = 2 + self.S - 1   # index into state vector after level+trend+seasonals
        _inclusion_threshold = 0.05
        for i in range(self.p):
            idx = _beta_start + i
            if idx < len(self.state):
                beta_i = self.state[idx]
                # Soft-threshold: shrink near-zero betas toward 0 (exclude them)
                if abs(beta_i) < _inclusion_threshold:
                    self.state[idx] *= 0.5   # 50% shrinkage per step

    # ------------------------------------------------------------------
    def _adapt_sigma_obs(self):
        """Adaptive observation noise sigma from rolling residual variance.

        REF: Scott & Varian (2014) use Inverse-Gamma prior on sigma_obs.
             Here we approximate with an online rolling variance update every 10 steps.
        """
        if len(self._residuals) >= 20:
            rolling_var = float(np.var(self._residuals[-20:]) + 1e-6)
            # Smooth update: don't jump sigma_obs abruptly
            self.R = 0.8 * self.R + 0.2 * rolling_var

    # ------------------------------------------------------------------
    @property
    def level(self) -> float:
        return float(self.x[0])

    @property
    def slope(self) -> float:
        """Posterior slope estimate — used as kf_trends[metric]."""
        return float(self.x[1])

    @property
    def n_obs(self) -> int:
        return self._n_obs


# ===========================================================================
# BSTSFeedback  (transplanted from run.py v1.0.13, race_type-aware)
# ===========================================================================
class BSTSFeedback:
    """EMA-smoothed, Kalman-informed reward-weight adjuster.

    v1.3.1 (2026-04-27): NaN guard in update().
      Non-finite metric values (nan/inf) are now skipped in the EMA loop.
      Previously a nan reward EMA would propagate into _run_kalmans() and
      poison all downstream Kalman slopes to nan.

    v1.3.0 (2026-04-27): Per-class compliance signals integrated.
      Recognises brake_field_compliance_gradient and race_line_compliance_gradient
      from CombinedBrakeField (brake_field.py v1.3.0) and wires them directly
      through EMA + Kalman trends → adjust_weights() reward shaping.

      Object classes tracked (from SwinUNetPP.classify_sectors_for_brake_field()):
        0 center_line | 1 curb | 2 obstacle | 3 bot
      Their per-sector mask probabilities are accepted as update() kwargs:
        swin_curb_prob, swin_obs_prob, swin_bot_prob

    v1.1.1: self._kf lazily spawns BSTSKalmanFilter per metric.
    update() calls _run_kalmans() so kf_trends/kf_betas are always populated.

    REF:
      Scott, S. L. & Varian, H. R. (2014). Predicting the present with Bayesian
        structural time series. Int. J. Math. Model. Numer. Optim., 5(1-2), 4-23.
      Brodersen, K. H. et al. (2015). Inferring causal impact using Bayesian
        structural time-series models. Ann. Appl. Stat., 9(1), 247-274.
      Durbin, J. & Koopman, S. J. (2012). Time Series Analysis by State Space
        Methods (2nd ed.). Oxford University Press.
      Kalman, R. E. (1960). A new approach to linear filtering and prediction
        problems. J. Basic Eng., 82(1), 35-45.
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        feedback_strength: float = 0.15,
        race_type: str = "TIME_TRIAL",
        kf_obs_noise: float = 1.0,
        kf_level_noise: float = 0.05,
        kf_slope_noise: float = 0.005,
    ):
        self.ema      = {}
        self.alpha    = float(ema_alpha)
        self.strength = float(feedback_strength)
        self.race_type = _norm_race_type(race_type)

        # KF hyperparams
        self._KF_OBS_NOISE   = float(kf_obs_noise)
        self._KF_LEVEL_NOISE = float(kf_level_noise)
        self._KF_SLOPE_NOISE = float(kf_slope_noise)

        # Lazily instantiated KF instances: metric -> BSTSKalmanFilter
        self._kf: dict = {}

        # Public: populated by _run_kalmans() after each update()
        # v1.1.1: inline Kalman instances — lazily created by _run_kalmans()
        # Previously populated only by run.py analyze_logs pipeline (1x/episode, often 0).
        # Now populated every update() call via BSTSKalmanFilter.
        self._kf_instances: dict = {}
        self.kf_trends: dict = {}   # metric -> posterior slope (float)
        self.kf_betas:  dict = {}   # metric -> normalised slope/level ratio

        # History buffer for get_trend_vector()
        self._history: dict = {}
        self._HIST_LEN = 20

    # ------------------------------------------------------------------
    def set_race_type(self, race_type: str) -> None:
        self.race_type = _norm_race_type(race_type)

    # ------------------------------------------------------------------
    def model(self, metric: str, period: int = 100) -> "_NullModel":
        """Backward-compat stub for run.py bsts_season = bsts_feedback.model(...)."""
        return _NullModel()

    # ------------------------------------------------------------------
    def update(self, metrics_dict: dict, *, step=None, race_type_tag=None) -> None:
        """Feed new scalar metrics into EMA state and run Kalman update.

        v1.3.1: Non-finite values (nan/inf) are skipped in the EMA loop.
        A single nan metric (e.g., reward=nan from 0/0 ep_return) would
        otherwise propagate into _run_kalmans() and poison the Kalman state.
        """
        if race_type_tag is not None:
            self.race_type = _norm_race_type(str(race_type_tag))

        # v1.1.5c FIX-Q: normalise key aliases before ingesting so EMA/KF have both names.
        # avg_jerk → jerk_rms alias: adjust_weights._slope("jerk_rms") works without data.
        # track_progress_m → track_progress alias: bsts_metrics sends both; KF uses one.
        _KEY_ALIASES = {'avg_jerk': 'jerk_rms', 'track_progress_m': 'track_progress'}
        _aliased = dict(metrics_dict)
        for _ak, _av_key in _KEY_ALIASES.items():
            if _ak in _aliased:
                _aliased.setdefault(_av_key, _aliased[_ak])
        for k, v in _aliased.items():
            if not isinstance(v, (int, float)):
                continue
            v = float(v)
            # v1.3.1: skip non-finite values — do not let nan/inf enter EMA or Kalman
            if not math.isfinite(v):
                continue
            if k not in self.ema:
                self.ema[k] = v
            else:
                self.ema[k] = self.alpha * v + (1.0 - self.alpha) * self.ema[k]
            if k not in self._history:
                self._history[k] = collections.deque(maxlen=self._HIST_LEN)
            self._history[k].append(v)

        # v1.1.1: run Kalman update on every EMA metric
        self._run_kalmans()
        
    # ------------------------------------------------------------------
    # v1.1.4c: Episode-level-only metrics (computed once/ep, not per-step).
    # These use a slower KF (lower level_noise) because their signal is smoother
    # — one observation per episode vs tens of per-step observations for others.
    # Without this, the KF for race_line_adherence over-reacts to each single
    # episode observation, producing noisy slopes.
    _EP_LEVEL_METRICS = frozenset({
        'race_line_adherence', 'brake_compliance',
        'compliance_gradient', 'brake_field_compliance_gradient',
        'race_line_compliance_gradient', 'track_progress',
        'avg_speed_centerline', 'smoothness_steering_rate',
    })

    def _run_kalmans(self) -> None:
        """Update one KF per EMA metric; populate kf_trends/kf_betas.

        v1.1.4c: Episode-level metrics use slower KF (level_noise/10).
        v1.1.1: Called at the end of every update(). Previously kf_trends/kf_betas
        were only populated by run.py's analyze_logs pipeline (once per episode,
        inside a try-block with n=1 buffer → meaningless slopes).

        REF: Scott & Varian (2014) BSTS per-metric Kalman trend components.
        REF: George & McCulloch (1993) spike-and-slab shrinkage.
        """
        for metric, val in self.ema.items():
            if metric not in self._kf_instances:
                # v1.1.4c: episode-level metrics need slower KF to avoid
                # over-reacting to single-episode observations.
                if metric in self._EP_LEVEL_METRICS:
                    _ln = self._KF_LEVEL_NOISE * 0.1   # 10x slower adaptation
                    _sn = self._KF_SLOPE_NOISE * 0.1
                else:
                    _ln = self._KF_LEVEL_NOISE
                    _sn = self._KF_SLOPE_NOISE
                self._kf_instances[metric] = BSTSKalmanFilter(
                    obs_noise=self._KF_OBS_NOISE,
                    level_noise=_ln,
                    slope_noise=_sn,
                )
            self._kf_instances[metric].update(float(val))
            self.kf_trends[metric] = self._kf_instances[metric].slope
        new_betas = {}
        for metric, kf in self._kf_instances.items():
            if kf.n_obs >= 2 and abs(kf.level) > 1e-9:
                beta = kf.slope / (abs(kf.level) + 1e-9)
                if abs(beta) < 0.05:   # spike-and-slab soft shrinkage
                    beta *= 0.5
                new_betas[metric] = beta
        self.kf_betas = new_betas
        
    # ------------------------------------------------------------------
    def get_trend_vector(self, race_type_filter=None) -> dict:
        """Return {metric: linear_slope} over last 5 history observations.

        v1.1.4c NaN guards:
          1. n < 2: return 0.0 (np.diff of 1-element = empty → np.mean([]) = nan).
          2. Filter non-finite values from arr before polyfit.
             A single nan in the history window poisons np.polyfit → nan slope.
          3. polyfit result checked: if not finite → 0.0.
        REF: Welch & Bishop (2006) Kalman filter — measurement gating skips nan.
        """
        out = {}
        window = 5
        for k, dq in self._history.items():
            arr = list(dq)[-window:]
            # v1.1.4c: strip non-finite values before regression
            arr = [v for v in arr if v == v and v not in (float('inf'), float('-inf'))]
            n   = len(arr)
            if n < 2:
                out[k] = 0.0
                continue
            x = np.arange(n, dtype=float)
            try:
                slope = float(np.polyfit(x, arr, 1)[0])
                if not math.isfinite(slope):
                    slope = 0.0
            except Exception:
                slope = 0.0
            out[k] = slope
        return out

    # ------------------------------------------------------------------
    def _apply_curriculum_phase_gate(self, w):
        """
        v1.1.5d PATCH-Q: Phase A/B/C curriculum weight gate.
        REF: almakhayita2025curriculum — progress-first then line-following.
             Designed divergence brake_compliance >> race_line_compliance in Phase B.
             almakhayita2025curriculum note: 'Low jerk with notable acceleration
             changes near corners is Phase1 target. Convergence of brake-field
             and race-line compliance signals Phase2 onset.'
        avg_track_progress is EMA of (ep_centerline_progress_m / ep_track_length_m)
        populated by PATCH-V in run.py bsts_feedback.update().
        """
        import math
        avg_tp = float(self.ema.get('avg_track_progress', 0.0))
        if not math.isfinite(avg_tp):
            avg_tp = 0.0

        if avg_tp < 0.50:
            # Phase A: Survival — creep forward, minimise rl/speed pressure
            w['progress']    = max(w.get('progress', 0.0), 0.45)
            w['braking']     = max(w.get('braking', 0.0), 0.12)
            w['racing_line'] = min(w.get('racing_line', 0.99), 0.04)
            w['curv_speed']  = min(w.get('curv_speed', 0.99), 0.02)
            w['min_speed']   = min(w.get('min_speed', 0.99), 0.04)
        elif avg_tp < 0.90:
            # Phase B: Divergence — brake weight intentionally > raceline
            # Creates designed gap between brake_compliance and race_line_compliance
            _frac = (avg_tp - 0.50) / 0.40   # 0.0 at Phase B entry, 1.0 at exit
            _brk_floor = 0.18 * (1 - _frac) + 0.12 * _frac
            _rl_floor  = 0.04 * (1 - _frac) + 0.10 * _frac
            _spd_floor = 0.02 * (1 - _frac) + 0.08 * _frac
            w['braking']     = max(w.get('braking', 0.0), _brk_floor)
            w['racing_line'] = max(w.get('racing_line', 0.0), _rl_floor)
            w['curv_speed']  = max(w.get('curv_speed', 0.0), _spd_floor)
        # Phase C (>=0.90): no clamp — weights evolve freely toward convergence
        return w

    def adjust_weights(self, base_weights, *, race_type_filter=None) -> dict:
        """Return re-normalised reward weights adjusted by BSTS+Kalman feedback.

        v1.2.0: Compliance-gradient-aware shaping.
        Boost magnitude is PROPORTIONAL to (1 - compliance_gradient) so the
        reward signal targets exactly the behaviour the agent is weakest at.
        New signals: brake_field_compliance_gradient, race_line_compliance_gradient.
        REF: Ng, Harada & Russell (1999); Scott & Varian (2014).
        """
        effective_race_type = (
            _norm_race_type(str(race_type_filter))
            if race_type_filter is not None
            else self.race_type
        )
        if isinstance(base_weights, dict):
            w = {k: float(v) if isinstance(v, (int, float)) else 0.1
                 for k, v in base_weights.items()}
        else:
            w = {}

        s   = self.strength
        cr  = self.ema.get("crash_rate",        0.0)
        otr = self.ema.get("offtrack_rate",      0.0)
        spd = self.ema.get("avg_speed",          2.0)
        ccr = self.ema.get("corner_crash_rate",  0.0)

        def _ema(k, d=0.0):
            return float(self.ema.get(k, d))
        def _slope(k):
            try:
                return float(self.kf_trends.get(k, 0.0))
            except Exception:
                return 0.0

        # ── Layer 0: Compliance gradient shaping (v1.2.0, PRIMARY) ─────────
        _bf_cg  = _ema("brake_field_compliance_gradient",  1.0)
        _rl_cg  = _ema("race_line_compliance_gradient",    1.0)
        _bf_def = float(max(0.0, 1.0 - _bf_cg))
        _rl_def = float(max(0.0, 1.0 - _rl_cg))

        if _bf_def > 0.05:
            w["braking"]    = w.get("braking",    0.08) + s * _bf_def * 0.6
            w["safe_speed"] = w.get("safe_speed", 0.06) + s * _bf_def * 0.3
        if _slope("brake_field_compliance_gradient") < -0.005:
            w["braking"]    = w.get("braking", 0.08) + s * abs(_slope("brake_field_compliance_gradient")) * 20.0

        if _rl_def > 0.05:
            w["racing_line"] = w.get("racing_line", 0.04) + s * _rl_def * 0.7
            w["heading"]     = w.get("heading",     0.10) + s * _rl_def * 0.3
        if _slope("race_line_compliance_gradient") < -0.005:
            w["racing_line"] = w.get("racing_line", 0.04) + s * abs(_slope("race_line_compliance_gradient")) * 20.0

        # ── Layer 1: Kalman trend signals ───────────────────────────────────
        for metric, trend_val in self.kf_trends.items():
            if metric == "crash_rate" and trend_val > 0.01:
                w["obstacle"] = w.get("obstacle", 0.1) + s * 0.3
                w["center"]   = w.get("center",   0.1) + s * 0.2
            elif metric == "off_track_rate" and trend_val > 0.01:
                w["center"]  = w.get("center",  0.1) + s * 0.4
                w["heading"] = w.get("heading", 0.1) + s * 0.3

        # ── Layer 1b: Kalman regression — v_perp and urgency ───────────────
        for metric, beta in self.kf_betas.items():
            if "v_perp" in metric and abs(beta) > 0.1:
                w["braking"]       = w.get("braking",       0.08) + s * abs(beta) * 0.3
                w["speed_steering"]= w.get("speed_steering",0.08) + s * abs(beta) * 0.2
            if "urgency" in metric and abs(beta) > 0.1:
                w["braking"]       = w.get("braking", 0.08) + s * abs(beta) * 0.25
            if "race_line_compliance_gradient" in metric and abs(beta) > 0.1:
                w["racing_line"]   = w.get("racing_line", 0.04) + s * abs(beta) * 0.3

        # ── Layer 2: EMA threshold rules ────────────────────────────────────
        if cr > 0.3:
            b = s * min(cr, 1.0)
            w["obstacle"] = w.get("obstacle", 0.1) + b * 0.5
            w["center"]   = w.get("center",   0.1) + b * 0.3
        if otr > 0.2:
            b = s * min(otr, 1.0)
            w["center"]  = w.get("center",  0.1) + b * 0.4
            w["heading"] = w.get("heading", 0.1) + b * 0.3
        if spd < 1.5:
            w["curv_speed"] = w.get("curv_speed", 0.15) + s * 0.3
        _ep_prog = _ema("avg_progress", 0.0)
        if ccr > 0.2 and _ep_prog > 10.0:
            w["corner"] = w.get("corner", 0.1) + s * min(ccr, 1.0) * 0.5

        ssr = _ema("avg_safe_speed_ratio", 1.0)
        ter = _ema("avg_turn_entry_ratio", 1.0)
        rle = _ema("avg_racing_line_err",  0.0)
        _ssr_thresh = {"TIME_TRIAL": 1.2, "OBJECT_AVOIDANCE": 1.05, "HEAD_TO_BOT": 1.1}.get(effective_race_type, 1.2)
        if ssr > _ssr_thresh:
            w["safe_speed"] = w.get("safe_speed", 0.06) + s * min(ssr - 1.0, 0.5) * 0.3
        if ter > 1.1:
            w["safe_speed"] = w.get("safe_speed", 0.06) + s * 0.2
        if rle > 0.4:
            w["racing_line"] = w.get("racing_line", 0.04) + s * min(rle, 1.0) * 0.2

        if _ema("avg_speed_centerline", 0.0) > 0 and _slope("avg_speed_centerline") < -0.01:
            w["progress"] = w.get("progress", 0.12) + s * 0.20
        # v1.1.5c FIX-Q: bsts_metrics sends 'avg_jerk'; adjust_weights read 'jerk_rms' → zero slope.
        # REF: Balaban (2018) — jerk characterises intentionality of braking input.
        _jerk_slope = min(_slope("jerk_rms"), _slope("avg_jerk"))
        if _jerk_slope < -0.01:
            w["jerk"]     = w.get("jerk", 0.03) + s * 0.15
        if _slope("late_corner_entry") < -0.01 or _slope("early_corner_exit") < -0.01:
            w["corner"]   = w.get("corner", 0.10) + s * 0.30
        if _slope("steer_speed_coordination") < -0.01:
            w["speed_steering"] = w.get("speed_steering", 0.08) + s * 0.15
        if _slope("waypoint_coverage") < -0.005:
            w["progress"] = w.get("progress", 0.12) + s * 0.10

        # ── Layer 3: race_type multipliers ──────────────────────────────────
        if effective_race_type == "OBJECT_AVOIDANCE":
            w["obstacle"]    = max(w.get("obstacle", 0.0), 0.12) * 1.3
            w["braking"]     = w.get("braking",  0.08) * 1.5
            w["center"]      = w.get("center",   0.10) * 1.1
        elif effective_race_type == "HEAD_TO_BOT":
            w["obstacle"]       = max(w.get("obstacle", 0.0), 0.15) * 1.5
            w["steering"]       = w.get("steering",       0.02) * 2.0
            w["speed_steering"] = w.get("speed_steering", 0.08) * 1.3
            w["corner"]         = w.get("corner",         0.10) * 1.2

        w['progress'] = max(w.get('progress', 0.0), 0.08)  # Ng et al. (1999) progress floor
        total = sum(w.values())
        if total > 0:
            return {k: v / total for k, v in w.items()}
        return w


# ---------------------------------------------------------------------------
# Module-level helpers for BSTSSeasonal
# ---------------------------------------------------------------------------
def _ewma(y, alpha=0.02):
    """Exponentially-weighted moving average (EWMA) trend extraction.
    REF: Cleveland et al. (1990) STL decomposition — trend component.
    """
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def _ols(X, y):
    """OLS via normal equations with ridge regularisation (lambda=1e-4).
    REF: Durbin & Koopman (2012) §3.2 regression component in state-space.
    """
    try:
        XtX = X.T @ X + 1e-4 * np.eye(X.shape[1])
        beta = np.linalg.solve(XtX, X.T @ y)
        return beta, X @ beta
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1]), np.zeros(len(y))


def _load_jsonl(path):
    """Load a JSONL file into a list of dicts. Silently skips malformed lines."""
    rows = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def _apply_denim():
    """Apply denim theme to matplotlib rcParams."""
    if not HAS_MPL:
        return
    rcParams.update({
        'axes.facecolor':   BG_LIGHT,
        'figure.facecolor': BG_LIGHT,
        'axes.edgecolor':   DENIM_DARK,
        'axes.labelcolor':  DENIM_DARK,
        'xtick.color':      DENIM_DARK,
        'ytick.color':      DENIM_DARK,
        'text.color':       DENIM_DARK,
        'grid.color':       DENIM_MID,
        'grid.alpha':       0.3,
        'axes.grid':        True,
    })


# ===========================================================================
# BSTSSeasonal  — Batch BSTS decomposition + live per-step buffer
# ===========================================================================
class BSTSSeasonal:
    """Batch BSTS decomposition with live per-step buffer for run.py.

    Provides:
      record_step()       -- buffer per-step observations
      fit_from_jsonl()    -- full BSTS fit from episode JSONL log
      get_trend()         -- cached trend dict (phase, slope, kf_level, ...)
      get_season()        -- cached season dict (worst_segments, segments)
      get_seasonal()      -- alias for get_season()

    REF:
      Cleveland, R. B. et al. (1990). STL: A seasonal-trend decomposition
        procedure based on Loess. J. Off. Stat., 6(1), 3-73.
      Scott, S. L. & Varian, H. R. (2014). Predicting the present with
        Bayesian structural time series. Int. J. Math. Model. Numer.
        Optim., 5(1-2), 4-23.
      Brodersen, K. H. et al. (2015). Inferring causal impact using
        Bayesian structural time-series models. Ann. Appl. Stat., 9(1),
        247-274.
      Durbin, J. & Koopman, S. J. (2012). Time Series Analysis by State
        Space Methods (2nd ed.). Oxford University Press.
    """

    STEP_BUFFER_MAXLEN      = 10_000
    STEP_BUFFER_MIN_EPISODES = 5
    TREND_WINDOW             = 10

    def __init__(self, n_segments=12, save_dir='results', alpha=0.02):
        self.n_segments = int(n_segments)
        self.save_dir   = str(save_dir)
        self.alpha      = float(alpha)
        os.makedirs(self.save_dir, exist_ok=True)

        self._step_buf:    collections.deque = collections.deque(maxlen=self.STEP_BUFFER_MAXLEN)
        self._ep_buf:      list  = []
        self._current_ep:  dict  = {}
        self._ep_count:    int   = 0
        self._last_results: dict = {}
        self._cached_trend: dict = {}
        self._cached_season: dict = {'worst_segments': [], 'segments': {}}

    def record_step(self, progress=0.0, speed=0.0, steering=0.0,
                    heading_err=0.0, raceline_err=0.0, reward=0.0,
                    context=None, action=None, lidar_min=0.0, wp_idx=0, **kwargs):
        self._step_buf.append({
            'progress':     float(progress),
            'speed':        float(speed),
            'steering':     float(steering),
            'heading_err':  float(heading_err),
            'raceline_err': float(raceline_err),
            'reward':       float(reward),
            'lidar_min':    float(lidar_min),
            'wp_idx':       int(wp_idx),
        })
        ep = self._current_ep
        ep.setdefault('rewards', []).append(float(reward))
        ep.setdefault('speeds',  []).append(float(speed))
        ep.setdefault('rl_errs', []).append(float(raceline_err))
        ep.setdefault('max_prog', 0.0)
        if float(progress) > ep['max_prog']:
            ep['max_prog'] = float(progress)

    def _flush_episode(self, lap_completed: bool = False):
        ep = self._current_ep
        if not ep.get('rewards'):
            self._current_ep = {}
            return
        self._ep_count += 1
        rewards = ep['rewards']
        speeds  = ep['speeds']
        rl_errs = ep['rl_errs']
        self._ep_buf.append({
            'episode':              self._ep_count,
            'ep_return':            float(sum(rewards)),
            'position_in_lap':      float(ep.get('max_prog', 0.0)) / 100.0,
            'lap_completed':        float(bool(lap_completed)),
            'avg_speed_centerline': float(np.mean(speeds)) if speeds else 0.0,
            'avg_racing_line_err':  float(np.mean(rl_errs)) if rl_errs else 0.0,
            'offtrack_rate':        float(ep.get('offtrack_rate', 0.0)),
            'avg_jerk':             float(ep.get('avg_jerk', 0.0)),
            'avg_safe_speed_ratio': float(ep.get('avg_safe_speed_ratio', 1.0)),
        })
        self._current_ep = {}

    def _fit_from_step_buffer(self):
        rows = self._ep_buf
        if len(rows) < self.STEP_BUFFER_MIN_EPISODES:
            return
        self._fit_rows(rows)

    def _fit_rows(self, rows: list):
        if len(rows) < 10:
            return
        ep = np.array([r.get('episode', i) for i, r in enumerate(rows)], dtype=float)
        X_cols, reg_used = [], []
        for k in REGRESSOR_KEYS:
            col = np.array([r.get(k, np.nan) for r in rows], dtype=float)
            if np.isfinite(col).sum() > len(col) * 0.2:
                med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
                col = np.where(np.isfinite(col), col, med)
                mu, sd = col.mean(), col.std() + 1e-9
                X_cols.append((col - mu) / sd)
                reg_used.append(k)
        if not X_cols:
            return
        X = np.column_stack([np.ones(len(rows))] + X_cols)
        results = {'regressors': reg_used, 'series': {}}
        for sk in SUCCESS_KEYS:
            y = np.array([r.get(sk, np.nan) for r in rows], dtype=float)
            if not np.isfinite(y).any():
                continue
            med = np.nanmedian(y)
            y   = np.where(np.isfinite(y), y, med)
            trend = _ewma(y, self.alpha)
            detr  = y - trend
            e_min, e_max = ep.min(), ep.max() + 1e-9
            seg_idx = np.clip(
                ((ep - e_min) / (e_max - e_min) * self.n_segments).astype(int),
                0, self.n_segments - 1
            )
            season, seg_means = np.zeros_like(y), {}
            for s in range(self.n_segments):
                mask = seg_idx == s
                if mask.any():
                    m = detr[mask].mean()
                    season[mask] = m
                    seg_means[s] = m
            resid_ts     = y - trend - season
            beta, yhat_r = _ols(X, resid_ts)
            residual     = resid_ts - yhat_r
            results['series'][sk] = dict(
                y=y, trend=trend, season=season, regression=yhat_r,
                residual=residual, beta=beta, seg_means=seg_means, ep=ep,
            )
        self._last_results = results
        self._update_trend_cache(results)
        self._update_season_cache(results, rows)

    def _update_trend_cache(self, results: dict):
        sk = 'ep_return'
        if sk not in results.get('series', {}):
            keys = list(results.get('series', {}).keys())
            if not keys:
                return
            sk = keys[0]
        d     = results['series'][sk]
        trend = d['trend']
        window = min(len(trend), self.TREND_WINDOW)
        if window < 4:
            self._cached_trend = {}
            return
        recent = trend[-window:]
        slope, _ = np.polyfit(np.arange(window, dtype=float), recent, 1)
        rel_slope = slope / (abs(np.mean(recent)) + 1e-9)
        if   rel_slope >  0.005: phase = 'improving'
        elif rel_slope < -0.005: phase = 'declining'
        else:                    phase = 'plateau'
        self._cached_trend = {
            'phase':        phase,
            'return_slope': float(rel_slope),
            'trend_last':   float(trend[-1]),
            'trend_mean':   float(np.mean(recent)),
        }

    def _update_season_cache(self, results: dict, rows: list):
        sk = 'ep_return'
        if sk not in results.get('series', {}):
            keys = list(results.get('series', {}).keys())
            if not keys:
                return
            sk = keys[0]
        d         = results['series'][sk]
        seg_means = d.get('seg_means', {})
        if not seg_means:
            return
        sorted_segs    = sorted(seg_means.items(), key=lambda kv: kv[1])
        worst_segments = [int(sid) for sid, _ in sorted_segs[:4]]
        ep_arr  = d['ep']
        e_min, e_max = ep_arr.min(), ep_arr.max() + 1e-9
        seg_idx_all = np.clip(
            ((ep_arr - e_min) / (e_max - e_min) * self.n_segments).astype(int),
            0, self.n_segments - 1
        )
        segments_info = {}
        for seg_id in range(self.n_segments):
            mask = seg_idx_all == seg_id
            if not mask.any():
                continue
            seg_rows  = [rows[i] for i in np.where(mask)[0]]
            crashes   = sum(1 for r in seg_rows if r.get('lap_completed', 1.0) < 0.5)
            avg_speed = float(np.mean([r.get('avg_speed_centerline', 0.0) for r in seg_rows]))
            segments_info[seg_id] = {
                'crashes':   int(crashes),
                'avg_speed': avg_speed,
                'n_eps':     int(len(seg_rows)),
            }
        self._cached_season = {
            'worst_segments': worst_segments,
            'segments':       segments_info,
        }

    def get_trend(self) -> dict:
        return dict(self._cached_trend)

    def get_season(self) -> dict:
        return dict(self._cached_season)

    def get_seasonal(self) -> dict:
        return self.get_season()

    def fit_from_jsonl(self, jsonl_path, out_png=None):
        rows = _load_jsonl(jsonl_path)
        if len(rows) < 10:
            print(f'[BSTS] too few rows ({len(rows)}) in {jsonl_path}')
            return None
        all_rows = rows + self._ep_buf
        if out_png is None:
            out_png = os.path.join(self.save_dir, 'bsts_decomposition.png')
        self._fit_rows(all_rows)
        results = self._last_results
        if not results:
            return None
        _apply_denim()
        ep = np.array([r.get('episode', i) for i, r in enumerate(all_rows)], dtype=float)
        self._plot(results, ep, out_png)
        for sk, d in results.get('series', {}).items():
            reg_used = results.get('regressors', [])
            coef_str = ', '.join(
                f'{k}={b:+.3f}' for k, b in zip(['int'] + reg_used, d['beta'])
            )
            print(
                f'[BSTS] {sk}: y_mean={d["y"].mean():.3f} '
                f'trend_last={d["trend"][-1]:.3f} '
                f'season_amp={float(np.ptp(list(d["seg_means"].values()))):.3f} '
                f'resid_std={d["residual"].std():.3f} | {coef_str}'
            )
        print(f'[BSTS] trend_cache={self._cached_trend}')
        return results

    def _plot(self, results, ep, out_png):
        if not HAS_MPL:
            return
        series = results['series']
        n_s    = len(series)
        if n_s == 0:
            return
        fig, axes = plt.subplots(4, n_s, figsize=(5.5 * n_s, 11), squeeze=False)
        panel_titles = [
            'Observed + Trend',
            'Season (per-restart)',
            'Regression (intermediary)',
            'Residual',
        ]
        reg_used = results['regressors']
        for j, (sk, d) in enumerate(series.items()):
            c1, c2 = COMPLEMENTARY_PAIRS[j % len(COMPLEMENTARY_PAIRS)]
            axes[0, j].plot(ep, d['y'],     lw=0.5, alpha=0.35, color=MUTED_GRAY, label='observed')
            axes[0, j].plot(ep, d['trend'], lw=2.2, color=c1,   label='trend')
            axes[0, j].set_title(f'{sk}\n{panel_titles[0]}', fontsize=9, fontweight='bold', color=DENIM_DARK)
            axes[0, j].legend(loc='best', fontsize=7, framealpha=0.8)
            axes[1, j].bar(
                range(self.n_segments),
                [d['seg_means'].get(s, 0) for s in range(self.n_segments)],
                color=c2, alpha=0.7, edgecolor=DENIM_DARK, linewidth=0.5,
            )
            axes[1, j].axhline(0, color=DENIM_DARK, lw=0.5)
            axes[1, j].set_title(panel_titles[1], fontsize=9, color=DENIM_DARK)
            axes[1, j].set_xlabel('restart segment', fontsize=7)
            axes[2, j].plot(ep, d['regression'], lw=1.2, color=TERRA_COTTA)
            coef_txt = '\n'.join(f'{k}: {b:+.3f}' for k, b in zip(reg_used, d['beta'][1:]))
            axes[2, j].text(
                0.02, 0.97, coef_txt, transform=axes[2, j].transAxes,
                fontsize=6.5, va='top', family='monospace', color=DENIM_DARK,
                bbox=dict(boxstyle='round', facecolor=BG_LIGHT, alpha=0.85, edgecolor=DENIM_MID),
            )
            axes[2, j].set_title(panel_titles[2], fontsize=9, color=DENIM_DARK)
            axes[3, j].plot(ep, d['residual'], lw=0.5, color=MUTED_GRAY, alpha=0.6)
            axes[3, j].axhline(0, color=DENIM_DARK, lw=0.5)
            axes[3, j].set_title(panel_titles[3], fontsize=9, color=DENIM_DARK)
            axes[3, j].set_xlabel('episode', fontsize=8)
        fig.suptitle(
            'BSTS Decomposition: Success ~ Trend + Season(per-restart) + Regression(intermediary) + Residual',
            fontsize=11, fontweight='bold', color=DENIM_DARK, y=0.99,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[BSTS-PLOT] saved {out_png}')
