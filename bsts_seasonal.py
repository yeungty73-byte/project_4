"""BSTS-style decomposition: Success ~ Trend + Season(per-restart) + Regression(intermediary) + Residual.

Success metrics (regressands):
  - position_in_lap (continuous waypoint fraction 0-1)
  - lap_completed (binary per episode, rolling-averaged)
  - ep_return
  - avg_ang_vel_centerline (angular velocity w/r track centerline)

Intermediary regressors:
  - avg_racing_line_err (raceline compliance, dynamic w/r barriers)
  - brake_line_compliance (decel before barriers)
  - avg_jerk (acceleration drasticness |da/dt|)
  - avg_safe_speed_ratio
  - offtrack_rate
  - curvature_x_speed

Denim theme palette. R-bsts style panels: trend, season, regression, residual.

v1.0.13 changes:
  - record_step() now buffers per-step data in a rolling deque (no longer a no-op)
  - get_trend() returns real phase/slope computed from last fit_from_jsonl decomposition
  - get_season() returns real worst_segments + per-segment crash/speed map
  - get_seasonal() alias added (run.py uses this name in one call-site)
  - fit_from_jsonl() caches decomposition results into self._last_results
  - _fit_from_step_buffer() added: periodic fit from record_step() buffer without needing jsonl
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

# ---- denim theme palette ----
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

# Harmonized with analyze_logs.py (single source of truth used by run.py and live_bsts_plot.py)
from analyze_logs import SUCCESS_METRICS as SUCCESS_KEYS, INTERMEDIARY_METRICS as REGRESSOR_KEYS


def _apply_denim():
    rcParams.update({
        'figure.facecolor': BG_LIGHT,
        'axes.facecolor':   WHITE_SMOKE,
        'axes.edgecolor':   DENIM_DARK,
        'axes.labelcolor':  DENIM_DARK,
        'xtick.color':      DENIM_DARK,
        'ytick.color':      DENIM_DARK,
        'text.color':       DENIM_DARK,
        'grid.color':       MUTED_GRAY,
        'grid.alpha':       0.3,
        'axes.grid':        True,
        'font.family':      'sans-serif',
    })


def _load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _ewma(x, alpha=0.02):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _ols(X, y):
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, X @ beta
    except Exception:
        return np.zeros(X.shape[1]), np.zeros_like(y)


class BSTSSeasonal:
    """Batch BSTS decomposition + live per-step buffer for run.py integration.

    Constructor signature (unchanged from original):
        BSTSSeasonal(n_segments=12, save_dir='results', alpha=0.02)

    New in v1.0.13:
        - record_step() buffers data; _fit_from_step_buffer() fits from buffer
        - get_trend()    -> real phase dict after any fit
        - get_season()   -> real worst_segments + segments dict after any fit
        - get_seasonal() -> alias for get_season() (run.py uses this name)
        - fit_from_jsonl() now caches into self._last_results
    """

    # How many steps to keep in the rolling step buffer before auto-fitting
    STEP_BUFFER_MAXLEN = 5000
    # Minimum episodes (rows) required before attempting a fit from step buffer
    STEP_BUFFER_MIN_EPISODES = 20
    # Episode-level rolling window for trend phase detection
    TREND_WINDOW = 30

    def __init__(self, n_segments=12, save_dir='results', alpha=0.02):
        self.n_segments = int(n_segments)
        self.save_dir   = str(save_dir)
        self.alpha      = float(alpha)
        os.makedirs(self.save_dir, exist_ok=True)

        # --- v1.0.13: live state ---
        # Per-step rolling buffer for _fit_from_step_buffer()
        self._step_buf: collections.deque = collections.deque(maxlen=self.STEP_BUFFER_MAXLEN)
        # Episode-level summary accumulator (populated by _flush_episode_to_buf)
        self._ep_buf: list = []
        self._current_ep: dict = {}
        self._ep_count: int = 0

        # Cached decomposition results (populated by fit_from_jsonl or _fit_from_step_buffer)
        self._last_results: dict = {}
        # Cached trend dict returned by get_trend()
        self._cached_trend: dict = {}
        # Cached season dict returned by get_season()
        self._cached_season: dict = {'worst_segments': [], 'segments': {}}

    # ------------------------------------------------------------------
    # v1.0.13: per-step live buffer
    # ------------------------------------------------------------------
    def record_step(
        self,
        progress: float = 0.0,
        speed: float = 0.0,
        steering: float = 0.0,
        heading_err: float = 0.0,
        raceline_err: float = 0.0,
        reward: float = 0.0,
        context=None,
        action=None,
        lidar_min: float = 0.0,
        wp_idx: int = 0,
        **kwargs,
    ):
        """Buffer one step of telemetry for periodic BSTS fitting."""
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
        # Accumulate into current episode summary
        ep = self._current_ep
        ep.setdefault('rewards', []).append(float(reward))
        ep.setdefault('speeds',  []).append(float(speed))
        ep.setdefault('rl_errs', []).append(float(raceline_err))
        ep.setdefault('max_prog', 0.0)
        if float(progress) > ep['max_prog']:
            ep['max_prog'] = float(progress)

    def _flush_episode(self, lap_completed: bool = False):
        """Called at episode end to commit current episode to _ep_buf."""
        ep = self._current_ep
        if not ep.get('rewards'):
            self._current_ep = {}
            return
        self._ep_count += 1
        rewards = ep['rewards']
        speeds  = ep['speeds']
        rl_errs = ep['rl_errs']
        self._ep_buf.append({
            'episode':               self._ep_count,
            'ep_return':             float(sum(rewards)),
            'position_in_lap':       float(ep.get('max_prog', 0.0)) / 100.0,
            'lap_completed':         float(bool(lap_completed)),
            'avg_speed_centerline':  float(np.mean(speeds)) if speeds else 0.0,
            'avg_racing_line_err':   float(np.mean(rl_errs)) if rl_errs else 0.0,
            'offtrack_rate':         float(ep.get('offtrack_rate', 0.0)),
            'avg_jerk':              float(ep.get('avg_jerk', 0.0)),
            'avg_safe_speed_ratio':  float(ep.get('avg_safe_speed_ratio', 1.0)),
        })
        self._current_ep = {}

    def _fit_from_step_buffer(self):
        """Fit BSTS decomposition from the episode buffer accumulated via record_step().
        Updates self._cached_trend and self._cached_season.
        """
        rows = self._ep_buf
        if len(rows) < self.STEP_BUFFER_MIN_EPISODES:
            return
        self._fit_rows(rows)

    def _fit_rows(self, rows: list):
        """Core fitting logic shared by fit_from_jsonl and _fit_from_step_buffer."""
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
            y = np.where(np.isfinite(y), y, med)
            trend = _ewma(y, self.alpha)
            detr  = y - trend
            e_min, e_max = ep.min(), ep.max() + 1e-9
            seg_idx = np.clip(
                ((ep - e_min) / (e_max - e_min) * self.n_segments).astype(int),
                0, self.n_segments - 1
            )
            season = np.zeros_like(y)
            seg_means = {}
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
                residual=residual, beta=beta, seg_means=seg_means,
                ep=ep,
            )
        self._last_results = results
        self._update_trend_cache(results)
        self._update_season_cache(results, rows)

    # ------------------------------------------------------------------
    # v1.0.13: trend + season cache update helpers
    # ------------------------------------------------------------------
    def _update_trend_cache(self, results: dict):
        """Derive phase + slope from ep_return trend series."""
        sk = 'ep_return'
        if sk not in results.get('series', {}):
            # fall back to first available success key
            keys = list(results.get('series', {}).keys())
            if not keys:
                return
            sk = keys[0]
        d = results['series'][sk]
        trend = d['trend']
        if len(trend) < self.TREND_WINDOW:
            window = len(trend)
        else:
            window = self.TREND_WINDOW
        if window < 4:
            self._cached_trend = {}
            return
        recent = trend[-window:]
        x = np.arange(window, dtype=float)
        slope, intercept = np.polyfit(x, recent, 1)
        rel_slope = slope / (abs(np.mean(recent)) + 1e-9)
        if rel_slope > 0.005:
            phase = 'improving'
        elif rel_slope < -0.005:
            phase = 'declining'
        else:
            phase = 'plateau'
        self._cached_trend = {
            'phase':        phase,
            'return_slope': float(rel_slope),
            'trend_last':   float(trend[-1]),
            'trend_mean':   float(np.mean(recent)),
        }

    def _update_season_cache(self, results: dict, rows: list):
        """Populate worst_segments from per-segment detrended residuals."""
        sk = 'ep_return'
        if sk not in results.get('series', {}):
            keys = list(results.get('series', {}).keys())
            if not keys:
                return
            sk = keys[0]
        d = results['series'][sk]
        seg_means = d.get('seg_means', {})
        if not seg_means:
            return
        # Sort segments by detrended mean ascending → worst = most negative
        sorted_segs = sorted(seg_means.items(), key=lambda kv: kv[1])
        worst_segments = [int(seg_id) for seg_id, _ in sorted_segs[:4]]
        # Build per-segment stat dict from episode rows
        ep_arr = d['ep']
        n = len(rows)
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
            seg_rows = [rows[i] for i in np.where(mask)[0]]
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

    # ------------------------------------------------------------------
    # Public API consumed by run.py
    # ------------------------------------------------------------------
    def get_trend(self) -> dict:
        """Return latest trend dict for run.py reward/hp scheduling.
        Keys: phase ('improving'|'plateau'|'declining'), return_slope, trend_last, trend_mean.
        Returns {} until first fit has been performed.
        """
        return dict(self._cached_trend)

    def get_season(self) -> dict:
        """Return season decomposition dict for run.py segment weighting.
        Keys: worst_segments (list[int]), segments (dict[int -> {crashes, avg_speed, n_eps}]).
        """
        return dict(self._cached_season)

    def get_seasonal(self) -> dict:
        """Alias for get_season() — run.py uses this name in one call-site."""
        return self.get_season()

    # ------------------------------------------------------------------
    # Batch BSTS fit from JSONL (called periodically from run.py)
    # ------------------------------------------------------------------
    def fit_from_jsonl(self, jsonl_path, out_png=None):
        rows = _load_jsonl(jsonl_path)
        if len(rows) < 10:
            print(f'[BSTS] too few rows ({len(rows)}) in {jsonl_path}')
            return None
        # Merge episode buffer rows too (local steps not yet flushed to jsonl)
        all_rows = rows + self._ep_buf
        if out_png is None:
            out_png = os.path.join(self.save_dir, 'bsts_decomposition.png')
        # Core fit
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
                f'{k}={b:+.3f}'
                for k, b in zip(['int'] + reg_used, d['beta'])
            )
            print(
                f'[BSTS] {sk}: y_mean={d["y"].mean():.3f} '
                f'trend_last={d["trend"][-1]:.3f} '
                f'season_amp={float(np.ptp(list(d["seg_means"].values()))):.3f} '
                f'resid_std={d["residual"].std():.3f} | {coef_str}'
            )
        print(f'[BSTS] trend_cache={self._cached_trend}')
        return results

    # ------------------------------------------------------------------
    # Plotting (unchanged from original)
    # ------------------------------------------------------------------
    def _plot(self, results, ep, out_png):
        series = results['series']
        n_s = len(series)
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
            axes[0, j].plot(ep, d['y'], lw=0.5, alpha=0.35, color=MUTED_GRAY, label='observed')
            axes[0, j].plot(ep, d['trend'], lw=2.2, color=c1, label='trend')
            axes[0, j].set_title(
                f'{sk}\n{panel_titles[0]}', fontsize=9, fontweight='bold', color=DENIM_DARK
            )
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
            coef_txt = '\n'.join(
                f'{k}: {b:+.3f}'
                for k, b in zip(reg_used, d['beta'][1:])
            )
            axes[2, j].text(
                0.02, 0.97, coef_txt,
                transform=axes[2, j].transAxes,
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
