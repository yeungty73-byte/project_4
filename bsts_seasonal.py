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
"""
import json, os, numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---- denim theme palette ----
DENIM_DARK  = '#1B3A5C'
DENIM_MID   = '#3A6B8C'
DENIM_BRIGHT= '#5B9BD5'
AMBER       = '#D4A03C'
TERRA_COTTA = '#C75B39'
GOLD_WARM   = '#E8C167'
SAGE_GREEN  = '#6BB38A'
MUTED_PURPLE= '#9B6EB7'
BG_DARK     = '#2E2E2E'
BG_LIGHT    = '#F0EDE6'
MUTED_GRAY  = '#8C8C8C'
WHITE_SMOKE = '#F5F5F5'

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
        'axes.facecolor': WHITE_SMOKE,
        'axes.edgecolor': DENIM_DARK,
        'axes.labelcolor': DENIM_DARK,
        'xtick.color': DENIM_DARK,
        'ytick.color': DENIM_DARK,
        'text.color': DENIM_DARK,
        'grid.color': MUTED_GRAY,
        'grid.alpha': 0.3,
        'axes.grid': True,
        'font.family': 'sans-serif',
    })

def _load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def _ewma(x, alpha=0.02):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return x
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    return out

def _ols(X, y):
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, X @ beta
    except Exception:
        return np.zeros(X.shape[1]), np.zeros_like(y)

class BSTSSeasonal:
    def __init__(self, n_segments=12, save_dir='results', alpha=0.02):
        self.n_segments = n_segments
        self.save_dir = save_dir
        self.alpha = alpha
        os.makedirs(save_dir, exist_ok=True)

    def fit_from_jsonl(self, jsonl_path, out_png=None):
        rows = _load_jsonl(jsonl_path)
        if len(rows) < 10:
            print(f'[BSTS] too few rows ({len(rows)}) in {jsonl_path}')
            return None
        ep = np.array([r.get('episode', i) for i, r in enumerate(rows)], dtype=float)
        # --- build regressor matrix ---
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
            print('[BSTS] no usable regressors'); return None
        X = np.column_stack([np.ones(len(rows))] + X_cols)
        results = {'regressors': reg_used, 'series': {}}
        for sk in SUCCESS_KEYS:
            y = np.array([r.get(sk, np.nan) for r in rows], dtype=float)
            if not np.isfinite(y).any(): continue
            med = np.nanmedian(y)
            y = np.where(np.isfinite(y), y, med)
            trend = _ewma(y, self.alpha)
            detr = y - trend
            e_min, e_max = ep.min(), ep.max() + 1e-9
            seg_idx = np.clip(((ep - e_min) / (e_max - e_min) * self.n_segments).astype(int),
                              0, self.n_segments - 1)
            season = np.zeros_like(y)
            seg_means = {}
            for s in range(self.n_segments):
                mask = seg_idx == s
                if mask.any():
                    m = detr[mask].mean()
                    season[mask] = m
                    seg_means[s] = m
            resid_ts = y - trend - season
            beta, yhat_reg = _ols(X, resid_ts)
            residual = resid_ts - yhat_reg
            results['series'][sk] = dict(
                y=y, trend=trend, season=season, regression=yhat_reg,
                residual=residual, beta=beta, seg_means=seg_means)
        if out_png is None:
            out_png = os.path.join(self.save_dir, 'bsts_decomposition.png')
        _apply_denim()
        self._plot(results, ep, out_png)
        for sk, d in results['series'].items():
            coef_str = ', '.join(f'{k}={b:+.3f}' for k, b in zip(['int'] + reg_used, d['beta']))
            print(f'[BSTS] {sk}: y_mean={d["y"].mean():.3f} trend_last={d["trend"][-1]:.3f} '
                  f'season_amp={float(np.ptp(list(d["seg_means"].values()))):.3f} '
                  f'resid_std={d["residual"].std():.3f} | {coef_str}')
        return results

    def _plot(self, results, ep, out_png):
        series = results['series']
        n_s = len(series)
        if n_s == 0: return
        fig, axes = plt.subplots(4, n_s, figsize=(5.5 * n_s, 11), squeeze=False)
        panel_titles = ['Observed + Trend', 'Season (per-restart)', 'Regression (intermediary)', 'Residual']
        reg_used = results['regressors']
        for j, (sk, d) in enumerate(series.items()):
            c1, c2 = COMPLEMENTARY_PAIRS[j % len(COMPLEMENTARY_PAIRS)]
            # Row 0: observed + trend
            axes[0, j].plot(ep, d['y'], lw=0.5, alpha=0.35, color=MUTED_GRAY, label='observed')
            axes[0, j].plot(ep, d['trend'], lw=2.2, color=c1, label='trend')
            axes[0, j].set_title(f'{sk}\n{panel_titles[0]}', fontsize=9, fontweight='bold', color=DENIM_DARK)
            axes[0, j].legend(loc='best', fontsize=7, framealpha=0.8)
            # Row 1: season
            axes[1, j].bar(range(self.n_segments), [d['seg_means'].get(s, 0) for s in range(self.n_segments)],
                           color=c2, alpha=0.7, edgecolor=DENIM_DARK, linewidth=0.5)
            axes[1, j].axhline(0, color=DENIM_DARK, lw=0.5)
            axes[1, j].set_title(panel_titles[1], fontsize=9, color=DENIM_DARK)
            axes[1, j].set_xlabel('restart segment', fontsize=7)
            # Row 2: regression
            axes[2, j].plot(ep, d['regression'], lw=1.2, color=TERRA_COTTA)
            coef_txt = '\n'.join(f'{k}: {b:+.3f}' for k, b in zip(reg_used, d['beta'][1:]))
            axes[2, j].text(0.02, 0.97, coef_txt, transform=axes[2, j].transAxes,
                            fontsize=6.5, va='top', family='monospace', color=DENIM_DARK,
                            bbox=dict(boxstyle='round', facecolor=BG_LIGHT, alpha=0.85, edgecolor=DENIM_MID))
            axes[2, j].set_title(panel_titles[2], fontsize=9, color=DENIM_DARK)
            # Row 3: residual
            axes[3, j].plot(ep, d['residual'], lw=0.5, color=MUTED_GRAY, alpha=0.6)
            axes[3, j].axhline(0, color=DENIM_DARK, lw=0.5)
            axes[3, j].set_title(panel_titles[3], fontsize=9, color=DENIM_DARK)
            axes[3, j].set_xlabel('episode', fontsize=8)
        fig.suptitle(
            'BSTS Decomposition: Success ~ Trend + Season(per-restart) + Regression(intermediary) + Residual',
            fontsize=11, fontweight='bold', color=DENIM_DARK, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[BSTS-PLOT] saved {out_png}')

    def record_step(self, *args, **kwargs):
        """Accept per-step data from run.py (no-op for batch BSTS)."""
        pass

    def get_trend(self):
        """Return latest trend dict for run.py reward scheduling."""
        return {}

    def get_season(self):
        """Return season decomposition dict for run.py segment weighting."""
        return {'worst_segments': [], 'segments': {}}
