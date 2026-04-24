#!/usr/bin/env python3
"""v1.0.6 Live Dashboard – denim-themed, ribbon plots across cluster fleet,
pooled violin + Bayesian nc-t, success metrics front-and-center."""
import json, glob, os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as sp_stats
warnings.filterwarnings('ignore')

# ---- Denim palette (from denim_theme.py) ----
DENIM_DARK   = '#1B3A5C'
DENIM_MID    = '#3A6B8C'
DENIM_BRIGHT = '#5B9BD5'
AMBER        = '#D4A03C'
TERRA_COTTA  = '#C75B39'
SAGE_GREEN   = '#6BB38A'
MUTED_PURPLE = '#9B6EB7'
GOLD_WARM    = '#E8C167'
BG_LIGHT     = '#F0EDE6'  # off-white denim
BG_PANEL     = '#EAE4DA'  # panel bg
SPINE_CLR    = '#8C8C8C'
TICK_CLR     = '#5A5A5A'
LABEL_CLR    = '#3A3A3A'
TITLE_CLR    = DENIM_DARK
SUPTITLE_CLR = DENIM_DARK
RIBBON_ALPHA = 0.25

# Success metrics (top row, front-and-center)
SUCCESS_METRICS = [
    ('progress',             'Track Progress',        '%',    (0, 1),   DENIM_BRIGHT),
    ('avg_racing_line_err',  'Racing-Line Error',     'm',    None,     TERRA_COTTA),
    ('avg_safe_speed_ratio', 'Safe-Speed Ratio',      'ratio',None,     SAGE_GREEN),
]
# Intermediary metrics (second row)
INTERMEDIARY_METRICS = [
    ('avg_speed',            'Avg Speed',             'm/s',  None,     MUTED_PURPLE),
    ('offtrack_rate',        'Off-Track Rate',        'ratio',(0, 1),   TERRA_COTTA),
    ('ep_brake_frac',        'Brake Fraction',        'ratio',(0, 1),   DENIM_MID),
]
TAIL_N = 500
REFRESH = 30

_POS = [a for a in sys.argv[1:] if not a.startswith('--')]
LOG_DIR = _POS[0] if len(_POS) > 0 else 'results'
OUT_DIR = _POS[1] if len(_POS) > 1 else 'results/live'

def load_all_clusters(log_dir):
    """Load ALL v7/v200_metrics_*.jsonl files, return dict of {file_id: [eps]}"""
    clusters = {}
    for fp in sorted(glob.glob(os.path.join(log_dir, 'v[27]*_metrics_*.jsonl'))):
        eps = []
        if os.path.getsize(fp) < 100:
            continue
        try:
            for line in open(fp):
                line = line.strip()
                if line:
                    try: eps.append(json.loads(line))
                    except json.JSONDecodeError: pass
        except Exception: pass
        if eps and os.path.getsize(fp) > 100:
            clusters[fp] = eps
    return clusters

def safe(eps, key, default=0.0):
    return [e.get(key, default) for e in eps]

def ema(vals, alpha=0.08):
    if not vals: return []
    out = [vals[0]]
    for v in vals[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out

def fit_nct(data):
    if len(data) < 10: return None
    try:
        params = sp_stats.nct.fit(data[:200], method='mm')
        return params  # (df, nc, loc, scale)
    except Exception:
        return None

def pool_ribbon(clusters, key, default=0.0):
    """Align episode-indexed series across clusters → median + IQR ribbon."""
    all_series = []
    for eps in clusters.values():
        s = np.array(safe(eps, key, default), dtype=float)
        all_series.append(s)
    if not all_series: return None, None, None, None
    max_len = max(len(s) for s in all_series)
    # Pad shorter series with NaN
    mat = np.full((len(all_series), max_len), np.nan)
    for i, s in enumerate(all_series):
        mat[i, :len(s)] = s
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        median = np.nanmedian(mat, axis=0)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
    x = np.arange(max_len)
    return x, median, q25, q75

def pool_tail(clusters, key, tail_n=500, default=0.0):
    """Pool last `tail_n` episodes across all clusters for violin + nc-t."""
    pooled = []
    for eps in clusters.values():
        vals = safe(eps, key, default)
        pooled.extend(vals[-tail_n:])
    return np.array(pooled, dtype=float)

def plot_dashboard(clusters, out_dir):
    n_total = sum(len(v) for v in clusters.values())
    n_clusters = len(clusters)
    all_metrics = SUCCESS_METRICS + INTERMEDIARY_METRICS
    n_metrics = len(all_metrics)
    n_cols = 3
    n_rows_main = (n_metrics + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 5 * n_rows_main + 0.8), facecolor=BG_LIGHT)
    gs = GridSpec(n_rows_main, n_cols, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (key, title, unit, ylim, color) in enumerate(all_metrics):
        row = idx // n_cols
        col = idx % n_cols

        # ---- Left: ribbon time-series ----
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(BG_PANEL)
        x, med, q25, q75 = pool_ribbon(clusters, key)
        if x is not None and len(x) > 0:
            sm = np.array(ema(med.tolist()))
            ax.fill_between(x, q25, q75, alpha=RIBBON_ALPHA, color=color, linewidth=0)
            ax.plot(x, sm, color=color, linewidth=1.5, alpha=0.9)
        ax.set_title(f'{title} ({unit})', fontsize=9, color=LABEL_CLR, fontweight='bold')
        if ylim: ax.set_ylim(*ylim)
        ax.tick_params(labelsize=6, colors=TICK_CLR)
        for s in ax.spines.values(): s.set_color(SPINE_CLR)
        ax.set_xlabel('Episode', fontsize=6, color=TICK_CLR)

        # ---- Violin overlay on time-series (last TAIL_N eps) ----
        tail_data = pool_tail(clusters, key, TAIL_N)
        if x is not None and len(x) > 0 and len(tail_data) > 5:
            ax2 = ax.twinx()
            vx = x[-1] + max((x[-1] - x[0]) * 0.03, 1)
            vw = max((x[-1] - x[0]) * 0.05, 1)
            parts = ax2.violinplot(tail_data, positions=[vx], showmedians=True, widths=vw)
            for pc in parts['bodies']:
                pc.set_facecolor(color); pc.set_alpha(0.25)
            for k in ('cmeans','cmedians','cbars','cmins','cmaxes'):
                if k in parts: parts[k].set_color(SPINE_CLR); parts[k].set_linewidth(0.6)
            ax2.set_ylim(ax.get_ylim()); ax2.set_yticks([])
            for sp in ax2.spines.values(): sp.set_visible(False)

    fig.suptitle(
        f'DeepRacer Live Dashboard   '
        f'[{n_clusters} cluster(s), {n_total:,} total eps]',
        color=SUPTITLE_CLR, fontsize=13, fontweight='bold', y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'dashboard.png'), dpi=140,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f'[DASHBOARD] Saved dashboard.png ({n_clusters} clusters, {n_total} eps)', flush=True)

def console_summary(clusters):
    n = sum(len(v) for v in clusters.values())
    if n == 0:
        print('[DASHBOARD] No data yet', flush=True)
        return
    # Latest cluster, last ep
    latest = list(clusters.values())[-1]
    e = latest[-1]
    parts = [f"ep={e.get('episode','?')}"]
    for key, title, unit, _, _ in SUCCESS_METRICS:
        parts.append(f"{title[:12]}={e.get(key,0):.3f}")
    print(f'[DASH] {" | ".join(parts)}', flush=True)

if __name__ == '__main__':
    print(f'[DASHBOARD] Starting | log_dir={LOG_DIR} out={OUT_DIR} refresh={REFRESH}s')
    while True:
        try:
            clusters = load_all_clusters(LOG_DIR)
            console_summary(clusters)
            if clusters:
                plot_dashboard(clusters, OUT_DIR)
        except Exception as e:
            print(f'[DASHBOARD] Error: {e}', flush=True)
        time.sleep(REFRESH)
