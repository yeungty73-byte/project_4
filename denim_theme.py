
# denim_theme.py -- v19 denim-and-complementary-colors palette
"""Unified color theme: denim blue primary with warm complementary accents.

Denim blue family:    #1B3A5C (dark), #3A6B8C (mid), #5B9BD5 (bright)
Complementary warm:   #D4A03C (amber), #C75B39 (terra cotta), #E8C167 (gold)
Neutral:              #2E2E2E (bg dark), #F0EDE6 (bg light), #8C8C8C (muted)
Accent:               #6BB38A (sage green), #9B6EB7 (muted purple)
"""

DENIM_DARK   = "#1B3A5C"
# REF: Cleveland, R. B. et al. (1990). STL decomposition. | Gelman, A. et al. (2013). Bayesian Data Analysis (3rd ed.). Chapman and Hall/CRC.
DENIM_MID    = "#3A6B8C"
DENIM_BRIGHT = "#5B9BD5"
AMBER        = "#D4A03C"
TERRA_COTTA  = "#C75B39"
GOLD_WARM    = "#E8C167"
SAGE_GREEN   = "#6BB38A"
MUTED_PURPLE = "#9B6EB7"
BG_DARK      = "#2E2E2E"
BG_LIGHT     = "#F0EDE6"
MUTED_GRAY   = "#8C8C8C"
WHITE_SMOKE  = "#F5F5F5"

# Ordered palette for cycling through series
SERIES_COLORS = [
    DENIM_BRIGHT, AMBER, TERRA_COTTA, SAGE_GREEN,
    MUTED_PURPLE, GOLD_WARM, DENIM_MID, DENIM_DARK,
]

def apply_theme(fig=None, ax=None):
    """Apply denim theme to matplotlib figure/axes."""
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.gcf()
    fig.patch.set_facecolor(BG_DARK)
    axes = fig.get_axes() if ax is None else ([ax] if not hasattr(ax, "__iter__") else list(ax))
    for a in axes:
        a.set_facecolor(BG_DARK)
        a.tick_params(colors=WHITE_SMOKE, which="both")
        a.xaxis.label.set_color(WHITE_SMOKE)
        a.yaxis.label.set_color(WHITE_SMOKE)
        a.title.set_color(DENIM_BRIGHT)
        for spine in a.spines.values():
            spine.set_color(MUTED_GRAY)
    return fig

def get_color(idx):
    return SERIES_COLORS[idx % len(SERIES_COLORS)]
