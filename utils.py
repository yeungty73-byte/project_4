import os, re
import json
import yaml
import torch
import random
import shutil
try:
    import enlighten
except ImportError:
    class _DummyCounter:
        def update(self, *a, **kw): pass
        def close(self, *a, **kw): pass
    class _DummyStatusBar:
        def update(self, *a, **kw): pass
        def close(self, *a, **kw): pass
    class _DummyManager:
        def counter(self, *a, **kw): return _DummyCounter()
        def status_bar(self, *a, **kw): return _DummyStatusBar()
    enlighten = type('module', (), {'get_manager': lambda: _DummyManager(), 'Justify': type('Justify', (), {'CENTER': 'center'})})(  )
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import gymnasium as gym
from loguru import logger
from gymnasium import spaces
import matplotlib.pyplot as plt
from gymnasium.wrappers import (
    RecordVideo,
    FlattenObservation,
    RecordEpisodeStatistics
)
from IPython.display import Video, display, clear_output

from agents import Agent


try:
    PROGRESS_MANAGER = enlighten.get_manager()
except (TypeError, ValueError):
    PROGRESS_MANAGER = None
FS_TICK: int = 12
FS_LABEL: int = 18
PLOT_DPI: int=1200
PLOT_FORMAT: str='pdf'
RC_PARAMS: dict = {
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'xtick.color': 'black',
    'ytick.color': 'black',
}
ENVIRONMENT_PARAMS_PATH: str='configs/environment_params.yaml'
ENVIRONMENT_NAME: str='deepracer-v0'
MAX_DEMO_STEPS: int = 1_000
MAX_EVAL_STEPS: int = 1_000
EVAL_EPISODES: int = 5
ONLY_CPU: bool = False
SEED: int=42


def set_seed(seed: int=SEED):
    '''set seed for reproducibility'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f'Random seed set as {seed}.')


def device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    if ONLY_CPU:
        device = 'cpu'
    logger.info(f'Using {device} device.')
    return torch.device(device)


def make_environment(
        environment_name: str = ENVIRONMENT_NAME,
        seed: int = SEED,
        reward_function=None,
        **kwargs
    ):
    """Create and wrap the DeepRacer gym environment.

    v1.1.2 FIX: reward_function defaults to _identity_reward (returns 1.0)
    so that run.py's AnnealingScheduler-shaped reward is the sole signal
    and there is no dual-reward contamination from the gym env.

    Pass a callable explicitly to override for standalone eval / demo runs.
    """
    from configs.reward_function import _identity_reward

    _rf = _identity_reward if reward_function is None else reward_function

    environment = gym.make(environment_name, reward_function=_rf, **kwargs)

    environment = RecordEpisodeStatistics(
        FlattenObservation(environment)
    )

    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)

    return environment


def get_world_name(
    environment_params_path: str=ENVIRONMENT_PARAMS_PATH
    ):
    with open(environment_params_path, 'r') as f:
        environment_params = yaml.safe_load(f)
    if 'WORLD_NAME' not in environment_params:
        raise ValueError(
            f'WORLD_NAME not defined in {environment_params_path}'
        )
    return environment_params['WORLD_NAME']


def get_race_type(
    environment_params_path: str = ENVIRONMENT_PARAMS_PATH
    ):
    """Return 'time_trial' | 'obstacle_avoidance' | 'head_to_bot'.

    v1.1.2 FIX: prefers the RACE_TYPE string key when present in the YAML
    (authoritative).  Falls back to NUMBER_OF_OBSTACLES / NUMBER_OF_BOT_CARS
    count-based logic for backwards compatibility.

    Original code silently ignored RACE_TYPE and would raise ValueError for
    any bot-car count other than exactly 3 — crashing eval for h2b.yaml
    which had NUMBER_OF_BOT_CARS: 4.
    """
    with open(environment_params_path, 'r') as f:
        ep = yaml.safe_load(f)

    # Primary: explicit RACE_TYPE string in YAML (most configs now have this)
    _race_type_map = {
        'TIME_TRIAL':        'time_trial',
        'OBJECT_AVOIDANCE':  'obstacle_avoidance',
        'HEAD_TO_BOT':       'head_to_bot',
    }
    if 'RACE_TYPE' in ep:
        rt = str(ep['RACE_TYPE']).upper().strip()
        if rt in _race_type_map:
            return _race_type_map[rt]
        logger.warning(
            f"get_race_type: unrecognised RACE_TYPE='{ep['RACE_TYPE']}' in "
            f"{environment_params_path}; falling back to count-based logic."
        )

    # Fallback: count-based (original logic, extended to bots >= 1)
    obstacles = int(ep.get('NUMBER_OF_OBSTACLES', 0))
    bots      = int(ep.get('NUMBER_OF_BOT_CARS', 0))
    if obstacles == 0 and bots == 0:
        return 'time_trial'
    elif obstacles == 6 and bots == 0:
        return 'obstacle_avoidance'
    elif obstacles == 0 and bots >= 1:
        return 'head_to_bot'
    else:
        raise ValueError(
            f'Cannot determine race type: NUMBER_OF_OBSTACLES={obstacles}, '
            f'NUMBER_OF_BOT_CARS={bots} in {environment_params_path}. '
            f'Add a RACE_TYPE key to the YAML.'
        )


def demo(
        agent: Agent,
        environment_name: str=ENVIRONMENT_NAME,
        directory: str='./demos'
    ):
    race_type = get_race_type(
        environment_params_path=ENVIRONMENT_PARAMS_PATH
    )
    world_name = get_world_name(
        environment_params_path=ENVIRONMENT_PARAMS_PATH
    )

    demo_device = torch.device('cpu')
    agent.eval().to(demo_device)
    os.makedirs(directory, exist_ok=True)

    demo_environment = make_environment(
        environment_name, render_mode='rgb_array'
    )

    demo_environment = RecordVideo(
        demo_environment,
        video_folder=directory,
        episode_trigger=lambda _: True,
        name_prefix=f'{world_name}-{race_type}-{agent.name}'
    )

    observation, _ = demo_environment.reset()

    if PROGRESS_MANAGER is not None:
        demo_progress = PROGRESS_MANAGER.counter(
        total=MAX_DEMO_STEPS, desc=f'{world_name} {race_type} demo', unit='steps', leave=False
    )
    for t in range(MAX_DEMO_STEPS):
        action = agent.get_action(torch.Tensor(observation)[None, :])
        if not isinstance(action, np.ndarray) and torch.is_tensor(action):
            action = action.cpu().detach().numpy()
        if isinstance(demo_environment.action_space, spaces.Discrete):
            action = action.item()
        observation, _, terminated, truncated, _ = demo_environment.step(action)
        demo_progress.update()
        demo_progress.refresh()
        if terminated or truncated:
            break

    demo_environment.close()
    demo_progress.close()

    filtered_videos = sorted(
        f for f in os.listdir(directory)
        if (
            f.endswith('.mp4')
            and agent.name in f
            and world_name in f
            and race_type in f
        )
    )
    if len(filtered_videos) == 0:
        logger.warning('No videos found!')
        return

    video_path = os.path.join(directory, filtered_videos[-1])
    clear_output(wait=True)
    display(Video(video_path, embed=True))


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_command(command):
    result=subprocess.run(command, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.returncode:
        logger.error(result.stderr)
    else:
        logger.warning(result.stderr)


def evaluate_track(
        agent: Agent,
        world_name: str,
        environment_name: str=ENVIRONMENT_NAME,
        directory: str='./evaluations'
    ):
    race_type = get_race_type(
        environment_params_path=ENVIRONMENT_PARAMS_PATH
    )
    logger.info(f'Starting {race_type} evaluation on {world_name} track.')

    run_command([
        '/bin/bash',
        './scripts/restart_deepracer.sh',
        '-E', 'true',
        '-W', world_name,
    ])

    eval_device = torch.device('cpu')
    agent.eval().to(eval_device)
    os.makedirs(directory, exist_ok=True)

    eval_environment = make_environment(ENVIRONMENT_NAME)
    observation, _ = eval_environment.reset()

    eval_metrics = {'progress': [], 'lap_time': []}
    if PROGRESS_MANAGER is not None:
        evaluation_progress = PROGRESS_MANAGER.counter(
        total=EVAL_EPISODES, desc=f'Evaluating {world_name}', unit='episodes'
    )
    for episode in range(EVAL_EPISODES):
        if PROGRESS_MANAGER is not None:
            episode_progress = PROGRESS_MANAGER.counter(
            total=MAX_EVAL_STEPS, desc=f'Episode {episode}', unit='steps', leave=False
        )
        for t in range(MAX_EVAL_STEPS):
            action = agent.get_action(torch.Tensor(observation)[None, :])
            if not isinstance(action, np.ndarray) and torch.is_tensor(action):
                action = action.cpu().detach().numpy()
            if isinstance(eval_environment.action_space, spaces.Discrete):
                action = action.item()
            observation, reward, terminated, truncated, info = eval_environment.step(action)
            episode_progress.update()
            episode_progress.refresh()
            done = terminated or truncated
            if done or t == MAX_EVAL_STEPS - 1:
                progress = info.get('reward_params', {}).get('progress', 0.0)
                lap = lap_time(info)
                eval_metrics['progress'].append(progress)
                eval_metrics['lap_time'].append(lap)
                logger.info(f'Episode {episode}:\t progress: {progress}\t lap_time: {lap}')
                observation, info = eval_environment.reset()
                break
        episode_progress.close()
        evaluation_progress.update()
        evaluation_progress.refresh()
    evaluation_progress.close()
    eval_environment.close()

    try:
        with open(f'{directory}/{race_type}-{agent.name}.json', '+r') as f:
            all_metrics = json.load(f)
    except:
        all_metrics = {}

    all_metrics.update({world_name: eval_metrics})
    with open(f'{directory}/{race_type}-{agent.name}.json', '+w') as f:
        json.dump(all_metrics, f)

    return eval_metrics


def evaluate(
        agent: Agent,
        environment_name: str=ENVIRONMENT_NAME,
        directory: str='./evaluations'
    ):
    race_type = get_race_type(environment_params_path=ENVIRONMENT_PARAMS_PATH)

    eval_world_names = ([
        'reInvent2019_wide',
        'reInvent2019_track',
        'Vegas_track',
    ])

    eval_device = torch.device('cpu')
    agent.eval().to(eval_device)
    os.makedirs(directory, exist_ok=True)

    if PROGRESS_MANAGER is not None:
        status = PROGRESS_MANAGER.status_bar(
        status_format=race_type + u' {fill}Evaluating {track}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER, track='<track>',
        autorefresh=True, min_delta=0.5
    )
    eval_metrics = {}
    for world_name in eval_world_names:
        status.update(track=world_name)
        status.refresh()
        eval_metrics[world_name] = evaluate_track(
            agent=agent,
            world_name=world_name,
            environment_name=environment_name,
            directory=directory
        )
    status.close()

    with open(f'{directory}/{race_type}-{agent.name}.json', '+w') as f:
        json.dump(eval_metrics, f)

    run_command(['/bin/bash', './scripts/restart_deepracer.sh'])
    return eval_metrics


def plot_metrics(
    data,
    title,
    directory: str = './plots',
    bsts_report: dict = None,
):
    """Publication-quality metric visualisation with STL decomposition.

    Renders a 4-panel figure per metric key in *data*:
      Panel 1 - Observed series with EMA overlay + 95% Bayesian credible interval
      Panel 2 - STL trend component (LOESS-extracted)
      Panel 3 - STL seasonal/cyclic component
      Panel 4 - Residual with +/- 2-sigma control limits

    If *bsts_report* is supplied, a fifth panel shows Kalman-filtered level.

    References:
        Cleveland et al. (1990) STL decomposition
        Hyndman & Athanasopoulos (2021) Forecasting ch.3
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    try:
        from statsmodels.tsa.seasonal import STL
        _HAS_STL = True
    except ImportError:
        _HAS_STL = False
    try:
        from denim_theme import (apply_theme, get_color, DENIM_BRIGHT,
            AMBER, TERRA_COTTA, SAGE_GREEN, MUTED_PURPLE, GOLD_WARM,
            DENIM_MID, DENIM_DARK, BG_DARK, WHITE_SMOKE, MUTED_GRAY)
    except ImportError:
        BG_DARK, WHITE_SMOKE, MUTED_GRAY = "#2E2E2E", "#F5F5F5", "#8C8C8C"
        DENIM_BRIGHT, AMBER, TERRA_COTTA = "#5B9BD5", "#D4A03C", "#C75B39"
        SAGE_GREEN, MUTED_PURPLE, GOLD_WARM = "#6BB38A", "#9B6EB7", "#E8C167"
        DENIM_MID, DENIM_DARK = "#3A6B8C", "#1B3A5C"
        def apply_theme(fig=None, ax=None): pass
        def get_color(i): return [DENIM_BRIGHT, AMBER, TERRA_COTTA][i % 3]

    try:
        from IPython.display import clear_output
    except ImportError:
        def clear_output(wait=False): pass

    PLOT_DPI = 150
    PLOT_FORMAT = "png"
    os.makedirs(directory, exist_ok=True)

    def ema(y, span=20):
        a = 2.0 / (span + 1)
        out = np.empty_like(y, dtype=float)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = a * y[i] + (1 - a) * out[i - 1]
        return out

    def bayesian_credible_interval(y, span=20, ci=0.95):
        """Bayesian credible interval via conjugate Normal-InverseGamma posterior.
        References:
            Murphy (2007) Conjugate Bayesian analysis of the Gaussian distribution
            Gelman et al. (2013) Bayesian Data Analysis, 3rd ed., Ch. 2-3
        """
        from scipy import stats as sp_stats
        n = len(y)
        lo = np.empty(n)
        hi = np.empty(n)
        mu_0 = np.mean(y[:min(10, n)])
        kappa_0 = 1.0
        alpha_0 = 2.0
        beta_0 = np.var(y[:min(10, n)]) + 1e-6
        half_alpha = (1 - ci) / 2
        for t in range(n):
            window = y[max(0, t - span):t + 1]
            nw = len(window)
            y_bar = np.mean(window)
            s2 = np.var(window, ddof=1) if nw > 1 else beta_0
            kappa_n = kappa_0 + nw
            mu_n = (kappa_0 * mu_0 + nw * y_bar) / kappa_n
            alpha_n = alpha_0 + nw / 2.0
            beta_n = (beta_0 + 0.5 * nw * s2
                      + 0.5 * kappa_0 * nw * (y_bar - mu_0)**2 / kappa_n)
            df = 2 * alpha_n
            scale = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))
            lo[t] = sp_stats.t.ppf(half_alpha, df, loc=mu_n, scale=scale)
            hi[t] = sp_stats.t.ppf(1 - half_alpha, df, loc=mu_n, scale=scale)
        return lo, hi

    if isinstance(data, dict):
        metric_names = list(data.keys())
    else:
        metric_names = [title]
        data = {title: data}

    for key in metric_names:
        series = np.asarray(data[key], dtype=float)
        n = len(series)
        if n < 8:
            continue

        has_bsts = (bsts_report is not None
                    and key in bsts_report.get("decompositions", {}))
        n_panels = 5 if has_bsts else 4
        fig = plt.figure(figsize=(14, 3.2 * n_panels))
        gs = gridspec.GridSpec(n_panels, 1, hspace=0.35,
                               height_ratios=[2] + [1]*(n_panels-1))
        axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

        x = np.arange(n)
        ema_line = ema(series)
        lo, hi = bayesian_credible_interval(series)

        plt.style.use('ggplot')
        plt.rcParams.update({
            'figure.facecolor': BG_DARK,
            'axes.facecolor': BG_DARK,
            'text.color': WHITE_SMOKE,
            'axes.labelcolor': WHITE_SMOKE,
            'xtick.color': WHITE_SMOKE,
            'ytick.color': WHITE_SMOKE,
            'axes.edgecolor': MUTED_GRAY,
            'grid.color': '#3A3A3A',
            'grid.alpha': 0.3,
            'font.family': 'serif',
        })

        ax = axes[0]
        ax.fill_between(x, lo, hi, alpha=0.18, color=DENIM_BRIGHT,
                        label="95% Bayesian credible interval")
        ax.plot(x, series, linewidth=0.4, alpha=0.45, color=MUTED_GRAY, label="raw")
        ax.plot(x, ema_line, linewidth=1.8, color=DENIM_BRIGHT, label="EMA-20")
        ax.set_title(f"{key}  (observed)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.6)

        try:
            from scipy import stats as sp_stats
            half = n // 2
            t_stat, p_val = sp_stats.ttest_ind(series[:half], series[half:], equal_var=False)
            cohens_d = ((np.mean(series[half:]) - np.mean(series[:half]))
                        / np.sqrt((np.var(series[:half]) + np.var(series[half:])) / 2 + 1e-12))
            s_mk = sum(1 for i in range(n) for j in range(i+1, n) if series[j] > series[i])
            s_mk -= sum(1 for i in range(n) for j in range(i+1, n) if series[j] < series[i])
            tau = 2.0 * s_mk / (n * (n - 1)) if n > 1 else 0
            sig_str = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.annotate(
                f"Welch t={t_stat:.2f} p={p_val:.4f}{sig_str}  d={cohens_d:.3f}  MK tau={tau:.3f}",
                xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=6.5, color=GOLD_WARM, alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.3", fc=DENIM_DARK, alpha=0.7, ec=MUTED_GRAY))
        except Exception:
            pass
        ax.set_ylabel(key, fontsize=9)

        if _HAS_STL and n >= 14:
            period = min(max(2, n // 5), n // 2)
            if period % 2 == 0:
                period += 1
            try:
                stl = STL(series, period=period, robust=True)
                res = stl.fit()
                trend_c, seasonal_c, resid_c = res.trend, res.seasonal, res.resid
            except Exception:
                trend_c = ema_line
                seasonal_c = np.zeros(n)
                resid_c = series - ema_line
        else:
            trend_c = ema_line
            seasonal_c = np.zeros(n)
            resid_c = series - ema_line

        ax1 = axes[1]
        ax1.plot(x, trend_c, linewidth=1.6, color=AMBER, label="trend")
        ax1.fill_between(x, trend_c - np.std(resid_c), trend_c + np.std(resid_c),
                         alpha=0.12, color=AMBER)
        ax1.set_title("Trend (LOESS)", fontsize=10)
        ax1.set_ylabel("level", fontsize=8)

        ax2 = axes[2]
        ax2.bar(x, seasonal_c, width=1.0, color=SAGE_GREEN, alpha=0.6, label="seasonal")
        ax2.axhline(0, color=MUTED_GRAY, linewidth=0.5, linestyle="--")
        ax2.set_title("Seasonal / Cyclic Component", fontsize=10)
        ax2.set_ylabel("deviation", fontsize=8)

        ax3 = axes[3]
        sigma2 = np.std(resid_c) * 2
        ax3.plot(x, resid_c, linewidth=0.6, color=TERRA_COTTA, alpha=0.7)
        ax3.axhline(sigma2, color=MUTED_PURPLE, linewidth=0.8, linestyle=":", label="+2σ")
        ax3.axhline(-sigma2, color=MUTED_PURPLE, linewidth=0.8, linestyle=":")
        ax3.axhline(0, color=MUTED_GRAY, linewidth=0.5, linestyle="--")
        ax3.fill_between(x, -sigma2, sigma2, alpha=0.06, color=MUTED_PURPLE)
        ax3.set_title("Residual + Control Limits", fontsize=10)
        ax3.set_ylabel("residual", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)

        if has_bsts:
            dec = bsts_report["decompositions"][key]
            ax4 = axes[4]
            lvl = np.asarray(dec["levels"])
            pred = np.asarray(dec["predictions"])
            ax4.plot(x[:len(lvl)], lvl, linewidth=1.4, color=GOLD_WARM, label="Kalman level")
            ax4.plot(x[:len(pred)], pred, linewidth=0.9, color=DENIM_MID,
                     linestyle="--", label="BSTS prediction")
            ax4.set_title("BSTS Kalman Decomposition", fontsize=10)
            ax4.legend(loc="upper left", fontsize=7)
            ax4.set_ylabel("level", fontsize=8)
            drivers = bsts_report.get("intermediary_drivers", {}).get(key, [])
            if drivers:
                driver_txt = "  ".join(
                    f"{nm.replace(chr(95)+chr(109)+chr(101)+chr(97)+chr(110),chr(32))}={c:.3f}"
                    for nm, c in drivers[:3]
                )
                ax4.annotate(f"Top drivers: {driver_txt}",
                             xy=(0.02, 0.05), xycoords="axes fraction",
                             fontsize=7, color=WHITE_SMOKE, alpha=0.85,
                             bbox=dict(boxstyle="round,pad=0.3",
                                       fc=DENIM_DARK, alpha=0.7))

        apply_theme(fig, axes)
        for a in axes:
            a.set_xlim(0, n)
            a.tick_params(labelsize=7)
        axes[-1].set_xlabel("Episode", fontsize=9)
        fig.suptitle(f"{title} :: {key}", fontsize=13, fontweight="bold",
                     y=0.995, color=DENIM_BRIGHT)
        clear_output(wait=True)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", key)
        out_path = f"{directory}/{title}_{safe_key}.{PLOT_FORMAT}"
        fig.savefig(out_path, dpi=PLOT_DPI, format=PLOT_FORMAT,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot_metrics] saved {out_path}")


def lap_time(info):
    if info.get('reward_params', {}).get('progress', 0) >= 100:
        if isinstance(info['episode']['t'], np.ndarray):
            return info['episode']['t'].mean()
        else:
            return info['episode']['t']
    else:
        return np.nan


class BSTSLogger:
    METRIC_KEYS = ["time_sec", "ep_return", "reason",
                   "avg_return", "avg_time", "bsts_score"]

    def __init__(self, win=50, log_dir="./logs"):
        import csv, collections
        self._win = win
        self._episodes = []
        self._returns  = collections.deque(maxlen=win)
        self._times    = collections.deque(maxlen=win)
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "bsts_log.csv")
        self._f = open(self._path, "a", newline="")
        self._w = csv.writer(self._f)
        if os.path.getsize(self._path) == 0:
            self._w.writerow(["episode"] + self.METRIC_KEYS)

    def record(self, time_sec, ep_return, reason):
        ep = len(self._episodes) + 1
        self._episodes.append((time_sec, ep_return, reason))
        self._returns.append(ep_return)
        self._times.append(time_sec)
        avg_r = float(np.mean(self._returns))
        avg_t = float(np.mean(self._times)) if self._times else 0.0
        bsts  = avg_r / max(avg_t, 1e-3)
        self._w.writerow([ep, round(time_sec,3), round(ep_return,4),
                          reason, round(avg_r,4), round(avg_t,3), round(bsts,5)])
        self._f.flush()

    def summary(self):
        if not self._returns:
            return "BSTS: no episodes yet"
        avg_r = float(np.mean(self._returns))
        avg_t = float(np.mean(self._times))
        bsts  = avg_r / max(avg_t, 1e-3)
        return (f"n={len(self._episodes)} avg_ret={avg_r:.3f} "
                f"avg_t={avg_t:.1f}s bsts={bsts:.4f}")

    def close(self):
        self._f.close()


class EpisodeMetricsAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self._steps  = []
        self._ep_num = 0

    def record_step(self, state):
        self._steps.append(state)

    def end_episode(self, ep_progress, ep_return, ep_length,
                    terminated_reason="unknown"):
        speeds  = [s.get("speed", 0.0) for s in self._steps]
        dists   = [s.get("distance_from_center", 0.0) for s in self._steps]
        crashes = sum(1 for s in self._steps if s.get("is_crashed", False))
        summary = {
            "ep":                self._ep_num,
            "ep_progress":       ep_progress,
            "ep_return":         ep_return,
            "ep_length":         ep_length,
            "terminated_reason": terminated_reason,
            "n_steps":           len(self._steps),
            "crash_events":      crashes,
            "avg_speed":         float(np.mean(speeds)) if speeds else 0.0,
            "max_speed":         float(np.max(speeds))  if speeds else 0.0,
            "avg_dist_center":   float(np.mean(dists))  if dists  else 0.0,
        }
        self._ep_num += 1
        self._steps = []
        return summary


# =============================================================================
# Integrated from research_modules.py
# =============================================================================

from collections import deque
import random as _random


class ReplayBuffer:
    """Experience replay buffer for off-policy correction.
    REF: Fujimoto et al. (2018) TD3. ICML.
    """
    def __init__(self, cap=200000):
        self.buf = deque(maxlen=cap)

    def push(self, *t):
        self.buf.append(t)

    def sample(self, n):
        b = _random.sample(self.buf, n)
        return tuple(np.array(x, dtype=np.float32) for x in zip(*b))

    def __len__(self):
        return len(self.buf)


class BSTSTracker:
    """Bayesian Structural Time Series tracker for episode diagnostics."""
    def __init__(self, win=50):
        self.win = win
        self.rets = deque(maxlen=win)
        self.laps = deque(maxlen=win)
        self.events = []

    def record(self, lap_time, ep_return, term_reason):
        self.rets.append(ep_return)
        if lap_time > 0:
            self.laps.append(lap_time)
        self.events.append(term_reason)

    def trend(self):
        if len(self.rets) < 4:
            return 0.0
        half = len(self.rets) // 2
        r = list(self.rets)
        return float(np.mean(r[half:]) - np.mean(r[:half]))

    def summary(self):
        laps = list(self.laps) if self.laps else [0]
        return dict(
            mean_ret=float(np.mean(list(self.rets))) if self.rets else 0,
            std_ret=float(np.std(list(self.rets))) if self.rets else 0,
            mean_lap=float(np.mean(laps)),
            trend=self.trend(),
            events=len(self.events),
        )
