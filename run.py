import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages")); import deepracer_gym
# REF: Balaji, B. et al. (2020). DeepRacer: Autonomous Racing Platform for Sim2Real RL. IEEE ICRA.
# REF: Salazar, J. et al. (2024). Deep RL for Autonomous Driving in AWS DeepRacer. Information, 15(2).
# REF: Samant, N. & Deshpande, A. (2020). How we broke into the top 1% of AWS DeepRacer. Building Fynd.
import math, socket, hashlib
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import yaml
import time
import signal
import torch
import os
import json
import datetime
import numpy as np
import gymnasium as gym
import harmonized_metrics as _hm
from loguru import logger
import subprocess
import csv as _csv
from munch import munchify
from torch.utils.tensorboard import SummaryWriter

from agents import PPOAgent, RandomAgent
# REF: Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
from context_aware_agent import ContextAwarePPOAgent, compute_intermed_targets
from failure_analysis import FailurePointSampler
from brake_field import BrakeField
# REF: Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
from stuck_tracker import StuckTracker
# REF: Kolter, J. Z. & Ng, A. Y. (2009). Near-Bayesian exploration in polynomial time. ICML.
from corner_analysis import lookahead_curvature_scan, curvature_radius, optimal_speed
# REF: Garlick, J. & Middleditch, A. (2022). Real-time optimal racing line generation. IEEE Trans. Games.
# Research-module integration (phased out to corner_analysis + utils)
# REF: Yang, S. et al. (2023). COMPSAC. | Haarnoja, T. et al. (2018). ICML.
# REF: Fujimoto, S. et al. (2018). ICML. | Garlick, J. & Middleditch, A. (2022).
from corner_analysis import (
    LineOfSightReward,    # Garlick, J., & Middleditch, A. (2022)
    CornerAnalyzer,       # Yang, S. et al. (2023). COMPSAC
    OvertakeAnalyzer,     # Yang, S. et al. (2023). REUNS
)
from corner_analysis import (
    compute_braking_reward,
    compute_turn_alignment_reward,
    get_stuck_antecedent_bonus,
        build_racing_line_map,
    racing_line_reward,
)
from federated_pool import FederatedPool
# REF: Lillicrap, T. P. et al. (2016). Continuous control with deep reinforcement learning. ICLR.
from race_line_engine import MultiRaceLineEngine
# REF: Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE Trans. SSC, 4(2), 100-107.
from bsts_seasonal import BSTSFeedback
# REF: Scott, S. L. & Varian, H. R. (2014). Predicting the present with Bayesian structural time series. Int. J. Math. Model. Numer. Optim., 5(1-2), 4-23.
live_analyze = lambda *a,**k: None  # live_metrics.py purged (stubbed)
try:
    from live_dashboard import console_summary as live_summary
except Exception:
    live_summary = None
from analyze_logs import (
# REF: Brodersen, K. H. et al. (2015). Inferring causal impact using Bayesian structural time-series models. Ann. Appl. Stat., 9(1), 247-274.
    BSTSKalmanFilter,
    extract_intermediary_metrics,
    episode_summary_metrics,
    bsts_compliance_report,
    compute_anneal_recommendations,
    INTERMEDIARY_METRICS,
    SUCCESS_METRICS,
    compute_optimal_race_line,
    score_race_line_compliance,
)
from td3_sac_ensemble import TD3SACEnsemble
from utils import (
    device,
    set_seed,
    BSTSLogger,
    EpisodeMetricsAccumulator,
    BSTSTracker,          # AWS. (2020). DeepRacer log analysis
    ReplayBuffer,         # Fujimoto, S. et al. (2018). ICML
    make_environment,
)

_device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(_device_str)   # explicit torch.device object, not a lambda/function

# v210: ContextAwarePPOAgent helpers
def process_action(rawaction, actionspace):
    """Squeeze batch dim; for Discrete → argmax-then-clamp; for Box → clip."""
    a = np.asarray(rawaction)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]

    if isinstance(actionspace, gym.spaces.Discrete):
        # a might be logits (shape [n]) or a scalar index
        if a.ndim >= 1 and a.size == actionspace.n:
            # logits → pick best action
            idx = int(np.argmax(a))
        else:
            # already a scalar index — just clamp it
            idx = int(np.round(a.item() if hasattr(a, 'item') else float(a)))
        return int(np.clip(idx, 0, actionspace.n - 1))
    else:
        # Continuous Box
        actdim = actionspace.shape[0]
        if a.ndim == 0:
            a = np.array([float(a)])
        a = a.copy().astype(np.float32)
        if actdim >= 2 and a.size >= 2:
            # remap throttle channel from tanh [-1,1] → env [0,1]
            a[1] = (a[1] + 1.0) / 2.0
        return np.clip(a, actionspace.low, actionspace.high).astype(np.float32)

def compute_track_curvature(waypoints, closest, lookahead=5):
    """Return (curvature, safe_speed). Safe for empty/short waypoint lists."""
    if not waypoints or len(waypoints) < 3:
        return 0.0, 4.0
    n = len(waypoints)
    idx = closest[1] if len(closest) > 1 else 0
    p0 = waypoints[(idx - lookahead) % n]
    p1 = waypoints[idx]
    p2 = waypoints[(idx + lookahead) % n]
    ax, ay = p0[0] - p1[0], p0[1] - p1[1]
    bx, by = p2[0] - p1[0], p2[1] - p1[1]
    cx, cy = p2[0] - p0[0], p2[1] - p0[1]
    cross = abs(ax * by - ay * bx)
    d01 = (ax**2 + ay**2)**0.5 + 1e-8
    d12 = (bx**2 + by**2)**0.5 + 1e-8
    d02 = (cx**2 + cy**2)**0.5 + 1e-8
    curv = 2.0 * cross / (d01 * d12 * d02 + 1e-8)
    safe_speed = min(4.0, max(1.0, 0.8 * (1.0 / (curv + 1e-6))**0.5))
    return curv, safe_speed


def compute_racing_line_offset(waypoints, closest, tw, lookahead=8):
    """Return lateral offset from centerline for racing line (-0.7 to 0.7)."""
    if not waypoints or len(waypoints) < 3:
        return 0.0
    n = len(waypoints)
    idx = closest[1] if len(closest) > 1 else 0
    p0 = waypoints[idx]
    pa = waypoints[(idx + lookahead) % n]
    pb = waypoints[(idx - 3) % n]
    dx1, dy1 = p0[0] - pb[0], p0[1] - pb[1]
    dx2, dy2 = pa[0] - p0[0], pa[1] - p0[1]
    cross = dx1 * dy2 - dy1 * dx2
    mag = (dx1**2 + dy1**2)**0.5 * (dx2**2 + dy2**2)**0.5 + 1e-8
    return max(-0.7, min(0.7, -cross / mag * 3.0))


def _compute_crash_v_perp(speed, heading, closest_wp, waypoints):
    """Velocity component perpendicular to track tangent at closest waypoint."""
    if not waypoints or not closest_wp or len(closest_wp) < 2:
        return 0.0
    try:
        dx = waypoints[closest_wp[1]][0] - waypoints[closest_wp[0]][0]
        dy = waypoints[closest_wp[1]][1] - waypoints[closest_wp[0]][1]
        delta = math.radians(heading - math.degrees(math.atan2(dy, dx)))
        return abs(speed * math.sin(delta))
    except Exception:
        return 0.0


def _compute_crash_v_tang(speed, heading, closest_wp, waypoints):
    """Velocity component tangent to track at closest waypoint."""
    if not waypoints or not closest_wp or len(closest_wp) < 2:
        return 0.0
    try:
        dx = waypoints[closest_wp[1]][0] - waypoints[closest_wp[0]][0]
        dy = waypoints[closest_wp[1]][1] - waypoints[closest_wp[0]][1]
        delta = math.radians(heading - math.degrees(math.atan2(dy, dx)))
        return abs(speed * math.cos(delta))
    except Exception:
        return 0.0

# v5: Track geometry helpers

def compute_racing_line_offset(waypoints, closest, tw, lookahead=8):
    if not waypoints or len(waypoints) < 3: return 0.0
    n = len(waypoints); idx = closest[1] if len(closest) > 1 else 0
    p0 = waypoints[idx]; pa = waypoints[(idx + lookahead) % n]; pb = waypoints[(idx - 3) % n]
    dx1, dy1 = p0[0] - pb[0], p0[1] - pb[1]; dx2, dy2 = pa[0] - p0[0], pa[1] - p0[1]
    cross = dx1 * dy2 - dy1 * dx2; mag = (dx1**2 + dy1**2)**0.5 * (dx2**2 + dy2**2)**0.5 + 1e-8
    return max(-0.7, min(0.7, -cross / mag * 3.0))

DEVICE = device()
HYPER_PARAMS_PATH: str = 'configs/hyper_params.yaml'


def tensor(x: np.array, dtype=torch.float, dev=DEVICE) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=dev)


def zeros(x: tuple, dtype=torch.float, dev=DEVICE) -> torch.Tensor:
    return torch.zeros(x, dtype=dtype, device=dev)

def obs_to_array(obs):
    """Convert observation to numpy array, handling dict observations."""
    if isinstance(obs, dict):
        arrays = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if hasattr(v, 'flatten'):
                arrays.append(v.flatten())
            else:
                arrays.append(np.array(v).flatten())
        return np.concatenate(arrays)
    if hasattr(obs, '__array__'):
        return np.asarray(obs)
    return np.array(obs)


def obs_to_2d_img(obs, H=120, W=160):
    """v1.1.0: Restore DeepRacer camera obs back to 2D spatial tensor (H,W).
    DeepRacer camera: 120x160 grayscale = 19200 floats (first slice of flat obs).
    Gemini (2026-04-25) confirmed: spatial structure required for Swin-style attention.
    The SelfAttention1D in utransformer.py operates on 1D sequences, but shifted-window
    attention needs 2D grid topology. This function gives a safe reshape entry point.
    Returns: np.ndarray (H, W) float32, or None if obs too short.
    """
    flat = obs_to_array(obs)
    n = H * W  # 19200
    if flat.size < n:
        return None
    return flat[:n].reshape(H, W).astype(np.float32)



# ============================================================
# Meta-Annealing Scheduler (3 dimensions)
# ============================================================
# REF: Almakhayita, S. K. et al. (2025). Reward design and hyperparameter tuning for generalizable deep RL agents. PLoS ONE, 20(6).
class AnnealingScheduler:
    """Manages 3 annealing dimensions: reward_weights, hyperparams, architecture."""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.explore_end = 0.25
        self.transition_end = 0.60

        # PHASE -1: Survival bootstrap (0–5% of training)
        # ONE objective: advance. No racing line, no speed gate.
        # Without this the agent learns to oscillate for stuck_bonus near start.
        self.rw_phase_m1 = {
            "center":         0.05,
            "heading":        0.18,   # v1.1.0: boosted from 0.10 — point FORWARD is critical in bootstrap
            "racing_line":    0.02,
            "braking":        0.01,   # v1.1.0: reduced from 0.05 to fund heading
            "progress":       0.62,   # DOMINANT — just advance
            "corner":         0.02,
            "speed_steering": 0.02,
            "curv_speed":     0.00,
            "min_speed":      0.06,   # v1.1.0: reduced from 0.10 to fund heading
            "completion":     0.02,
            "decel":          0.00,
            "obstacle":       0.00,
            "steering":       0.00,
        }
        # PHASE 0: Completion-first (5–30% of training)
        # Positioning, heading, brake-field, racing line dominate. Speed suppressed.
        self.rw_phase0 = {
            "center":         0.22,
            "heading":        0.20,
            "racing_line":    0.18,
            "braking":        0.15,
            "progress":       0.14,
            "corner":         0.06,
            "speed_steering": 0.03,
            "curv_speed":     0.01,
            "min_speed":      0.01,
            "completion":     0.00,
            "decel":          0.00,
            "obstacle":       0.00,
            "steering":       0.00,
        }
        # PHASE 1: Positioning+Speed coupling (30â€“65%)
        # Racing line compliance gates speed reward via speed_steering
        self.rw_phase1 = {
            "center":       0.14,
            "heading":      0.12,
            "racing_line":  0.14,
            "braking":      0.10,
            "progress":     0.16,
            "corner":       0.07,
            "speed_steering": 0.10, # rising â€” speed only rewarded when steering is clean
            "curv_speed":   0.08,   # curvature-aware speed, still modest
            "min_speed":    0.06,
            "completion":   0.05,
            "decel":        0.03,
            "obstacle":     0.03,
            "steering":     0.02,
        }
        # PHASE 2: Speed-optimized (65â€“100%)
        # Full speed rewards, position/heading fade, completion bonus peaks
        self.rw_phase2 = {
            "center":       0.07,
            "heading":      0.06,
            "racing_line":  0.05,
            "braking":      0.07,
            "progress":     0.18,
            "corner":       0.04,
            "speed_steering": 0.10,
            "curv_speed":   0.18,   # speed reward fully live
            "min_speed":    0.10,
            "completion":   0.10,
            "decel":        0.02,
            "obstacle":     0.02,
            "steering":     0.01,
        }

        self.rw_start = self.rw_phase0   # existing code reads rw_start/rw_end
        self.rw_end   = self.rw_phase1   # will swap to phase2 at 65%
        
    def _sigmoid_blend(self, step, start_frac, end_frac, k=12.0):
        mid = (start_frac + end_frac) / 2.0
        t = step / self.total_steps
        return 1.0 / (1.0 + math.exp(-k * (t - mid)))
    
    def get_reward_weights(self, step):
        t = step / max(self.total_steps, 1)
        if t < 0.05:
            # Phase -1 → 0: survival bootstrap
            alpha = t / 0.05
            src, dst = self.rw_phase_m1, self.rw_phase0
        elif t < 0.30:
            # Phase 0 → 1: positioning+compliance
            alpha = (t - 0.05) / 0.25
            src, dst = self.rw_phase0, self.rw_phase1
        elif t < 0.65:
            # Phase 1 → 2: speed coupling
            alpha = (t - 0.30) / 0.35
            src, dst = self.rw_phase1, self.rw_phase2
        else:
            alpha = min((t - 0.65) / 0.35, 1.0)
            src, dst = self.rw_phase2, self.rw_phase2

        weights = {}
        for k in set(list(src.keys()) + list(dst.keys())):
            s = float(src.get(k, 0.0))
            e = float(dst.get(k, 0.0))
            weights[k] = s * (1.0 - alpha) + e * alpha

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()} if total > 0 else weights

    def get_hyperparams(self, step):
        t = step / self.total_steps
        base_lr, min_lr = 3e-4, 1e-4  # v11: higher LR
        cycle_len = self.total_steps / 3
        cycle_pos = (step % cycle_len) / cycle_len
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * cycle_pos))
        ent_coef = 0.02 + (0.01 - 0.02) * self._sigmoid_blend(step, 0.1, 0.5)  # annealed
        clip_coef = 0.25 + (0.15 - 0.25) * t  # annealed clip
        gae_lambda = 0.95 + (0.98 - 0.95) * t
        return {"lr": lr, "ent_coef": ent_coef, "clip_coef": clip_coef, "gae_lambda": gae_lambda}

    def get_architecture_params(self, step):
        dropout = 0.15 + (0.02 - 0.15) * self._sigmoid_blend(step, 0.2, 0.6)
        return {"dropout": dropout}


# v27: barrier-relative velocity helpers
def _compute_crash_v_perp(speed, heading, closest_wp, waypoints):
    if not waypoints or not closest_wp or len(closest_wp) < 2: return 0.0
    try:
        dx = waypoints[closest_wp[1]][0] - waypoints[closest_wp[0]][0]
        dy = waypoints[closest_wp[1]][1] - waypoints[closest_wp[0]][1]
        delta = math.radians(heading - math.degrees(math.atan2(dy, dx)))
        return abs(speed * math.sin(delta))
    except: return 0.0

def _compute_crash_v_tang(speed, heading, closest_wp, waypoints):
    if not waypoints or not closest_wp or len(closest_wp) < 2: return 0.0
    try:
        dx = waypoints[closest_wp[1]][0] - waypoints[closest_wp[0]][0]
        dy = waypoints[closest_wp[1]][1] - waypoints[closest_wp[0]][1]
        delta = math.radians(heading - math.degrees(math.atan2(dy, dx)))
        return abs(speed * math.cos(delta))
    except: return 0.0



# --- v205 auto-bootstrap: start deepracer container if needed ---
def _auto_bootstrap_deepracer(max_wait=60):
    """Auto-start deepracer Docker/Apptainer container and discover GYM_PORT."""
    import subprocess, time, shutil
    script_dir = os.path.dirname(os.path.abspath(__file__))
    start_sh = os.path.join(script_dir, 'start_deepracer.sh')
    env_sh = os.path.join(script_dir, 'env_for_client.sh')
    # Check if gym bridge is already reachable
    if _preflight_gym_bridge():
        return True
    print('[v205] gym bridge not reachable, attempting auto-bootstrap...', flush=True)
    # Try to start the container
    if os.path.exists(start_sh) and shutil.which('bash'):
        try:
            subprocess.run(['bash', start_sh], timeout=600, capture_output=True)
            print('[v205] start_deepracer.sh executed', flush=True)
        except Exception as e:
            print(f'[v205] start_deepracer.sh failed: {e}', flush=True)
    # Wait for gym bridge with retries
    for i in range(max_wait // 5):
        time.sleep(5)
        # Try to source env_for_client.sh to discover GYM_PORT
        if os.path.exists(env_sh) and shutil.which('bash'):
            try:
                result = subprocess.run(
                    ['bash', '-c', f'source {env_sh} 2>/dev/null && echo GYM_PORT=$GYM_PORT'],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    if line.startswith('GYM_PORT='):
                        port = line.split('=',1)[1].strip()
                        if port and port != '0':
                            os.environ['GYM_PORT'] = port
                            print(f'[v205] auto-discovered GYM_PORT={port}', flush=True)
            except Exception:
                pass
        if _preflight_gym_bridge():
            return True
        print(f'[v205] waiting for gym bridge... ({(i+1)*5}s/{max_wait}s)', flush=True)
    print('[v205] auto-bootstrap failed: gym bridge still unreachable', flush=True)
    return False
# --- end v205 auto-bootstrap ---

# --- v202 preflight ---
def _preflight_gym_bridge():
    """Verify GYM_PORT is reachable before training starts."""
    import os, socket
    host=os.environ.get('GYM_HOST','127.0.0.1')
    port=int(os.environ.get('GYM_PORT','8888'))
    try:
        with socket.create_connection((host,port), timeout=5):
            print(f'[preflight v202] gym bridge {host}:{port} OPEN', flush=True); return True
    except Exception as e:
        print(f'[preflight v202] gym bridge {host}:{port} UNREACHABLE: {e}', flush=True); return False
# --- end v202 preflight ---


# ============================================================
# v212: Embedded 9-Phase Orchestrator — yaml-aware
# 3 tracks x 3 variants = 9 training phases
# Each phase maps directly to an environment_params yaml file.
# _apply_phase_env passes the CORRECT yaml path to make_environment,
# so WORLD_NAME / RACE_TYPE / NUMBER_OF_OBSTACLES are always right.
# ============================================================
import itertools as _itertools, hashlib as _hashlib, socket as _socket

# Canonical 3x3 phase identifiers
TRACKS   = ["reinvent2019_wide", "reinvent2019_track", "vegas_track"]
VARIANTS = ["time_trial", "obstacle", "h2h"]
_ALL_PHASES = list(_itertools.product(TRACKS, VARIANTS))

# Maps (track_key, variant_key) -> configs/ yaml filename
# yaml filenames match what deepracer-gym starter code ships (plus the 6 new ones below)
_PHASE_YAML_REGISTRY = {
    ("reinvent2019_wide",  "time_trial"): "environment_params_tt_reinvent.yaml",
    ("reinvent2019_wide",  "obstacle"):   "environment_params_oa.yaml",
    ("reinvent2019_wide",  "h2h"):        "environment_params_h2h_reinvent.yaml",
    ("reinvent2019_track", "time_trial"): "environment_params_tt_reinvent_track.yaml",
    ("reinvent2019_track", "obstacle"):   "environment_params_oa_reinvent_track.yaml",
    ("reinvent2019_track", "h2h"):        "environment_params_h2h_reinvent_track.yaml",
    ("vegas_track",        "time_trial"): "environment_params_tt_vegas.yaml",
    ("vegas_track",        "obstacle"):   "environment_params_oa_vegas.yaml",
    ("vegas_track",        "h2h"):        "environment_params_h2h_vegas.yaml",
}

# Full env-param spec per phase (used as fallback if yaml doesn't exist)
_PHASE_ENV_PARAMS = {
    ("reinvent2019_wide",  "time_trial"): {"WORLD_NAME": "reInvent2019_wide",  "RACE_TYPE": "TIME_TRIAL",       "NUMBER_OF_OBSTACLES": 0, "NUMBER_OF_BOT_CARS": 0},
    ("reinvent2019_wide",  "obstacle"):   {"WORLD_NAME": "reInvent2019_wide",  "RACE_TYPE": "OBJECT_AVOIDANCE", "NUMBER_OF_OBSTACLES": 6, "NUMBER_OF_BOT_CARS": 0},
    ("reinvent2019_wide",  "h2h"):        {"WORLD_NAME": "reInvent2019_wide",  "RACE_TYPE": "HEAD_TO_BOT",      "NUMBER_OF_OBSTACLES": 0, "NUMBER_OF_BOT_CARS": 3},
    ("reinvent2019_track", "time_trial"): {"WORLD_NAME": "reInvent2019_track", "RACE_TYPE": "TIME_TRIAL",       "NUMBER_OF_OBSTACLES": 0, "NUMBER_OF_BOT_CARS": 0},
    ("reinvent2019_track", "obstacle"):   {"WORLD_NAME": "reInvent2019_track", "RACE_TYPE": "OBJECT_AVOIDANCE", "NUMBER_OF_OBSTACLES": 6, "NUMBER_OF_BOT_CARS": 0},
    ("reinvent2019_track", "h2h"):        {"WORLD_NAME": "reInvent2019_track", "RACE_TYPE": "HEAD_TO_BOT",      "NUMBER_OF_OBSTACLES": 0, "NUMBER_OF_BOT_CARS": 3},
    ("vegas_track",        "time_trial"): {"WORLD_NAME": "Vegas_track",        "RACE_TYPE": "TIME_TRIAL",       "NUMBER_OF_OBSTACLES": 0, "NUMBER_OF_BOT_CARS": 0},
    ("vegas_track",        "obstacle"):   {"WORLD_NAME": "Vegas_track",        "RACE_TYPE": "OBJECT_AVOIDANCE", "NUMBER_OF_OBSTACLES": 6, "NUMBER_OF_BOT_CARS": 0},
    ("vegas_track",        "h2h"):        {"WORLD_NAME": "Vegas_track",        "RACE_TYPE": "HEAD_TO_BOT",      "NUMBER_OF_OBSTACLES": 0, "NUMBER_OF_BOT_CARS": 3},
}


def _resolve_phase_yaml(track, variant, configs_dir="configs"):
    """Return absolute path to the yaml for this phase.
    If the file doesn't exist, write it from _PHASE_ENV_PARAMS so training never stalls.
    """
    fname = _PHASE_YAML_REGISTRY.get((track, variant))
    if fname is None:
        raise KeyError(f"[ORCH] No yaml registered for ({track}, {variant})")
    path = os.path.join(configs_dir, fname)
    if not os.path.exists(path):
        # Auto-generate missing yaml from the spec table
        params = _PHASE_ENV_PARAMS[(track, variant)]
        os.makedirs(configs_dir, exist_ok=True)
        with open(path, "w") as _f:
            _f.write("---\n")
            for k, v in params.items():
                _f.write(f"{k}: {v}\n")
        logger.warning(f"[ORCH] Auto-generated missing yaml: {path}")
    return path


def _build_phase_schedule(total_timesteps, cluster_ids=None):
    """Return ordered list of phase dicts for this node.
    Single-cluster (laptop): all 9 phases, equal timestep budget, round-robin.
    Multi-cluster (PACE-ICE): each node gets a permuted ordering via hostname hash.
    Delegates to cluster_orchestrator.get_phase_assignment() if importable.
    """
    my_host  = _socket.gethostname()
    my_hash  = int(_hashlib.md5(my_host.encode()).hexdigest(), 16)
    n_phases = len(_ALL_PHASES)
    phase_ts = total_timesteps // n_phases

    if cluster_ids and len(cluster_ids) > 1:
        try:
            from cluster_orchestrator import get_phase_assignment
            phases = get_phase_assignment(my_host, _ALL_PHASES, cluster_ids)
            logger.info(f"[ORCH] Multi-cluster: {len(cluster_ids)} nodes, "
                        f"{len(phases)} phases delegated to {my_host}")
        except ImportError:
            offset = my_hash % n_phases
            order  = list(range(n_phases))
            order  = order[offset:] + order[:offset]
            phases = [_ALL_PHASES[i] for i in order]
            logger.info(f"[ORCH] Multi-cluster fallback permute offset={offset}")
    else:
        phases = list(_ALL_PHASES)
        logger.info(f"[ORCH] Single-cluster: {n_phases} phases x {phase_ts} steps, host={my_host}")

    return [
        {
            "track":    t,
            "variant":  v,
            "timesteps": phase_ts,
            "phase_id": f"{t}__{v}",
            "index":    i,
            "yaml_path": _resolve_phase_yaml(t, v),
        }
        for i, (t, v) in enumerate(phases)
    ]


def _apply_phase_env(args, phase, current_env=None):
    """Switch to a new phase: update env vars, close old env, spin up fresh one.
    make_environment() always takes the gym ID ("deepracer-v0"), NOT a yaml path.
    Track/variant switching is done by writing the phase yaml path into the
    module-level ENVIRONMENT_PARAMS_PATH used by get_world_name() / get_race_type(),
    and by setting WORLD_NAME / RACE_TYPE env vars for the sim bridge.
    """
    import utils as _utils

    yaml_path = phase.get("yaml_path") or _resolve_phase_yaml(phase["track"], phase["variant"])

    # 1. Tell the deepracer sim which track/variant to use
    env_params = _PHASE_ENV_PARAMS[(phase["track"], phase["variant"])]
    os.environ["WORLD_NAME"]          = env_params["WORLD_NAME"]
    os.environ["RACE_TYPE"]           = env_params["RACE_TYPE"]
    os.environ["NUMBER_OF_OBSTACLES"] = str(env_params["NUMBER_OF_OBSTACLES"])
    os.environ["NUMBER_OF_BOT_CARS"]  = str(env_params["NUMBER_OF_BOT_CARS"])

    # 2. Patch utils module-level path so get_world_name()/get_race_type() resolve correctly
    _utils.ENVIRONMENT_PARAMS_PATH = yaml_path

    logger.info(
        f"[ORCH] Phase {phase['index']+1}/9: {phase['phase_id']}  "
        f"yaml={yaml_path}  WORLD={env_params['WORLD_NAME']}  "
        f"RACE={env_params['RACE_TYPE']}"
    )

    if current_env is not None:
        try:
            current_env.close()
        except Exception:
            pass

    # 3. gym.make always gets the registered env ID, not the yaml path
    env_name = getattr(args, "environment_name",
                   getattr(args, "gym_env_id", "deepracer-v0"))
    return make_environment(env_name)

HTM_PILOT_EPISODES = 50   # collect 50 clean HTM completions
MIN_PILOT_PROGRESS = 80.0  # only keep episodes that hit 80%+

def harvest_htm_pilots(env, htm_agent, td3sac, n_episodes=12, min_progress=2.0):  # v1.1.4: default lowered from 5.0
    """
    Run deterministic pilot and seed replay for BC/TD3.
    v1.1.1 fixes:
      - BCPilot now outputs env-space [steer, throttle∈[0,1]] directly.
        Do NOT call process_action() on BCPilot outputs — it would re-remap throttle.
        For HTMPilotDriver (if available), keep process_action for safety.
      - _bc_is_bc_pilot flag: skip process_action remap for BCPilot instances.
    """
    _is_bc_pilot = isinstance(htm_agent, BCPilot)  # v1.1.1: detect BCPilot
    pilot_count, stored = 0, 0
    for ep in range(max(n_episodes * 3, n_episodes)):
        obs, info = env.reset()
        rp = info.get('reward_params', {}) if isinstance(info, dict) else {}
        _bc_progress_cache = {}
        _bc_progress_state = reset_episode_centerline_progress(rp, _bc_progress_cache)
        ep_buf, ep_prog = [], 0.0
        terminated, truncated = False, False
        while not (terminated or truncated):
            raw_action = htm_agent.act(rp)
            # v1.1.1: BCPilot outputs env-space actions — skip process_action remap
            if _is_bc_pilot:
                action = np.clip(np.asarray(raw_action, dtype=np.float32),
                                 env.action_space.low, env.action_space.high)
            else:
                action = process_action(raw_action, env.action_space)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_rp = info.get('reward_params', {}) if isinstance(info, dict) else {}
            _, _, prog_pct, _, _bc_progress_state = update_episode_centerline_progress(
                next_rp, _bc_progress_cache, _bc_progress_state)
            ep_prog = max(ep_prog, float(prog_pct))
            act_t = torch.tensor(np.asarray(action), dtype=torch.float32)
            # v1.1.4: store RAW obs in replay — agent encoder expects obs_dim=38464 (raw flat array).
            # v1.1.3 compact12 fix resolved NameError but introduced shape mismatch (256x12 vs 38464x128).
            # compact12 is used only for lightweight reward-shaping signals, NOT for TD3/actor training.
            # REF: AWS (2020) — DeepRacer obs_space is flat float32 of shape (38464,).
            obs_t  = torch.tensor(obs_to_array(obs),      dtype=torch.float32)
            nobs_t = torch.tensor(obs_to_array(next_obs), dtype=torch.float32)
            # v1.1.0: normalize pixel obs before replay storage to prevent critic NaN
            # Raw DeepRacer obs values are in [0,255]; must be [0,1] for stable critic training.
            _obs_scale = obs_t.abs().max().item()
            if _obs_scale > 2.0:
                obs_t  = obs_t  / max(_obs_scale, 1.0)
                nobs_t = nobs_t / max(nobs_t.abs().max().item(), 1.0)
            _bc_reward = float(max(-10.0, min(10.0, reward)))
            ep_buf.append((obs_t, act_t, _bc_reward, nobs_t, float(terminated or truncated)))
            obs, rp = next_obs, next_rp
        if ep_buf and ep_prog >= float(min_progress):
            for transition in ep_buf:
                td3sac.store_transition(*transition)
                stored += 1
            pilot_count += 1
            logger.info(f"BC pilot {pilot_count}/{n_episodes}: centerline_progress={ep_prog:.1f}% steps={len(ep_buf)}")
        if pilot_count >= n_episodes:
            break
    logger.info(f"BC pilot harvest done: {pilot_count} episodes, {stored} transitions, replay={len(td3sac.replay)}")
    return pilot_count

def pretrain_td3_bc(td3sac, ppo_agent, bc_steps=2000):
    """
    Supervised pre-training: minimize |Q(s,a_htm) - r + γ·min_Q(s')| on BC buffer.
    No policy noise — these are expert actions, not noisy TD3 rollouts.
    """
    _nan_streak = 0
    for step in range(bc_steps):
        if len(td3sac.replay) < 256:
            break
        # v1.1.0: policy_noise=0.05 — target policy smoothing essential for Q stability
        # REF: Fujimoto et al. (2018) §4.2
        result = td3sac.update_critics(ppo_agent, batch_size=256, policy_noise=0.05)
        _cl = result.get('critic_loss', 0.0)
        # v1.1.0: NaN streak bail-out — if obs are corrupted, abort early instead of 
        # wasting 8 min of CPU. After 20 consecutive NaN batches, the replay is poisoned.
        if not (isinstance(_cl, float) and _cl == _cl):  # float nan check
            _nan_streak += 1
            if _nan_streak >= 20:
                logger.warning(f"[BC pretrain] NaN streak={_nan_streak} at step {step} — "
                               f"aborting pretrain. Check obs normalization.")
                break
        else:
            _nan_streak = 0
        if step % 200 == 0:
            logger.info(f"BC pretrain step {step}: critic_loss={_cl:.4f}, "
                        f"replay_size={result.get('replay_size',0)}, nan_streak={_nan_streak}")
    logger.info("BC pre-training complete — TD3 critics bootstrapped on expert trajectories")

def preflightgymbridge():
    """Verify GYMPORT is reachable before training starts."""
    import os, socket
    host = os.environ.get('GYMHOST', '127.0.0.1')
    port = int(os.environ.get('GYMPORT', 8888))
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"preflight v202 gym bridge {host}:{port} OPEN", flush=True)
            return True
    except Exception as e:
        print(f"preflight v202 gym bridge {host}:{port} UNREACHABLE {e}", flush=True)
        return False
# --- v213: BCPilot — internal deterministic expert for BC seeding ---
# Replaces htm_reference.HTMPilotDriver when unavailable. No external deps.
# Bugs fixed:
#   1. Throttle was in tanh [-1,1] space but process_action() re-maps (t+1)/2, so
#      braking command -0.5 became 0.25 env throttle — car never braked.
#      Fix: output throttle directly in env space [0,1]; skip process_action remapping.
#   2. No heading alignment recovery — car spawns perpendicular to track (~-77deg),
#      BCPilot tried racing-line steering immediately → immediate offtrack.
#      Fix: if |heading_to_track| > 45deg, apply recovery steering override first.
#   3. Waypoint reversal not handled — if car spawns reversed (heading~180deg from
#      track tangent), pilot was steering toward BEHIND waypoints.
#      Fix: detect reversal from is_reversed flag; apply countersteering.

class BCPilot:
    """Deterministic BC pilot. No htm_reference dependency, no OOB possible.
    v1.1.1: Fixed throttle space bug + heading recovery + reversal handling.
    """
    def __init__(self, waypoints, track_width=0.6, track_variant="timetrial"):
        self.waypoints = list(waypoints)
        self.track_width = float(track_width)
        self.track_variant = track_variant

    def _heading_to_track(self, rp):
        """Compute signed angle between car heading and track tangent (deg)."""
        import math as _math
        waypoints = rp.get("waypoints", self.waypoints)
        closest = rp.get("closest_waypoints", [0, 1])
        heading = float(rp.get("heading", 0.0))
        if len(waypoints) < 2:
            return 0.0
        try:
            n = len(waypoints)
            p0 = waypoints[closest[0] % n]
            p1 = waypoints[closest[1] % n]
            track_angle = _math.degrees(_math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            diff = heading - track_angle
            # Normalize to [-180, 180]
            while diff > 180:  diff -= 360
            while diff < -180: diff += 360
            return diff
        except Exception:
            return 0.0

    def act(self, rp: dict):
        import numpy as _np
        import math as _math
        waypoints = rp.get("waypoints", self.waypoints)
        closest = rp.get("closest_waypoints", [0, 1])
        speed = float(rp.get("speed", 0.0))
        dist_ctr = float(rp.get("distance_from_center", 0.0))
        is_left = bool(rp.get("is_left_of_center", False))
        tw = float(rp.get("track_width", self.track_width))
        is_reversed = bool(rp.get("is_reversed", False))

        if len(waypoints) < 3:
            # v1.1.1: output in env space [steer in [-1,1], throttle in [0,1]]
            return _np.array([0.0, 0.3], dtype=_np.float32)

        # --- Heading recovery: if >45deg off track tangent, align first ---
        hdiff = self._heading_to_track(rp)
        if abs(hdiff) > 15.0:
            # v1.1.1: threshold 25deg->15deg; gain /45 (was /60); throttle 0.35->0.15
            # ROOT CAUSE: car spawns at -77.7deg from track tangent. At throttle=0.35 + speed=4m/s,
            # steer=1.0 cannot overcome lateral momentum -> car slides into barrier unrecovered.
            # Fix: slow crawl during realignment so steering torque actually bites.
            steer_sign = -1.0 if hdiff > 0 else 1.0
            steer_strength = min(1.0, abs(hdiff) / 45.0)  # v1.1.1: sharper corrective gain
            return _np.array([steer_sign * steer_strength, 0.15], dtype=_np.float32)  # v1.1.1: slow crawl

        # --- Reversal recovery ---
        if is_reversed:
            # v1.1.1: reversed + large hdiff -> compound error needs stronger steer + slow throttle.
            # At reversed=-180deg AND spawn hdiff=-77.7deg, /90 gain was too gentle.
            hdiff_rev = self._heading_to_track(rp)
            steer_rev = _np.clip(-hdiff_rev / 60.0, -1.0, 1.0)   # v1.1.1: sharper gain /60 (was /90)
            _rev_throttle = 0.20 if abs(hdiff_rev) > 30.0 else 0.35  # v1.1.1: slow when misaligned
            return _np.array([steer_rev, _rev_throttle], dtype=_np.float32)

        # --- Normal racing line control ---
        # v1.1.0: speed-adaptive lookahead — faster speed = look further ahead
        # At 1.6m/s (slow): 8 wps; at 4m/s (max): 20 wps. Prevents late-braking.
        _la_wps = max(8, min(20, int(speed * 5)))
        try:
            _, _, safe_speed, dist_to_corner = lookahead_curvature_scan(
                waypoints, closest, max_lookahead=_la_wps
            )
        except Exception:
            safe_speed, dist_to_corner = 2.5, 5.0

        try:
            rl_offset = compute_racing_line_offset(waypoints, closest, tw, lookahead=max(6, _la_wps // 2))
        except Exception:
            rl_offset = 0.0

        lat_err = (rl_offset * tw / 2.0) - dist_ctr * (1 if is_left else -1)
        # v1.1.0: track-width-adaptive steering gain — prevents overcorrection on narrow tracks
        _steer_gain = max(0.8, min(1.5, 0.6 / max(tw, 0.3)))
        steering = float(_np.clip(lat_err / max(tw * 0.5, 0.01) * _steer_gain, -1.0, 1.0))

        # v1.1.1: throttle directly in env space [0,1] — NO process_action remapping
        # braker<0.4 → needs to brake → throttle=0.05 (near zero)
        # braker>=0.4 and speed < safe_speed*0.85 → accelerate → throttle=0.8
        # otherwise → maintain → throttle=0.45
        try:
            braker = compute_braking_reward(speed, safe_speed, dist_to_corner)
        except Exception:
            braker = 1.0 if speed < safe_speed * 0.95 else 0.5

        # v1.1.0: multiplicative throttle — no subtractive penalties (freeze trap avoided)
        # braker in [0,1]: 1=free (max speed), 0=hard-brake. Multiplicative attenuation:
        #   throttle = base * (floor_frac + (1-floor_frac) * braker)
        # floor_frac=0.40 → at braker=0: 0.55*0.40=0.22 (never stops); at braker=1: 0.55
        # REF: Ng et al. (1999) reward shaping — avoid policies that exploit negative signals.
        _throttle_base = 0.55 if speed < safe_speed * 0.85 else 0.42
        _floor_frac = 0.40  # v1.1.0: raised from 0.33→0.40; floor is ~0.22, not 0.18
        throttle = _throttle_base * (_floor_frac + (1.0 - _floor_frac) * float(braker))
        throttle = max(0.22, throttle)  # hard floor — car must always creep forward

        return _np.array([steering, throttle], dtype=_np.float32)

# v1.1.1: Lightweight observation preprocessor
# obs_dim=38464 is a flattened LIDAR/camera image — too large for BC bootstrap.
# Extract 12 scalar features from reward_params for a compact RL state.

def extract_compact_obs(obs_raw, rp: dict, waypoints, closest) -> np.ndarray:
    """
    v1.1.1: Extract 12 interpretable scalar features from reward_params.
    Falls back to zeros for missing fields. Always returns shape (12,).
    Use this alongside or instead of raw obs for RL state.
    """
    import math as _math
    try:
        speed = float(rp.get("speed", 0.0))
        dist_ctr = float(rp.get("distance_from_center", 0.0))
        tw = float(rp.get("track_width", 1.0))
        heading = float(rp.get("heading", 0.0))
        is_left = 1.0 if rp.get("is_left_of_center", False) else -1.0
        progress = float(rp.get("progress", 0.0)) / 100.0
        is_reversed = 1.0 if rp.get("is_reversed", False) else 0.0
        is_offtrack = 1.0 if rp.get("is_offtrack", False) else 0.0

        # Heading error to track tangent
        if waypoints and len(waypoints) >= 2 and len(closest) >= 2:
            n = len(waypoints)
            p0 = waypoints[closest[0] % n]
            p1 = waypoints[closest[1] % n]
            track_angle = _math.degrees(_math.atan2(p1[1]-p0[1], p1[0]-p0[0]))
            hdiff = heading - track_angle
            while hdiff > 180: hdiff -= 360
            while hdiff < -180: hdiff += 360
            heading_err = hdiff / 180.0  # normalized [-1,1]
            # Lateral position normalized
            lat_pos = dist_ctr / max(tw * 0.5, 0.01) * is_left  # [-1,1]
        else:
            heading_err = 0.0
            lat_pos = 0.0

        # Curvature ahead
        try:
            from corner_analysis import lookahead_curvature_scan
            _, _, safe_speed, dist_to_corner = lookahead_curvature_scan(
                waypoints, closest, max_lookahead=10)
            curv_signal = (speed - safe_speed) / max(safe_speed, 0.1)  # >0 means too fast
            dist_corner_norm = min(dist_to_corner / 5.0, 1.0)
        except Exception:
            curv_signal = 0.0
            dist_corner_norm = 1.0

        speed_norm = speed / 4.0  # normalize to [0,1]

        return np.array([
            speed_norm,         # 0: speed
            lat_pos,            # 1: lateral position [-1,1]
            heading_err,        # 2: heading error to track [-1,1]
            curv_signal,        # 3: overspeed vs safe_speed
            dist_corner_norm,   # 4: distance to next corner
            progress,           # 5: lap progress [0,1]
            is_reversed,        # 6: reversed flag
            is_offtrack,        # 7: offtrack flag
            is_left,            # 8: left/right of center
            float(closest[0] % len(waypoints)) / max(len(waypoints),1),  # 9: wp position
            tw / 2.0,           # 10: half track width (scale)
            0.0,                # 11: reserved
        ], dtype=np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)

# --- v1.0.13: centerline arc-length progress + bootstrap reward controller ---
def _arc_track_length(waypoints) -> float:
    if not waypoints or len(waypoints)<2: return 100.0
    pts=[(float(w[0]),float(w[1])) for w in waypoints if len(w)>=2]
    n=len(pts)
    if n<2: return 100.0
    return max(sum(math.sqrt((pts[(i+1)%n][0]-pts[i][0])**2+(pts[(i+1)%n][1]-pts[i][1])**2) for i in range(n)),1.0)

def _build_centerline_cache(waypoints, cache):
    n = len(waypoints)
    key = id(waypoints)
    if cache.get("key") == key and cache.get("n") == n:
        return cache
    seg = []
    cum = [0.0]
    for i in range(n):
        x0, y0 = float(waypoints[i][0]), float(waypoints[i][1])
        x1, y1 = float(waypoints[(i + 1) % n][0]), float(waypoints[(i + 1) % n][1])
        d = math.hypot(x1 - x0, y1 - y0)
        seg.append(d)
        cum.append(cum[-1] + d)
    cache.clear()
    cache.update({"key": key, "n": n, "seg": seg, "cum": cum, "total": max(cum[-1], 1.0)})
    return cache

def centerline_arc_position_from_reward_params(rp: dict, cache: dict | None = None):
    """Absolute projected arc coordinate in [0, track_len). Not episode progress."""
    if cache is None:
        cache = {}
    waypoints = rp.get("waypoints", []) if isinstance(rp, dict) else []
    closest = rp.get("closest_waypoints", [0, 1]) if isinstance(rp, dict) else [0, 1]
    if not waypoints or len(waypoints) < 2:
        raw = float(rp.get("progress", 0.0) or 0.0) if isinstance(rp, dict) else 0.0
        return max(0.0, raw), 100.0
    n = len(waypoints)
    _build_centerline_cache(waypoints, cache)
    try:
        prev_i = int(closest[0]) % n
        next_i = int(closest[1]) % n
    except Exception:
        prev_i, next_i = 0, 1
    x = float(rp.get("x", waypoints[prev_i][0]) or waypoints[prev_i][0])
    y = float(rp.get("y", waypoints[prev_i][1]) or waypoints[prev_i][1])
    x0, y0 = float(waypoints[prev_i][0]), float(waypoints[prev_i][1])
    x1, y1 = float(waypoints[next_i][0]), float(waypoints[next_i][1])
    vx, vy = x1 - x0, y1 - y0
    denom = vx * vx + vy * vy
    frac = 0.0 if denom <= 1e-12 else max(0.0, min(1.0, ((x - x0) * vx + (y - y0) * vy) / denom))
    arc_m = cache["cum"][prev_i] + frac * cache["seg"][prev_i]
    total_m = cache["total"]
    return float(max(0.0, min(total_m, arc_m))), float(total_m)

def reset_episode_centerline_progress(rp: dict, cache: dict | None = None):
    """Initialize episode-relative progress at this reset/spawn location."""
    arc_m, total_m = centerline_arc_position_from_reward_params(rp, cache or {})
    return {"start_arc": arc_m, "last_arc": arc_m, "unwrapped": 0.0,
            "best_m": 0.0, "total_m": total_m, "last_delta_m": 0.0}

def update_episode_centerline_progress(rp: dict, cache: dict | None, state: dict | None):
    """
    Episode progress = forward centerline displacement since reset/spawn.
    This deliberately does NOT treat waypoint 70/120 as 58% progress if the car spawned there.
    """
    if cache is None:
        cache = {}
    if state is None or "start_arc" not in state:
        state = reset_episode_centerline_progress(rp, cache)
        return 0.0, state["total_m"], 0.0, 0.0, state
    arc_m, total_m = centerline_arc_position_from_reward_params(rp, cache)
    last_arc = float(state.get("last_arc", arc_m))
    delta = arc_m - last_arc
    if delta > 0.5 * total_m:
        delta -= total_m
    elif delta < -0.5 * total_m:
        delta += total_m
    # A single sim step cannot legitimately cover 20% of the track; treat that as reset/teleport noise.
    if abs(delta) > 0.20 * total_m:
        delta = 0.0
    state["last_arc"] = arc_m
    state["total_m"] = total_m
    state["last_delta_m"] = delta
    state["unwrapped"] = float(state.get("unwrapped", 0.0)) + delta
    progress_m = max(0.0, min(total_m, float(state.get("unwrapped", 0.0))))
    state["best_m"] = max(float(state.get("best_m", 0.0)), progress_m)
    pct = 100.0 * state["best_m"] / max(total_m, 1e-6)
    return float(state["best_m"]), float(total_m), float(max(0.0, min(100.0, pct))), float(delta), state

def centerline_progress_from_reward_params(rp: dict, cache: dict | None = None):
    """Compatibility wrapper: returns absolute arc percent, not safe for episode progress."""
    arc_m, total_m = centerline_arc_position_from_reward_params(rp, cache or {})
    return arc_m, total_m, 100.0 * arc_m / max(total_m, 1e-6)

class BootstrapRewardController:
    """Adaptive initial-stage shaping: force completion-first rewards until laps appear."""
    def __init__(self, window=100, max_train_frac=0.60):
        self.window = int(window)
        self.max_train_frac = float(max_train_frac)
        self.recent = []
        self.best_progress = 0.0
        self.completions = 0

    def update_episode(self, progress_pct, completed=False):
        progress_pct = float(progress_pct or 0.0)
        self.best_progress = max(self.best_progress, progress_pct)
        self.completions += int(bool(completed))
        self.recent.append((progress_pct, bool(completed)))
        if len(self.recent) > self.window:
            self.recent = self.recent[-self.window:]

    def active(self, global_step, total_steps):
        if total_steps <= 0:
            return True
        recent_completions = sum(1 for _, done in self.recent if done)
        return (global_step / float(total_steps) < self.max_train_frac
                and recent_completions == 0
                and self.best_progress < 95.0)

    def weights(self, base):
        # Preserve all keys but make progress/min_speed/heading dominate.
        keys = set(base.keys()) | {
            "center", "heading", "racing_line", "braking", "progress",
            "corner", "speed_steering", "curv_speed", "min_speed",
            "completion", "decel", "obstacle", "steering"
        }
        w = {k: 0.0 for k in keys}
        # v1.1.0: boosted progress weight, added strong heading to keep car on track
        w.update({
            "progress": 0.62,   # v1.1.0 was 0.55 — get there, ONLY objective
            "heading": 0.18,    # v1.1.0 was 0.13 — point forward!
            "center": 0.08,     # v1.1.0 was 0.11
            "min_speed": 0.06,  # v1.1.0 was 0.10
            "braking": 0.03,
            "completion": 0.02,
            "racing_line": 0.01,
            "corner": 0.00,
            "speed_steering": 0.00,
            "curv_speed": 0.00,
            "decel": 0.00,
            "obstacle": 0.00,
            "steering": 0.00,
        })
        s = sum(w.values()) or 1.0
        return {k: v / s for k, v in w.items()}


def run(hparams):
    # v203: allow bypass when GYM_BRIDGE_OPTIONAL=1 (Ed #586 local-loop mode)
    # v205: auto-bootstrap deepracer container
    _auto_bootstrap_deepracer()
    if not _preflight_gym_bridge():
        if os.environ.get('GYM_BRIDGE_OPTIONAL','0') == '1':
            print('[v203] gym bridge missing but GYM_BRIDGE_OPTIONAL=1 -- continuing with local ROS env', flush=True)
        else:
            raise SystemExit('[v202] aborting: sim gym bridge unreachable (set GYM_PORT or source env_for_client.sh)')
    start_time = time.time()
    # v24: Track name/variant detection for logging
    _track_name = 'unknown'
    _track_variant = 'regular'
    for _ai, _av in enumerate(sys.argv):
        if _av == '--track' and _ai+1 < len(sys.argv):
            _track_name = sys.argv[_ai+1]
        if _av in ('--h2b', '--obstacle'):
            _track_variant = _av.lstrip('-')
    logger.info(f'v24: Track={_track_name} variant={_track_variant}')

    # load hyper-params
    with open(HYPER_PARAMS_PATH, 'r') as file:
        default_hparams = yaml.safe_load(file)

    final_hparams = default_hparams.copy()
    final_hparams.update(hparams)
    args = munchify(final_hparams)

    # --- Track identification ---
    try:
        import os as _os; _base_dir = _os.path.dirname(_os.path.abspath(__file__)); _env_params_path = getattr(args, 'environment_params_path', _os.path.join(_base_dir, 'configs', 'environment_params.yaml'))
        with open(_env_params_path) as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line.startswith('WORLD_NAME'):
                    _track_name = _line.split(':',1)[1].strip().strip('"').strip("'")
                elif _line.startswith('RACE_TYPE'):
                    _track_variant = _line.split(':',1)[1].strip().strip('"').strip("'").lower()
    except Exception as _e:
        print(f'Track ID error: {_e}')
    hp = args  # alias for hyperparameter access

    run_name = (
        f"{args.environment}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        'hyperparameters',
        "|param|value|\n|-|-|\n%s" % (
            "\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()]
            )
        ),
    )

    set_seed(args.seed)
    _host_hash = int(hashlib.md5(socket.gethostname().encode()).hexdigest(), 16) % 10000
    args.seed = args.seed + _host_hash
    set_seed(args.seed)
    logger.info(f"Diversified seed: {args.seed} (host_offset={_host_hash})")
    # --- v211: 9-phase orchestrator bootstrap ---
    _cluster_ids = [c.strip() for c in os.environ.get("CLUSTER_IDS", "").split(",") if c.strip()]
    _phase_schedule = _build_phase_schedule(args.total_timesteps, _cluster_ids or None)
    logger.info(f"[ORCH] Schedule ({len(_phase_schedule)} phases): {[p['phase_id'] for p in _phase_schedule]}")
    _current_phase_idx = 0
    _phase = _phase_schedule[0]
    env = _apply_phase_env(args, _phase, current_env=None)
    _phase_steps_remaining = _phase["timesteps"]
    logger.debug(f"action_space type: {type(env.action_space).__name__}")
    logger.debug(f"action_space: {env.action_space}")
    logger.debug(f"has .n: {hasattr(env.action_space, 'n')}, .n={getattr(env.action_space, 'n', None)}")
    logger.debug(f"has .low: {hasattr(env.action_space, 'low')}, shape={getattr(env.action_space, 'shape', None)}")
    # --- v6: stuck tracker ---
    stuck_tracker = StuckTracker(save_path=os.path.join("runs", run_name, "stuck_stats.json"))

    # --- v211: ALL dims resolved BEFORE any agent instantiation ---
    _obs_dim = env.observation_space.shape[0] if hasattr(env, "observation_space") else 64
    _is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if _is_discrete:
        _act_dim       = 1
        _act_n         = int(env.action_space.n)
        _act_dim_agent = _act_n   # ContextAwarePPOAgent discrete head = n logits
    else:
        _act_dim       = env.action_space.shape[0]
        _act_n         = None
        _act_dim_agent = _act_dim
    logger.info(f"v211 dims: obs={_obs_dim}  discrete={_is_discrete}  act_dim={_act_dim}  act_n={_act_n}")

    # PPO agent — safe: _obs_dim defined above
    agent = ContextAwarePPOAgent(
        obs_dim=_obs_dim,
        act_dim=_act_dim_agent,
        name="ctx_ppo_agent",
    ).to(DEVICE)

    # --- v19: TD3+SAC critic ensemble ---
    td3sac = TD3SACEnsemble(obs_dim=_obs_dim, act_dim=_act_dim_agent, hidden=256,
                            gamma=hp.get('gamma', 0.99), tau=0.005, 
                            lr=hp.get('lr', 3e-4), device=DEVICE,)
    _td3sac_update_freq = 4
    logger.info(f"v211: TD3+SAC init obs={_obs_dim} act={_act_dim}")
    
    # --- v4: Load checkpoint if available ---
    checkpoint_path = os.environ.get("CHECKPOINT", "ppo_agent_best.torch")
    # RandomAgent: baseline reference - not used for training actions
    # wired so PPOAgent + RandomAgent are both live imports (not dangling)
    _random_agent = RandomAgent(env, name="random_baseline")
    _ppo_baseline = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=2,  # v32: 2D continuous (steering, throttle/brake)
        name="ppo_baseline"
    )
    logger.info(f"Agents: main={agent.name} baseline={_ppo_baseline.name} random={_random_agent.name}")
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                agent.load_state_dict(ckpt['state_dict'])
                logger.info(f"v3 Loaded state_dict checkpoint from {checkpoint_path}")
            else:
                # Legacy: full-object pickle (old format)
                agent = ckpt.to(DEVICE)
                logger.info(f"v3 Loaded legacy object checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Checkpoint load failed ({e}), training from scratch")
    else:
        logger.info("v3 No checkpoint found, training from scratch")
    # NaN guard: if checkpoint has NaN weights, reinit
    _nan_params = [n for n, p in agent.named_parameters() if p.isnan().any()]
    if _nan_params:
        logger.warning(f"Checkpoint has NaN in {len(_nan_params)} params: {_nan_params[:3]}. Reinitializing weights.")
        agent._init_weights()
    else:
        logger.info("v3: Checkpoint weights look clean — no NaN detected")

    # v8: Federated checkpoint pool
    pool_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "federated_pool")
    pool = FederatedPool(pool_dir=pool_dir, max_pool_size=5)
    pool.load_manifest()

    # v16: Race-line engine (plural lines: time-trial + obstacle-avoidance)
        # REF: Garlick, S. & Bradley, A. (2022). Real-time optimal trajectory planning for autonomous vehicles. Vehicle System Dynamics, 60(12).
    race_engine = None  # v18: lazy-init after waypoints available

    # v16: BSTS Seasonal tracker (season=lap position, trend=epoch)
    bsts_feedback = BSTSFeedback(
        ema_alpha=0.05,
        feedback_strength=0.15,
        race_type=_track_variant,   # race-type tagging live from init
    )
    # v1.1.0: replace _NullModel stub with real BSTSSeasonal instance
    # bsts_feedback.model() returns _NullModel (backward-compat only) — record_step()/_flush_episode()
    # were silently no-ops, so BSTS-Kalman trends stayed 0.0 forever.
    try:
        from bsts_seasonal import BSTSSeasonal as _BSTSSeasonal
        bsts_season = _BSTSSeasonal(n_segments=12, save_dir=os.path.join("results", run_name), alpha=0.02)
        logger.info("[BSTS] BSTSSeasonal instantiated — record_step/_flush_episode now active")
    except Exception as _e_bsts:
        bsts_season = bsts_feedback.model("ep_return", period=100)  # fallback to null
        logger.warning(f"[BSTS] BSTSSeasonal import failed, using NullModel: {_e_bsts}") 

    # v16: Race-line blend factor: starts at 0.0 (all rigid), anneals to 1.0 (all race-line)
    _rl_blend = 0.0

    _RESEARCH_MODULES = True  # Activate research module integrations
    # v17 BSTS wiring imports
    try:
        live_analyze = lambda *a,**k: None  # live_metrics.py purged (stubbed)
        from failure_analysis import FailurePointSampler
        _ANALYSIS_MODULES = True
    except Exception:
        _ANALYSIS_MODULES = False
    # === RESEARCH INIT Haarnoja et al. (2018)Fujimoto et al. (2018)Garlick & Middleditch (2022) ===
    _los = LineOfSightReward(lookahead=5, weight=0.3) if _RESEARCH_MODULES else None
    _corners = CornerAnalyzer() if _RESEARCH_MODULES else None
    _overtake = OvertakeAnalyzer(safe=1.5, bonus=2.0) if _RESEARCH_MODULES else None
    _bsts = BSTSTracker(win=50) if _RESEARCH_MODULES else None
    # BSTSLogger: CSV episode logger wired to _bsts.record() calls below
    _bsts_log = BSTSLogger(win=50, log_dir="results") if _RESEARCH_MODULES else None
    _rbuf = ReplayBuffer(cap=200000) if _RESEARCH_MODULES else None
    print(f"[RESEARCH] active={_RESEARCH_MODULES}")
    # REF: Haarnoja et al. (2018) adaptive entropy temperature alpha
    # Target entropy = -|A| (action dimension)
    _target_entropy = -float(agent.actor.mu_head.out_features) if hasattr(agent, 'actor') and hasattr(agent.actor, 'mu_head') else -2.0
    _log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    _alpha_optim = torch.optim.Adam([_log_alpha], lr=3e-4)
    optimizer = torch.optim.Adam(
        agent.parameters(), lr=float(args.learning_rate), eps=1e-5,
    )

    # PPO hyperparameters
    num_steps = getattr(args, 'num_steps', 512)
    num_minibatches = getattr(args, 'num_minibatches', 4)
    update_epochs = getattr(args, 'update_epochs', 4)
    vf_coef = getattr(args, 'vf_coef', 0.5)
    max_grad_norm = getattr(args, 'max_grad_norm', 0.5)
    target_kl = getattr(args, 'target_kl', None)
    norm_adv = getattr(args, 'norm_adv', True)
    clip_vloss = getattr(args, 'clip_vloss', True)

    total_timesteps = args.total_timesteps
    _rl_blend_rate = 1.0 / max(total_timesteps, 1)  # linear anneal over training
    batch_size = num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // num_steps

    # --- v4: Meta-Annealing Scheduler ---
    scheduler = AnnealingScheduler(total_timesteps)
    bootstrap_rewards = BootstrapRewardController(window=100, max_train_frac=0.60)
    
        # --- v4: Failure Point Sampler ---
    sampler = FailurePointSampler(save_dir='results', num_segments=10, max_samples=50)
    # EpisodeMetricsAccumulator: step-level state accumulator for braking/speed analysis
    ep_metrics = EpisodeMetricsAccumulator()
    _track_progress_cache = {}
    _episode_progress_state = {}

    # --- v200: JSONL metrics file for BSTS analysis ---
    os.makedirs('results', exist_ok=True)
    jsonl_path = f'results/v200_metrics_{socket.gethostname()}_{int(time.time())}.jsonl'
    jsonl_file = open(jsonl_path, 'w')

    # --- Live dashboard subprocess (matplotlib, headless PNG updates every 30s) ---
    _dash_env = {**__import__('os').environ, 'LOG_DIR': 'results', 'DASH_OUT': 'results/dashboard', 'DASH_REFRESH': '30'}
    try:
        _dash_proc = subprocess.Popen(
            ['python3','-u','live_bsts_plot.py','--mode','all','--log-dir','results','--out-dir','results/live','--refresh','60'],
            env=_dash_env, stdout=open('results/dashboard.log','w'), stderr=subprocess.STDOUT
        )
        logger.info(f'[DASHBOARD] Started PID={_dash_proc.pid}')
    except Exception as _de:
        logger.warning(f'[DASHBOARD] Could not start: {_de}')
        _dash_proc = None

    # --- BSTS metrics CSV for reward weight history ---
    os.makedirs('results', exist_ok=True)
    _bsts_csv_path = 'results/bsts_metrics.csv'
    _bsts_csv_header_written = os.path.exists(_bsts_csv_path)
    _bsts_csv_f = open(_bsts_csv_path, 'a', newline='')
    _bsts_csv_keys = [
        'episode', 'global_step',
        'race_type', 'track_name', 'track_variant',
        'track_arc_m', 'track_progress_pct', 'track_progress_arc_m',
        'crash_rate', 'offtrack_rate', 'avg_speed',
        'corner_crash_rate', 'avg_safe_speed_ratio', 'avg_racing_line_err',
        'rw_center', 'rw_heading', 'rw_curv_speed', 'rw_progress',
        'rw_completion', 'rw_corner', 'rw_braking', 'rw_min_speed', 'rw_racing_line',
    ]
    _bsts_csv_writer = _csv.DictWriter(_bsts_csv_f, fieldnames=_bsts_csv_keys, extrasaction='ignore')
    if not _bsts_csv_header_written:
        _bsts_csv_writer.writeheader()
        _bsts_csv_f.flush()

    logger.info(f"PPO v7 BSTS+Dynamics+StuckFix Training: {total_timesteps} timesteps, {num_updates} updates")
    logger.info(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}")
    logger.info(f"JSONL metrics: {jsonl_path}")

    # --- v4: Sub-reward accumulators (per episode) ---
    ep_rewards_components = {
        "center": [], "speed": [], "steering": [],
        "progress": [], "efficiency": [], "heading": [],
            "braking": [], "turn_align": []
    }
    ep_speeds = []
    ep_headings = []  # v26
    ep_closest_wps = []  # v26
    ep_dist_from_center = []
    ep_offtrack_count = 0
    ep_step_count = 0
    ep_offtrack_steps = 0  # v39: offtrack grace period
    cumulative_ep_reward = 0.0
    _prev_prog_tracker = 0.0  # v23
    _bsts_alpha = 0.7  # v39: 70/30 shaped/env blend (was 0.90)
    bsts_metrics = {}
    bsts_feedback = BSTSFeedback(
        ema_alpha=0.05,
        feedback_strength=0.15,
        race_type=_track_variant,   # race-type tagging live from init
    )
    bsts_feedback._all_summaries = []  # accumulate all episode summaries for periodic BSTS report
    # Kalman-filter BSTS for proper trend/seasonal/regression decomposition
    _kf_bsts = {sm: BSTSKalmanFilter(seasonal_period=8, n_regressors=len(INTERMEDIARY_METRICS),
                    sigma_obs=1.0, sigma_level=0.1, sigma_trend=0.01,
                    sigma_season=0.05, sigma_beta=0.01)
                for sm in SUCCESS_METRICS}
    _kf_episode_buffer = []  # accumulate per-episode summaries for online BSTS
    bsts_feedback.kf_trends = {}
    bsts_feedback.kf_betas = {}
    # v4-bsts accumulators
    ep_heading_diffs = []
    _ep_step_log = []  # collect per-step dicts for extract_intermediary_metrics
    ep_steerings_raw = []
    _init_rp = {}
    ep_prev_speed   = 0.0  # seed from actual first obs speed
    _step_speed_snap = 0.0   # v1.0.14: single snapshot, updated ONCE at bottom of step
    ep_prev_accel   = None   # v1.0.14: for jerk computation
    ep_decel_penalties = []
    ep_safe_speed_ratios = []
    ep_racing_line_errors = []
    ep_recovery_steps = 0
    ep_in_recovery = False
    ep_turn_entry_speeds = []
    ep_positions = []
    ep_first_offtrack_step = None
    ep_progress_hist = []
    ep_progress = 0.0
    ep_progress_pct = 0.0
    ep_centerline_progress_m = 0.0
    ep_track_length_m = 100.0
    ep_track_progress_pct = 0.0
    ep_start_time = time.time()  # v24: lap time tracking
    ep_reversed_count = 0
    ep_zero_speed_count = 0
    ep_context_preds = []
    ep_lidar_mins = []
    ep_barrier_proximities = []
    ep_nearest_objects = []
    ep_crash_ctx = None  # context at crash
    ep_crash_speed = None  # speed at crash
    ep_crash_heading = None  # v26
    ep_crash_closest_wp = None  # v26
    ep_crash_lidar_min = None  # lidar min at crash
    ep_corner_speeds = []
    ep_graze_count = 0
    # v29: crash-antecedent kinematics ring buffer
    _ANTE_WIN = 20
    ep_ante_buf = []
    ep_prev_steer = 0.0
    # v37 per-episode lists — must exist before hard-truncation block
    ep_ang_vel_centerline: list = []
    ep_jerk_abs: list = []
    ep_brake_before_barrier: list = []

    # Initialize env to get observation shape
    # Retry env.reset() - recreate env on failure (ZMQ socket state)
    for _retry in range(10):
        try:
            # v206: cross-platform timeout (SIGALRM not available on Windows)
            if hasattr(signal, 'SIGALRM'):
                def _timeout_handler(signum, frame):
                    raise TimeoutError('env.reset() timed out after 600s')
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(600)
                _init_obs, _init_info = env.reset()
                signal.alarm(0)
            else:
                import threading
                _reset_result = [None, None]
                _reset_exc = [None]
                def _reset_worker():
                    try:
                        _reset_result[0], _reset_result[1] = env.reset()
                    except Exception as e:
                        _reset_exc[0] = e
                t = threading.Thread(target=_reset_worker, daemon=True)
                t.start()
                t.join(timeout=600)
                if t.is_alive():
                    raise TimeoutError('env.reset() timed out after 600s (Windows)')
                if _reset_exc[0]:
                    raise _reset_exc[0]
                _init_obs, _init_info = _reset_result
            logger.info(f"env.reset() succeeded on attempt {_retry+1}")
            break
        except Exception as e:
            logger.warning(f"env.reset() attempt {_retry+1}/10 failed: {e}")
            try:
                env.close()
            except:
                pass
            time.sleep(30)
            env = _apply_phase_env(args, _phase)  # recreate env with fresh ZMQ socket
    else:
        raise RuntimeError("env.reset() failed after 10 retries")
    obs_shape = _init_obs.shape if hasattr(_init_obs, "shape") else (len(_init_obs),)
        # --- HTM Behavioural Cloning seed (safe: after first env.reset) ---
    try:
        from htm_reference import HTMPilotDriver
        _init_rp = _init_info.get("reward_params", {}) if isinstance(_init_info, dict) else {}
        _env_waypoints = _init_rp.get("waypoints", [])
        _total_track_arc = _arc_track_length(_env_waypoints)
        logger.info(f"[ARC] Track arc: {_total_track_arc:.2f}m ({len(_env_waypoints)} wps)")
        htm_pilot = HTMPilotDriver(
            waypoints=_env_waypoints,
            track_width=float(_init_rp.get("track_width", 0.6)),
            track_variant=_track_variant,
        )
        # v1.1.4: min_progress=2.0 — accept near-any motion; BCPilot tops out at 5-25%
        n_harvested = harvest_htm_pilots(env, htm_pilot, td3sac,
                                         n_episodes=12, min_progress=2.0)
        if n_harvested >= 5:  # v1.1.4: was 10 — BCPilot only hits 5-12 successful eps
            pretrain_td3_bc(td3sac, agent, bc_steps=2000)
    except ImportError:
        # BCPilot: internal fallback, no external dependency
        logger.warning("[HTM] htm_reference not found – falling back to BCPilot")
        _init_rp = _init_info.get("reward_params", {}) if isinstance(_init_info, dict) else {}
        _env_waypoints = _init_rp.get("waypoints", [])
        _total_track_arc = _arc_track_length(_env_waypoints)
        logger.info(f"[ARC] Track arc: {_total_track_arc:.2f}m ({len(_env_waypoints)} wps)")
        if len(_env_waypoints) >= 10:
            _bc_pilot = BCPilot(
                waypoints=_env_waypoints,
                track_width=float(_init_rp.get("track_width", 0.6)),
                track_variant=_track_variant,
            )
            n_harvested = harvest_htm_pilots(env, _bc_pilot, td3sac,
                                             n_episodes=12, min_progress=2.0)  # v1.1.4: lowered from 5.0
            if n_harvested >= 5:  # v1.1.4: was 10 — BCPilot only hits 5-12 successful eps
                pretrain_td3_bc(td3sac, agent, bc_steps=2000)
        else:
            logger.warning("[BC] waypoints empty at init — skipping BC harvest")
    except Exception as _htm_e:
        # v213: also fall back to BCPilot on runtime crash (OOB etc)
        logger.warning(f"[HTM] BC harvest runtime error: {_htm_e} – falling back to BCPilot")
        _init_rp = _init_info.get("reward_params", {}) if isinstance(_init_info, dict) else {}
        _env_waypoints = _init_rp.get("waypoints", [])
        if len(_env_waypoints) >= 10:
            _bc_pilot = BCPilot(
                waypoints=_env_waypoints,
                track_width=float(_init_rp.get("track_width", 0.6)),
                track_variant=_track_variant,
            )
            n_harvested = harvest_htm_pilots(env, _bc_pilot, td3sac,
                                             n_episodes=12, min_progress=2.0)  # v1.1.4: lowered from 5.0  # v1.1.0: was 50/80 — too strict, never seeds replay
            if n_harvested >= 5:  # v1.1.4: was 10 — BCPilot only hits 5-12 successful eps
                pretrain_td3_bc(td3sac, agent, bc_steps=2000)
        else:
            logger.warning("[BC] waypoints empty at init — skipping BC harvest")
    # rollout storage
    obs = zeros((num_steps,) + obs_shape)
    # v211: correct dtype+shape — discrete needs long, continuous needs float32
    if _is_discrete:
        actions = torch.zeros((num_steps,), dtype=torch.long).to(DEVICE)
    else:
        actions = torch.zeros((num_steps, _act_dim), dtype=torch.float32).to(DEVICE)

    logprobs = zeros((num_steps,))
    rewards = zeros((num_steps,))
    dones = zeros((num_steps,))
    values = zeros((num_steps,))
    _race_map = None  # built lazily on first step with waypoints
    context_labels = zeros((num_steps,), dtype=torch.long)  # track context class

    # start
    global_step = 0
    episode_count = 0
    observation, info = env.reset()
    _episode_progress_state = reset_episode_centerline_progress(info.get('reward_params', {}) 
                                                                if isinstance(info, dict) else {}, 
                                                                _track_progress_cache)
    next_obs = tensor(obs_to_array(observation))
    next_done = torch.zeros(1, device=DEVICE)
    best_return = float('-inf')
    _curvature = 0.0  # default for context label
    _reset_rp = info.get("reward_params", {}) if isinstance(info, dict) else {}
    ep_prev_speed = float(_reset_rp.get("speed", 0.0))
    _decel = 0.0; _speed_ratio = 0.0; _racing_line_err = 0.0
    ep_return = float('-inf')

    for update in range(1, num_updates + 1):
        # --- v211: phase advance ---
        _phase_steps_remaining -= num_steps
        if _phase_steps_remaining <= 0 and _current_phase_idx < len(_phase_schedule) - 1:
            _current_phase_idx += 1
            _phase = _phase_schedule[_current_phase_idx]
            env = _apply_phase_env(args, _phase, current_env=env)
            _phase_steps_remaining = _phase["timesteps"]
            _track_name    = _phase["track"]
            _track_variant = _phase["variant"]
            observation, info = env.reset()
            _episode_progress_state = reset_episode_centerline_progress(info.get('reward_params', {}) 
                                                                        if isinstance(info, dict) else {}, 
                                                                        _track_progress_cache)
            next_obs   = tensor(obs_to_array(observation))
            next_done  = torch.zeros(1, device=DEVICE)
        # --- v4: Get annealed hyperparams ---
        hp = scheduler.get_hyperparams(global_step)
        # v16: BSTS trend-aware annealing adjustments
        # v1.0.13: get_trend_vector() returns {metric: slope_float} over last 5 obs
        _bt = bsts_feedback.get_trend_vector(race_type_filter=_track_variant)
        _prog_slope  = float(_bt.get('progress', 0.0))
        _speed_slope = float(_bt.get('speed', 0.0))
        _rew_slope   = float(_bt.get('reward', 0.0))

        if _rew_slope < -0.005:                           # reward declining → explore
            hp['ent_coef'] = min(hp['ent_coef'] * 2.0, 0.08)
        elif _rew_slope < 0.0:                            # plateau
            hp['ent_coef'] = min(hp['ent_coef'] * 1.5, 0.05)
            hp['lr']       = min(hp['lr'] * 1.3, 5e-4)
        elif _rew_slope > 0.005 and _prog_slope > 0.0:   # strong improvement → exploit
            hp['ent_coef'] = max(hp['ent_coef'] * 0.8, 0.005)
        # --- v6: architecture annealing ---
        _worst_br = min((s.breakout_rate for s in stuck_tracker.stats.values() if s.total_episodes >= 3), default=0.5)
        hp['ent_coef'] = hp['ent_coef'] + 0.03 * (1.0 - _worst_br) ** 2
        hp['clip_coef'] = max(0.05, hp['clip_coef'] - 0.03 * (1.0 - _worst_br))
        arch = scheduler.get_architecture_params(global_step)
        rw = scheduler.get_reward_weights(global_step)
        if bootstrap_rewards.active(global_step, total_timesteps):
            rw = bootstrap_rewards.weights(rw)
        # v16: BSTS season-aware reward weight adjustment
        '''
        _bs = bsts_season.get_season() if hasattr(bsts_season, 'get_season') else {}
        _worst_segs = _bs.get('worst_segments', [])
        if _worst_segs:
            # Boost corner/braking weights if worst segments have high crash rates
            for _seg_id in _worst_segs[:3]:
                _seg_info = _bs.get('segments', {}).get(_seg_id, {})
                if _seg_info.get('crashes', 0) > 2:
                    rw['braking'] = rw.get('braking', 0.08) * 1.3
                    rw['turn_align'] = rw.get('turn_align', 0.06) * 1.2    
            # Renormalize weights
            _rw_total = sum(rw.values())
            if _rw_total > 0:
                rw = {k: v / _rw_total for k, v in rw.items()}                        
        '''
        # v1.0.13: worst-segment crash boosting now driven by BSTSFeedback.adjust_weights()
        # which already runs below. No separate season query needed.
        # The crash_rate/corner_crash_rate EMA paths in adjust_weights() handle this.

        # Apply annealed learning rate
        for pg in optimizer.param_groups:
            pg['lr'] = hp['lr']

        # Apply dropout annealing to agent if supported
        for module in agent.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = arch['dropout']

        writer.add_scalar('charts/learning_rate', hp['lr'], global_step)
        writer.add_scalar('annealing/ent_coef', hp['ent_coef'], global_step)
        writer.add_scalar('annealing/clip_coef', hp['clip_coef'], global_step)
        writer.add_scalar('annealing/gae_lambda', hp['gae_lambda'], global_step)
        writer.add_scalar('annealing/dropout', arch['dropout'], global_step)
        for rk, rv in rw.items():
            writer.add_scalar(f'annealing/rw_{rk}', rv, global_step)

        # ---- collect rollout ----
        rp = {}  # Fix: init rp before step loop to avoid UnboundLocalError
        _cwps = [0, 1]  # v27: init to avoid UnboundLocalError
        for step in range(num_steps):
            _raw_action = None
            _step_action = None
            global_step += 1
            # Safe defaults
            _approaching_stuck = False
            _stuck_bonus = 0.0
            _stuck_wp = -1
            _lidar_min = 1.0
            _nearest_obj = 999.0
            _objects_dist = []
            _waypoints = []
            _closest = [0, 1]
            _heading = 0.0
            _speed = 0.0
            _prog = 0.0
            _offtrack = False
            _ddiff = 0.0
            _brake_r = 0.0
            _turn_align_r = 0.0
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value, ctx_logits, _intermed_pred = agent.get_action_and_value(
                    next_obs.unsqueeze(0)
                )
            # v212 FIX: discrete agent returns shape [1] or [26]; must reduce to scalar for storage
            if _is_discrete:
                _action_idx = action.argmax(-1).squeeze() if action.ndim > 0 and action.shape[-1] > 1 else action.squeeze()
                actions[step] = _action_idx.long()
            else:
                actions[step] = action.squeeze(0)  # remove batch dim for continuous [1, act_dim] -> [act_dim]
            logprobs[step] = logprob
            values[step] = value


            # v40: FIX action rescaling -- env expects throttle in [0,1] but tanh outputs [-1,1]
            # v40.4: proper action post-processing
            #   (1) squeeze batch dim so shape is (action_dim,)
            #   (2) detect continuous vs discrete via action_space (runtime, not cached _act_dim)
            #   (3) remap throttle [-1,1] -> [0,1] when act_dim==2 (steering, throttle)
            #   (4) clip to action_space.low/high

            # --- v213 FIX: use process_action for ALL action dispatch (discrete + continuous) ---
            raw_action = action.cpu().numpy()          # ← must be BEFORE process_action call
            _step_action = process_action(raw_action, env.action_space)
            # v40.2: removed duplicate env.step that used un-remapped action (caused 100% stuck episodes)
            # v40.3: comprehensive action dispatch telemetry for first 20 steps of each episode
            try:
                _ep_step = int(ep_step_count) if 'ep_step_count' in dir() else global_step
            except Exception:
                _ep_step = global_step
            # v1.0.13: telemetry only for first 3 steps of first 3 episodes
            if episode_count < 3 and ep_step_count < 3:
                try:
                    as_ = env.action_space
                    as_info = f"Box(shape={as_.shape})" if hasattr(as_, 'low') else f"Discrete(n={getattr(as_,'n','?')})"
                    logger.debug(f"TELEMETRY gs={global_step} ep={episode_count} step={ep_step_count} "
                                f"act_space={as_info} raw={type(raw_action).__name__}:{raw_action!r}")
                except Exception as e:
                    logger.warning(f"TELEMETRY fail {e}")
            observation, reward, terminated, truncated, info = env.step(_step_action)
            # v1.0.13: continuous arc-progress update — runs every step, not just episode end
            _rp_now = info.get("reward_params", {}) if isinstance(info, dict) else {}
            ep_centerline_progress_m, ep_track_length_m, ep_progress_pct, _prog_delta, _episode_progress_state = \
                update_episode_centerline_progress(_rp_now, _track_progress_cache, _episode_progress_state)
            ep_progress = ep_centerline_progress_m
            # logger.info(f"[TRUNCATE] ep_step={ep_step_count} >= 500, forcing truncation")
            # Debug: log reward_params for first 3 steps
            if global_step <= 3:
                logger.info(f"[DEBUG] step={global_step} rp={info.get('reward_params', {})} ep_status={info.get('episode_status', None)}")
            if global_step < 10 or global_step % 100 == 0: logger.info(f"step={global_step} reward={reward:.4f} terminated={terminated} truncated={truncated} info_keys={list(info.keys()) if info else None}")
            # --- research: Line-of-Sight reward (Garlick & Middleditch, 2022) ---
            _is_rev = False
            if rp and info and isinstance(info, dict):
                _v_perp_barrier = 0.0  # v39c: safe default before v_perp computed
                _v_perp_safe = 999.0  # v39c: safe default (always passes guard)
                _wps = rp.get("waypoints", [])
                _cwps = rp.get("closest_waypoints", [0, 1])
                _x = rp.get("x", 0.0)
                _y = rp.get("y", 0.0)
                _hdg = rp.get("heading", 0.0)
                _spd = rp.get("speed", 0.0)
                # v213: Phase -1 alive bonus — stay on track is enough
                _t_frac = global_step / max(total_timesteps, 1)
                if _t_frac < 0.05 and not _offtrack and not _is_stuck:
                    reward += 0.03
                if _wps and len(_wps) > 1:
                    _los_r = _los.compute(_x, _y, _hdg, _wps, _cwps[0])
                    # v39: Un-muted LOS with brake-line guard
                    if _v_perp_barrier <= _v_perp_safe:
                        reward += float(_los_r) * 0.15
                # REF: BSTS overtake tracking (AWS, 2020)
                # REF: Yang et al. (2023b) overtake reward for safe passing
                if _overtake is not None:
                    _bot_prog = rp.get("progress", 0.0)
                    _ot_r = _overtake.compute(_bot_prog, 0.0, min(_lidar) if _lidar else 1.0)
                    # v39: Un-muted overtake with brake-line guard
                    if _v_perp_barrier <= _v_perp_safe:
                        reward += 0.10 * float(_ot_r)
                    # REF: Corner analyzer curvature-based speed reward (Coulom, 2002; Yang et al., 2023)
                    _curv = curvature_radius(_wps, _cwps[0])
                    _opt_spd = optimal_speed(_curv)
                    _spd_err = abs(_spd - _opt_spd)
                    # v39: Un-muted curv speed with brake-line guard
                    if _v_perp_barrier <= _v_perp_safe:
                        reward += max(0.0, 0.15 * (1.0 - _spd_err / max(_opt_spd, 0.1)))
                # REF: Yang et al. (2023a) corner classification reward
                _corner_cls_r = _corners.corner_reward(_spd, _curv)
                # v39: Un-muted corner_cls_r with brake-line guard (was MUTED v25)
                # REF: Tian et al. (2024) balanced reward for safer cornering
                # REF: Ng, Harada & Russell (1999) potential-based shaping
                if _v_perp_barrier <= _v_perp_safe:
                    reward += 0.1 * _corner_cls_r

            # --- v4: Apply annealed reward weights ---
            _decel_r = 0.0; _speed_steer_r = 0.0; _min_speed_r = 0.0; _racing_line_r = 0.0
            _center_r = 0.0; _speed_r = 0.0; _steer_r = 0.0
            _prog_r = 0.0; _eff_r = 0.0; _head_r = 0.0; _comp_r = 0.0
            rp = info.get("reward_params", {})
            # v23: diagnostic
            if ep_step_count == 1 and global_step < 600:
                logger.info(f"[REWARD_DIAG] rp keys={list(rp.keys())}, rp={rp}")
            # v23: base progress reward
            _prog_raw = float(rp.get("progress", 0.0) or 0.0)
            _center_m, _total_m, _center_pct, _center_delta_m, _episode_progress_state = \
                update_episode_centerline_progress(rp, _track_progress_cache, _episode_progress_state)
            _prog = _center_pct
            ep_centerline_progress_m = max(ep_centerline_progress_m, _center_m)
            ep_track_length_m = max(ep_track_length_m, _total_m)
            ep_track_progress_pct = max(ep_track_progress_pct, _center_pct)
            _offtrack = rp.get("is_offtrack", False)
            _is_stuck = rp.get("is_stuck", False) if "is_stuck" in rp else False
            _speed = rp.get("speed", 0.0)
            _steer = abs(rp.get("steering_angle", 0))
            _delta_prog = _prog - _prev_prog_tracker
            _bootstrap_active = bootstrap_rewards.active(global_step, total_timesteps)
            if _delta_prog > 0:
                # v1.1.0: amplified bootstrap progress signal (was 3.0 → 8.0)
                 reward += _delta_prog * (8.0 if _bootstrap_active else 1.5)
            elif _bootstrap_active and ep_step_count > 3:
                reward = reward * 0.88  # v1.1.0: no-progress ATTENUATION during bootstrap (12% cut per step, no additive penalty)
            if _bootstrap_active and not _offtrack and not _is_stuck:
                reward += 0.05  # v1.1.0: stronger alive bonus during bootstrap (was 0.02)
                # v1.1.0: heading-to-waypoint bonus: fire for first 8 steps of episode
                if ep_step_count <= 8 and "_head_r" in dir() and _head_r > 0.5:
                    reward += 0.20 * _head_r  # strong initial heading alignment incentive
            _prev_prog_tracker = _prog
            # v23: alive bonus
            # if not _offtrack and not _is_stuck:  # MUTED v25
            # reward += 0.01  # MUTED v25
            # v25: MUTED rigid speed incentive, replaced by racing-line-only
            # v39: Re-enabled speed reward (was MUTED v25), now v_perp-aware
            if _speed > 0.1: reward += min(_speed, 4.0) * 0.05  # v39: halved from 0.1
            ep_step_count += 1  # v28: FIX missing increment
            
            _brake_field = BrakeField()
            if rp:
                _speed = rp.get("speed", 0)
                # v4-bsts: barrier proximity from LIDAR and objects
                _lidar = rp.get('lidar', [])
                _lidar_min = min(_lidar) if _lidar else 1.0
                _objects_dist = rp.get('objects_distance', [])
                _nearest_obj = min(_objects_dist) if _objects_dist else 999.0
                _track_width = rp.get('track_width', 0.6)
                _dist_from_center = rp.get('distance_from_center', 0.0)
                _barrier_proximity = max(0, 1.0 - (_lidar_min / max(_track_width, 0.1)))
                _dist = rp.get("distance_from_center", 0)
                _tw = rp.get("track_width", 1)
                _steer = abs(rp.get("steering_angle", 0))
                _prog = rp.get("progress", 0)
                _steps = rp.get("steps", 1)
                _offtrack = rp.get("is_offtrack", False)
                _aot = rp.get("all_wheels_on_track", True)
                _heading = rp.get("heading", 0)
                _waypoints = rp.get("waypoints", [])
                # v16: Lazy-init BrakeField alongside race engine
                if _waypoints and race_engine is None:
                    race_engine = MultiRaceLineEngine(_waypoints)
                    logger.info(f"[V16] MultiRaceLineEngine initialized with {len(_waypoints)} waypoints")
                    _brake_field.set_waypoints(np.array(_waypoints)) # v222
                    logger.info(f"[V41] BrakeField waypoints set: {len(_waypoints)} wps")

                # === Build racing line map lazily on first step ===
                # v39: Force track discovery before meaningful training
                if _race_map is None and ep_step_count < 3 and _waypoints:
                    reward = 0.01  # minimal reward until track layout known
                if _race_map is None and _waypoints and len(_waypoints) > 5:
                    _race_map = build_racing_line_map(_waypoints, _tw, v_max=4.0)
                    logger.info(f'[RACE_MAP] Built racing line map: {len(_race_map)} waypoints')
                _closest = rp.get("closest_waypoints", [0, 1])

                # --- v5: Track geometry ---
                _curvature, _safe_speed = compute_track_curvature(_waypoints, _closest)
                # --- v6: lookahead curvature scan ---
                _max_curv_ahead, _max_curv_wp, _safe_speed_ahead, _dist_to_corner = \
                    lookahead_curvature_scan(_waypoints, _closest, max_lookahead=15)
                _brake_r = compute_braking_reward(_speed, _safe_speed_ahead, _dist_to_corner)
                _turn_align_r = compute_turn_alignment_reward(_heading, _waypoints, _closest)
                _approaching_stuck, _stuck_bonus, _stuck_wp = \
                    get_stuck_antecedent_bonus(stuck_tracker, _waypoints, _closest)
                _speed_ratio = _speed / max(_safe_speed, 0.5)
                _racing_line_offset = compute_racing_line_offset(_waypoints, _closest, _tw)
                _optimal_dist = _racing_line_offset * (_tw / 2.0)
                _actual_lateral = _dist_from_center * (1 if rp.get("is_left_of_center", False) else -1)
                _racing_line_err = abs(_actual_lateral - _optimal_dist) / max(_tw / 2.0, 0.1)
                _decel = ep_prev_speed - _speed if ep_prev_speed > 0 else 0.0
                # compute sub-rewards
                _center_pct = _dist / (0.5 * _tw) if _tw > 0 else 0
                _center_r = max(0, 1.0 - _center_pct)
                # V12: REF: Curvature-aware Gaussian speed reward (Gonzalez2020)
                _curv_abs = max(abs(_curvature), 1e-6)
                _turn_radius = 1.0 / _curv_abs if _curvature != 0 else 100.0
                _v_optimal = min(4.0, max(0.5, 1.2 * (_turn_radius ** 0.25)))  # v20 REF:Gonzalez2020
                _speed_delta = _speed - _v_optimal
                _speed_r = 5.0 * math.exp(-0.5 * (_speed_delta / 0.8) ** 2)
                if _speed >= 0.8 * _v_optimal: _speed_r += 1.0  # anti-creep bonus
                _steer_r = max(0, 1.0 - _steer / 30.0)
                _prog_r = (_prog / 100.0) * 10.0
                _eff = _prog / _steps if _steps > 0 else 0
                _eff_r = min(_eff * 10.0, 1.0)

                # heading reward
                if _waypoints and len(_closest) >= 2:
                    _nwp = _waypoints[_closest[1]]
                    _pwp = _waypoints[_closest[0]]
                    _tdir = math.degrees(math.atan2(_nwp[1]-_pwp[1], _nwp[0]-_pwp[0]))
                    _ddiff = abs(_tdir - _heading)
                    if _ddiff > 180:
                        _ddiff = 360 - _ddiff
                    _head_r = max(0, 1.0 - _ddiff / 30.0)
                else:
                    _head_r = 0.0

                # v5: deceleration smoothness (always computed)
                # v1.1.0: jerk-informed brake intent (replaces flat _decel_r which punished hard braking)
                # REF: Balaban et al. (2018) Jerk as indicator of driving intent. Vehicle System Dynamics.
                # REF: Brayshaw & Harrison (2005) Quasi-steady state lap sim. Proc. IMechE Part D.
                _jerk_now = abs((_speed - _step_speed_snap) / 0.1 - (ep_prev_accel or 0.0)) / 0.1   # da/dt in m/s³ (dt≈0.1s)

                # Is the agent in a brake approach zone?
                _in_brake_approach = (
                    '_brake_potential' in dir() and _brake_potential > 0.3   # inside brake field
                    and '_dist_from_center' in dir() and _dist_from_center < 0.4 * max(_tw, 0.1)  # on track
                )

                if _in_brake_approach and _accel < -1.0:
                    # Intentional braking in brake zone: reward sharp onset (jerk onset = intentional)
                    # Capped at jerk=50 m/s³ (physical max for 1/18-scale at 4 m/s)
                    # REF: Balaban 2018 — jerk characterises intentionality of braking input
                    _decel_r = min(_jerk_now / 50.0, 1.0) * 0.8 + 0.2   # floor 0.2, ceiling 1.0
                elif _in_brake_approach and _accel >= 0:
                    # Throttle in brake zone — attenuate (multiplicative, NOT subtractive)
                    _decel_r = 0.4   # 60% of full brake-zone weight
                else:
                    # Not in brake zone — smooth driving preferred; excess random jerk attenuated
                    # Multiplicative: frozen agent near 0 gets no cut; moving agent with jerk gets cut
                    _decel_r = 1.0 / (1.0 + max(0.0, _jerk_now - 20.0) * 0.02)
                # V13: Speed-steering harmony (REF: Gonzalez2020)
                _steer_angle = abs(rp.get("steering_angle", 0))
                # Reward low steering at high speed (smooth driving)
                if _speed > 2.0 and _steer_angle < 10:
                    _speed_steer_r = 2.0  # smooth high-speed driving
                elif _speed > 1.5 and _steer_angle < 15:
                    _speed_steer_r = 1.5
                elif _steer_angle < 20:
                    _speed_steer_r = 1.0
                else:
                    _speed_steer_r = 0.3  # still positive, just less
                # V13: Speed maintenance reward (positive only)
                if _speed >= 2.5:
                    _min_speed_r = 3.0 * (_speed / 4.0)  # scale up with speed
                elif _speed >= 1.5:
                    _min_speed_r = 1.0 + (_speed - 1.5)
                elif _speed >= 0.5:
                    _min_speed_r = 0.3
                else:
                    _min_speed_r = 0.1  # still positive, just minimal
                _racing_line_r = max(0, 1.0 - _racing_line_err)
                # completion bonus
                _comp_r = (100.0 + 50.0 * min(1.0, max(0, (sum(ep_speeds)/max(len(ep_speeds),1)) - 1.0) / 3.0)) if _prog >= 100.0 else (_prog / 100.0) * 5.0 * (1.0 + 0.5 * min(1.0, _speed / 3.0))

                # === v13: Racing Line Proximity Reward (replaces penalty-based system) ===
                if _race_map is not None and _waypoints:
                    _wp_idx = _closest[0] if len(_closest) > 0 else 0
                    _rl_reward = racing_line_reward(
                        _race_map, _wp_idx, _speed, _dist_from_center,
                        _heading, _tw, rp.get('is_left_of_center', False),
                        _waypoints, _closest)
                    # Progress reward: pure positive, scales with completion
                    _prog_r = (_prog / 100.0) * 10.0
                    # Completion bonus: massive reward for finishing
                    _comp_r = (100.0 + 50.0 * min(1.0, max(0, (sum(ep_speeds)/max(len(ep_speeds),1)) - 1.0) / 3.0)) if _prog >= 100.0 else (_prog / 100.0) * 5.0 * (1.0 + 0.5 * min(1.0, _speed / 3.0))
                    # v20: Full sub-reward integration with _rl_blend annealing
                    # REF: Scott & Varian (2014) BSTS informs blend schedule
                    # REF: Gonzalez2020 curvature-aware speed reward integrated
                    _env_signal = (
                        rw.get('curv_speed',0.12) * _speed_r      +
                        rw.get('heading',   0.10) * _head_r       +
                        rw.get('braking',   0.08) * _decel_r      +
                        rw.get('min_speed', 0.10) * _min_speed_r  +
                        rw.get('corner',    0.06) * _speed_steer_r+
                        rw.get('racing_line',0.10)* _racing_line_r+
                        rw.get('progress',  0.15) * _prog_r       +
                        0.05 * _comp_r
                    )
                    shaped_reward = (1.0-_rl_blend)*_env_signal + _rl_blend*_rl_reward
                else:
                    # Fallback: use env-signal without race map
                    _env_signal = (
                        rw.get('curv_speed',0.12) * _speed_r      +
                        rw.get('heading',   0.10) * _head_r       +
                        rw.get('braking',   0.08) * _decel_r      +
                        rw.get('min_speed', 0.10) * _min_speed_r  +
                        rw.get('corner',    0.06) * _speed_steer_r+
                        rw.get('racing_line',0.10)* _racing_line_r+
                        rw.get('progress',  0.15) * _prog_r       +
                        0.05 * _comp_r
                    )
                    shaped_reward = _env_signal  # v39: FIX use env_signal when no race_map (was _rl_reward which may be undefined)
                    
                                        # v40: Speed gate - harsh penalty for staying still
                    # v221: brake-field aware speed gate
                    # Speed reward only fires at full strength when:
                    #   (a) not in brake field, OR (b) already braking correctly
                    # v222: Coherent speed gate via actual BrakeField.step() API
                    # So here we derive it from the same source directly:
                    _is_braking_now = bool(_speed < _step_speed_snap - 0.05)

                    try:
                        _bf_step         = _brake_field.step(
                                            wp_idx=_closest[0] if _closest else 0,
                                            speed=_speed,
                                            is_braking=_is_braking_now
                                        )
                        _in_brake_field  = _bf_step['in_brake_field']   # True if BrakeField.potential > 0
                        _brake_potential = _bf_step['brake_potential']   # Phi in [0, 1]
                        _bf_ok           = (not _in_brake_field) or _is_braking_now
                    except Exception:
                        _in_brake_field  = False
                        _brake_potential = 0.0
                        _bf_ok           = True

                    _position_compliance = max(0.0, 1.0 - (_dist_from_center / max(_track_width * 0.5, 0.1)))
                    _rl_compliance       = max(0.0, 1.0 - _racing_line_err)

                    _speed_gate = (
                        max(0.1, min(1.0, (_speed - 0.3) / 1.2))   # base: speed > 0.3 to get full gate
                        * (0.4 + 0.6 * _position_compliance)         # off-centerline: floor at 40%
                        * (0.5 + 0.5 * _rl_compliance)               # off-raceline: floor at 50%
                        * (1.0 if _bf_ok else max(0.3, 1.0 - _brake_potential))  # brake-zone: scales with how deep in field
                    )
                    # v1.1.0: removed flat -0.02 subtractive drag (caused freeze trap on near-zero reward agents)
                    # Instead: pure multiplicative speed gate — low speed_gate scales reward toward 0, never below
                    shaped_reward = shaped_reward * _speed_gate  # attenuation only, no additive penalty
                # Blend: mostly racing-line shaped, small env signal
                # v18: BSTS-driven alpha mixing (race-line compliance vs env signal)
                # v1.1.0: during bootstrap, reduce BSTS alpha so raw progress signal dominates
                # shaped_reward speed_gate fires near-zero in early chaos → don't let it drown out progress delta
                _eff_alpha = 0.30 if _bootstrap_active else _bsts_alpha  # 30% shaped vs 70% raw during bootstrap
                reward = (1.0 - _eff_alpha) * reward + _eff_alpha * shaped_reward  # v32: unclipped
                # v18: wire stuck-antecedent bonus into reward
                # v39: Un-muted _stuck_bonus (was MUTED v25)
                if _approaching_stuck and _stuck_bonus > 0:
                     # v1.1.0: use small stuck_bonus during bootstrap (10% strength) so agent DOES get escape signal
                     _sb_frac = 0.1 if _bootstrap_active else 1.0
                     reward += min(float(_stuck_bonus) * _sb_frac, 0.5 if not _bootstrap_active else 0.05)
                if _speed < 1.0 and _approaching_stuck:
                    reward += 0.5  # extra bonus for cautious approach near stuck zones
                # v19: SAC exploration bonus
                # ICM/SAC curiosity: only reward novelty that comes WITH forward progress
                try:
                    _sac_ex = td3sac.exploration_bonus(None, None, log_prob=None)
                    # Gate: curiosity bonus only fires if we made progress this step
                    _progress_gate = max(0.0, min(1.0, _d_prog * 20.0))  # 0→1 over 0.05 progress units
                    reward += float(_sac_ex) * 0.05 * _progress_gate
                except Exception:
                    pass
                # --- v6: adaptive reward + stuck tracking ---
                _cur_wp = _closest[0] if len(_closest) > 0 else 0
                anneal = stuck_tracker.get_annealing_params(_cur_wp)
                _moved_fwd = _prog > stuck_tracker._prev_progress
                # --- v38: incompatible-behavior reward (speed IS the reward floor) ---
                _is_rev = rp.get('is_reversed', False)
                try:
                    _d_prog = float(_prog) - float(stuck_tracker._prev_progress)
                    _spd = float(_speed) if _speed is not None else 0.0
                    # 1) Forward velocity bonus: speed * progress_delta (moving + progressing = best)
                    reward += 3.0 * _d_prog * max(_spd, 0.1)
                    # 2) Pure speed floor: ANY forward motion > standing still, always
                    #    This is the key anti-creep: reward = f(speed), so 0 speed = minimum reward
                    reward += 0.08 * min(_spd, 4.0)  # caps at 3 m/s to avoid pure speed hacking
                    # 3) Standing still is STRICTLY worst: explicit per-step cost for near-zero speed
                    #    Agent pays -0.08/step for creeping. At 200 steps/ep thats -16.0 ep_return.
                    #    But crashing at speed at least earns the speed bonus before dying.
                    if _spd < 0.15:
                        reward = reward * 0.70  # v1.1.0: creep ATTENUATION (30% cut, not subtractive — avoids freeze trap)
                    # 4) Raceline compliance bonus when moving (reward the incompatible behavior)
                    if _spd > 0.3 and ep_dist_from_center:
                        _ctr = abs(float(ep_dist_from_center[-1]))
                        reward += 0.25 * max(0.0, 1.0 - _ctr / 0.5)  # on-line + moving = bonus  # v39: boosted from 0.06
                    # 5) Reverse is attenuated but NOT additively penalized (v1.1.0: multiplicative)
                    if _is_rev:
                        reward = reward * 0.80  # 20% cut for reversing; frozen agent near 0 gets ~0 cut
                except Exception:
                    pass

                # v39: offtrack grace period - don't count as stuck for first 10 offtrack steps
                if _offtrack:
                    ep_offtrack_steps += 1
                else:
                    ep_offtrack_steps = max(0, ep_offtrack_steps - 1)  # recover
                _offtrack_stuck = _offtrack and ep_offtrack_steps > 10  # v39: grace period
                # v1.1.0: v_perp offtrack ATTENUATION — multiplicative, not additive
                # Division avoids freeze-to-avoid-penalty trap (frozen agent near 0 reward gets 0 cut)
                # At v_perp=0: factor=1.0. At v_perp=3.6: factor≈0.41 (59% reward cut).
                if _offtrack:
                    _vperp_val = float(_v_perp_barrier) if "_v_perp_barrier" in dir() else 1.0
                    _offtrack_attn = 1.0 / (1.0 + max(0.0, _vperp_val) * 0.40)
                    reward = reward * _offtrack_attn  # never subtracts; just scales earned reward down
                _is_stuck = (_speed < 0.3) or _offtrack_stuck or _is_rev  # v39: uses grace
                if anneal['reward_boost'] > 1.0:
    #                     # reward += (anneal['reward_boost'] - 1.0) * 0.2 * _prog_r  # MUTED v25
                    pass  # v25 muted
                stuck_tracker.step_update(
                    wp_idx=_cur_wp, is_stuck=_is_stuck,
                    moved_forward=_moved_fwd, step_reward=reward,
                    speed=_speed, crashed=terminated,
                    reversed_flag=_is_rev, offtrack=_offtrack)
                stuck_tracker._prev_progress = _prog
                if _is_stuck and not terminated and not truncated:
                    _thresh = stuck_tracker.get_early_term_threshold(_cur_wp)
                    if ep_step_count > _thresh:
                        truncated = True

            # Hard max-step truncation fallback
            if ep_step_count > 500 and not terminated and not truncated:
                truncated = True
                logger.info(f"[HARD_TRUNC] ep_step={ep_step_count} > 500")
                ep_rewards_components["center"].append(_center_r)
                ep_rewards_components["speed"].append(_speed_r)
                ep_rewards_components["steering"].append(_steer_r)
                ep_rewards_components["progress"].append(_prog_r)
                ep_rewards_components["efficiency"].append(_eff_r)
                ep_rewards_components["heading"].append(_head_r)
                ep_rewards_components["braking"].append(_brake_r)
                ep_rewards_components["turn_align"].append(_turn_align_r)
                ep_speeds.append(_speed)
                ep_headings.append(_heading)  # v26
                ep_closest_wps.append(_cwps)  # v26
                ep_dist_from_center.append(_dist)
                _accel = (_speed - _step_speed_snap) / 0.1
                try:  # v37 per-step
                    if ep_heading_diffs:
                        ep_ang_vel_centerline.append(abs(float(ep_heading_diffs[-1])))
                except Exception: pass
                try:
                    if ep_prev_accel is not None:
                        _jerk = abs(_accel - ep_prev_accel) / 0.1   # true da/dt in m/s³
                        ep_jerk_abs.append(_jerk)
                    ep_prev_accel = _accel   # update AFTER use
                except Exception:
                    pass
                try:  # v37 brake compliance
                    if ep_barrier_proximities and float(ep_barrier_proximities[-1]) < 1.0 and ep_prev_speed is not None:
                        _sp_now2 = float(info.get("speed", ep_speeds[-1] if ep_speeds else 0.0))
                        _dec = float(ep_prev_speed) - _sp_now2
                        if _dec > 0: ep_brake_before_barrier.append(_dec)
                except Exception: pass
                # v3-bsts per-step tracking
                ep_heading_diffs.append(_ddiff if "_ddiff" in dir() else 0.0)
                ep_steerings_raw.append(_steer)
                # Collect step dict for extract_intermediary_metrics
                try:
                    # v1.1.0 FIX: x/y/heading/speed live in rp (reward_params), NOT top-level info
                    # top-level info keys are: reward_params, episode_status, goal
                    # Using _step_info.get('x') was ALWAYS 0 — causing all BSTS zeros.
                    _rp_log = rp if rp else (info.get("reward_params", {}) if isinstance(info, dict) else {})
                    # v1.1.0: record_step for BSTSSeasonal live buffer
                    try:
                        if bsts_season is not None and hasattr(bsts_season, 'record_step'):
                            _rl_err_step = float(_racing_line_err) if '_racing_line_err' in dir() else 0.0
                            bsts_season.record_step(
                                progress=float(_prog) if '_prog' in dir() else 0.0,
                                speed=float(_speed) if '_speed' in dir() else 0.0,
                                steering=float(action[0]) if hasattr(action, '__getitem__') else 0.0,
                                heading_err=float(_heading) * 3.14159 / 180.0 if '_heading' in dir() else 0.0,
                                raceline_err=_rl_err_step,
                                reward=float(reward),
                                lidar_min=float(_lidar_min_step) if '_lidar_min_step' in dir() else 1.0,
                                wp_idx=int(_closest[0]) if isinstance(_closest, (list,tuple)) and _closest else 0,
                            )
                    except Exception:
                        pass
                    _ep_step_log.append({
                        'x':                    float(_rp_log.get('x', 0.0)),
                        'y':                    float(_rp_log.get('y', 0.0)),
                        'heading':              float(_rp_log.get('heading', 0.0)),
                        'speed':                float(_speed),   # already resolved above
                        'steering_angle':       float(action[0]) if hasattr(action, '__getitem__') else 0.0,
                        'throttle':             float(action[1]) if hasattr(action, '__getitem__') and len(action) > 1 else 0.0,
                        'reward':               float(reward),
                        'all_wheels_on_track':  bool(_rp_log.get('all_wheels_on_track', True)),
                        'distance_from_center': float(_dist_from_center) if '_dist_from_center' in dir() else float(_rp_log.get('distance_from_center', 0.0)),
                        'closest_waypoint':     int(_closest[0]) if isinstance(_closest, (list, tuple)) and len(_closest) > 0 else -1,  # v1.1.0: fix: was 'closest_wp_idx'[1]; hm needs 'closest_waypoint'[0]
                        'dist_to_raceline':   float(_racing_line_err) if '_racing_line_err' in dir() else 0.0,  # v1.1.0: was missing → race_line_adherence=0
                        # v1.1.0: fields _translate_step() / extract_intermediary_metrics() need
                        'braking':              int(_braking_intent) if '_braking_intent' in dir() else 0,
                        'is_offtrack':          bool(_offtrack),
                        'progress':             float(_prog),
                        'heading_diff':         float(ep_heading_diffs[-1]) if ep_heading_diffs else 0.0,
                        'safe_speed_ratio':     float(_speed_ratio) if '_speed_ratio' in dir() else 1.0,
                        'racing_line_offset':   float(_racing_line_err) if '_racing_line_err' in dir() else 0.0,
                        'in_corner':            bool(_rp_log.get('is_turn', False)),
                        'track_width':          float(_tw) if '_tw' in dir() else float(_rp_log.get('track_width', 1.0)),
                        # v1.1.0: jerk for brake_intent scoring
                        'accel':                float(_accel) if '_accel' in dir() else 0.0,
                        'v_perp':               float(_v_perp_barrier) if '_v_perp_barrier' in dir() else 0.0,
                        'in_brake_field':       bool(_in_brake_field) if '_in_brake_field' in dir() else False,
                        'brake_potential':      float(_brake_potential) if '_brake_potential' in dir() else 0.0,
                    })
                except Exception:
                    pass
                ep_positions.append((rp.get("x", 0.0), rp.get("y", 0.0)))
                ep_progress_hist.append(_prog)
                if (_offtrack or not _aot) and ep_first_offtrack_step is None:
                    ep_first_offtrack_step = ep_step_count
                if rp.get("is_reversed", False):
                    ep_reversed_count += 1
                if _speed < 0.01:
                    ep_zero_speed_count += 1
                if _offtrack or not _aot:
                    ep_offtrack_count += 1

            
            # v13: On-track bonus (no penalties, only rewards)
            rp_v7 = info.get("reward_params", {})
            # v7: Speed bonus for staying on track
            if rp_v7.get("speed", 0) > 2.0:
                reward += 2.0 * min(rp_v7.get("speed", 0), 4.0)
            cumulative_ep_reward += reward
            # v4: context-aware step tracking
            _ctx = int(agent.get_context(tensor(observation).unsqueeze(0))[0])
            ep_context_preds.append(_ctx)
            ep_lidar_mins.append(_lidar_min)
            ep_barrier_proximities.append(_barrier_proximity)
            # v29: crash-antecedent kinematics
            _accel = (_speed - _step_speed_snap) / 0.1  # m/s^2 (dt~0.1s)
            ep_prev_speed = _speed
            _af = action.flatten() if hasattr(action, 'flatten') else (np.array(action).flatten() if hasattr(action, '__iter__') else [action])
            _act_steer = float(_af[0].item() if hasattr(_af[0], 'item') else float(_af[0]))
            _act_throttle = float(_af[1].item() if hasattr(_af[1], "item") else float(_af[1])) if len(_af) > 1 else 0.0  # v32: throttle/brake command, negative=brake
            _steer_rate = (abs(_act_steer) - ep_prev_steer) / 0.1
            ep_prev_steer = abs(_act_steer)
            # v_perp to barrier: component of velocity perpendicular to track tangent
            _v_perp_barrier = _compute_crash_v_perp(_speed, _heading, _closest, _waypoints) if _waypoints else 0.0
            _v_tang_barrier = _compute_crash_v_tang(_speed, _heading, _closest, _waypoints) if _waypoints else 0.0
            _dist_barrier = _lidar_min * max(_track_width, 0.1)  # approx metres to wall
            # stopping distance: d = v_perp^2 / (2*a_max) where a_max~3.0 m/s^2
            _A_MAX_BRAKE = 3.0
            _stop_dist = (_v_perp_barrier**2) / (2*_A_MAX_BRAKE + 1e-8)
            _braking_intent = 1 if (_act_throttle < -0.1 or _accel < -0.5) else 0  # v32: agent brake OR physical decel  # did agent attempt to brake?
            # v29: STOPPING-DISTANCE CALCULUS REWARD
            # Reward agent for braking when stop_dist >= dist_to_barrier
            # This teaches the agent to anticipate braking distance
            if _dist_barrier < _stop_dist * 1.5 and _dist_barrier > 0.01:
                # We need to brake! Reward deceleration, penalize acceleration
                if _accel < -0.3:  # actively braking
                    reward += 2.0 * min(abs(_accel)/3.0, 1.0)  # proportional to brake force
                elif _accel < 0:  # coasting/light brake
                    reward += 0.5
                # Bonus for reducing v_perp toward 0
                if len(ep_ante_buf) >= 2:
                    _prev_vp = ep_ante_buf[-1].get('v_perp', _v_perp_barrier)
                    if _v_perp_barrier < _prev_vp:  # v_perp decreasing
                        reward += 1.5 * (_prev_vp - _v_perp_barrier)
            elif _dist_barrier > _stop_dist * 2.0 and _v_perp_barrier < 0.3:
                # Safe zone with low v_perp: reward maintaining speed
                reward += 0.5 * min(_speed / 4.0, 1.0)
            _ante_rec = {'step': ep_step_count, 'speed': _speed, 'accel': round(_accel,3),
                'steer': abs(_act_steer), 'steer_rate': round(_steer_rate,3),
                'v_perp': round(_v_perp_barrier,3), 'v_tang': round(_v_tang_barrier,3),
                'stuck_antecedent': _approaching_stuck, 'stuck_bonus': round(_stuck_bonus,3), 'stuck_wp': _stuck_wp,  # v39: was dangling
                'dist_barrier': round(_dist_barrier,3), 'stop_dist': round(_stop_dist,3),
                'braking': _braking_intent, 'heading': round(_heading,1)}
            ep_ante_buf.append(_ante_rec)
            if len(ep_ante_buf) > _ANTE_WIN:
                ep_ante_buf.pop(0)
            if _objects_dist:
                ep_nearest_objects.append(_nearest_obj)
                ep_decel_penalties.append(_decel_r)
                ep_safe_speed_ratios.append(_speed_ratio)
                ep_racing_line_errors.append(_racing_line_err)
                if _offtrack or _dist_from_center > 0.4 * _tw: ep_in_recovery = True
                elif ep_in_recovery and _dist_from_center < 0.2 * _tw: ep_in_recovery = False
                if ep_in_recovery: ep_recovery_steps += 1
                if len(ep_heading_diffs)>=2 and abs(ep_heading_diffs[-1])>5.0 and abs(ep_heading_diffs[-2])<=5.0: ep_turn_entry_speeds.append(_speed_ratio)
            # Track corner speeds (high heading diff = corner)
            if len(ep_heading_diffs) > 0 and abs(ep_heading_diffs[-1]) > 5.0:
                ep_corner_speeds.append(_speed)
            # Track graze events (curb context but still on track)
            if _ctx == 1 and not _offtrack:
                ep_graze_count += 1
            
                        # --- v4: Record step for failure analysis ---
            ep_metrics.record_step(rp)  # EpisodeMetricsAccumulator: v_perp braking analysis
            sampler.record_step({
                    'x': rp.get('x', 0.0), 'y': rp.get('y', 0.0),
                    'heading': rp.get('heading', 0), 'speed': _speed,
                    'steering_angle': _steer, 'progress': _prog,
                    'closest_waypoints': _closest,
                    'is_offtrack': _offtrack, 'is_crashed': rp.get('is_crashed', False),
                    'is_reversed': rp.get('is_reversed', False),
                    'distance_from_center': _dist, 'track_width': _tw,
                    'action': action.cpu().numpy().item() if action.numel() == 1 else action.cpu().numpy(), 'reward': reward,
                    'center_r': _center_r, 'heading_r': _head_r,
                    'speed_r': _speed_r, 'step': global_step,
                    'context_pred': int(agent.get_context(tensor(observation).unsqueeze(0))[0]) if hasattr(agent, 'get_context') else -1,
                    'lidar_min': _lidar_min,
                    'nearest_object': _nearest_obj,
                    'barrier_proximity': _barrier_proximity,
                    'track_width': _track_width,
                    'dist_from_center': _dist_from_center,
                })
            # v16: BSTS seasonal per-step tracking
            # v222: per-step BSTS update via BSTSFeedback.update()
            # This feeds into bsts_feedback.model(metric) Kalman filters
            # and drives get_trend_vector() and adjust_weights() correctly.
            bsts_feedback.update(
                {
                    'progress':  float(_prog),
                    'speed':     float(_speed),
                    'steering':  float(abs(_steer)),
                    'reward':    float(reward),
                },
                step=global_step,
                race_type_tag=_track_variant,
            )
            if episode_count % 10 == 0:
                print(f"[KALMAN_CHECK] n_kfs={len(bsts_feedback._kf_instances)}, "
                    f"kf_trends={dict(list(bsts_feedback.kf_trends.items())[:3])}")
                
            _rl_blend = min(1.0, _rl_blend + _rl_blend_rate)
            _step_speed_snap = _speed  # v1.1.1: update AFTER reward, ensures next step sees correct prev speed
            rewards[step] = tensor(np.array(reward))
            # v19: off-policy replay store
            try:
                _nobs_t = next_obs.cpu() if isinstance(next_obs, torch.Tensor) else torch.tensor(obs_to_array(observation), dtype=torch.float32)
                # Store the continuous action tensor (what the critic sees), not the discretized env action
                _td3_action = action.squeeze(0).detach().cpu()
                td3sac.store_transition(obs[step], _td3_action, reward, next_obs, terminated or truncated)
            except Exception:
                pass
            # context label: 0=straight, 1=left_curve, 2=right_curve
            context_labels[step] = 0 if not rp or abs(_curvature) < 0.01 else (1 if _curvature > 0 else 2)

            next_obs = tensor(obs_to_array(observation))
            next_done = tensor(np.array(float(terminated or truncated)))

            if terminated or truncated:
                ep_return = cumulative_ep_reward
                ep_length = ep_step_count
                # ep_progress = max(ep_track_progress_pct, _final_pct)
                # _raw_final_progress = float(_final_rp.get("progress", 0.0) or 0.0)
                _bad_end = bool(_rp_now.get("is_offtrack", False) or _rp_now.get("is_crashed", False) or _rp_now.get("is_reversed", False))
                # lap_completed = 1.0 if (ep_progress >= 99.0 and not _bad_end and ep_step_count >= 60) else 0.0
                # v1.0.13: ep_progress / ep_progress_pct already set per-step above
                # raw_prog still needed for lap gate only
                raw_prog = info.get("reward_params", {}).get("progress", 0.0) if isinstance(info, dict) else 0.0
                ep_progress_pct = max(ep_progress_pct, float(raw_prog))   # take sim pct as floor for lap gate
                lap_completed = 1.0 if ep_progress_pct >= 100.0 and not _bad_end else 0.0
                bootstrap_rewards.update_episode(ep_progress_pct, lap_completed > 0.0)
                # v1.1.0: removed duplicate update_episode (was using metres not pct) — single call above
                lap_time_sec = time.time() - ep_start_time  # v24: wall-clock episode time
                episode_count += 1
                # v1.1.1: SAC alpha update — was never called, alpha stuck at 1.0
                if hasattr(td3sac, 'update_alpha') and len(logprobs) > 0:
                    _mean_logprob = float(logprobs[:ep_step_count].mean().item())
                    td3sac.update_alpha(_mean_logprob)
                # --- v6: episode stuck update ---
                # v1.1.0: progressive escaped threshold — starts at 5% (early training), grows to 20%
                _t_frac_ep = global_step / max(total_timesteps, 1)
                _escaped_thresh = 5.0 + 15.0 * min(_t_frac_ep / 0.30, 1.0)  # 5% @ t=0 → 20% @ t=30%
                _escaped = ep_progress_pct > _escaped_thresh  # v1.1.0
                if stuck_tracker._cur_stuck_cluster is not None:
                    _escaped = ep_progress_pct> (stuck_tracker._cur_stuck_cluster / 120.0 * 100.0 + 15.0)
                stuck_tracker.episode_update(
                    entry_wp=_closest[0] if len(_closest) > 0 else 0,
                    ep_return=ep_return, ep_progress=ep_progress,
                    escaped_stuck=_escaped)
                if episode_count % 50 == 0:
                    try:
                        stuck_tracker.print_report()
                    except Exception as e:
                        logger.warning(f"print_report failed: {e}")

                    # v7: Save BSTS data
                    if hasattr(stuck_tracker, 'save_to_json'):
                        try:
                            stuck_tracker.save_to_json("results")
                        except Exception as e:
                            logger.warning(f"save_to_json failed: {e}")
                
                            # v4: End episode for failure analysis
                term_reason = "crashed" if info.get("reward_params",{}).get("is_crashed",False) else ("offtrack" if info.get("reward_params",{}).get("is_offtrack",False) else ("completed" if ep_progress>=95 else "stuck")); sampler.end_episode(ep_progress, ep_return, ep_length, terminated_reason=term_reason)

                # === BSTS AWS (2020)TheRayG (2020) ===
                if _bsts is not None:
                    _bsts.record(ep_step_count*0.1, ep_return, term_reason)
                if _bsts_log is not None:
                    _bsts_log.record(ep_step_count*0.1, ep_return, term_reason)
                # EpisodeMetricsAccumulator: produce braking/speed summary
                _ep_summary = ep_metrics.end_episode(
                    ep_progress, ep_return, ep_step_count,
                    terminated_reason=term_reason
                )
                if global_step % 200 == 0: print(f"[BSTS] ep={episode_count} {_bsts.summary()}")
                # v4-bsts: crash-site diagnostics
                if term_reason == "crashed" or term_reason == "offtrack":
                    ep_crash_ctx = ep_context_preds[-1] if ep_context_preds else -1
                    ep_crash_speed = ep_speeds[-1] if ep_speeds else 0.0
                    ep_crash_heading = ep_headings[-1] if ep_headings else 0.0  # v26
                    ep_crash_closest_wp = ep_closest_wps[-1] if ep_closest_wps else [0,1]  # v26
                    ep_crash_lidar_min = ep_lidar_mins[-1] if ep_lidar_mins else 1.0
                    # v29: CRASH FORENSICS - dump antecedent kinematics
                    _ante_n = len(ep_ante_buf)
                    _ante_brakes = sum(1 for r in ep_ante_buf if r.get('braking',0))
                    _ante_mean_accel = sum(r.get('accel',0) for r in ep_ante_buf)/max(_ante_n,1)
                    _ante_mean_vperp = sum(r.get('v_perp',0) for r in ep_ante_buf)/max(_ante_n,1)
                    _ante_max_vperp = max((r.get('v_perp',0) for r in ep_ante_buf), default=0)
                    _ante_min_dist = min((r.get('dist_barrier',9) for r in ep_ante_buf), default=9)
                    _ante_final_vperp = ep_ante_buf[-1].get('v_perp',0) if ep_ante_buf else 0
                    _ante_final_stopdist = ep_ante_buf[-1].get('stop_dist',0) if ep_ante_buf else 0
                    logger.warning(
                        f'[CRASH_FORENSICS] ep={episode_count} step={global_step} wp={ep_crash_closest_wp} spd={ep_crash_speed:.2f} hdg={ep_crash_heading:.1f} '
                        f'ante_window={_ante_n} brake_steps={_ante_brakes}/{_ante_n} '
                        f'mean_accel={_ante_mean_accel:.2f} mean_v_perp={_ante_mean_vperp:.3f} '
                        f'max_v_perp={_ante_max_vperp:.3f} final_v_perp={_ante_final_vperp:.3f} '
                        f'min_dist_barrier={_ante_min_dist:.3f} final_stop_dist={_ante_final_stopdist:.3f} '
                        f'crash_speed={ep_crash_speed:.2f}')
                    # Dump full antecedent buffer for detailed analysis
                    for _ai, _ar in enumerate(ep_ante_buf):
                        logger.info(f'  [ANTE t-{_ante_n-_ai}] {_ar}')
                # --- v4: Log stuck position ---
                if ep_progress_pct< 100.0 or term_reason in ('offtrack', 'crashed'):
                    _rp = info.get('reward_params', {})
                    logger.warning(
                        f'[STUCK] step={global_step}, '
                        f'progress={ep_progress:.1f}%%, '
                        f'pos=({_rp.get("x", 0):.2f}, {_rp.get("y", 0):.2f}), '
                        f'heading={_rp.get("heading", 0):.1f}, '
                        f'closest_wp={_rp.get("closest_waypoints", [0,1])}, '
                        f'speed={_rp.get("speed", 0):.2f}, '
                        f'offtrack={_rp.get("is_offtrack", False)}, '
                        f'crashed={_rp.get("is_crashed", False)}, '
                        f'reversed={_rp.get("is_reversed", False)}'
                    )

                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=round(et)))
                logger.info(
                    f'step={global_step}, '
                    f'episodic_return={ep_return}, '
                    f'episodic_length={ep_length}, '
                    f'time_elapsed={et}, '
                    f'progress={ep_progress:.1f}%, '
                f'lap_time={lap_time_sec:.1f}s, track={_track_name}({_track_variant}), term={term_reason}, '
                )

                writer.add_scalar('charts/track_progress', ep_progress, global_step)
                writer.add_scalar('charts/track_progress_pct',   ep_progress_pct,   global_step)
                writer.add_scalar('charts/track_progress_m', ep_centerline_progress_m, global_step)
                writer.add_scalar('charts/track_length_m', ep_track_length_m, global_step)
                writer.add_scalar('charts/lap_completed', lap_completed, global_step)
                writer.add_scalar('charts/episodic_return', ep_return, global_step)
                writer.add_scalar('charts/episodic_length', ep_length, global_step)
                
                            # v4: Periodic save of failure analysis
                if episode_count % 25 == 0:
                    sampler.save()

                        # --- v5: BSTS JSONL metrics (per episode) ---
                    _crashed = 1 if rp.get('is_crashed', False) else 0
                    _min_dist = min(ep_dist_from_center) if ep_dist_from_center else 0.0
                    _graze = 1 if (_min_dist < 0.15 * _tw and not _crashed) else 0
                    # curvature*speed: mean(|heading_diff|)*mean(speed)
                    _mean_hdiff = sum(abs(h) for h in ep_heading_diffs)/max(len(ep_heading_diffs),1)
                    _mean_speed = sum(ep_speeds)/max(len(ep_speeds),1)
                    _curv_x_spd = _mean_hdiff * _mean_speed
                    # early-entry-late-exit: track apex timing via progress rate
                    _prog_diffs = [ep_progress_hist[i+1]-ep_progress_hist[i] for i in range(len(ep_progress_hist)-1)] if len(ep_progress_hist)>1 else [0]
                    _eele = max(_prog_diffs) - min(_prog_diffs) if _prog_diffs else 0.0
                        # --- v27 harmonized metrics (success + intermediary; all up==good) ---
                    try:
                        _hm_track_width = float(rp.get('track_width', 1.0)) if rp else 1.0
                        _hm_n_wp = len(rp.get('waypoints', [])) if rp else None
                        _hm_out = _hm.compute_all(_ep_step_log, float(_prog), n_waypoints=_hm_n_wp, track_width=_hm_track_width, waypoints=_waypoints if '_waypoints' in dir() else None)  # v1.1.0: pass waypoints for arc-progress
                    except Exception as _e_hm:
                        _hm_out = {}
                    # v213: race-type tag for plot/CSV differentiation
                    _race_type_tag = {
                        'time_trial': 'tt',
                        'obstacle':   'oa',
                        'h2h':        'h2h',
                    }.get(_track_variant, _track_variant)
                    bsts_row = {
                        'episode': episode_count,
                        'global_step': global_step,
                        "variant": _track_variant,
                        "track": _track_name, 
                        'crash': _crashed,
                        'graze': _graze,
                        'curvature_x_speed': round(_curv_x_spd, 4),
                        'early_entry_late_exit': round(_eele, 4),
                        'progress': round(ep_progress, 2),
                        'ep_return': round(ep_return, 4),
                        'ep_brake_frac': round(_ep_summary.get('brake_fraction',0.0),4) if _ep_summary else 0.0,
                        'ep_speed_mean': round(_ep_summary.get('mean_speed',0.0),4) if _ep_summary else 0.0,
                        'ep_speed_std':  round(_ep_summary.get('speed_var',0.0)**0.5,4) if _ep_summary else 0.0,
                                        'lap_time_sec': round(lap_time_sec,2),
                        # v213: race-type / arc-length telemetry
                        'race_type':             _race_type_tag,
                        'track_arc_m':           round(ep_track_length_m, 2),
                        'track_progress_pct':    round(ep_progress_pct, 2),
                        'track_progress_arc_m':  round(ep_centerline_progress_m, 2),
                'term_reason': term_reason,
                'track_name': _track_name,
                'track_variant': _track_variant,
                    # v29 crash antecedent forensics
                    'ante_brake_steps': sum(1 for r in ep_ante_buf if r.get('braking',0)),
                    'ante_mean_accel': round(sum(r.get('accel',0) for r in ep_ante_buf)/max(len(ep_ante_buf),1),3),
                    'ante_mean_vperp': round(sum(r.get('v_perp',0) for r in ep_ante_buf)/max(len(ep_ante_buf),1),3),
                    'ante_final_vperp': round(ep_ante_buf[-1].get('v_perp',0),3) if ep_ante_buf else 0,
                    'ante_final_stopdist': round(ep_ante_buf[-1].get('stop_dist',0),3) if ep_ante_buf else 0,
                        'rl_blend':      round(_rl_blend,4),
                        'env_signal':    round(_env_signal if '_env_signal' in dir() else 0.0,4),
                        'avg_speed': round(_mean_speed, 4),
                        'min_dist_from_center': round(_min_dist, 4),
                        'offtrack_rate': round(ep_offtrack_count/max(ep_step_count,1), 4),
                        'reversed_count': ep_reversed_count,
                        'zero_speed_count': ep_zero_speed_count,
                    'avg_safe_speed_ratio': round(sum(ep_safe_speed_ratios)/max(len(ep_safe_speed_ratios),1), 4),
                    'avg_racing_line_err': round(sum(ep_racing_line_errors)/max(len(ep_racing_line_errors),1), 4),
                    'avg_decel_penalty': round(sum(ep_decel_penalties)/max(len(ep_decel_penalties),1), 4),
                        'recovery_steps': ep_recovery_steps,
                    'avg_turn_entry_ratio': round(sum(ep_turn_entry_speeds)/max(len(ep_turn_entry_speeds),1), 4) if ep_turn_entry_speeds else 0.0,
                    }
                    bsts_row.update(_hm_out)
                    jsonl_file.write(json.dumps(bsts_row) + '\n')
                    jsonl_file.flush()


                    # --- v4: Log sub-reward breadcrumbs ---
                    if ep_step_count > 0:
                        for comp_name, comp_vals in ep_rewards_components.items():
                            if comp_vals:
                                avg_val = sum(comp_vals) / len(comp_vals)
                                writer.add_scalar(f"rewards/{comp_name}", avg_val, global_step)
                        if ep_speeds:
                            writer.add_scalar("behavior/avg_speed", sum(ep_speeds)/len(ep_speeds), global_step)
                        if ep_dist_from_center:
                            writer.add_scalar("behavior/avg_dist_from_center", sum(ep_dist_from_center)/len(ep_dist_from_center), global_step)
                        writer.add_scalar("behavior/offtrack_rate", ep_offtrack_count / ep_step_count, global_step)
                        writer.add_scalar("behavior/ep_steps", ep_step_count, global_step)
                    # v4: context-aware tensorboard logging
                        # v4-bsts: compute metrics BEFORE reset
                    bsts_metrics = {
                        'crash_rate': float(terminated),
                        'offtrack_rate': float(ep_offtrack_count / max(ep_step_count, 1)),
                        'avg_speed': sum(ep_speeds)/max(len(ep_speeds),1) if ep_speeds else 2.0,
                        'corner_crash_rate': float(terminated and ep_step_count < 50),
                    'avg_safe_speed_ratio': round(sum(ep_safe_speed_ratios) / max(len(ep_safe_speed_ratios), 1), 4),
                    'avg_racing_line_err':  round(sum(ep_racing_line_errors) / max(len(ep_racing_line_errors), 1), 4),
                    'ctx_obstacle_ratio': sum(1 for h in ep_context_preds if h == 2) / max(len(ep_context_preds), 1),
                    'ctx_curb_ratio': sum(1 for h in ep_context_preds if h == 1) / max(len(ep_context_preds), 1),
                    'ctx_clear_ratio': sum(1 for h in ep_context_preds if h == 0) / max(len(ep_context_preds), 1),
                    'avg_corner_speed': sum(ep_corner_speeds) / max(len(ep_corner_speeds), 1) if ep_corner_speeds else 0.0,
                    'graze_count': ep_graze_count,
                    'crash_ctx': ep_crash_ctx if ep_crash_ctx is not None else -1,
                    'crash_speed': ep_crash_speed if ep_crash_speed is not None else 0.0,
                    'crash_lidar_min': ep_crash_lidar_min if ep_crash_lidar_min is not None else 1.0,

                    # v26: barrier-relative velocity at crash
                    'crash_v_perp_barrier': _compute_crash_v_perp(ep_crash_speed, ep_crash_heading, ep_crash_closest_wp, _waypoints) if ep_crash_speed is not None else 0.0,
                    'crash_v_tang_barrier': _compute_crash_v_tang(ep_crash_speed, ep_crash_heading, ep_crash_closest_wp, _waypoints) if ep_crash_speed is not None else 0.0,
                    'avg_lidar_min': sum(ep_lidar_mins)/max(len(ep_lidar_mins),1) if ep_lidar_mins else 1.0,
                    'avg_barrier_proximity': sum(ep_barrier_proximities)/max(len(ep_barrier_proximities),1) if ep_barrier_proximities else 0.0,
                    'avg_nearest_object': sum(ep_nearest_objects)/max(len(ep_nearest_objects),1) if ep_nearest_objects else 999.0,
                    # v29 crash antecedent metrics
                    'crash_ante_brake_steps': sum(1 for r in ep_ante_buf if r.get('braking',0)) if ep_ante_buf else 0,
                    'crash_ante_mean_accel': sum(r.get('accel',0) for r in ep_ante_buf)/max(len(ep_ante_buf),1),
                    'crash_ante_mean_vperp': sum(r.get('v_perp',0) for r in ep_ante_buf)/max(len(ep_ante_buf),1),
                    'crash_ante_final_vperp': ep_ante_buf[-1].get('v_perp',0) if ep_ante_buf else 0,
                    'crash_ante_final_stopdist': ep_ante_buf[-1].get('stop_dist',0) if ep_ante_buf else 0,
                        'avg_ang_vel_centerline': (sum(ep_ang_vel_centerline)/len(ep_ang_vel_centerline)) if ('ep_ang_vel_centerline' in dir() and ep_ang_vel_centerline) else 0.0,  # v37
                        'avg_jerk': (sum(ep_jerk_abs)/len(ep_jerk_abs)) if ('ep_jerk_abs' in dir() and ep_jerk_abs) else 0.0,  # v37
                        'max_jerk': (max(ep_jerk_abs) if ('ep_jerk_abs' in dir() and ep_jerk_abs) else 0.0),  # v37
                        'brake_line_compliance': (sum(ep_brake_before_barrier)/len(ep_brake_before_barrier)) if ('ep_brake_before_barrier' in dir() and ep_brake_before_barrier) else 0.0,  # v37
                        'position_in_lap': float(ep_progress),  # v1.0.14 centerline pct
                        'completion_pct': float(ep_progress),
                        'track_progress_m': float(ep_centerline_progress_m),
                        'track_length_m': float(ep_track_length_m),
                        'lap_completed': float(lap_completed),  # v1.0.14 strict completion gate
                    }
                    # v1.1.0: merge harmonized_metrics output into bsts_metrics
                    # so avg_speed_centerline, track_progress, race_line_adherence reach BSTS EMA
                    if _hm_out:
                        bsts_metrics.update({
                            'avg_speed_centerline': float(_hm_out.get('avg_speed_centerline', 0.0)),
                            'track_progress':       float(_hm_out.get('track_progress', 0.0)),
                            'race_line_adherence':  float(_hm_out.get('race_line_adherence', 0.0)),
                            'brake_compliance':     float(_hm_out.get('brake_compliance', 0.0)),
                            'smoothness_jerk_rms':  float(_hm_out.get('smoothness_jerk_rms', 0.0)),
                        })
                    bsts_feedback.update(bsts_metrics)
                    if episode_count % 10 == 0:
                        print(f"[KALMAN_CHECK] n_kfs={len(bsts_feedback._kf_instances)}, "
                            f"kf_trends={dict(list(bsts_feedback.kf_trends.items())[:3])}")
                # --- Compute BSTS-adjusted reward weights ---
                # v1.1.1: skip BSTS weight adjustment during bootstrap.
                # BSTS was overwriting bootstrap progress weights -> rw_adj:prog collapsed 0.127->0.026.
                if _bootstrap_active:
                    _adjusted_rw = rw  # bootstrap weights unchanged during bootstrap phase
                else:
                    _adjusted_rw = bsts_feedback.adjust_weights(rw)
                # --- Rich BSTS console log every episode ---
                _bsts_ctx = f"ctx=clear:{bsts_metrics.get('ctx_clear_ratio',0):.2f}/curb:{bsts_metrics.get('ctx_curb_ratio',0):.2f}/obs:{bsts_metrics.get('ctx_obstacle_ratio',0):.2f}"
                print(
                    f'[BSTS ep={episode_count:5d} step={global_step:7d} Track={_track_name}({_track_variant})] '
                    f'ret={ep_return:8.2f} prog={ep_progress:5.1f}% spd={bsts_metrics.get("avg_speed",0):.2f} '
                    f'crash={bsts_metrics.get("crash_rate",0):.2f} otr={bsts_metrics.get("offtrack_rate",0):.2f} '
                    f'ssr={bsts_metrics.get("avg_safe_speed_ratio",1):.2f} rle={bsts_metrics.get("avg_racing_line_err",0):.2f} '
                    f'term={term_reason} lap_t={lap_time_sec:.1f}s {_bsts_ctx} '
                    f'rw_adj=center:{_adjusted_rw.get("center",0):.3f}/prog:{_adjusted_rw.get("progress",0):.3f}/corner:{_adjusted_rw.get("corner",0):.3f}',
                    flush=True
                )
                # Log to BSTS CSV
                try:
                    _bsts_csv_writer.writerow({
                        'episode': episode_count, 'global_step': global_step,
                        'race_type':            _race_type_tag,
                        'track_name':           _track_name,
                        'track_variant':        _track_variant,
                        'track_arc_m':          round(ep_track_length_m, 2),
                        'track_progress_pct':   round(ep_progress_pct, 2),
                        'track_progress_arc_m': round(ep_centerline_progress_m, 2),
                        'crash_rate': bsts_metrics.get('crash_rate',0),
                        'offtrack_rate': bsts_metrics.get('offtrack_rate',0),
                        'avg_speed': bsts_metrics.get('avg_speed',0),
                        'corner_crash_rate': bsts_metrics.get('corner_crash_rate',0),
                        'avg_safe_speed_ratio': bsts_metrics.get('avg_safe_speed_ratio',1),
                        'avg_racing_line_err': bsts_metrics.get('avg_racing_line_err',0),
                        **{f'rw_{k}': _adjusted_rw.get(k,0) for k in ['center','heading','curv_speed','progress','completion','corner','braking','min_speed','racing_line']}
                    })
                    _bsts_csv_f.flush()
                    # v28: Periodic console summary from live_dashboard
                    if live_summary is not None and episode_count % 500 == 0:
                        try:
                            live_summary(bsts_feedback, global_step, episode_count)
                        except Exception:
                            pass  # non-critical
                except Exception as _ce:
                    pass
                # Use BSTS-adjusted weights for next episode reward shaping
                rw = _adjusted_rw
            # === Online Kalman BSTS update ===
                try:
                    ep_data = {'steps': _ep_step_log, 'completion_pct': bsts_metrics.get('completion_pct', 0),
                               'termination_reason': term_reason}
                    # Wire: extract real intermediary metrics from episode step log
                    try:
                        intermediary = extract_intermediary_metrics({'steps': _ep_step_log})
                    except Exception:
                        intermediary = {m: [0] for m in INTERMEDIARY_METRICS}  # fallback
                    summary = episode_summary_metrics(ep_data, intermediary)
                    # Override with actual bsts_metrics values
                    summary['lap_completion_pct'] = bsts_metrics.get('completion_pct', 0)
                    summary['reward_per_step'] = bsts_metrics.get('avg_reward', 0)
                    summary['off_track_rate'] = bsts_metrics.get('offtrack_rate', 0)
                    summary['crash_rate'] = 1.0 if term_reason == 'crashed' else 0.0
                    _kf_episode_buffer.append(summary)
                    if hasattr(bsts_feedback, "_all_summaries"): bsts_feedback._all_summaries.append(summary)
                    # v1.1.1: flush Kalman every episode (was 10). Eps are 14-29 steps each;
                    # at batch=10, Kalman was dead (all zeros) for 200+ episodes.
                    if len(_kf_episode_buffer) >= 1:
                        # import numpy as np  # use global
                        reg_names = [f'{m}_mean' for m in INTERMEDIARY_METRICS]
                        n = len(_kf_episode_buffer)
                        X = np.zeros((n, len(reg_names)))
                        for t in range(n):
                            for j, rn in enumerate(reg_names):
                                X[t, j] = _kf_episode_buffer[t].get(rn, 0.0)
                        X_std = X.std(axis=0) + 1e-8
                        X_norm = (X - X.mean(axis=0)) / X_std
                        for sm in SUCCESS_METRICS:
                            y = np.array([_kf_episode_buffer[t].get(sm, 0) for t in range(n)])
                            result = _kf_bsts[sm].filter_series(y, X_norm)
                            bsts_feedback.kf_trends[sm] = float(result['trends'][-1])
                            if result.get('betas') is not None:
                                for j, rn in enumerate(reg_names):
                                    bsts_feedback.kf_betas[rn] = float(result['betas'][-1][j])
                        _kf_episode_buffer.clear()

                        # === BSTS Compliance Report (every 50 episodes) ===
                        if episode_count > 0 and episode_count % 50 == 0:
                            try:
                                _bsts_matrix = list(_kf_episode_buffer) if _kf_episode_buffer else []
                                # Use accumulated summaries from all prior episodes
                                if hasattr(bsts_feedback, '_all_summaries'):
                                    _bsts_matrix = bsts_feedback._all_summaries
                                if len(_bsts_matrix) >= 10:
                                    _bsts_rpt = bsts_compliance_report(_bsts_matrix)
                                    _anneal_recs = compute_anneal_recommendations(_bsts_rpt, _bsts_matrix)
                                    logger.info(f"[BSTS-Report] trend={_bsts_rpt.get('trend','?')} "
                                                f"seasonal={_bsts_rpt.get('seasonal_period','?')} "
                                                f"LR_rec={_anneal_recs.get('learning_rate','?')} "
                                                f"residual_std={_anneal_recs.get('residual_std',0):.4f}")
                                    for sm, trend in _bsts_rpt.get('per_metric_trends', {}).items():
                                        logger.info(f"  {sm}: {trend}")
                                    for rec in _bsts_rpt.get('recommendations', []):
                                        logger.info(f"  [REC] {rec.get('action','')}")
                                    # Apply anneal recommendations to reward weights
                                    for k, v in _anneal_recs.get('reward_weight_adjustments', {}).items():
                                        logger.info(f"  [ANNEAL] {k}: {v}")
                            except Exception as _be:
                                logger.debug(f"BSTS report skip: {_be}")

                        # === Race Line Analysis (every 100 episodes) ===
                        if episode_count > 0 and episode_count % 100 == 0:
                            try:
                                _wps = []
                                if hasattr(bsts_feedback, '_all_summaries'):
                                    for _ep in bsts_feedback._all_summaries[-20:]:
                                        if _ep.get('lap_completion_pct', 0) > 80:
                                            _wps = [(s.get('x',0), s.get('y',0)) for s in _ep.get('_steps', [])]
                                            break
                                if not _wps and hasattr(env, 'waypoints'):
                                    _wps = [(w[0], w[1]) for w in env.waypoints]
                                if _wps and len(_wps) >= 4:
                                    _rl = compute_optimal_race_line(_wps)
                                    logger.info(f"[RaceLine] brake_integral={_rl.get('brake_zone_integral',0):.3f} ")
                                _rl_eps = [{"steps": e.get("_steps", []), "completion_pct": e.get("lap_completion_pct", 0)} for e in bsts_feedback._all_summaries[-20:]]
                                _rl_score = score_race_line_compliance(_rl_eps, _rl)
                                logger.info(f"[RaceLine] perp_v={_rl_score.get(chr(39)+chr(97)+chr(118)+chr(103)+chr(95)+chr(112)+chr(101)+chr(114)+chr(112)+chr(95)+chr(118)+chr(39), 0):.4f}")
                            except Exception as _re:
                                logger.debug(f"Race line analysis skip: {_re}")
                        logger.info(f"[BSTS-Kalman] trends={bsts_feedback.kf_trends} betas_top={dict(list(bsts_feedback.kf_betas.items())[:3])}")
                except Exception as e:
                    # v1.1.0: was bare `pass` — silent death of ALL Kalman signal.
                    # Now logs at DEBUG so we can see what's failing without spamming console.
                    logger.debug(f"[BSTS-Kalman] update failed ep={episode_count}: {type(e).__name__}: {e}")

                                # TensorBoard context/crash/barrier scalars — only when we have context data
                if ep_context_preds:
                    writer.add_scalar('context/obstacle_ratio',  bsts_metrics.get('ctx_obstacle_ratio', 0),  global_step)
                    writer.add_scalar('context/curb_ratio',      bsts_metrics.get('ctx_curb_ratio', 0),      global_step)
                    writer.add_scalar('context/avg_corner_speed',bsts_metrics.get('avg_corner_speed', 0),    global_step)
                    writer.add_scalar('context/graze_count',     bsts_metrics.get('graze_count', 0),         global_step)
                    writer.add_scalar('crash/ctx_at_crash',      bsts_metrics.get('crash_ctx', -1),          global_step)
                    writer.add_scalar('crash/speed_at_crash',    bsts_metrics.get('crash_speed', 0),         global_step)
                    writer.add_scalar('crash/lidar_min_at_crash',bsts_metrics.get('crash_lidar_min', 1),     global_step)
                    writer.add_scalar('barrier/avg_lidar_min',   bsts_metrics.get('avg_lidar_min', 1),       global_step)
                    writer.add_scalar('barrier/avg_proximity',   bsts_metrics.get('avg_barrier_proximity', 0),global_step)
                    writer.add_scalar('barrier/avg_nearest_obj', bsts_metrics.get('avg_nearest_object', 999),global_step)

                # Reset accumulators — ALWAYS, not gated on ep_context_preds
                _brake_field.reset()
                ep_rewards_components  = {k: [] for k in ep_rewards_components}
                ep_speeds              = []
                ep_headings            = []
                ep_closest_wps         = []
                ep_dist_from_center    = []
                ep_offtrack_count      = 0
                ep_offtrack_steps      = 0
                ep_step_count          = 0
                cumulative_ep_reward   = 0.0
                _prev_prog_tracker     = 0.0
                ep_progress            = 0.0
                ep_progress_pct        = 0.0
                ep_centerline_progress_m = 0.0
                ep_track_length_m      = 100.0
                ep_track_progress_pct  = 0.0
                ep_context_preds       = []
                ep_lidar_mins          = []
                ep_barrier_proximities = []
                ep_nearest_objects     = []
                ep_crash_ctx           = None
                ep_crash_speed         = None
                ep_crash_heading       = None
                ep_crash_closest_wp    = None
                ep_crash_lidar_min     = None
                ep_corner_speeds       = []
                ep_graze_count         = 0
                ep_heading_diffs = []
                ep_ang_vel_centerline = []  # v37
                _trunc_rp = info.get("reward_params", {}) if isinstance(info, dict) else {}
                ep_prev_speed = float(_trunc_rp.get("speed", 0.0))        
                _step_speed_snap = ep_prev_speed
                ep_jerk_abs = []  # v37
                ep_brake_before_barrier = []  # v37
                ep_steerings_raw = []
                ep_decel_penalties = []
                ep_safe_speed_ratios = []
                ep_racing_line_errors = []
                # v1.1.0: flush BSTSSeasonal episode buffer before resetting
                if bsts_season is not None and hasattr(bsts_season, '_flush_episode'):
                    try:
                        bsts_season._flush_episode(lap_completed > 0.5)
                        # periodic Kalman fit from step buffer (every 20 episodes)
                        if episode_count % 20 == 0 and len(getattr(bsts_season, '_ep_buf', [])) >= getattr(bsts_season, 'STEP_BUFFER_MIN_EPISODES', 20):
                            bsts_season._fit_from_step_buffer()
                            _bsts_trend = bsts_season.get_trend() if hasattr(bsts_season, 'get_trend') else {}
                            logger.info(f"[BSTS-Kalman] ep={episode_count} trend={_bsts_trend}")
                    except Exception as _fe:
                        logger.debug(f"BSTS flush error: {_fe}")
                _ep_step_log = []
                ep_recovery_steps = 0
                ep_in_recovery = False
                ep_turn_entry_speeds = []
                ep_positions = []
                ep_first_offtrack_step = None
                ep_progress_hist = []
                ep_reversed_count = 0
                ep_zero_speed_count = 0
                ep_offtrack_steps = 0 
                ep_start_time = time.time()  # v24: reset lap timer
                # v17-bsts feedback: wire analysis -> annealing
                if bsts_season is not None:
                    try:
                        # get_trend_vector returns {metric: 5-step linear slope}
                        _tv   = bsts_feedback.get_trend_vector(race_type_filter=_track_variant)
                        trend = float(_tv.get('reward', 0.0))     # use reward trend as proxy for return trend
                        # seasonal component is internal to the Kalman RLS inside BSTSFeedback.model()
                        # — not separately extractable; set to 0.0 here
                        seasonal = 0.0
                        m = {
                            'ep_return':    ep_return,
                            'trend':        trend,
                            'seasonal':     seasonal,
                            'episode':      episode_count,
                            'crash_rate':   ep_offtrack_count / max(ep_step_count, 1),
                            'avg_speed':    float(np.mean(ep_speeds)) if ep_speeds else 0.0,
                        }
                        bsts_feedback.update(m, step=global_step, race_type_tag=_track_variant)
                        if hasattr(bsts_feedback, 'adjust_weights'):
                            adj = bsts_feedback.adjust_weights(scheduler.get_reward_weights(global_step),
                                                            race_type_filter=_track_variant)
                            scheduler.rw_end = adj
                        if hasattr(td3sac, 'update_alpha') and 'logprobs' in locals() and len(logprobs) > 0:
                            _ep_mean_logp = float(logprobs[:min(ep_step_count, len(logprobs))].mean().item())
                            td3sac.update_alpha(_ep_mean_logp)
                    except Exception as e:
                        logger.debug(f"BSTS feedback skip: {e}")
                # v17: periodic live analysis and race-line phase-out
                if episode_count % 50 == 0 and episode_count > 0:
                    try:
                        import os as _os
                        _lp = _os.path.join("results", "metrics.jsonl")
                        if _os.path.exists(_lp) and _ANALYSIS_MODULES:
                            _a = live_analyze(_lp)
                            logger.info(f"Live analysis ep{episode_count}: {_a}")
                    except Exception as _e:
                        logger.debug(f"Live analysis skip: {_e}")

                for _rtry in range(3):
                    try:
                        observation, info = env.reset()
                        # v1.0.14: init arc-progress state for NEW episode spawn position
                        _new_rp = info.get("reward_params", {}) if isinstance(info, dict) else {}
                        _episode_progress_state = reset_episode_centerline_progress(
                            _new_rp, _track_progress_cache
                        )
                        break
                    except Exception as e:
                        logger.warning(f"mid-training reset attempt {_rtry+1}/3 failed: {e}")
                        if _rtry < 2:
                            # Simple retry first — ZMQ may just need a moment
                            time.sleep(5)
                        else:
                            # All 3 simple retries exhausted — full ZMQ rebuild
                            logger.warning("mid-training: all retries failed, rebuilding env with ZMQ teardown")
                            try:
                                env.close()
                            except Exception:
                                pass
                            time.sleep(10)  # let OS release the socket before rebind
                            env = _apply_phase_env(args, _phase)
                            # Brief preflight to confirm port 8888 is back up before handing off
                            for _wait in range(6):
                                if preflightgymbridge():
                                    break
                                time.sleep(5)
                else:
                    raise RuntimeError("mid-training env.reset unrecoverable after 3 retries")
                next_obs = tensor(obs_to_array(observation))
                next_done = torch.zeros(1, device=DEVICE)
            if ep_progress_pct> 10 and ep_return > best_return:  # v41: must have >10% progress to be best
                    best_return = ep_return
                    torch.save({
                        'state_dict': agent.state_dict(),
                        'class': agent.__class__.__name__,
                        '_obs_dim': _obs_dim,
                        'actdim': _act_dim_agent,
                    }, f"{agent.name}best.torch")

                    # LOAD (line ~684)

                    pool.add_checkpoint(agent, ep_return, episode_count)  # v8
                    logger.info(f'New best model saved: return={best_return}')


        # ---- compute GAE ----

        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0))

        advantages = zeros((num_steps,))
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * hp['gae_lambda'] * nextnonterminal * lastgaelam
        returns = advantages + values

        # ---- PPO update ----
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1).long() if _is_discrete else actions  # V33: keep 2D for continuous; V211
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_context_labels = context_labels.reshape(-1).long()

        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, new_ctx_logits, _new_intermed = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > hp['clip_coef']).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - hp['clip_coef'], 1 + hp['clip_coef']
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                if clip_vloss:
                    # REF: Fujimoto et al. (2018) target policy smoothing: clipped noise on value targets
                    _smooth_noise = torch.randn_like(b_returns[mb_inds]).clamp(-0.5, 0.5) * 0.1
                    _smoothed_returns = (b_returns[mb_inds] + _smooth_noise).detach()
                    v_loss_unclipped = (newvalue - _smoothed_returns) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -hp['clip_coef'], hp['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(
                        v_loss_unclipped, v_loss_clipped
                    ).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                # REF: Haarnoja et al. (2018) update adaptive temperature
                if _RESEARCH_MODULES and '_log_alpha' in dir():
                    _alpha_loss = -(_log_alpha * (entropy_loss.detach() + _target_entropy))
                    _alpha_optim.zero_grad(); _alpha_loss.backward(); _alpha_optim.step()
                    hp['ent_coef'] = float(_log_alpha.exp().detach().clamp(0.01, 1.0))
                ctx_loss = torch.nn.functional.cross_entropy(new_ctx_logits, b_context_labels[mb_inds])
                # v1.1.1: add intermed_loss so the intermediary head actually learns
                # REF: Hettiarachchi et al. (2024) U-Transformer auxiliary losses.
                _device_resolved = device if isinstance(device, torch.device) else torch.device(str(device))
                _intermed_tgts_raw = compute_intermed_targets(
                    params if 'params' in dir() else {},
                    race_line_engine=race_line_eng if 'race_line_eng' in locals() else None
                )
                _intermed_targets_mb = _intermed_tgts_raw.to(_device_resolved).unsqueeze(0).expand(len(mb_inds), -1)
                _intermed_loss = torch.nn.functional.mse_loss(
                    _new_intermed, _intermed_targets_mb.detach()
                )
                # v1.1.1: intermed head needs a gradient signal or ctx_emb stays random.
                # Use self-consistency target: intermediary output should be stable across steps
                # (L2 toward running EMA of its own outputs — no external ground truth needed).
                if not hasattr(agent, '_intermed_ema'):
                    agent._intermed_ema = None
                if agent._intermed_ema is None:
                    agent._intermed_ema = _new_intermed.detach().mean(0)
                else:
                    agent._intermed_ema = (0.95 * agent._intermed_ema +
                                        0.05 * _new_intermed.detach().mean(0))
                intermed_consistency_loss = F.mse_loss(
                    _new_intermed,
                    agent._intermed_ema.unsqueeze(0).expand_as(_new_intermed).detach()
                )
                loss = (0.3 * pg_loss
                        - hp['ent_coef'] * entropy_loss
                        + vf_coef * v_loss
                        + 0.1 * ctx_loss
                        + 0.03 * intermed_consistency_loss)  # gentle — don't overwhelm PPO

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.parameters(), max_grad_norm
                )
                optimizer.step()
                # --- v19: TD3+SAC critic update ---
                if global_step % _td3sac_update_freq == 0 and len(td3sac.replay) >= 256:
                    _em = td3sac.update_critics(agent, batch_size=256)
                    if _em["critic_loss"] > 0:
                        writer.add_scalar("td3sac/critic_loss", _em["critic_loss"], global_step)
                        writer.add_scalar("td3sac/alpha", _em["alpha"], global_step)
                        writer.add_scalar("td3sac/q1_mean", _em["q1_mean"], global_step)
                        writer.add_scalar("td3sac/replay_size", _em["replay_size"], global_step)
                # --- v20: TD3 actor update (PRIMARY policy optimizer) ---
                if global_step % (_td3sac_update_freq * 2) == 0:
                    _td3_actor = td3sac.update_actor(agent, batch_size=256)
                    print(f"[TD3_ACTOR] obs_max={obs_b.abs().max():.2f}, "
                        f"bc_loss={bc_loss.item():.4f}, finite={torch.isfinite(bc_loss)}")
                    # v1.1.1: use td3_ready flag; blend bc_loss into PPO loss
                    # rather than running a separate optimizer step (avoids
                    # double-stepping the optimizer and corrupting GAE returns).
                    # REF: Fujimoto et al. (2021) TD3+BC §3 — lambda * BC term.
                    if _td3_actor.get("td3_ready") and _td3_actor.get("td3_bc_loss") is not None:
                        _td3_bc = _td3_actor["td3_bc_loss"]
                        if torch.isfinite(_td3_bc):
                            # Blend into current minibatch loss (loss was already .backward()'d
                            # above — we need a fresh backward for the bc term only)
                            optimizer.zero_grad()
                            (0.05 * _td3_bc).backward()
                            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                            optimizer.step()
                            writer.add_scalar("td3sac/actor_bc_loss", _td3_actor["td3_bc_loss_val"], global_step)
                            writer.add_scalar("td3sac/q1_mean_actor", _td3_actor["td3_q1_mean"], global_step)

            if target_kl is not None and approx_kl > target_kl:
                break

        # logging
        writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)
        writer.add_scalar('losses/approx_kl', approx_kl.item(), global_step)
        writer.add_scalar('losses/clipfrac', np.mean(clipfracs), global_step)
        writer.add_scalar('losses/context_loss', ctx_loss.item(), global_step)
        writer.add_scalar(
            'charts/SPS',
            int(global_step / (time.time() - start_time)),
            global_step,
        )

        # periodic save
        if update % 10 == 0:
            torch.save({
                'state_dict': agent.state_dict(),
                'class': agent.__class__.__name__,
                '_obs_dim': _obs_dim,
                'actdim': _act_dim_agent,
            }, f"{agent.name}best.torch")
            logger.info(
                f'Update {update}/{num_updates}, '
                f'step={global_step}, '
                f'policy_loss={pg_loss.item():.4f}, '
                f'value_loss={v_loss.item():.4f}'
            )

    pool.save_manifest()  # v8: persist pool
    # final save
    torch.save({
            'state_dict': agent.state_dict(),
            'class': agent.__class__.__name__,
            '_obs_dim': _obs_dim,
            'actdim': _act_dim_agent,
        }, f"{agent.name}best.torch")
    sampler.save()  # v4: Final save of failure analysis
    # --- v201 guard: detect silent empty-episode / no-sim failure ---
    try:
        jsonl_file.flush()
        import os as _os
        _sz = _os.path.getsize(jsonl_path)
        if _sz == 0:
            logger.error(f'[GUARD] JSONL is EMPTY ({jsonl_path}): no episodes terminated. '
                         f'Likely the DeepRacer sim (docker) is not running on this node. '
                         f'Run start_deepracer.sh on a docker-capable PACE session.')
        else:
            logger.info(f'[GUARD] JSONL size={_sz} bytes -> episodes were recorded.')
    except Exception as _ge:
        logger.warning(f'[GUARD] size check failed: {_ge}')
    jsonl_file.close()
    logger.info(f'Model {agent.name} saved. Training complete.')
    logger.info(f'JSONL metrics saved to {jsonl_path}')
    env.close()
    writer.close()
    # Shutdown dashboard
    try:
        if '_dash_proc' in dir() and _dash_proc:
            _dash_proc.terminate()
            logger.info('[DASHBOARD] Terminated dashboard process')
    except: pass
    try:
        _bsts_csv_f.close()
    except: pass


if __name__ == "__main__":
    run({})
