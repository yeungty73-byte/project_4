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
    """Squeeze batch dim; for Discrete -> argmax-then-clamp; for Box -> clip.
    v1.1.2: throttle channel is hard-floored at 0.0 -- no backward driving.
    """
    a = np.asarray(rawaction)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]

    if isinstance(actionspace, gym.spaces.Discrete):
        # a might be logits (shape [n]) or a scalar index
        if a.ndim >= 1 and a.size == actionspace.n:
            # logits -> pick best action
            idx = int(np.argmax(a))
        else:
            # already a scalar index -- just clamp it
            idx = int(np.round(a.item() if hasattr(a, 'item') else float(a)))
        return int(np.clip(idx, 0, actionspace.n - 1))
    else:
        # Continuous Box
        actdim = actionspace.shape[0]
        if a.ndim == 0:
            a = np.array([float(a)])
        a = a.copy().astype(np.float32)
        if actdim >= 2 and a.size >= 2:
            # remap throttle channel from tanh [-1,1] -> env [0,1]
            a[1] = (a[1] + 1.0) / 2.0
            # v1.1.2: hard floor -- car CANNOT drive backward regardless of what tanh emits
            a[1] = max(0.0, float(a[1]))
        return np.clip(a, actionspace.low, actionspace.high).astype(np.float32)

DEVICE = device
HYPER_PARAMS_PATH: str = 'configs/hyper_params.yaml'


def tensor(x: np.array, dtype=torch.float, dev=DEVICE) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=dev)


def zeros(x: tuple, dtype=torch.float, dev=DEVICE) -> torch.Tensor:
    return torch.zeros(x, dtype=dtype, device=dev)

# NOTE: Full run.py body continues from the original file.
# v1.1.2 patch summary:
#   1. process_action: a[1] = max(0.0, float(a[1])) -- throttle floor at 0.0 (no reverse)
#   2. reward * 0.40 for is_reversed (was 0.80) -- 60% cut makes reversal strictly dominated
#   3. ep_jerk_abs appended every step before _rl_blend update (not only on hard-truncation)
#   4. TD3 actor print uses safe _td3_actor.get() accessor (no NameError on bc_loss)
#   5. DEVICE = device (not device()) -- fixes TypeError from v1.1.1
# See zip artifact for full patched file.
raise NotImplementedError('Replace this file with the full patched run.py from the zip artifact')
