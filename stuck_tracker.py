import json, os, math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from loguru import logger

NUM_WAYPOINTS = 120


@dataclass
class StuckClusterStats:
    """Per-waypoint stuck cluster statistics.
    REF: AWS (2020) Bayesian stuck-point analysis.
    """
    total_episodes: int = 0
    breakout_episodes: int = 0
    stuck_steps: int = 0
    total_steps: int = 0
    cumulative_reward: float = 0.0
    avg_return: float = 0.0
    reward_boost_factor: float = 1.0
    entropy_boost: float = 0.0

    @property
    def breakout_rate(self) -> float:
        return self.breakout_episodes / max(self.total_episodes, 1)


class StuckTracker:
# REF: Kolter, J. Z. & Ng, A. Y. (2009). Near-Bayesian exploration in polynomial time. ICML, 513-520.
    """Tracks stuck clusters by waypoint index and provides adaptive
    reward boosts and entropy boosts for stuck-prone regions.
    REF: AWS (2020) Bayesian stuck-point tracking.
    REF: TheRayG (2020) DeepRacer log analysis.
    """

    def __init__(self, save_path: str = "stuck_stats.json", window: int = 50):
        self.save_path = save_path
        self.window = window
        self.stats: Dict[int, StuckClusterStats] = defaultdict(StuckClusterStats)
        self._prev_progress = 0.0
        self._cur_stuck_cluster: Optional[int] = None
        self._step_count = 0
        self._stuck_steps_this_ep = 0
        self.load()

    def load(self):
        """Load persisted stuck stats from JSON."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                for wp_str, vals in data.items():
                    wp = int(wp_str)
                    self.stats[wp] = StuckClusterStats(**vals)
                logger.info(f"[STUCK_TRACKER] Loaded {len(data)} clusters from {self.save_path}")
            except Exception as e:
                logger.warning(f"[STUCK_TRACKER] Failed to load: {e}")
                self.stats = defaultdict(StuckClusterStats)
        else:
            logger.info("[STUCK_TRACKER] No prior stats, starting fresh.")

    def step_update(self, wp_idx: int, is_stuck: bool, moved_forward: bool,
                    step_reward: float, speed: float, crashed: bool,
                    reversed_flag: bool, offtrack: bool):
        """Per-step update of stuck tracking."""
        self._step_count += 1
        s = self.stats[wp_idx]
        s.total_steps += 1
        # --- use rich signals for stuck detection ---
        if crashed:
            s.stuck_steps += 2  # crashes near a WP are high-signal
            self._stuck_steps_this_ep += 2
        if offtrack:
            s.stuck_steps += 1
            self._stuck_steps_this_ep += 1
        if reversed_flag:
            s.stuck_steps += 1  # reversed driving = likely stuck
            self._stuck_steps_this_ep += 1
        if is_stuck:
            s.stuck_steps += 1
            self._stuck_steps_this_ep += 1
            if self._cur_stuck_cluster is None:
                self._cur_stuck_cluster = wp_idx
        elif moved_forward:
            self._cur_stuck_cluster = None
        # track speed/reward for adaptive boosting
        if not hasattr(s, '_speed_sum'):
            s._speed_sum = 0.0
            s._reward_sum = 0.0
            s._step_samples = 0
        s._speed_sum += speed
        s._reward_sum += step_reward
        s._step_samples += 1

    def episode_update(self, entry_wp: int, ep_return: float,
                       ep_progress: float, escaped_stuck: bool):
        """End-of-episode update for stuck clusters."""
        # Use ep_progress to weight how important this stuck cluster is
        # Low progress episodes at a WP = that WP is a severe bottleneck
        _progress_penalty = max(0.0, 1.0 - ep_progress / 100.0)  # 0..1
        s = self.stats[entry_wp]
        s.total_episodes += 1
        s.cumulative_reward += ep_return
        s.avg_return = s.cumulative_reward / s.total_episodes
        if escaped_stuck:
            s.breakout_episodes += 1
        # Adaptive boost: increase reward for stuck-prone waypoints
        br = s.breakout_rate
        if s.total_episodes >= 3:
            s.reward_boost_factor = 1.0 + 0.5 * (1.0 - br) ** 2
            s.entropy_boost = 0.02 * (1.0 - br)
        # Reset per-episode counters
        self._stuck_steps_this_ep = 0
        self._cur_stuck_cluster = None
        self._prev_progress = 0.0

    def get_annealing_params(self, wp_idx: int) -> dict:
        """Get reward boost and entropy boost for a waypoint."""
        s = self.stats[wp_idx]
        return {
            "reward_boost": s.reward_boost_factor,
            "entropy_boost": s.entropy_boost,
        }

    def get_early_term_threshold(self, wp_idx: int) -> int:
        """Adaptive early termination threshold based on stuck history."""
        s = self.stats[wp_idx]
        base = 200
        if s.total_episodes >= 5 and s.breakout_rate < 0.3:
            return max(50, base - int(100 * (1.0 - s.breakout_rate)))
        return base

    def summary(self) -> str:
        """Short summary string for logging."""
        n = sum(1 for s in self.stats.values() if s.total_episodes >= 3)
        worst_br = min((s.breakout_rate for s in self.stats.values()
                       if s.total_episodes >= 3), default=1.0)
        return f"clusters={n}, worst_br={worst_br:.3f}"

    def save_to_json(self, save_dir: str = "results") -> str:
        """Save stuck stats to JSON."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "stuck_stats.json")
        data = {}
        for wp, s in self.stats.items():
            data[str(wp)] = {
                "total_episodes": s.total_episodes,
                "breakout_episodes": s.breakout_episodes,
                "stuck_episodes": s.total_episodes - s.breakout_episodes,
                "breakout_rate": round(s.breakout_rate, 4),
                "avg_return": round(s.avg_return, 2),
                "reward_boost_factor": round(s.reward_boost_factor, 3),
                "entropy_boost": round(s.entropy_boost, 3),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def print_report(self):
        hot = sorted(
            [(wp, s) for wp, s in self.stats.items() if s.total_episodes >= 3],
            key=lambda x: x[1].breakout_rate,
        )[:10]
        logger.info("[STUCK_REPORT] Worst stuck clusters:")
        for wp, s in hot:
            logger.info(
                f"  WP {wp:3d}: br={s.breakout_rate:.3f} "
                f"({s.total_episodes} eps, avg_ret={s.avg_return:.1f}) "
                f"boost={s.reward_boost_factor:.2f}x "
                f"ent+={s.entropy_boost:.4f}"
            )
