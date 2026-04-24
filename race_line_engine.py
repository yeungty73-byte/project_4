# race_line_engine.py
# Multi-line optimal path engine: time-trial, obstacle-aware, bot-overtake
# REF: Hart et al. (1968) A*; TheRayG (2020) deepracer-log-analysis;
#      Gonzalez (2020); Coulom (2002) curvature calculus
import math, numpy as np
from typing import List, Tuple, Optional, Dict

def _curvature_radius(wpts, idx, w=3):
    n = len(wpts)
    if n < 3: return 999.0
    p0 = np.array(wpts[(idx-w)%n][:2])
    p1 = np.array(wpts[idx%n][:2])
    p2 = np.array(wpts[(idx+w)%n][:2])
    d1 = p1-p0; d2 = p2-p1
    cross = abs(d1[0]*d2[1]-d1[1]*d2[0])
    nm = max(np.linalg.norm(d1),1e-6)
    return 1.0/(cross/(nm**3+1e-9)+1e-6)

def _optimal_speed(r, C=1.5, vmin=0.5, vmax=4.0):
    return float(np.clip(math.sqrt(max(C*r,0)), vmin, vmax))

class RaceLine:
# REF: Garlick, J. & Middleditch, A. (2022). Real-time optimal racing line generation. IEEE Trans. Games.
    """A single candidate racing line with per-waypoint offset, speed target, heading."""
    def __init__(self, wpts, lateral_offsets: np.ndarray, name='time_trial'):
        self.name = name
        self.n = len(wpts)
        self.wpts = np.array([w[:2] for w in wpts])
        self.offsets = np.clip(lateral_offsets, -0.9, 0.9)  # fraction of half-width
        self.speeds = np.zeros(self.n)
        self.headings = np.zeros(self.n)
        self._compute()

    def _compute(self):
        n = self.n
        for i in range(n):
            r = _curvature_radius(self.wpts, i)
            self.speeds[i] = _optimal_speed(r)
            p0 = self.wpts[(i-1)%n]; p2 = self.wpts[(i+1)%n]
            self.headings[i] = math.atan2(p2[1]-p0[1], p2[0]-p0[0])

    def reward(self, wp_idx, car_lat_pos, car_speed, car_heading, track_width) -> float:
        """Reward for following this race line at wp_idx.
        car_lat_pos: signed fraction of half-width (left=positive)
        """
        target_offset = self.offsets[wp_idx]
        target_speed = self.speeds[wp_idx]
        target_heading = self.headings[wp_idx]
        # lateral proximity reward: Gaussian around target offset
        lat_err = car_lat_pos - target_offset
        lat_r = math.exp(-0.5*(lat_err/(0.5*track_width))**2)
        # speed proximity reward: Gaussian around optimal speed
        spd_err = car_speed - target_speed
        spd_r = math.exp(-0.5*(spd_err/0.6)**2)
        # heading alignment reward
        hdg_diff = abs(math.atan2(
            math.sin(car_heading - target_heading),
            math.cos(car_heading - target_heading)
        ))
        hdg_r = math.exp(-0.5*(hdg_diff/0.3)**2)
        return (0.45*lat_r + 0.35*spd_r + 0.20*hdg_r)


class MultiRaceLineEngine:
    """
    Maintains multiple race lines:
      1. time_trial: pure inner-apex optimal line (minimize curvature)
      2. obstacle_avoid: dynamically shifted away from detected obstacles/barriers
      3. bot_overtake: set up overtake angle on the bot's racing line
    Combines them with a soft-max blend based on context.
    REF: TheRayG (2020); Gonzalez (2020); Haarnoja (2018) entropy-weighted action selection
    """
    LINE_TIME_TRIAL = 'time_trial'
    LINE_OBSTACLE   = 'obstacle_avoid'
    LINE_BOT        = 'bot_overtake'

    def __init__(self, waypoints: list, track_width: float = 0.6):
        self.waypoints = waypoints
        self.track_width = track_width
        self.n = len(waypoints)
        self._lines: Dict[str, RaceLine] = {}
        self._obstacle_offsets = np.zeros(self.n)  # dynamic, updated per step
        self._bot_offsets      = np.zeros(self.n)
        self._initialized = False

    def initialize(self):
        if not self.waypoints or self.n < 5:
            return
        # 1. Time-trial: apex-seeking inside-out offset
        tt_offsets = self._compute_apex_offsets()
        self._lines[self.LINE_TIME_TRIAL] = RaceLine(self.waypoints, tt_offsets, self.LINE_TIME_TRIAL)
        # 2. Obstacle: start identical to time_trial, will be modified dynamically
        self._lines[self.LINE_OBSTACLE] = RaceLine(self.waypoints, tt_offsets.copy(), self.LINE_OBSTACLE)
        # 3. Bot overtake: start on opposite side of track from time-trial
        bot_offsets = -tt_offsets * 0.6  # opposite side, softer
        self._lines[self.LINE_BOT] = RaceLine(self.waypoints, bot_offsets, self.LINE_BOT)
        self._initialized = True

    def _compute_apex_offsets(self) -> np.ndarray:
        """Compute apex-seeking lateral offsets per waypoint.
        Tight corners -> go wide before, apex inside, wide after (negative on left-turn).
        REF: Coulom (2002); Gonzalez (2020)
        """
        offsets = np.zeros(self.n)
        for i in range(self.n):
            r = _curvature_radius(self.waypoints, i, w=4)
            # curvature sign: cross product determines left vs right turn
            p0 = np.array(self.waypoints[(i-3)%self.n][:2])
            p1 = np.array(self.waypoints[i][:2])
            p2 = np.array(self.waypoints[(i+3)%self.n][:2])
            d1 = p1-p0; d2 = p2-p1
            cross_sign = 1.0 if (d1[0]*d2[1]-d1[1]*d2[0]) > 0 else -1.0
            # tight corners: offset toward apex (inside)
            if r < 20:  tightness = min(0.7, 8.0/r)
            elif r < 50: tightness = min(0.4, 4.0/r)
            else:        tightness = 0.05  # near-straight: stay near center
            offsets[i] = -cross_sign * tightness
        # smooth offsets
        from scipy.ndimage import uniform_filter1d
        try:
            offsets = uniform_filter1d(offsets, size=5, mode='wrap')
        except Exception:
            pass
        return offsets

    def update_obstacle_line(self, wp_idx: int, lidar_min: float,
                              barrier_prox: float, nearest_obj: float, context: int):
        """Dynamically shift obstacle-avoidance line away from detected hazards.
        context: 1=curb, 2=obstacle, 3=corner, 0=clear, 4=straight
        """
        if not self._initialized: return
        tt_offset = self._lines[self.LINE_TIME_TRIAL].offsets[wp_idx]
        if context == 2 and nearest_obj < 1.5:  # obstacle close
            # shift opposite to time_trial
            min_clear = (0.10+0.15)/max(self.track_width/2,0.1)
            shift = max(min_clear, min(0.5, 1.2/max(nearest_obj,0.3)))
            self._lines[self.LINE_OBSTACLE].offsets[wp_idx] = -tt_offset + shift*np.sign(-tt_offset+1e-6)
        elif context == 1 and lidar_min < 0.4:  # near curb
            # pull back toward center
            self._lines[self.LINE_OBSTACLE].offsets[wp_idx] = tt_offset * (1.0 - barrier_prox)
        else:
            # fall back to time_trial
            alpha = 0.1
            self._lines[self.LINE_OBSTACLE].offsets[wp_idx] = (
                alpha * tt_offset + (1-alpha)*self._lines[self.LINE_OBSTACLE].offsets[wp_idx]
            )
        self._lines[self.LINE_OBSTACLE].offsets[wp_idx] = float(
            np.clip(self._lines[self.LINE_OBSTACLE].offsets[wp_idx], -0.9, 0.9)
        )

    def update_bot_line(self, wp_idx: int, bot_progress: float, own_progress: float,
                         bot_x: float = 0.0, bot_y: float = 0.0):
        """Update bot-overtake line: position to overtake safely.
        Places car on opposite side from bot when within 10% progress of bot.
        REF: Yang (2023) overtake shaping
        """
        if not self._initialized: return
        gap = own_progress - bot_progress
        if abs(gap) < 10.0:  # bot nearby
            # estimate bot lateral position from its waypoint offset (use opposite)
            # --- Use bot_x, bot_y if available for precise bot lateral position ---
            BOT_HALF_WIDTH = 0.10  # ~0.20m car width / 2 (1/18 scale DeepRacer)
            if bot_x != 0.0 or bot_y != 0.0:
                # compute bot lateral position on track from (bot_x, bot_y)
                n = len(self.waypoints)
                wp_c = np.array(self.waypoints[wp_idx % n][:2])
                wp_n = np.array(self.waypoints[(wp_idx + 1) % n][:2])
                tang = wp_n - wp_c
                tang_len = np.linalg.norm(tang) + 1e-8
                tang_unit = tang / tang_len
                norm_unit = np.array([-tang_unit[1], tang_unit[0]])  # left normal
                bot_vec = np.array([bot_x, bot_y]) - wp_c
                bot_lat_pos = float(np.dot(bot_vec, norm_unit))  # signed lateral
                bot_est_offset = bot_lat_pos
                # store bot boundaries for downstream use
                self._bot_left_edge = bot_lat_pos + BOT_HALF_WIDTH
                self._bot_right_edge = bot_lat_pos - BOT_HALF_WIDTH
            else:
                bot_est_offset = self._lines[self.LINE_TIME_TRIAL].offsets[wp_idx]
                self._bot_left_edge = bot_est_offset + BOT_HALF_WIDTH
                self._bot_right_edge = bot_est_offset - BOT_HALF_WIDTH
            # position ourselves on opposite side
            overtake_offset = -np.sign(bot_est_offset) * 0.6
            alpha = max(0.0, 1.0 - abs(gap)/10.0)  # stronger as gap shrinks
            cur = self._lines[self.LINE_BOT].offsets[wp_idx]
            self._lines[self.LINE_BOT].offsets[wp_idx] = float(
                np.clip(alpha*overtake_offset + (1-alpha)*cur, -0.9, 0.9)
            )

    def get_combined_reward(
        self,
        wp_idx: int,
        car_lat_pos: float,  # signed lateral fraction
        car_speed: float,
        car_heading: float,
        track_width: float,
        context: int,
        lidar_min: float,
        nearest_obj: float,
        bot_progress: float = 0.0,
        own_progress: float = 0.0,
    ) -> Tuple[float, Dict]:
        """Compute blended multi-line reward with context-aware weights.
        Returns (combined_reward, {per_line_rewards})
        """
        if not self._initialized or not self._lines:
            return 0.0, {}
        # Context-based line weights
        # clear/straight: follow time_trial
        # obstacle/curb: emphasize obstacle-avoid line
        # bot nearby: blend in bot-overtake line
        w_tt  = 0.7
        w_obs = 0.2
        w_bot = 0.1
        if context == 2:   w_tt, w_obs, w_bot = 0.3, 0.6, 0.1  # obstacle
        elif context == 1: w_tt, w_obs, w_bot = 0.5, 0.4, 0.1  # curb
        elif context == 3: w_tt, w_obs, w_bot = 0.6, 0.3, 0.1  # corner
        # bot proximity modulation
        if bot_progress > 0 and abs(own_progress - bot_progress) < 15.0:
            bot_factor = max(0.0, 1.0 - abs(own_progress-bot_progress)/15.0) * 0.3
            w_bot += bot_factor
            total = w_tt + w_obs + w_bot
            w_tt  /= total; w_obs /= total; w_bot /= total
        line_rewards = {}
        total_r = 0.0
        for line_name, weight in [
            (self.LINE_TIME_TRIAL, w_tt),
            (self.LINE_OBSTACLE,   w_obs),
            (self.LINE_BOT,        w_bot),
        ]:
            if line_name in self._lines:
                r = self._lines[line_name].reward(
                    wp_idx, car_lat_pos, car_speed, car_heading, track_width
                )
                line_rewards[line_name] = r
                total_r += weight * r
        # --- proximity penalty: penalize high speed near obstacles/curbs ---
        # lidar_min ~0..1 (0=touching wall), nearest_obj ~0..inf (metres)
        _lidar_safe = min(1.0, max(0.0, lidar_min))  # clamp 0-1
        _obj_safe = min(1.0, nearest_obj / 1.5) if nearest_obj > 0 else 0.0
        _prox_factor = 0.5 + 0.5 * min(_lidar_safe, _obj_safe)  # 0.5..1.0
        total_r *= _prox_factor
        line_rewards['proximity'] = _prox_factor
        return float(total_r), line_rewards

    def get_target_speed(self, wp_idx: int, context: int) -> float:
        if not self._initialized or self.LINE_TIME_TRIAL not in self._lines:
            return 2.0
        if context in (1, 2, 3):
            line = self._lines.get(self.LINE_OBSTACLE, self._lines[self.LINE_TIME_TRIAL])
        else:
            line = self._lines[self.LINE_TIME_TRIAL]
        return float(line.speeds[wp_idx % self.n])

    def reset(self):
        self._initialized = False
        self._lines = {}
