# race_line_engine.py
# Multi-line optimal path engine: time-trial, obstacle-aware, bot-overtake.
# UPGRADED: object recognition + dimensions, bot motion projection,
#           car-dimension-aware apex offsets, object-permanence angle gating.
#
# REF: Hart et al. (1968) A*; TheRayG (2020) deepracer-log-analysis;
#      Gonzalez (2020); Coulom (2002) curvature calculus;
#      Heilmeier et al. (2020) minimum-curvature QP race line;
#      Waymo Motion Prediction (2021) constant-velocity projection.
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from config_loader import CFG

# ---------------------------------------------------------------------------
# Vehicle / object dimensions  (metres, 1/18-scale DeepRacer)
# ---------------------------------------------------------------------------
_CAR_WIDTH      = CFG.get("vehicle", {}).get("width",          0.20)
_CAR_HALF_W     = _CAR_WIDTH / 2.0                                     # 0.10 m
_CAR_SAFETY     = CFG.get("vehicle", {}).get("safety_lat",     0.05)  # lateral buffer
_SAFE_HALF_W    = _CAR_HALF_W + _CAR_SAFETY                           # 0.15 m
_CAR_LENGTH     = CFG.get("vehicle", {}).get("length",         0.28)
_CAR_FRONT_OVH  = _CAR_LENGTH / 2.0                                    # 0.14 m

# Object half-widths by type
_OBJ_HALF_W = {
    'bot':    _CAR_HALF_W,          # same as own car
    'cone':   CFG.get("vehicle", {}).get("cone_radius",     0.05),
    'static': 0.10,
    'curb':   0.02,
}


def _curvature_radius(wpts, idx, w=3):
    """Radius of curvature at waypoint idx using Menger formula.
    REF: Coulom (2002); Menger (1930).
    """
    n = len(wpts)
    if n < 5:
        return 999.0
    p0 = np.array(wpts[(idx - w) % n][:2], dtype=float)
    p1 = np.array(wpts[idx % n][:2],       dtype=float)
    p2 = np.array(wpts[(idx + w) % n][:2], dtype=float)
    d1 = p1 - p0
    d2 = p2 - p1
    cross = abs(d1[0]*d2[1] - d1[1]*d2[0])
    nm    = max(np.linalg.norm(d1), 1e-6)
    return 1.0 / (cross / (nm**3 + 1e-9) + 1e-6)


def _optimal_speed(r, C=1.5, vmin=0.5, vmax=4.0):
    return float(np.clip(math.sqrt(max(C * r, 0)), vmin, vmax))


def _cross_sign(wpts, i):
    """Returns +1 for left turn, -1 for right turn at waypoint i."""
    n  = len(wpts)
    p0 = np.array(wpts[(i-3) % n][:2], dtype=float)
    p1 = np.array(wpts[i][:2],         dtype=float)
    p2 = np.array(wpts[(i+3) % n][:2], dtype=float)
    d1 = p1 - p0
    d2 = p2 - p1
    return 1.0 if (d1[0]*d2[1] - d1[1]*d2[0]) > 0 else -1.0


def _apex_offset_car_aware(r, cross, track_half_w):
    """Apex-seeking lateral offset that accounts for car half-width.

    The car EDGE (not centre) must not exceed the track boundary.
    max_offset = track_half_w - _SAFE_HALF_W   (guarantees clearance).
    REF: Heilmeier et al. (2020) minimum-curvature QP.
    """
    if r > 200:
        return 0.0
    max_off = max(0.0, track_half_w - _SAFE_HALF_W)
    if r < 20:
        tightness = min(max_off, 8.0 / r)
    elif r < 50:
        tightness = min(max_off * 0.7, 4.0 / r)
    else:
        tightness = min(max_off * 0.3, 0.5)
    return float(-cross * tightness)


class ObjectRecord:
    """Describes a detected object for race-line computation.

    Supports:
      'bot'    — another DeepRacer (has heading + speed)
      'cone'   — static traffic cone
      'static' — generic static obstacle
      'curb'   — track boundary marker (treated as a constraint, not obstacle)

    Object-permanence:
      visible_angle_from() returns the angular span of the object corners
      from the car's current viewpoint.  A larger angle signals that the
      car is close or the object is large — the race-line shift scales with it.

    Bot motion projection (constant-velocity):
      projected_wp() finds the track waypoint the bot will occupy in
      `lookahead_t` seconds, so the overtake line is set up AHEAD of time.
      REF: Waymo Motion Prediction Challenge (2021).
    """

    def __init__(self, obj_type='static', x=0.0, y=0.0,
                 heading=0.0, speed=0.0, wp_idx=0):
        self.obj_type = obj_type
        self.x        = float(x)
        self.y        = float(y)
        self.heading  = float(heading)
        self.speed    = float(speed)
        self.wp_idx   = int(wp_idx)

    @property
    def half_width(self):
        return _OBJ_HALF_W.get(self.obj_type, 0.10)

    def safe_clearance(self):
        """Total lateral space needed: obj half-w + own car half-w + safety buffer."""
        return self.half_width + _SAFE_HALF_W

    def corner_points(self):
        """4 bounding-box corners in world frame."""
        hw = self.half_width
        hl = hw * 1.4 if self.obj_type == 'bot' else hw
        c, s = math.cos(self.heading), math.sin(self.heading)
        local = np.array([[ hl, hw], [ hl,-hw], [-hl,-hw], [-hl, hw]], dtype=float)
        rot   = np.array([[c, -s], [s, c]])
        return (rot @ local.T).T + np.array([self.x, self.y])

    def visible_angle_from(self, car_x, car_y):
        """Angular span of object corners from car position.
        Larger -> car is close or object is big -> scale race-line shift up.
        """
        corners  = self.corner_points()
        car_pos  = np.array([car_x, car_y])
        angles   = [math.atan2(c[1]-car_pos[1], c[0]-car_pos[0]) for c in corners]
        max_span = 0.0
        for i in range(len(angles)):
            for j in range(i+1, len(angles)):
                d = abs(angles[i] - angles[j])
                max_span = max(max_span, min(d, 2*math.pi - d))
        return max_span

    def projected_position(self, dt=1.5):
        """Constant-velocity projection for bots."""
        if self.obj_type == 'bot' and self.speed > 0.1:
            return (self.x + self.speed * math.cos(self.heading) * dt,
                    self.y + self.speed * math.sin(self.heading) * dt)
        return self.x, self.y

    def projected_wp_idx(self, wpts, dt=1.5):
        """Returns nearest waypoint index to projected position."""
        px, py = self.projected_position(dt)
        target = np.array([px, py])
        dists  = np.linalg.norm(np.array([w[:2] for w in wpts], dtype=float) - target, axis=1)
        return int(np.argmin(dists))


class RaceLine:
    # REF: Garlick & Middleditch (2022). Real-time optimal racing line. IEEE Trans. Games.
    """A single candidate racing line with per-waypoint offset, speed target, heading."""

    def __init__(self, wpts, lateral_offsets: np.ndarray,
                 name='time_trial', track_half_w: float = 0.3):
        self.name         = name
        self.n            = len(wpts)
        self.wpts         = np.array([w[:2] for w in wpts], dtype=float)
        self.track_half_w = track_half_w
        # Clamp: car edge must not exceed boundary
        max_off = max(0.0, track_half_w - _SAFE_HALF_W)
        self.offsets      = np.clip(lateral_offsets, -max_off, max_off)
        self.speeds       = np.zeros(self.n)
        self.headings     = np.zeros(self.n)
        self._compute()

    def _compute(self):
        n = self.n
        for i in range(n):
            r = _curvature_radius(self.wpts, i)
            self.speeds[i] = _optimal_speed(r)
            p0 = self.wpts[(i-1) % n]
            p2 = self.wpts[(i+1) % n]
            self.headings[i] = math.atan2(p2[1]-p0[1], p2[0]-p0[0])

    def reward(self, wp_idx, car_lat_pos, car_speed, car_heading, track_width) -> float:
        """Reward for following this race line at wp_idx.
        car_lat_pos: signed fraction of half-width (left positive).
        """
        target_offset  = self.offsets[wp_idx % self.n]
        target_speed   = self.speeds[wp_idx % self.n]
        target_heading = self.headings[wp_idx % self.n]
        half_w = track_width / 2.0
        lat_r  = math.exp(-0.5 * ((car_lat_pos - target_offset) / max(half_w, 0.1))**2)
        spd_r  = math.exp(-0.5 * ((car_speed   - target_speed)  / 0.6)**2)
        hdg_diff = abs(math.atan2(math.sin(car_heading - target_heading),
                                   math.cos(car_heading - target_heading)))
        hdg_r  = math.exp(-0.5 * (hdg_diff / 0.3)**2)
        return 0.45*lat_r + 0.35*spd_r + 0.20*hdg_r


class MultiRaceLineEngine:
    """
    Three race lines dynamically updated with object recognition + dimensions:
      1. time_trial     — pure min-curvature apex line (car-width-aware)
      2. obstacle_avoid — shifted away from static obstacles/curbs using
                          object half-width + car safe clearance bubble
      3. bot_overtake   — positions car on opposite side of bot's projected
                          path, updated T=1.5 s ahead of bot's location

    Object-permanence: visible_angle_from() scales the line shift so a
    nearby large object shifts the line MORE than a distant small one.

    REF: Heilmeier et al. (2020) min-curvature QP;
         Waymo Motion Prediction (2021) constant-velocity projection;
         Gonzalez (2020) DeepRacer reward shaping;
         Haarnoja et al. (2018) entropy-weighted action selection.
    """
    LINE_TIME_TRIAL = 'time_trial'
    LINE_OBSTACLE   = 'obstacle_avoid'
    LINE_BOT        = 'bot_overtake'

    def __init__(self, waypoints: list, track_width: float = 0.6):
        self.waypoints   = waypoints
        self.track_width = track_width
        self.half_w      = track_width / 2.0
        self.n           = len(waypoints)
        self._lines: Dict[str, RaceLine] = {}
        self._initialized = False
        self._objects: List[ObjectRecord] = []

    def initialize(self):
        if not self.waypoints or self.n < 5:
            return
        tt_offsets  = self._compute_apex_offsets()
        self._lines[self.LINE_TIME_TRIAL] = RaceLine(
            self.waypoints, tt_offsets, self.LINE_TIME_TRIAL, self.half_w)
        self._lines[self.LINE_OBSTACLE]   = RaceLine(
            self.waypoints, tt_offsets.copy(), self.LINE_OBSTACLE, self.half_w)
        bot_offsets = -tt_offsets * 0.6
        self._lines[self.LINE_BOT]        = RaceLine(
            self.waypoints, bot_offsets, self.LINE_BOT, self.half_w)
        self._initialized = True

    def _compute_apex_offsets(self) -> np.ndarray:
        """Car-dimension-aware apex offsets.
        The car EDGE stays within track_half_w.  REF: Heilmeier et al. (2020).
        """
        offsets = np.zeros(self.n)
        for i in range(self.n):
            r    = _curvature_radius(self.waypoints, i, w=4)
            cs   = _cross_sign(self.waypoints, i)
            offsets[i] = _apex_offset_car_aware(r, cs, self.half_w)
        # smooth
        try:
            from scipy.ndimage import uniform_filter1d
            offsets = uniform_filter1d(offsets, size=5, mode='wrap')
        except Exception:
            pass
        return offsets

    # ------------------------------------------------------------------
    # Object registration
    # ------------------------------------------------------------------
    def update_objects(self, objects):
        """Register current frame's objects (bots, cones, static, curbs).
        objects: List[ObjectRecord] or List[dict]
        Called each step from run.py before get_combined_reward().
        """
        parsed = []
        for o in objects:
            if isinstance(o, ObjectRecord):
                parsed.append(o)
            elif isinstance(o, dict):
                parsed.append(ObjectRecord(**{k: o[k] for k in
                    ('obj_type','x','y','heading','speed','wp_idx') if k in o}))
        self._objects = parsed

    # ------------------------------------------------------------------
    # Dynamic line updates
    # ------------------------------------------------------------------
    def update_obstacle_line(self, wp_idx, lidar_min, barrier_prox,
                             nearest_obj, context, car_x=0.0, car_y=0.0,
                             swin_clearance=None):   # v1.1.0
        """Shift obstacle-avoidance line using:
          - Object type -> known half-width -> minimum safe clearance
          - Object-permanence angle -> scale shift magnitude
          - context: 0=clear,1=curb,2=obstacle,3=corner,4=straight
        """
        if not self._initialized:
            return
        tt_off = self._lines[self.LINE_TIME_TRIAL].offsets[wp_idx % self.n]
        max_off = max(0.0, self.half_w - _SAFE_HALF_W)

        # Find closest non-bot obstacle to this WP
        closest_obs   = None
        closest_dist  = float('inf')
        for obj in self._objects:
            if obj.obj_type in ('cone', 'static', 'curb'):
                d = math.hypot(obj.x - self.waypoints[wp_idx % self.n][0],
                               obj.y - self.waypoints[wp_idx % self.n][1])
                if d < closest_dist:
                    closest_dist, closest_obs = d, obj

        if closest_obs is not None and closest_dist < 1.5:
            clearance = closest_obs.safe_clearance()
            raw_shift = max(clearance / self.half_w,
                            min(max_off, 1.2 / max(closest_dist, 0.3)))
            # Scale by angular span (object-permanence)
            if car_x != 0.0 or car_y != 0.0:
                vis = closest_obs.visible_angle_from(car_x, car_y)
                raw_shift *= min(2.0, 1.0 + vis / math.pi)
            # v1.1.0: use Swin left/right clearance to pick avoidance direction
            if (swin_clearance is not None and len(swin_clearance) >= 3
                    and closest_obs is not None):
                left_c  = float(swin_clearance[1])
                right_c = float(swin_clearance[2])
                # steer toward clearer side
                _avoid_sign = 1.0 if left_c >= right_c else -1.0
            else:
                _avoid_sign = np.sign(-tt_off + 1e-6)
            new_off = tt_off + raw_shift * _avoid_sign 
            new_off = float(np.clip(new_off, -max_off, max_off))
        elif context == 1 and lidar_min < 0.4:
            # Curb: pull toward center proportionally
            new_off = float(tt_off * (1.0 - barrier_prox))
        else:
            # Decay back to time-trial
            alpha   = 0.1
            cur     = self._lines[self.LINE_OBSTACLE].offsets[wp_idx % self.n]
            new_off = float(alpha * tt_off + (1 - alpha) * cur)

        self._lines[self.LINE_OBSTACLE].offsets[wp_idx % self.n] = \
            float(np.clip(new_off, -max_off, max_off))

    def update_bot_line(self, wp_idx: int, bot_progress: float,
                         own_progress: float,
                         bot_x: float = 0.0, bot_y: float = 0.0,
                         bot_heading: float = 0.0, bot_speed: float = 0.0,
                         car_x: float = 0.0, car_y: float = 0.0):
        """Position car to overtake bot using:
          - Projected bot position (T+1.5 s) to set up gap AHEAD of time
          - Bot half-width for minimum safe passing distance
          - Object-permanence angle to scale urgency
        REF: Waymo Motion Prediction (2021); Yang (2023) overtake shaping.
        """
        if not self._initialized:
            return
        gap     = own_progress - bot_progress
        max_off = max(0.0, self.half_w - _SAFE_HALF_W)

        if abs(gap) < 10.0:
            # Build an ObjectRecord for the bot
            bot_obj = ObjectRecord('bot', bot_x, bot_y, bot_heading, bot_speed, wp_idx)
            # Projected WP index
            proj_wp = bot_obj.projected_wp_idx(self.waypoints, dt=1.5)
            # Bot's projected lateral position at that WP
            wc  = np.array(self.waypoints[proj_wp % self.n][:2], dtype=float)
            wn  = np.array(self.waypoints[(proj_wp+1) % self.n][:2], dtype=float)
            tang = wn - wc
            tang_len = np.linalg.norm(tang) + 1e-8
            norm_unit = np.array([-tang[1], tang[0]]) / tang_len
            px, py    = bot_obj.projected_position(dt=1.5)
            bot_lat   = float(np.dot(np.array([px, py]) - wc, norm_unit))

            # Minimum safe passing offset
            clearance    = bot_obj.safe_clearance()
            overtake_off = bot_lat + clearance * (-1.0 if bot_lat >= 0 else 1.0)
            overtake_off = float(np.clip(overtake_off, -max_off, max_off))

            # Scale by object-permanence angle
            if car_x != 0.0 or car_y != 0.0:
                vis = bot_obj.visible_angle_from(car_x, car_y)
                alpha_vis = min(1.0, 0.5 + vis / math.pi)
            else:
                alpha_vis = max(0.0, 1.0 - abs(gap) / 10.0)

            cur = self._lines[self.LINE_BOT].offsets[wp_idx % self.n]
            new_off = alpha_vis * overtake_off + (1 - alpha_vis) * cur
            self._lines[self.LINE_BOT].offsets[wp_idx % self.n] = \
                float(np.clip(new_off, -max_off, max_off))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def get_combined_reward(
        self,
        wp_idx: int,
        car_lat_pos: float,
        car_speed: float,
        car_heading: float,
        track_width: float,
        context: int,
        lidar_min: float,
        nearest_obj: float,
        bot_progress: float = 0.0,
        own_progress: float = 0.0,
        car_x: float = 0.0,
        car_y: float = 0.0,
        swin_clearance: np.ndarray = None,   # v1.1.0: [front,left,right,rear]
    ) -> Tuple[float, Dict]:
        if not self._initialized or not self._lines:
            return 0.0, {}
        w_tt, w_obs, w_bot = 0.7, 0.2, 0.1
        if context == 2:   w_tt, w_obs, w_bot = 0.3, 0.6, 0.1
        elif context == 1: w_tt, w_obs, w_bot = 0.5, 0.4, 0.1
        elif context == 3: w_tt, w_obs, w_bot = 0.6, 0.3, 0.1
        if bot_progress > 0 and abs(own_progress - bot_progress) < 15.0:
            bf    = max(0.0, 1.0 - abs(own_progress-bot_progress)/15.0) * 0.3
            w_bot += bf
            tot   = w_tt + w_obs + w_bot
            w_tt /= tot; w_obs /= tot; w_bot /= tot
        line_rewards = {}
        total_r = 0.0
        for lname, wt in [(self.LINE_TIME_TRIAL, w_tt),
                           (self.LINE_OBSTACLE,   w_obs),
                           (self.LINE_BOT,        w_bot)]:
            if lname in self._lines:
                r = self._lines[lname].reward(
                    wp_idx, car_lat_pos, car_speed, car_heading, track_width)
                line_rewards[lname] = r
                total_r += wt * r
        # Proximity penalty
        _lidar_s  = float(np.clip(lidar_min, 0.0, 1.0))
        _obj_s    = min(1.0, nearest_obj/1.5) if nearest_obj > 0 else 1.0
        _prox_raw = 0.5 + 0.5 * min(_lidar_s, _obj_s)
        # v1.1.0: Swin clearance gate — multiplicative, no freeze trap
        if swin_clearance is not None and len(swin_clearance) >= 3:
            front_c = float(np.clip(swin_clearance[0], 0.0, 1.0))
            lat_c   = float(np.clip(min(swin_clearance[1],swin_clearance[2]), 0.0, 1.0))
            swin_gate = (0.4 + 0.6*front_c) * (0.7 + 0.3*lat_c)
        else:
            swin_gate = 1.0
        _prox = float(np.clip(_prox_raw * swin_gate, 0.05, 1.0))
        total_r  *= _prox
        line_rewards['proximity'] = _prox
        line_rewards['swin_gate'] = swin_gate
        return float(total_r), line_rewards

    def get_target_speed(self, wp_idx: int, context: int) -> float:
        if not self._initialized or self.LINE_TIME_TRIAL not in self._lines:
            return 2.0
        line = (self._lines.get(self.LINE_OBSTACLE, self._lines[self.LINE_TIME_TRIAL])
                if context in (1, 2, 3) else self._lines[self.LINE_TIME_TRIAL])
        return float(line.speeds[wp_idx % self.n])

    def reset(self):
        self._initialized = False
        self._lines       = {}
        self._objects     = []


# ============================================================
# v1.2.0: BC Pilot interface additions
# ============================================================

def get_active_line_for_bc_pilot(
    engine,
    wp_idx: int,
    car_x: float,
    car_y: float,
    car_speed: float,
    car_heading_rad: float,
    context: int = 0,
    bot_progress: float = 0.0,
    own_progress: float = 0.0,
) -> dict:
    """v1.2.0: single query point for BCPilot.act() to get active race line data.

    context: 0=clear, 1=curb, 2=obstacle, 3=bot-nearby

    Returns dict with:
      target_offset  : signed lateral offset from centreline (m)
      target_speed   : optimal speed at this WP (m/s)
      target_heading : track tangent (rad)
      active_line    : which line is dominant
      brake_zone     : bool — upcoming WPs need lower speed (pre-braking cue)

    REF: Heilmeier et al. (2020) Race-line speed targets as natural braking cues.
    REF: Betz et al. (2022) Autonomous racing survey. arXiv:2202.07008.
    """
    import numpy as _np
    if not getattr(engine, '_initialized', False) or not getattr(engine, '_lines', {}):
        return dict(target_offset=0.0, target_speed=2.0, target_heading=0.0,
                    active_line='time_trial', brake_zone=False)

    LINE_TT  = 'time_trial'
    LINE_OBS = 'obstacle_avoid'
    LINE_BOT = 'bot_overtake'

    if context == 2 and LINE_OBS in engine._lines:
        active_name = LINE_OBS
    elif context == 3 and LINE_BOT in engine._lines and abs(own_progress - bot_progress) < 15.0:
        active_name = LINE_BOT
    else:
        active_name = LINE_TT

    if active_name not in engine._lines:
        active_name = LINE_TT
    if active_name not in engine._lines:
        return dict(target_offset=0.0, target_speed=2.0, target_heading=0.0,
                    active_line='time_trial', brake_zone=False)

    line = engine._lines[active_name]
    n    = engine.n
    idx  = wp_idx % n
    target_offset  = float(line.offsets[idx])
    target_speed   = float(line.speeds[idx])
    target_heading = float(line.headings[idx])

    # Brake-zone lookahead: speed-adaptive
    brake_zone = False
    la = max(6, min(24, int(car_speed * 6)))
    for j in range(1, la + 1):
        nidx = (idx + j) % n
        if line.speeds[nidx] < car_speed * 0.85:
            brake_zone = True
            break

    return dict(
        target_offset=target_offset,
        target_speed=target_speed,
        target_heading=target_heading,
        active_line=active_name,
        brake_zone=brake_zone,
    )


def get_speed_targets_array(engine) -> "np.ndarray | None":
    """v1.2.0: export time_trial speed targets for BrakeField.set_waypoints().
    Returns np.ndarray(N,) or None if engine not initialized.
    """
    import numpy as _np
    if not getattr(engine, '_initialized', False):
        return None
    LINE_TT = 'time_trial'
    if LINE_TT not in getattr(engine, '_lines', {}):
        return None
    return engine._lines[LINE_TT].speeds.copy()
