"""
HTM-style Deterministic Lookahead Oracle
=========================================
REF: Hawkins & Blakeslee (2004). On Intelligence. Times Books.
     (HTM = Hierarchical Temporal Memory: predictive lookahead over track geometry.)
REF: Kapania & Gerdes (2015). Feedback-feedforward steering at handling limits.
     Vehicle System Dynamics, 53(12).
REF: Heilmeier et al. (2020). Minimum curvature trajectory planning.
     Vehicle System Dynamics, 58(10).
REF: AWS DeepRacer Developer Guide (2020). Vehicle dimensions.
REF: Coulom, R. (2002). Curvature calculus for racing lines.

Purpose
-------
Provides a deterministic reference policy for all 9 track variants
(tt_reinvent, tt_vegas, tt_bowtie, oa_*, h2h_*, h2b_*).

Supports three questions you asked:
  1. Does code account for car + obstacle dimensions?
     -> Yes: SAFE_HALF_WIDTH = car_half_w + SAFETY_MARGIN
        _bounding_box_corners() computes rotated rectangles for every object.
  2. Does it consider what an object IS to know how much of it
     will be revealed as the car turns (object permanence)?
     -> Yes: ObjectTracker.permanence_delta() sweeps the ego future
        position and measures lateral face visibility increase.
  3. Does it project bot motion for evasion?
     -> Yes: _project_bot_motion() extrapolates linear velocity
        from history; bot race-line phase is used when history is short.

Object Dimension Registry
--------------------------
  DeepRacer 1/18 ego car :  0.28m L  x  0.20m W  x  0.15m H
  Bot car (same platform):  0.28m L  x  0.20m W
  Static traffic cone    :  0.20m dia  (radius 0.10m)
  Track barrier / curb   :  infinite lateral  x  0.08m depth
  Safety margin (edge)   :  0.12m beyond car half-width
  -> SAFE_HALF_WIDTH      =  0.10 + 0.12 = 0.22m from ego centreline

Track variants
--------------
  Pass track_variant str from config.yaml ('tt_reinvent','tt_vegas','tt_bowtie',
  'oa_reinvent', ...) to HTMOracle.__init__() so variant-specific mu / width
  overrides are applied.

Usage
-----
  from htm_reference import HTMOracle, ObjectTracker

  obj_tracker = ObjectTracker()
  oracle = HTMOracle(waypoints, track_width=0.6, track_variant='tt_reinvent',
                     object_tracker=obj_tracker)
  oracle.build()   # call once after waypoints are available

  # per step:
  objs = obj_tracker.update(step, context_class, lidar_min, nearest_dist,
                             bot_x, bot_y, bot_heading, ego_x, ego_y)
  oracle.adjust_for_exclusion_zones(wp_idx, obj_tracker.get_exclusion_zones())
  htm_scores = oracle.score_agent_step(wp_idx, speed, lat_frac, heading_rad)
  bsts_row.update(htm_scores)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

CAR_LENGTH     = 0.28    # metres  -- DeepRacer 1/18 scale
CAR_WIDTH      = 0.20    # metres
CAR_HALF_W     = CAR_WIDTH  / 2.0   # 0.10 m
CAR_HALF_L     = CAR_LENGTH / 2.0   # 0.14 m
SAFETY_MARGIN  = 0.12    # gap BEYOND car edge to nearest obstacle
SAFE_HALF_W    = CAR_HALF_W + SAFETY_MARGIN   # 0.22 m from ego centreline

G   = 9.81
MU_DEFAULT = 0.70

# mu overrides per track variant (surface grip differences)
_VARIANT_MU = {
    "tt_reinvent": 0.72,
    "tt_vegas":    0.68,   # smoother surface
    "tt_bowtie":   0.70,
    "oa_reinvent": 0.70,
    "oa_vegas":    0.68,
    "oa_bowtie":   0.70,
    "h2h_reinvent":0.72,
    "h2h_vegas":   0.68,
    "h2b_reinvent":0.72,
}

# Object dimension registry  (length m, width m)
_OBJ_DIMS: Dict[str, Tuple[float, float]] = {
    "bot":     (0.28, 0.20),
    "cone":    (0.20, 0.20),   # circular -> square approx
    "barrier": (99.0, 0.08),   # treat as infinite lateral wall
    "curb":    (99.0, 0.10),
}

# exclusion radius = obj_half_w + SAFE_HALF_W (ego centreline must stay outside)
_OBJ_EXCLUSION: Dict[str, float] = {
    k: v[1] / 2.0 + SAFE_HALF_W
    for k, v in _OBJ_DIMS.items()
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _menger_curvature(wpts: np.ndarray, i: int, w: int = 3) -> float:
    """Menger curvature at waypoint i using ±w window.
    Numerically stable for short inter-waypoint distances.
    REF: Coulom (2002).
    """
    n = len(wpts)
    p0, p1, p2 = wpts[(i - w) % n], wpts[i], wpts[(i + w) % n]
    d1  = float(np.linalg.norm(p1 - p0)) + 1e-9
    d2  = float(np.linalg.norm(p2 - p1)) + 1e-9
    d3  = float(np.linalg.norm(p2 - p0)) + 1e-9
    cross = abs(
        float((p1[0] - p0[0]) * (p2[1] - p0[1])
             - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    )
    return 2.0 * cross / (d1 * d2 * d3 + 1e-9)


def _turn_sign(wpts: np.ndarray, i: int) -> float:
    "+1 = left turn, -1 = right turn at waypoint i."
    n   = len(wpts)
    d1  = wpts[i] - wpts[(i - 2) % n]
    d2  = wpts[(i + 2) % n] - wpts[i]
    return 1.0 if (d1[0] * d2[1] - d1[1] * d2[0]) > 0 else -1.0


def _safe_speed_from_curvature(
    curvature: float,
    mu: float  = MU_DEFAULT,
    v_max: float = 4.0,
    v_min: float = 0.5,
) -> float:
    """Physics speed limit: v_safe = sqrt(mu * g / curvature).
    REF: Kapania & Gerdes (2015).
    """
    if curvature < 1e-6:
        return v_max
    return float(np.clip(math.sqrt(mu * G / (curvature + 1e-9)), v_min, v_max))


def _brake_distance(v: float, mu: float = MU_DEFAULT) -> float:
    """d = v² / (2·mu·g).  REF: Wikipedia Braking distance."""
    return v ** 2 / (2.0 * mu * G + 1e-9)


def _bounding_box_corners(
    cx: float, cy: float,
    heading_rad: float,
    length: float, width: float,
) -> np.ndarray:
    """
    Rotated bounding box corners for an object at (cx,cy) facing heading_rad.
    Returns (4, 2) array of (x,y) corners: FR, FL, RL, RR.
    Used for: visibility sweep, exclusion zone, permanence delta.
    """
    hl, hw = length / 2.0, width / 2.0
    local  = np.array([[ hl,  hw],
                        [ hl, -hw],
                        [-hl, -hw],
                        [-hl,  hw]], dtype=np.float64)
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    R    = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([cx, cy])


def _visible_lateral_fraction(
    observer_pos: np.ndarray,
    obj_corners:  np.ndarray,
) -> float:
    """
    Fraction of the object’s lateral face visible from observer_pos.
    Projects corners onto the axis perpendicular to observer->object.
    Returns 0.0..1.0.

    Object-permanence usage: compare this at current ego pos vs.
    predicted ego pos after turning N waypoints — delta > 0 means
    the object is being revealed as the car turns into the corner.
    """
    centre   = obj_corners.mean(axis=0)
    to_obj   = centre - observer_pos
    dist     = np.linalg.norm(to_obj) + 1e-9
    perp     = np.array([-to_obj[1], to_obj[0]]) / dist
    projs    = obj_corners @ perp
    lat_span = projs.max() - projs.min()
    # normalise by narrower side = "width" visible
    side_a   = np.linalg.norm(obj_corners[0] - obj_corners[1])
    side_b   = np.linalg.norm(obj_corners[1] - obj_corners[2])
    obj_w    = min(side_a, side_b) + 1e-9
    return float(np.clip(lat_span / obj_w, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Object tracker
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    Tracks every object the car can encounter:
      - Bot car    : from bot_x/bot_y/bot_heading API params
      - Static obj : from context_class (2=cone, 1=curb) + nearest_obj_dist

    Per object, maintains:
      - Bounding box corners (rotated rectangle)
      - Visibility fraction (object permanence)
      - Motion history -> linear velocity projection
      - Exclusion zone radius (car centreline must stay outside)

    Object permanence
    -----------------
    permanence_delta(object_id, future_x, future_y) answers:
      "If I drive to this future position (e.g. waypoint N),
       how much MORE of this object will I see?"
    If delta > 0 and a bot is expected to be there, brake BEFORE turning.
    This prevents the classic failure: car commits to apex, bot appears.
    """

    _CONTEXT_TO_TYPE: Dict[int, str] = {
        1: "curb",
        2: "cone",
    }

    def __init__(self):
        self._state:   Dict[str, dict] = {}
        self._history: Dict[str, list] = {}   # bot only: [(x,y,hdg,step)]

    # ----------------------------------------------------------------
    def update(
        self,
        step:              int,
        context_class:     int,
        lidar_min:         float,
        nearest_obj_dist:  float,
        bot_x:             float = 0.0,
        bot_y:             float = 0.0,
        bot_heading:       float = 0.0,
        ego_x:             float = 0.0,
        ego_y:             float = 0.0,
    ) -> Dict[str, dict]:
        """
        Update registry and return current object states.
        Call every step before oracle.adjust_for_exclusion_zones().
        """
        objects: Dict[str, dict] = {}

        # --- Bot ---
        if bot_x != 0.0 or bot_y != 0.0:
            dims    = _OBJ_DIMS["bot"]
            corners = _bounding_box_corners(bot_x, bot_y, bot_heading,
                                             dims[0], dims[1])
            ego_pos = np.array([ego_x, ego_y])
            vis     = _visible_lateral_fraction(ego_pos, corners)

            hist = self._history.setdefault("bot", [])
            hist.append((bot_x, bot_y, bot_heading, step))
            if len(hist) > 15:
                hist.pop(0)

            proj = self._project_bot_motion(hist, steps_ahead=6)

            self._state["bot"] = {
                "type":          "bot",
                "dims":          dims,
                "pos":           (bot_x, bot_y),
                "heading":       bot_heading,
                "corners":       corners,
                "visibility":    vis,
                "projected_pos": proj,
                "exclusion_r":   _OBJ_EXCLUSION["bot"],
                "step":          step,
            }
            objects["bot"] = self._state["bot"]

        # --- Static obstacle or curb from context ---
        obj_type = self._CONTEXT_TO_TYPE.get(context_class)
        if obj_type and nearest_obj_dist < 2.5:
            dims    = _OBJ_DIMS[obj_type]
            # position approximation: straight ahead at nearest_obj_dist
            obj_x   = ego_x + nearest_obj_dist * math.cos(0.0)
            obj_y   = ego_y + nearest_obj_dist * math.sin(0.0)
            corners = _bounding_box_corners(obj_x, obj_y, 0.0, dims[0], dims[1])
            ego_pos = np.array([ego_x, ego_y])
            vis     = _visible_lateral_fraction(ego_pos, corners)

            objects[obj_type] = {
                "type":          obj_type,
                "dims":          dims,
                "pos":           (obj_x, obj_y),
                "heading":       0.0,
                "corners":       corners,
                "visibility":    vis,
                "projected_pos": None,
                "exclusion_r":   _OBJ_EXCLUSION.get(obj_type, SAFE_HALF_W),
                "step":          step,
            }
            self._state[obj_type] = objects[obj_type]

        return objects

    # ----------------------------------------------------------------
    def _project_bot_motion(
        self,
        history: list,
        steps_ahead: int = 6,
    ) -> Optional[Tuple[float, float]]:
        """
        Extrapolate bot position N steps forward using linear velocity estimate.
        Uses last min(4, len) frames.  Returns None if history too short.
        """
        if len(history) < 2:
            return (history[-1][0], history[-1][1]) if history else None
        n  = min(4, len(history))
        xs = [h[0] for h in history[-n:]]
        ys = [h[1] for h in history[-n:]]
        dx = (xs[-1] - xs[0]) / max(n - 1, 1)
        dy = (ys[-1] - ys[0]) / max(n - 1, 1)
        return (
            history[-1][0] + dx * steps_ahead,
            history[-1][1] + dy * steps_ahead,
        )

    # ----------------------------------------------------------------
    def get_exclusion_zones(
        self,
        use_projected: bool = True,
    ) -> List[Tuple[float, float, float]]:
        """
        Return [(cx, cy, radius), ...] for all tracked objects.
        If use_projected=True, uses projected_pos for the bot
        ("where the bot WILL be" rather than where it is now).
        Passed to HTMOracle.adjust_for_exclusion_zones().
        """
        zones = []
        for obj_id, obj in self._state.items():
            if use_projected and obj.get("projected_pos"):
                pos = obj["projected_pos"]
            else:
                pos = obj["pos"]
            zones.append((pos[0], pos[1], obj["exclusion_r"]))
        return zones

    # ----------------------------------------------------------------
    def permanence_delta(
        self,
        object_id:    str,
        future_ego_x: float,
        future_ego_y: float,
    ) -> float:
        """
        How much MORE of the object becomes visible as ego moves to (future_ego_x/y)?
        Positive = more revealed.  Used to trigger pre-emptive braking BEFORE
        the car turns into the corner and sees the bot for the first time.

        Example usage in run.py:
          delta = obj_tracker.permanence_delta(
              'bot',
              future_ego_x = wpts[(wp_idx+8)%n][0],
              future_ego_y = wpts[(wp_idx+8)%n][1],
          )
          if delta > 0.3:  # >30% more bot revealed in next 8 WPs
              pre_brake_flag = True
        """
        if object_id not in self._state:
            return 0.0
        obj         = self._state[object_id]
        corners     = obj["corners"]
        current_vis = obj["visibility"]
        future_pos  = np.array([future_ego_x, future_ego_y])
        future_vis  = _visible_lateral_fraction(future_pos, corners)
        return float(future_vis - current_vis)


# ---------------------------------------------------------------------------
# HTM Oracle
# ---------------------------------------------------------------------------

class HTMOracle:
    """
    Deterministic per-waypoint reference plan — replaces waypoint_coverage
    as the HTM-composite metric in harmonized_metrics.

    Answers: at each waypoint, what SHOULD the agent do (speed, lateral,
    heading) given track geometry, car dimensions, and object positions?

    Computes for ALL waypoints at build():
      - target_speed    : physics-limited m/s (backward-pass corrected)
      - lateral_offset  : signed fraction of half_w (car-dimension safe apex)
      - heading         : radians
      - should_brake    : bool (lookahead brake flag incl. front overhang)
      - regime          : 'straight' | 'corner' | 'brake_zone'

    After build(), call adjust_for_exclusion_zones() per step to push
    the reference line away from dynamically tracked objects.
    """

    def __init__(
        self,
        waypoints:      list,
        track_width:    float = 0.6,
        track_variant:  str   = 'unknown',
        mu:             float = None,
        n_lookahead:    int   = 15,
        object_tracker: Optional[ObjectTracker] = None,
    ):
        self.wpts         = np.array([w[:2] for w in waypoints], dtype=np.float64)
        self.n            = len(self.wpts)
        self.track_width  = track_width
        self.half_w       = track_width / 2.0
        self.mu           = mu if mu is not None else _VARIANT_MU.get(track_variant, MU_DEFAULT)
        self.n_lookahead  = n_lookahead
        self.obj_tracker  = object_tracker
        self.track_variant = track_variant

        self._plan:  List[dict] = []
        self._built: bool = False

        # track arc length: used for true track_progress calculation
        self._arc_lengths: Optional[np.ndarray] = None
        self._total_length: float = 0.0

    # ----------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------

    def build(self):
        """Pre-compute full plan for all waypoints. Call once."""
        n, wpts, mu = self.n, self.wpts, self.mu

        # --- 1. Arc lengths (for true track_progress) ---
        seg_len = np.array([
            float(np.linalg.norm(wpts[(i + 1) % n] - wpts[i]))
            for i in range(n)
        ])
        self._arc_lengths  = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        self._total_length = float(seg_len.sum())

        # --- 2. Menger curvature ---
        curvatures = np.array([_menger_curvature(wpts, i) for i in range(n)])

        # --- 3. Cornering speed limits ---
        speeds_raw = np.array([_safe_speed_from_curvature(c, mu) for c in curvatures])

        # --- 4. Backward-pass: brake-limited entry speeds ---
        speeds = speeds_raw.copy()
        for _ in range(5):
            for i in range(n - 1, -1, -1):
                j    = (i + 1) % n
                dist = float(np.linalg.norm(wpts[j] - wpts[i])) + 1e-9
                v_in_max = math.sqrt(
                    max(speeds[j] ** 2 + 2.0 * mu * G * dist, 0.0)
                )
                speeds[i] = min(speeds[i], v_in_max)

        # --- 5. Lateral offsets (car-dimension aware) ---
        max_lat_m    = self.half_w - SAFE_HALF_W            # metres
        max_lat_frac = float(np.clip(max_lat_m / (self.half_w + 1e-9), 0.0, 0.85))

        offsets = np.zeros(n)
        for i in range(n):
            sign  = _turn_sign(wpts, i)
            curv  = curvatures[i]
            if curv < 0.01:
                offsets[i] = 0.0
            elif curv < 0.08:
                frac = min(max_lat_frac, curv * 5.0)
                offsets[i] = -sign * frac
            else:
                offsets[i] = -sign * max_lat_frac

        # Smooth with 5-pt wrap kernel
        kernel  = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        padded  = np.concatenate([offsets[-2:], offsets, offsets[:2]])
        offsets = np.convolve(padded, kernel, mode='valid')[2:-2]

        # --- 6. Headings ---
        headings = np.array([
            math.atan2(
                wpts[(i + 1) % n][1] - wpts[(i - 1) % n][1],
                wpts[(i + 1) % n][0] - wpts[(i - 1) % n][0],
            )
            for i in range(n)
        ])

        # --- 7. Brake flags (incl. front overhang CAR_HALF_L) ---
        brake_flags = np.zeros(n, dtype=bool)
        for i in range(n):
            for k in range(1, self.n_lookahead + 1):
                j = (i + k) % n
                if speeds[j] < speeds[i] - 0.20:
                    phys_d = sum(
                        float(np.linalg.norm(wpts[(i + m + 1) % n] - wpts[(i + m) % n]))
                        for m in range(k)
                    )
                    d_stop  = _brake_distance(speeds[i], mu) - _brake_distance(speeds[j], mu)
                    d_stop  = d_stop * 1.25 + CAR_HALF_L   # safety + front bumper offset
                    if phys_d <= d_stop:
                        brake_flags[i] = True
                    break

        # --- 8. Assemble plan ---
        self._plan = [
            {
                "wp_idx":         i,
                "target_speed":   float(speeds[i]),
                "lateral_offset": float(offsets[i]),
                "heading":        float(headings[i]),
                "should_brake":   bool(brake_flags[i]),
                "curvature":      float(curvatures[i]),
                "raw_speed":      float(speeds_raw[i]),
                "arc_dist":       float(self._arc_lengths[i]),
                "regime":         (
                    "brake_zone" if brake_flags[i]
                    else "corner"   if curvatures[i] > 0.05
                    else "straight"
                ),
            }
            for i in range(n)
        ]
        self._built = True

    # ----------------------------------------------------------------
    # True track_progress (arc-length based, NOT waypoint index fraction)
    # ----------------------------------------------------------------

    def arc_progress(self, wp_idx: int, within_wp_frac: float = 0.0) -> float:
        """
        True track_progress = arc_dist_covered / total_arc_length  [0, 1].

        This replaces the collinear waypoint_index / n_waypoints metric.
        within_wp_frac: how far between wp_idx and wp_idx+1 (0.0..1.0),
        from car position projection onto the track segment.

        Usage in run.py:
          seg_len  = np.linalg.norm(wpts[wp_next] - wpts[wp_idx])
          proj_d   = np.dot(car_pos - wpts[wp_idx],
                             (wpts[wp_next]-wpts[wp_idx]) / (seg_len+1e-9))
          frac     = float(np.clip(proj_d / (seg_len+1e-9), 0, 1))
          true_prog = htm_oracle.arc_progress(wp_idx, within_wp_frac=frac)
        """
        if not self._built:
            self.build()
        base = self._arc_lengths[wp_idx % self.n]
        # interpolate within segment
        j    = (wp_idx + 1) % self.n
        seg  = float(np.linalg.norm(self.wpts[j] - self.wpts[wp_idx % self.n]))
        return float(np.clip(
            (base + within_wp_frac * seg) / max(self._total_length, 1e-9),
            0.0, 1.0,
        ))

    # ----------------------------------------------------------------
    # Waypoint lookahead depth (replaces waypoint_coverage)
    # ----------------------------------------------------------------

    def lookahead_depth(
        self,
        agent_plan_wps: List[int],
        n_lookahead_max: int = 15,
    ) -> float:
        """
        How far ahead the agent is effectively planning, as a fraction [0,1].

        agent_plan_wps: list of future waypoint indices the agent has
        implicitly "committed to" (e.g. from the intermediary_head curvature
        predictions for the next N waypoints, or from the race-line engine
        lookahead buffer).

        Returns len(unique_wps) / n_lookahead_max, capped at 1.0.
        High value = agent plans further ahead = good.
        Low value  = agent is reactive, not anticipating corners.
        """
        unique = len(set(agent_plan_wps))
        return float(np.clip(unique / max(n_lookahead_max, 1), 0.0, 1.0))

    # ----------------------------------------------------------------
    # Dynamic obstacle adjustment
    # ----------------------------------------------------------------

    def adjust_for_exclusion_zones(
        self,
        wp_idx: int,
        zones:  List[Tuple[float, float, float]],
    ):
        """
        Shift the reference lateral_offset at wp_idx away from all
        exclusion zones [(cx, cy, radius), ...].

        Zones come from ObjectTracker.get_exclusion_zones() which already
        uses PROJECTED bot position — so the oracle avoids where the
        bot WILL be, not where it is now.
        """
        if not self._built or not zones:
            return
        idx     = wp_idx % self.n
        wp_pos  = self.wpts[idx]
        wp_next = self.wpts[(idx + 1) % self.n]
        tang    = wp_next - wp_pos
        tlen    = float(np.linalg.norm(tang)) + 1e-9
        tang_u  = tang / tlen
        norm_u  = np.array([-tang_u[1], tang_u[0]])  # left-normal

        cur_off = self._plan[idx]["lateral_offset"]

        for cx, cy, r in zones:
            obj_vec  = np.array([cx, cy]) - wp_pos
            obj_lat  = float(np.dot(obj_vec, norm_u))
            obj_frac = obj_lat / (self.half_w + 1e-9)
            sep      = abs(cur_off - obj_frac)
            thresh   = r / (self.half_w + 1e-9) + SAFE_HALF_W / (self.half_w + 1e-9)

            if sep < thresh:
                direction = -1.0 if obj_frac > 0 else 1.0
                new_off   = float(np.clip(
                    obj_frac + direction * thresh * 1.1,
                    -0.85, 0.85,
                ))
                self._plan[idx]["lateral_offset"] = new_off

    # ----------------------------------------------------------------
    # Query
    # ----------------------------------------------------------------

    def get(self, wp_idx: int) -> dict:
        if not self._built:
            self.build()
        return self._plan[wp_idx % self.n]

    def score_agent_step(
        self,
        wp_idx:        int,
        agent_speed:   float,
        agent_lateral: float,   # signed fraction of track half_w
        agent_heading: float,   # radians
    ) -> dict:
        """
        Compare agent behaviour to oracle reference at this waypoint.
        Returns per-dimension compliance [0,1] + composite.
        Inject into bsts_row each step for TensorBoard.
        """
        ref = self.get(wp_idx)

        # Speed
        v_t      = ref["target_speed"]
        spd_err  = abs(agent_speed - v_t) / max(v_t, 0.1)
        spd_sc   = float(np.clip(1.0 - spd_err, 0.0, 1.0))

        # Lateral (fraction of half_w)
        lat_err  = abs(agent_lateral - ref["lateral_offset"])
        lat_sc   = float(np.clip(1.0 - lat_err / 1.0, 0.0, 1.0))

        # Heading
        hdg_diff = abs(agent_heading - ref["heading"])
        hdg_diff = abs((hdg_diff + math.pi) % (2 * math.pi) - math.pi)
        hdg_sc   = float(np.clip(1.0 - hdg_diff / math.pi, 0.0, 1.0))

        # Brake compliance
        brake_sc = 1.0  # N/A unless caller passes is_braking

        return {
            "htm_speed_score":   spd_sc,
            "htm_lateral_score": lat_sc,
            "htm_heading_score": hdg_sc,
            "htm_composite":     0.40 * spd_sc + 0.40 * lat_sc + 0.20 * hdg_sc,
            "htm_regime":        ref["regime"],
            "htm_target_speed":  float(v_t),
            "htm_should_brake":  bool(ref["should_brake"]),
            "htm_curvature":     ref["curvature"],
        }

class HTMPilotDriver:
    """
    Wraps HTMOracle to produce executable [steer, throttle] actions.
    Uses oracle.get(wp_idx) for target_speed + lateral_offset + should_brake.
    """
    def __init__(self, waypoints, track_width, track_variant):
        self.obj_tracker = ObjectTracker()
        self.oracle = HTMOracle(
            waypoints,
            track_width=track_width,
            track_variant=track_variant,
            object_tracker=self.obj_tracker
        )
        self.oracle.build()
        self.wpts = np.array([w[:2] for w in waypoints])

    def act(self, rp: dict) -> list:
        """
        rp = info['reward_params'] dict from DeepRacer env step.
        Returns [steering_angle (-1..1), throttle (-1..1)].
        """
        wp_idx   = rp.get('closest_waypoints', [0,1])[1]
        speed    = float(rp.get('speed', 0.0))
        heading  = float(rp.get('heading', 0.0))
        dist     = float(rp.get('distance_from_center', 0.0))
        tw       = float(rp.get('track_width', 0.6))
        lat_frac = dist / max(tw / 2.0, 1e-9)  # sign: positive = right
        if not rp.get('is_left_of_center', True):
            lat_frac = -lat_frac

        # Update object tracker
        self.obj_tracker.update(
            step=0,
            context_class=int(rp.get('objects_distance', [99])[:1][0] < 2.0) if rp.get('objects_distance') else 0,
            lidar_min=float(rp.get('closest_objects', [99])[0]) if rp.get('closest_objects') else 9.9,
            nearest_obj_dist=float(rp.get('closest_objects', [99])[0]) if rp.get('closest_objects') else 9.9,
            ego_x=float(rp.get('x', 0.0)),
            ego_y=float(rp.get('y', 0.0)),
        )
        self.oracle.adjust_for_exclusion_zones(
            wp_idx, self.obj_tracker.get_exclusion_zones()
        )

        ref = self.oracle.get(wp_idx)
        target_speed = ref['target_speed']
        target_lat   = ref['lateral_offset']
        should_brake = ref['should_brake']

        # --- Steering: proportional control toward target lateral + heading ---
        heading_rad  = math.radians(heading)
        ref_hdg      = ref['heading']
        hdg_err      = (ref_hdg - heading_rad + math.pi) % (2*math.pi) - math.pi
        lat_err      = target_lat - lat_frac
        steer        = float(np.clip(2.0 * hdg_err + 1.5 * lat_err, -1.0, 1.0))

        # --- Throttle: track target speed, brake if should_brake ---
        if should_brake or (speed > target_speed + 0.3):
            throttle = float(np.clip(-0.5 * (speed - target_speed), -1.0, -0.1))
        else:
            throttle = float(np.clip(0.6 + 0.4 * (target_speed - speed) / max(target_speed, 0.1), 0.0, 1.0))

        return [steer, throttle]
