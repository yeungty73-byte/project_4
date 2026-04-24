"""HTM-style Deterministic Lookahead Oracle.

REF: Hawkins, J. & Blakeslee, S. (2004). On Intelligence. Times Books.
     (HTM = Hierarchical Temporal Memory -- used here as a lookahead sweep
     metaphor: given track geometry + physics, compute what a perfect driver
     would do at every waypoint deterministically.)

REF: Kapania, N. R. & Gerdes, J. C. (2015). Design of a feedback-feedforward
     steering controller for accurate path tracking and stability at the
     limits of handling. Vehicle System Dynamics, 53(12).

REF: Heilmeier, A. et al. (2020). Minimum curvature trajectory planning and
     control for an autonomous race car. Vehicle System Dynamics, 58(10).

REF: AWS DeepRacer Developer Guide (2020). Vehicle dimensions.

Purpose
-------
Reference policy for evaluating whether the RL agent is learning
the right courses of action.  Use htm_composite as a per-step metric
in harmonized_metrics to replace the collinear waypoint_coverage.

Object dimension registry
--------------------------
  DeepRacer 1/18 scale car:  0.28m L x 0.20m W x 0.15m H
  Bot car (same platform):   0.28m L x 0.20m W
  Static traffic cone:       0.10m radius (circular footprint)
  Track barrier/curb:        continuous wall -- treated as infinite lateral extent

Object permanence
-----------------
For each dynamic object (bot), the oracle sweeps an angle from the car's
current heading to the heading at the apex of the upcoming corner, and
calculates how much MORE of the object's bounding box becomes visible
(i.e., occlusion changes).  This drives early braking / line adjustment
before the object is fully in view.

Bot motion projection
---------------------
Given the bot's current position + heading, the oracle projects its
position N steps ahead along its inferred race-line to produce a
predicted exclusion zone the ego car must avoid.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# DeepRacer 1/18 scale -- ego car
CAR_LENGTH      = 0.28    # metres
CAR_WIDTH       = 0.20    # metres
CAR_HALF_W      = CAR_WIDTH  / 2.0   # 0.10 m
CAR_HALF_L      = CAR_LENGTH / 2.0   # 0.14 m
SAFETY_MARGIN   = 0.12    # lateral gap the car EDGE must keep from any obstacle
SAFE_HALF_WIDTH = CAR_HALF_W + SAFETY_MARGIN   # 0.22 m from car centreline

# Object dimension registry
# key: context_class int (matches ContextHead in context_aware_agent.py)
#   0 = clear, 1 = curb, 2 = obstacle (static), 3 = corner_approach, 4 = straight
# bot detected separately via bot_x/bot_y API
_OBJ_DIMS: Dict[str, Tuple[float, float]] = {
    "bot":      (0.28, 0.20),   # length, width  metres
    "cone":     (0.10, 0.10),   # radius as square approximation
    "barrier":  (999.0, 0.05),  # infinite extent laterally, 5cm depth
    "curb":     (999.0, 0.08),  # raised curb
}

G   = 9.81   # m/s²
MU  = 0.70   # tyre-road friction coefficient (DeepRacer default)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _menger_curvature(wpts: np.ndarray, i: int, w: int = 3) -> float:
    """Menger curvature at waypoint i using ±w window.
    More numerically stable than cross/norm^3 for short segments.
    REF: Coulom (2002) curvature calculus.
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
    """Return +1 for left turn, -1 for right turn at waypoint i."""
    n   = len(wpts)
    d1  = wpts[i] - wpts[(i - 2) % n]
    d2  = wpts[(i + 2) % n] - wpts[i]
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    return 1.0 if cross > 0 else -1.0


def _safe_speed(curvature: float, v_max: float = 4.0, v_min: float = 0.5) -> float:
    """v_safe = sqrt(mu*g / curvature), capped to [v_min, v_max]."""
    if curvature < 1e-6:
        return v_max
    return float(np.clip(math.sqrt(MU * G / (curvature + 1e-9)), v_min, v_max))


def _brake_distance(v: float) -> float:
    """d_brake = v² / (2·mu·g).  REF: Wikipedia Braking distance."""
    return v ** 2 / (2.0 * MU * G + 1e-9)


def _bounding_box_corners(
    cx: float, cy: float,
    heading_rad: float,
    length: float, width: float,
) -> np.ndarray:
    """Return 4 corners of an axis-aligned bounding box rotated by heading.

    Returns shape (4, 2) array of (x, y) corner coordinates.
    Used for object permanence sweep.
    """
    hl, hw = length / 2.0, width / 2.0
    # local corners (front-right, front-left, rear-left, rear-right)
    local = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ])
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([cx, cy])


def _visible_fraction_from_angle(
    observer_pos: np.ndarray,
    obj_corners: np.ndarray,
) -> float:
    """Estimate fraction of object's lateral extent visible from observer.

    Projects object corners onto the perpendicular axis of the observer->object
    vector.  Returns 0..1 where 1 = full lateral face visible.
    """
    obj_centre = obj_corners.mean(axis=0)
    to_obj     = obj_centre - observer_pos
    dist       = np.linalg.norm(to_obj) + 1e-9
    # perpendicular axis
    perp = np.array([-to_obj[1], to_obj[0]]) / dist
    projections = obj_corners @ perp
    lateral_span = projections.max() - projections.min()
    # normalise by object width (approx)
    widths = np.linalg.norm(obj_corners[0] - obj_corners[1]), \
             np.linalg.norm(obj_corners[1] - obj_corners[2])
    obj_width = min(widths) + 1e-9
    return float(np.clip(lateral_span / obj_width, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Object tracker: identifies type, corners, motion projection
# ---------------------------------------------------------------------------

class ObjectTracker:
    """Per-step object state: type identification, corner geometry,
    motion projection, and object-permanence sweep.

    Object type is inferred from context_class (ContextHead output):
      0 = clear      -> no object
      1 = curb       -> barrier geometry
      2 = obstacle   -> static cone
      3 = corner     -> no movable object (track geometry)
      4 = straight   -> no object
    Bot is tracked separately when bot_x/bot_y are provided.
    """

    # context_class -> object type string
    _CONTEXT_TO_TYPE = {
        1: "curb",
        2: "cone",
    }

    def __init__(self):
        self._state: Dict[str, dict] = {}   # object_id -> last known state
        self._history: Dict[str, list] = {} # object_id -> [(x,y,heading,t)]

    def update(
        self,
        step: int,
        context_class: int,
        lidar_min: float,
        nearest_obj_dist: float,
        bot_x: float = 0.0,
        bot_y: float = 0.0,
        bot_heading: float = 0.0,
        ego_x: float = 0.0,
        ego_y: float = 0.0,
    ) -> Dict[str, dict]:
        """Update object registry and return current object states.

        Returns dict of {object_id: {type, corners, projected_pos, visibility}}
        """
        objects = {}

        # --- Bot ---
        if bot_x != 0.0 or bot_y != 0.0:
            dims   = _OBJ_DIMS["bot"]
            corners = _bounding_box_corners(bot_x, bot_y, bot_heading,
                                             dims[0], dims[1])
            ego_pos = np.array([ego_x, ego_y])
            vis     = _visible_fraction_from_angle(ego_pos, corners)
            hist    = self._history.setdefault("bot", [])
            hist.append((bot_x, bot_y, bot_heading, step))
            if len(hist) > 10:
                hist.pop(0)
            proj = self._project_bot_motion(hist, steps_ahead=5)
            self._state["bot"] = {
                "type":          "bot",
                "dims":          dims,
                "pos":           (bot_x, bot_y),
                "heading":       bot_heading,
                "corners":       corners,
                "visibility":    vis,
                "projected_pos": proj,   # predicted (x,y) 5 steps ahead
                "exclusion_r":   (dims[1] / 2.0) + SAFE_HALF_WIDTH,
            }
            objects["bot"] = self._state["bot"]

        # --- Static obstacle / curb from context ---
        obj_type = self._CONTEXT_TO_TYPE.get(context_class)
        if obj_type and nearest_obj_dist < 2.0:
            dims     = _OBJ_DIMS[obj_type]
            # Approximate object position along lidar_min bearing
            # (we don't have exact coords, so place it nearest_obj_dist ahead)
            obj_x    = ego_x + nearest_obj_dist * math.cos(0.0)  # forward axis approx
            obj_y    = ego_y + nearest_obj_dist * math.sin(0.0)
            corners  = _bounding_box_corners(obj_x, obj_y, 0.0, dims[0], dims[1])
            ego_pos  = np.array([ego_x, ego_y])
            vis      = _visible_fraction_from_angle(ego_pos, corners)
            objects[obj_type] = {
                "type":          obj_type,
                "dims":          dims,
                "pos":           (obj_x, obj_y),
                "heading":       0.0,
                "corners":       corners,
                "visibility":    vis,
                "projected_pos": None,  # static
                "exclusion_r":   (dims[1] / 2.0) + SAFE_HALF_WIDTH,
            }

        return objects

    def _project_bot_motion(
        self,
        history: list,
        steps_ahead: int = 5,
    ) -> Optional[Tuple[float, float]]:
        """Project bot position N steps ahead using average heading.

        Uses last 3 history frames to estimate velocity vector.
        If history is too short, returns last known position.
        """
        if len(history) < 2:
            if history:
                return (history[-1][0], history[-1][1])
            return None
        # average velocity over last min(3, len) frames
        n  = min(3, len(history))
        xs = [h[0] for h in history[-n:]]
        ys = [h[1] for h in history[-n:]]
        dx = (xs[-1] - xs[0]) / max(n - 1, 1)
        dy = (ys[-1] - ys[0]) / max(n - 1, 1)
        px = history[-1][0] + dx * steps_ahead
        py = history[-1][1] + dy * steps_ahead
        return (px, py)

    def get_exclusion_zones(self) -> List[Tuple[float, float, float]]:
        """Return list of (cx, cy, radius) exclusion circles for all tracked objects.
        Used by HTMOracle to shift race line away from obstacles.
        """
        zones = []
        for obj in self._state.values():
            pos = obj.get("projected_pos") or obj.get("pos")
            if pos:
                zones.append((pos[0], pos[1], obj["exclusion_r"]))
        return zones

    def permanence_delta(
        self,
        object_id: str,
        future_ego_x: float,
        future_ego_y: float,
    ) -> float:
        """How much MORE of the object becomes visible as ego moves to future pos.

        Returns delta in [0, 1]: positive = more revealed, negative = more occluded.
        Used to trigger early braking when a bot will be revealed mid-corner.
        """
        if object_id not in self._state:
            return 0.0
        obj = self._state[object_id]
        corners = obj["corners"]
        current_vis = obj["visibility"]
        future_ego  = np.array([future_ego_x, future_ego_y])
        future_vis  = _visible_fraction_from_angle(future_ego, corners)
        return float(future_vis - current_vis)


# ---------------------------------------------------------------------------
# HTM Oracle
# ---------------------------------------------------------------------------

class HTMOracle:
    """Deterministic per-waypoint reference plan.

    Computes: target_speed, lateral_offset (car-dimension aware),
    brake_flag, heading, regime (straight/corner/brake_zone)
    for ALL waypoints at initialisation.

    Optionally accepts live ObjectTracker to shift the reference line
    away from dynamic obstacles.

    Usage
    -----
    oracle = HTMOracle(waypoints, track_width=0.6)
    oracle.build()
    scores = oracle.score_agent_step(wp_idx, speed, lateral_frac, heading_rad)
    # -> {'htm_composite': float, 'htm_regime': str, ...}
    """

    def __init__(
        self,
        waypoints:    list,
        track_width:  float = 0.6,
        mu:           float = MU,
        n_lookahead:  int   = 15,
        object_tracker: Optional[ObjectTracker] = None,
    ):
        self.wpts        = np.array([w[:2] for w in waypoints], dtype=np.float64)
        self.n           = len(self.wpts)
        self.track_width = track_width
        self.half_w      = track_width / 2.0
        self.mu          = mu
        self.n_lookahead = n_lookahead
        self.obj_tracker = object_tracker
        self._plan: List[dict] = []
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """Pre-compute the full deterministic plan. Call once after init."""
        n    = self.n
        wpts = self.wpts

        # 1. Curvature at each WP
        curvatures = np.array([_menger_curvature(wpts, i) for i in range(n)])

        # 2. Cornering speed (physics limit)
        speeds_raw = np.array([_safe_speed(c) for c in curvatures])

        # 3. Backward pass: ensure entry speed allows braking to corner speed
        speeds = speeds_raw.copy()
        for _ in range(4):   # iterate to convergence
            for i in range(n - 1, -1, -1):
                j    = (i + 1) % n
                dist = float(np.linalg.norm(wpts[j] - wpts[i])) + 1e-9
                v_in_max = math.sqrt(
                    max(speeds[j] ** 2 + 2.0 * self.mu * G * dist, 0.0)
                )
                speeds[i] = min(speeds[i], v_in_max)

        # 4. Lateral offsets -- car-dimension-aware apex seeking
        #    max_offset = track half_width - SAFE_HALF_WIDTH
        #    so the car EDGE (not centreline) is at most at the edge
        max_lat = self.half_w - SAFE_HALF_WIDTH  # e.g. 0.30 - 0.22 = 0.08 m as fraction
        # as fraction of half_w:
        max_frac = max_lat / self.half_w if self.half_w > 1e-6 else 0.0
        max_frac = float(np.clip(max_frac, 0.0, 0.85))

        offsets = np.zeros(n)
        for i in range(n):
            curv  = curvatures[i]
            sign  = _turn_sign(wpts, i)
            if curv < 0.01:
                offsets[i] = 0.0       # straight: stay on centre
            elif curv < 0.1:
                tightness = min(max_frac, 2.0 * curv)
                offsets[i] = -sign * tightness
            else:
                offsets[i] = -sign * max_frac  # tight corner: full safe apex

        # Smooth offsets with wrap-around 5-point kernel
        kernel  = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        padded  = np.concatenate([offsets[-2:], offsets, offsets[:2]])
        offsets = np.convolve(padded, kernel, mode='valid')[2:-2]

        # 5. Headings
        headings = np.zeros(n)
        for i in range(n):
            p0 = wpts[(i - 1) % n]
            p2 = wpts[(i + 1) % n]
            headings[i] = math.atan2(p2[1] - p0[1], p2[0] - p0[0])

        # 6. Brake flags: lookahead for upcoming speed reduction
        brake_flags = np.zeros(n, dtype=bool)
        for i in range(n):
            for k in range(1, self.n_lookahead + 1):
                j = (i + k) % n
                if speeds[j] < speeds[i] - 0.25:
                    phys_d = sum(
                        float(np.linalg.norm(wpts[(i + m + 1) % n] - wpts[(i + m) % n]))
                        for m in range(k)
                    )
                    # Need to brake from speeds[i] down to speeds[j]
                    d_stop = _brake_distance(speeds[i]) - _brake_distance(speeds[j])
                    d_stop *= 1.2   # safety factor
                    # Also account for car front overhang (0.14m) -- brake BEFORE
                    # the front bumper reaches the corner
                    d_stop += CAR_HALF_L
                    if phys_d <= d_stop:
                        brake_flags[i] = True
                    break

        # 7. Assemble plan
        self._plan = [
            {
                "wp_idx":         i,
                "target_speed":   float(speeds[i]),
                "lateral_offset": float(offsets[i]),  # signed fraction of half_w
                "heading":        float(headings[i]),  # radians
                "should_brake":   bool(brake_flags[i]),
                "curvature":      float(curvatures[i]),
                "raw_speed":      float(speeds_raw[i]),
                "regime":         (
                    "brake_zone" if brake_flags[i]
                    else "corner" if curvatures[i] > 0.05
                    else "straight"
                ),
            }
            for i in range(n)
        ]
        self._built = True

    # ------------------------------------------------------------------
    # Dynamic obstacle adjustment
    # ------------------------------------------------------------------

    def adjust_for_exclusion_zones(
        self,
        wp_idx:  int,
        zones:   List[Tuple[float, float, float]],
    ):
        """Shift lateral_offset away from exclusion zones at this waypoint.

        zones : [(cx, cy, radius), ...] from ObjectTracker.get_exclusion_zones()
        Adjusts the oracle's reference offset so it avoids all known objects.
        Call this AFTER build() when new objects are detected.
        """
        if not self._built or not zones:
            return
        wp_pos  = self.wpts[wp_idx % self.n]
        wp_next = self.wpts[(wp_idx + 1) % self.n]
        tang    = wp_next - wp_pos
        tlen    = float(np.linalg.norm(tang)) + 1e-9
        tang_u  = tang / tlen
        norm_u  = np.array([-tang_u[1], tang_u[0]])  # left-normal

        cur_off = self._plan[wp_idx % self.n]["lateral_offset"]

        for cx, cy, r in zones:
            obj_vec    = np.array([cx, cy]) - wp_pos
            obj_lat    = float(np.dot(obj_vec, norm_u))  # signed lateral (m)
            obj_frac   = obj_lat / (self.half_w + 1e-9)  # fraction of half_w
            separation = abs(cur_off - obj_frac)

            if separation < (r / self.half_w + SAFE_HALF_WIDTH / self.half_w):
                # Too close -- push to opposite side
                push = (r / self.half_w + SAFE_HALF_WIDTH / self.half_w) * 1.1
                direction = -1.0 if obj_frac > 0 else 1.0
                new_off = float(np.clip(
                    obj_frac + direction * push,
                    -0.85, 0.85
                ))
                self._plan[wp_idx % self.n]["lateral_offset"] = new_off

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, wp_idx: int) -> dict:
        if not self._built:
            self.build()
        return self._plan[wp_idx % self.n]

    def score_agent_step(
        self,
        wp_idx:        int,
        agent_speed:   float,
        agent_lateral: float,   # signed fraction of half_w
        agent_heading: float,   # radians
    ) -> dict:
        """Compare agent behaviour to oracle at this waypoint.

        Returns dict with per-dimension compliance scores [0,1]
        and a composite score.  Plug into bsts_row in run.py.

        Parameters
        ----------
        wp_idx        : closest waypoint index
        agent_speed   : current m/s
        agent_lateral : signed fraction of track half_width (left=+)
        agent_heading : heading in radians
        """
        ref = self.get(wp_idx)

        # Speed compliance
        v_target = ref["target_speed"]
        spd_err  = abs(agent_speed - v_target) / max(v_target, 0.1)
        spd_score = float(np.clip(1.0 - spd_err, 0.0, 1.0))

        # Lateral compliance
        lat_err   = abs(agent_lateral - ref["lateral_offset"])
        lat_score = float(np.clip(1.0 - lat_err / max(1.0, 1e-6), 0.0, 1.0))

        # Heading compliance
        hdg_diff  = abs(agent_heading - ref["heading"])
        # wrap to [-pi, pi]
        hdg_diff  = abs((hdg_diff + math.pi) % (2 * math.pi) - math.pi)
        hdg_score = float(np.clip(1.0 - hdg_diff / math.pi, 0.0, 1.0))

        return {
            "htm_speed_score":   spd_score,
            "htm_lateral_score": lat_score,
            "htm_heading_score": hdg_score,
            "htm_composite":     0.40 * spd_score + 0.40 * lat_score + 0.20 * hdg_score,
            "htm_regime":        ref["regime"],
            "htm_target_speed":  float(v_target),
            "htm_should_brake":  bool(ref["should_brake"]),
        }
