"""Brake-field reward module — object-aware, car-dimension-aware.

Physics: braking distance d_brake = v^2 / (2 * mu * g)   [Wikipedia/Braking_distance]
The "brake field" is the region before each corner where the agent
SHOULD be decelerating, represented as a potential function:

  Phi(s) = clamp(1 - effective_dist / (safety * d_brake), 0, 1)

Key upgrades vs previous version:
  1. Car dimensions: front overhang (0.14 m) subtracted from effective distance
     so the brake zone starts when the CAR FRONT, not the centre-point, reaches
     the corner.  REF: AWS DeepRacer Developer Guide (2020) — 1/18-scale body.
  2. Object-aware corners: static obstacles and curbs inject synthetic corner
     waypoints so the brake field activates before any obstacle, not just track
     geometry corners.  Obstacle half-width (0.10 m) adds a safety bubble.
  3. Menger curvature replaces the cross-product sine approximation for
     numerically stable corner detection on short-segment waypoint lists.
     REF: Menger, K. (1930). Untersuchungen uber allgemeine Metrik.
  4. in_brake_field is only True when the car IS actually braking.
     (Bug fix: prior version exposed in_field even on non-braking steps.)

REF: Kapania, N. R. & Gerdes, J. C. (2015). Design of a feedback-feedforward
     steering controller for accurate path tracking. Vehicle System Dynamics.
REF: AWS DeepRacer Developer Guide (2020).
"""
import math
import numpy as np
from config_loader import CFG

# ---------------------------------------------------------------------------
# Vehicle constants (AWS DeepRacer 1/18-scale)
# ---------------------------------------------------------------------------
_CAR_LENGTH      = CFG.get("vehicle", {}).get("length",      0.28)   # m
_CAR_WIDTH       = CFG.get("vehicle", {}).get("width",       0.20)   # m
_CAR_HALF_WIDTH  = _CAR_WIDTH / 2.0                                    # 0.10 m
_CAR_FRONT_OVHG  = _CAR_LENGTH / 2.0                                   # 0.14 m
_CAR_SAFETY_LAT  = CFG.get("vehicle", {}).get("safety_lat",  0.05)   # extra lateral buffer
_SAFE_HALF_W     = _CAR_HALF_WIDTH + _CAR_SAFETY_LAT                  # 0.15 m

# Obstacle / bot dimensions
_BOT_HALF_WIDTH  = CFG.get("vehicle", {}).get("bot_half_width",  0.10)  # m
_CONE_RADIUS     = CFG.get("vehicle", {}).get("cone_radius",     0.05)  # m


def braking_distance(speed: float, mu: float = None, g: float = None) -> float:
    """d = v^2 / (2 * mu * g)"""
    _cfg = CFG.get("brake_field", {})
    mu = mu or _cfg.get("mu", 0.7)
    g  = g  or _cfg.get("g", 9.81)
    return speed ** 2 / (2.0 * mu * g + 1e-8)


def _menger_curvature(wpts: np.ndarray, i: int, w: int = 3) -> float:
    """Menger curvature at waypoint i.  Numerically stable for short segments.
    REF: Menger (1930); Coulom (2002) curvature calculus for racing lines.
    """
    n = len(wpts)
    p0 = wpts[(i - w) % n]
    p1 = wpts[i]
    p2 = wpts[(i + w) % n]
    d1 = np.linalg.norm(p1 - p0) + 1e-9
    d2 = np.linalg.norm(p2 - p1) + 1e-9
    d3 = np.linalg.norm(p2 - p0) + 1e-9
    cross = abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]))
    return 2.0 * cross / (d1 * d2 * d3 + 1e-9)


class ObstacleRecord:
    """Describes a detected obstacle so the brake field can project its position.

    obj_type: 'bot' | 'cone' | 'static' | 'curb'
    x, y    : world position (m)
    heading : direction of travel in radians (bots only; 0 for static)
    speed   : scalar speed m/s (bots only)
    wp_idx  : nearest track waypoint index

    For bots, we project future position over `lookahead_t` seconds to
    determine whether and where the paths will intersect.
    REF: Waymo Motion Prediction Challenge (2021) — constant-velocity projection.
    """
    __slots__ = ('obj_type', 'x', 'y', 'heading', 'speed', 'wp_idx')

    # Dimension lookup by type (half-width, half-length) in metres
    DIMS = {
        'bot':    (_BOT_HALF_WIDTH, _BOT_HALF_WIDTH * 1.4),
        'cone':   (_CONE_RADIUS,    _CONE_RADIUS),
        'static': (0.10,            0.10),
        'curb':   (0.02,            0.50),
    }

    def __init__(self, obj_type='static', x=0.0, y=0.0,
                 heading=0.0, speed=0.0, wp_idx=0):
        self.obj_type = obj_type
        self.x        = float(x)
        self.y        = float(y)
        self.heading  = float(heading)
        self.speed    = float(speed)
        self.wp_idx   = int(wp_idx)

    @property
    def half_width(self) -> float:
        return self.DIMS.get(self.obj_type, (0.10, 0.10))[0]

    @property
    def half_length(self) -> float:
        return self.DIMS.get(self.obj_type, (0.10, 0.10))[1]

    def safe_clearance(self) -> float:
        """Minimum lateral clearance needed for own car to pass: obj half-width
        + own car half-width + safety buffer.
        """
        return self.half_width + _SAFE_HALF_W

    def projected_position(self, dt: float = 1.0):
        """Constant-velocity projection for bots; static otherwise.
        Returns (px, py).
        """
        if self.obj_type == 'bot' and self.speed > 0.1:
            px = self.x + self.speed * math.cos(self.heading) * dt
            py = self.y + self.speed * math.sin(self.heading) * dt
            return px, py
        return self.x, self.y

    def corner_points(self) -> np.ndarray:
        """Returns 4 corner points of the object bounding box (world frame).
        Used for object-permanence angle calculation.
        """
        hw, hl = self.half_width, self.half_length
        c, s = math.cos(self.heading), math.sin(self.heading)
        # local corners: front-left, front-right, rear-right, rear-left
        local = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw],
        ])
        rot = np.array([[c, -s], [s, c]])
        world = (rot @ local.T).T + np.array([self.x, self.y])
        return world  # shape (4, 2)

    def visible_angle_from(self, car_x: float, car_y: float) -> float:
        """Angular subtended by object corners from car position (radians).
        Larger angle -> car is closer or object is bigger: increase caution.
        This implements 'object permanence': as car turns, how much MORE
        of the object will be exposed?
        """
        corners = self.corner_points()
        car_pos = np.array([car_x, car_y])
        angles = [math.atan2(c[1] - car_pos[1], c[0] - car_pos[0])
                  for c in corners]
        # Angular span of object from car viewpoint
        diffs = []
        for i in range(len(angles)):
            for j in range(i+1, len(angles)):
                d = abs(angles[i] - angles[j])
                diffs.append(min(d, 2*math.pi - d))
        return max(diffs) if diffs else 0.0


class BrakeField:
    """Computes brake-field potential and compliance per step.

    Object-aware: obstacles and bots inject synthetic brake waypoints.
    Car-dimension-aware: front overhang subtracted from effective distance.
    Bot-motion-aware: bot projected position used for dynamic brake zone.
    """

    def __init__(self, waypoints=None):
        _cfg = CFG.get("brake_field", {})
        self.mu          = _cfg.get("mu",                  0.7)
        self.g           = _cfg.get("g",                   9.81)
        self.safety      = _cfg.get("safety_margin",       1.2)
        self.lookahead   = _cfg.get("lookahead_waypoints", 8)
        self.lookahead_t = _cfg.get("bot_lookahead_s",     1.5)  # seconds for bot projection
        self.waypoints   = waypoints           # np.ndarray (N, 2)
        self._corner_indices   = None          # geometric corners
        self._obstacle_wps     = []            # synthetic obstacle brake points (wp_idx, physical_dist)
        self._braking_events   = []
        self._in_field_count   = 0
        self._obstacles        = []            # List[ObstacleRecord]

    def set_waypoints(self, waypoints: np.ndarray):
        self.waypoints = waypoints
        self._corner_indices = None

    def update_obstacles(self, obstacles):
        """Called each step with fresh obstacle list from env.
        obstacles: List[ObstacleRecord]  (or list of dicts with same fields)
        Bots: project position forward, find nearest WP, inject brake point.
        Static/curbs: find nearest WP, inject brake point with safety clearance.
        """
        if not isinstance(obstacles, list):
            obstacles = []
        parsed = []
        for o in obstacles:
            if isinstance(o, ObstacleRecord):
                parsed.append(o)
            elif isinstance(o, dict):
                parsed.append(ObstacleRecord(**{k: o[k] for k in ObstacleRecord.__slots__ if k in o}))
        self._obstacles = parsed
        self._obstacle_wps = []
        if self.waypoints is None or len(self.waypoints) < 3:
            return
        wpts = self.waypoints
        n    = len(wpts)
        for obs in self._obstacles:
            # Project bot position forward
            px, py = obs.projected_position(self.lookahead_t)
            target = np.array([px, py])
            # Find nearest waypoint to projected position
            dists  = np.linalg.norm(wpts - target, axis=1)
            nearest_wp = int(np.argmin(dists))
            # Physical distance from that WP to projected position
            phys_d = float(dists[nearest_wp])
            # Safety clearance: account for object + car dimensions
            clearance = obs.safe_clearance()
            # Effective brake distance = distance - clearance bubble - car front overhang
            effective_d = max(0.0, phys_d - clearance - _CAR_FRONT_OVHG)
            self._obstacle_wps.append((nearest_wp, effective_d, obs))

    def _find_corners(self):
        """Menger curvature corner detection — replaces cross-product sine approx."""
        if self.waypoints is None or len(self.waypoints) < 5:
            self._corner_indices = []
            return
        wpts = self.waypoints
        n    = len(wpts)
        threshold = CFG.get("brake_field", {}).get("corner_curvature_threshold", 0.08)
        self._corner_indices = [
            i for i in range(n)
            if _menger_curvature(wpts, i, w=3) > threshold
        ]

    def potential(self, wp_idx: int, speed: float,
                  car_x: float = 0.0, car_y: float = 0.0) -> float:
        """Compute brake-field potential Phi(s) in [0, 1].

        Accounts for:
          - geometric track corners (Menger curvature)
          - injected obstacle brake points (bots projected, static/curbs fixed)
          - car front overhang subtracted from effective distance
        """
        if self._corner_indices is None:
            self._find_corners()
        if self.waypoints is None:
            return 0.0

        d_brake_raw  = braking_distance(speed, self.mu, self.g)
        d_brake      = d_brake_raw * self.safety
        n            = len(self.waypoints)
        best_phi     = 0.0

        # --- geometric corners ---
        for ci in (self._corner_indices or []):
            idx_dist = (ci - wp_idx) % n
            if 0 < idx_dist <= self.lookahead:
                phys_dist = sum(
                    np.linalg.norm(self.waypoints[(wp_idx+j+1) % n]
                                   - self.waypoints[(wp_idx+j) % n])
                    for j in range(idx_dist)
                )
                # Subtract car front overhang: brake BEFORE front hits corner
                eff_dist = max(0.0, phys_dist - _CAR_FRONT_OVHG)
                if d_brake > 1e-6:
                    best_phi = max(best_phi,
                                   float(np.clip(1.0 - eff_dist / d_brake, 0.0, 1.0)))

        # --- obstacle brake points ---
        for (obs_wp, eff_d, obs) in self._obstacle_wps:
            # How far ahead is this obstacle WP from current position?
            idx_dist = (obs_wp - wp_idx) % n
            if 0 < idx_dist <= self.lookahead * 2:  # wider lookahead for bots
                # eff_d already has clearance + overhang subtracted
                if d_brake > 1e-6:
                    phi_obs = float(np.clip(1.0 - eff_d / d_brake, 0.0, 1.0))
                    # Object-permanence scale: larger visible angle -> stronger signal
                    if car_x != 0.0 or car_y != 0.0:
                        vis_angle = obs.visible_angle_from(car_x, car_y)
                        phi_obs  *= min(1.5, 1.0 + vis_angle / math.pi)
                    best_phi = max(best_phi, phi_obs)

        return best_phi

    def step(self, wp_idx: int, speed: float, is_braking: bool,
             car_x: float = 0.0, car_y: float = 0.0) -> dict:
        """Called each env step. Returns enrichment dict for step record."""
        phi = self.potential(wp_idx, speed, car_x, car_y)
        in_field = phi > 0.0
        if is_braking:
            self._braking_events.append(in_field)
            if in_field:
                self._in_field_count += 1
        return dict(
            brake_potential=phi,
            in_brake_field=(in_field and is_braking),  # only True when braking in zone
            is_braking=is_braking,
        )

    @property
    def compliance(self) -> float:
        if not self._braking_events:
            return 1.0
        return self._in_field_count / len(self._braking_events)

    def reset(self):
        self._braking_events   = []
        self._in_field_count   = 0
        self._obstacle_wps     = []
