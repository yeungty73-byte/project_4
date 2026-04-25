"""
Brake Field Engine — v2 (object-aware, car-dimension-corrected)
===============================================================
Physics:
  d_brake = v² / (2 · mu · g)          [Wikipedia: Braking distance]
  Corner approach: braking must start d_brake BEFORE the front bumper
  reaches the corner apex. Front bumper is CAR_HALF_L = 0.14m ahead
  of the car’s reported position.
  -> effective_d_brake = d_brake + CAR_HALF_L

Brake-field potential:
  Phi(s) = clip(1 - dist_to_corner / (safety * effective_d_brake), 0, 1)
  Phi=1.0 means the car is AT the ideal brake point.
  Phi<1.0 means the car has room before it must brake.
  Phi>0 AND is_braking -> correct braking placement.

Object-dimension integration:
  - Obstacles / bots detected via ObjectTracker push the effective corner
    forward: the car must brake for the OBJECT (if closer than the corner)
    not just for the track apex.
  - The obstacle’s exclusion radius (obj_half_w + SAFE_HALF_W) defines a
    virtual "wall"; dist_to_corner is the distance to that wall’s near edge.

Dynamic curvature:
  Each corner’s speed limit is computed via Menger curvature per waypoint,
  so the brake distance is variable per corner — tight hairpins demand
  earlier braking than gentle kinks.

Track-variant mu:
  Pass track_variant='tt_vegas' etc. to get surface-correct friction.
  See _VARIANT_MU in htm_reference.py (same registry).

REF: Wikipedia. Braking distance. https://en.wikipedia.org/wiki/Braking_distance
REF: Coulom (2002) curvature calculus.
REF: Kapania & Gerdes (2015) Vehicle System Dynamics, 53(12).
REF: AWS DeepRacer Developer Guide (2020).
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Tuple
from config_loader import CFG

# Car physical constants (DeepRacer 1/18 scale)
CAR_HALF_L   = 0.14    # front bumper offset from reported position (metres)
CAR_HALF_W   = 0.10
SAFETY_MARGIN = 0.12
SAFE_HALF_W  = CAR_HALF_W + SAFETY_MARGIN   # 0.22 m exclusion from ego centre

_VARIANT_MU = {
    "tt_reinvent": 0.72,
    "tt_vegas":    0.68,
    "tt_bowtie":   0.70,
    "oa_reinvent": 0.70,
    "oa_vegas":    0.68,
    "oa_bowtie":   0.70,
    "h2h_reinvent":0.72,
    "h2h_vegas":   0.68,
    "h2b_reinvent":0.72,
}


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def braking_distance(
    speed:   float,
    mu:      float = 0.70,
    g:       float = 9.81,
    add_overhang: bool = True,
) -> float:
    """
    d = v² / (2·mu·g)  +  CAR_HALF_L (front overhang).
    add_overhang=True ensures braking starts before the front bumper
    reaches the corner, not the GPS reported centre.
    REF: Wikipedia Braking distance.
    """
    d = speed ** 2 / (2.0 * mu * g + 1e-8)
    if add_overhang:
        d += CAR_HALF_L
    return d


def _menger_curvature(wpts: np.ndarray, i: int, w: int = 3) -> float:
    """Menger curvature at waypoint i using ±w window.  REF: Coulom (2002)."""
    n = len(wpts)
    p0, p1, p2 = wpts[(i - w) % n], wpts[i], wpts[(i + w) % n]
    d1 = float(np.linalg.norm(p1 - p0)) + 1e-9
    d2 = float(np.linalg.norm(p2 - p1)) + 1e-9
    d3 = float(np.linalg.norm(p2 - p0)) + 1e-9
    cross = abs(
        float((p1[0] - p0[0]) * (p2[1] - p0[1])
             - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    )
    return 2.0 * cross / (d1 * d2 * d3 + 1e-9)


def _corner_speed(curvature: float, mu: float, vmax: float = 4.0) -> float:
    """Physics speed limit at corner: v = sqrt(mu*g / curvature)."""
    if curvature < 1e-6:
        return vmax
    return float(np.clip(math.sqrt(mu * 9.81 / (curvature + 1e-9)), 0.5, vmax))


# ---------------------------------------------------------------------------
# BrakeField
# ---------------------------------------------------------------------------

class BrakeField:
    """
    Brake-field reward module with dynamic curvature, car-dimension
    correction, and object-exclusion-zone integration.

    Parameters
    ----------
    waypoints       : np.ndarray (N, 2) — set via set_waypoints() or __init__
    track_variant   : str — used for per-surface mu
    mu              : override friction (None = from variant or config)
    safety          : multiplier on d_brake for field extent (default 1.25)
    lookahead_m     : physical lookahead distance in metres (default 3.0)
    curvature_thresh: minimum Menger curvature to classify as a corner
    exclusion_zones : list of (cx, cy, radius) from ObjectTracker
    """

    def __init__(
        self,
        waypoints:          Optional[np.ndarray] = None,
        track_variant:      str   = 'unknown',
        mu:                 Optional[float] = None,
        safety:             float = 1.25,
        lookahead_m:        float = 3.0,
        curvature_thresh:   float = 0.05,
    ):
        _cfg = CFG.get("brake_field", {})
        self.mu               = mu or _VARIANT_MU.get(track_variant,
                                    _cfg.get("mu", 0.70))
        self.g                = _cfg.get("g", 9.81)
        self.safety           = safety
        self.lookahead_m      = lookahead_m
        self.curvature_thresh = curvature_thresh
        self.track_variant    = track_variant

        self.waypoints: Optional[np.ndarray] = None
        self._corners:  List[dict] = []   # {idx, speed_limit, arc_dist}
        self._arc_dists: Optional[np.ndarray] = None

        self._braking_events: List[bool] = []
        self._in_field_count: int  = 0
        self._exclusion_zones: List[Tuple[float, float, float]] = []

        if waypoints is not None:
            self.set_waypoints(np.asarray(waypoints, dtype=np.float64))

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------

    def set_waypoints(self, waypoints: np.ndarray):
        """Load track waypoints and rebuild corner registry."""
        self.waypoints   = waypoints[:, :2]
        self._arc_dists  = None
        self._corners    = []
        self._build_corners()

    def set_exclusion_zones(
        self,
        zones: List[Tuple[float, float, float]],
    ):
        """
        Inject live exclusion zones from ObjectTracker.
        Call every step BEFORE step() so brake points account for obstacles.

        zones : [(cx, cy, radius), ...]  -- ObjectTracker.get_exclusion_zones()
        """
        self._exclusion_zones = zones

    # ----------------------------------------------------------------
    # Internal build
    # ----------------------------------------------------------------

    def _build_arc_dists(self):
        wpts = self.waypoints
        n    = len(wpts)
        segs = np.array([
            float(np.linalg.norm(wpts[(i + 1) % n] - wpts[i]))
            for i in range(n)
        ])
        self._arc_dists  = np.concatenate([[0.0], np.cumsum(segs[:-1])])
        self._total_arc  = float(segs.sum())

    def _build_corners(self):
        """
        Identify corner waypoints using Menger curvature.
        For each corner, compute: waypoint index, physics speed limit,
        arc distance from start.
        """
        if self.waypoints is None or len(self.waypoints) < 5:
            return
        self._build_arc_dists()
        wpts = self.waypoints
        n    = len(wpts)

        # 1. Raw curvature per WP
        curvs = np.array([_menger_curvature(wpts, i) for i in range(n)])

        # 2. Identify local maxima above threshold
        raw = []
        for i in range(n):
            if curvs[i] > self.curvature_thresh:
                # local max check
                prev_c = curvs[(i - 1) % n]
                next_c = curvs[(i + 1) % n]
                if curvs[i] >= prev_c and curvs[i] >= next_c:
                    raw.append(i)

        # 3. Merge nearby corners (< 0.5m apart = same corner)
        merged = []
        if raw:
            group = [raw[0]]
            for ci in raw[1:]:
                arc_gap = self._arc_dists[ci] - self._arc_dists[group[-1]]
                if arc_gap < 0.5:
                    group.append(ci)
                else:
                    apex = max(group, key=lambda x: curvs[x])
                    merged.append(apex)
                    group = [ci]
            apex = max(group, key=lambda x: curvs[x])
            merged.append(apex)

        self._corners = [
            {
                "idx":         ci,
                "curvature":   float(curvs[ci]),
                "speed_limit": _corner_speed(curvs[ci], self.mu),
                "arc_dist":    float(self._arc_dists[ci]),
            }
            for ci in merged
        ]

    # ----------------------------------------------------------------
    # Dynamic virtual corners from exclusion zones
    # ----------------------------------------------------------------

    def _virtual_corners_from_zones(
        self,
        wp_idx:   int,
        cur_speed: float,
    ) -> List[dict]:
        """
        Treat each active exclusion zone as a virtual "corner":
        the car must brake as if it were a tight corner.
        Returns list of {dist, speed_limit} for obstacles within lookahead.
        """
        if not self._exclusion_zones or self.waypoints is None:
            return []
        wpts    = self.waypoints
        ego_pos = wpts[wp_idx % len(wpts)]
        virtual = []
        for cx, cy, r in self._exclusion_zones:
            obj_pos = np.array([cx, cy])
            dist    = float(np.linalg.norm(obj_pos - ego_pos)) - r  # to near edge
            dist    = max(dist, 0.0)
            if dist < self.lookahead_m * 1.5:   # objects slightly beyond lookahead count
                # treat exclusion zone as v=0 (must stop / deviate)
                # conservative: use v=0.5 min speed
                virtual.append({"dist": dist, "speed_limit": 0.5})
        return virtual

    # ----------------------------------------------------------------
    # Potential function (public)
    # ----------------------------------------------------------------

    def potential(
        self,
        wp_idx: int,
        speed:  float,
    ) -> float:
        """
        Brake-field potential Phi(s) in [0, 1].
        Phi=1 = agent is exactly AT the ideal brake point.
        Phi<1 = still approaching it.
        0 = no corner in lookahead OR already past the brake point.

        Accounts for:
          - Menger-curvature-derived per-corner speed limits
          - Car front overhang (CAR_HALF_L)
          - Active exclusion zones (obstacles / bots)
          - Safety margin multiplier
        """
        if not self._corners and not self._exclusion_zones:
            return 0.0
        if self.waypoints is None:
            return 0.0

        d_brake_raw = braking_distance(speed, self.mu, self.g, add_overhang=True)
        targets: List[Tuple[float, float]] = []   # (dist_to_target, target_speed)

        # --- Track corners ---
        n      = len(self.waypoints)
        wpts   = self.waypoints
        for corner in self._corners:
            ci       = corner["idx"]
            arc_ci   = corner["arc_dist"]
            arc_now  = float(self._arc_dists[wp_idx % n])
            arc_diff = (arc_ci - arc_now) % self._total_arc   # forward distance
            if arc_diff < self.lookahead_m * 2.0:
                d_needed = (
                    braking_distance(speed, self.mu, self.g, True)
                    - braking_distance(corner["speed_limit"], self.mu, self.g, False)
                )
                d_needed *= self.safety
                targets.append((arc_diff, d_needed))

        # --- Virtual corners from objects ---
        for v in self._virtual_corners_from_zones(wp_idx, speed):
            d_needed = braking_distance(speed, self.mu, self.g, True) * self.safety
            targets.append((v["dist"], d_needed))

        if not targets:
            return 0.0

        # Take the closest target that requires braking
        best_phi = 0.0
        for dist_to_target, d_needed in targets:
            if d_needed < 1e-4:
                continue
            phi = float(np.clip(
                1.0 - dist_to_target / (d_needed + 1e-6),
                0.0, 1.0,
            ))
            best_phi = max(best_phi, phi)

        return best_phi

    # ----------------------------------------------------------------
    # Per-step call
    # ----------------------------------------------------------------

    def step(
        self,
        wp_idx:     int,
        speed:      float,
        is_braking: bool,
        exclusion_zones: Optional[List[Tuple[float, float, float]]] = None,
    ) -> dict:
        """
        Main per-step call.  Returns enrichment dict for bsts_row.

        Parameters
        ----------
        wp_idx          : current closest waypoint index
        speed           : current speed m/s
        is_braking      : True if throttle < 0 or speed decreasing
        exclusion_zones : latest ObjectTracker.get_exclusion_zones() (optional)

        Returns
        -------
        dict with:
          brake_potential   float [0,1]  - how urgently braking is needed
          in_brake_field    bool         - agent IS braking inside the field
          is_braking        bool         - pass-through
          brake_distance_m  float        - physics brake distance this step
          corner_speed_target float      - closest corner speed limit
        """
        if exclusion_zones is not None:
            self._exclusion_zones = exclusion_zones

        phi      = self.potential(wp_idx, speed)
        in_field = phi > 0.0

        if is_braking:
            self._braking_events.append(in_field)
            if in_field:
                self._in_field_count += 1

        # Find closest upcoming corner speed target
        cst = 4.0
        if self._corners and self.waypoints is not None:
            n        = len(self.waypoints)
            arc_now  = float(self._arc_dists[wp_idx % n])
            closest  = min(
                self._corners,
                key=lambda c: (c["arc_dist"] - arc_now) % self._total_arc,
            )
            cst = closest["speed_limit"]

        return dict(
            brake_potential      = phi,
            in_brake_field       = bool(in_field and is_braking),
            is_braking           = is_braking,
            brake_distance_m     = braking_distance(speed, self.mu, self.g),
            corner_speed_target  = cst,
        )

    # ----------------------------------------------------------------
    # Episode compliance metric
    # ----------------------------------------------------------------

    @property
    def compliance(self) -> float:
        """Fraction of braking events that occurred inside the brake field."""
        if not self._braking_events:
            return 1.0
        return self._in_field_count / len(self._braking_events)

    def reset(self):
        """Call at episode start."""
        self._braking_events = []
        self._in_field_count = 0
        self._exclusion_zones = []
