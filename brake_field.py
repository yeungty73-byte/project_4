"""brake_field.py — v1.3.0
===================================================================================
DESIGN: Per-class Vector Field Brake Field
===================================================================================
SwinUNetPP.forward() → 64-dim fused tensor.
Sector layout used here:
  [:16]  obstacle mask logits (sigmoid → probability each of 16 sectors occupied)
  [16:32] distance estimates per sector (normalised 0=far, 1=near)
  [32:64] SwinEncoder2D camera features (used for context only, not per-sector)

Four object classes, each with its own repulsive scalar field φᵢ:
  CLASS 0 — CENTER_LINE   : reference attractor (slightly attractive, guides race line)
  CLASS 1 — CURB          : short-range anisotropic repulsor; depends on approach angle
  CLASS 2 — OBSTACLE      : full braking repulsor; uses object-permanence dimension + angle
  CLASS 3 — BOT           : dynamic repulsor; projects bot's race line + v_perp constraint

Per-class BrakeVectorField contract
  .step(state) → {
      "v_perp_impact":         float,   # speed component perpendicular to object on impact [m/s]
      "required_decel":        float,   # m/s² needed so v_perp → 0 before contact
      "brake_urgency":         float ∈ [0,1],
      "compliance_gradient":   float ∈ [0,1],
      "field_magnitude":       float ∈ [0,1],
      "class":                 str,
  }

CombinedBrakeField
  .step(state) → union of all four fields + weighted scalar
  .compliance_gradient → episode mean, continuous [0,1]

Race-line harmonisation:
  CombinedBrakeField.race_line_safe_speed(wp_idx, car_speed, race_speeds) returns
  the maximum speed compatible with all active brake fields, so race_line_engine
  can clamp its speed profile before giving rewards.

REFERENCES
  Brayshaw & Harrison (2005) Quasi-steady-state braking point calculation.
    10th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference.
  Heinzmann & Zelinsky (2003) Quantified safety design for human-robot interaction.
    Autonomous Robots 15(2):55–69.  doi:10.1023/A:1024831613974
  Heilmeier et al. (2020) Minimum curvature trajectory planning and control for an
    autonomous race car. Vehicle System Dynamics 58(10):1497–1527.
  Liu et al. (2021) Swin Transformer. ICCV. arXiv:2103.14030
  Petit et al. (2021) U-Net Transformer. arXiv:2103.06104
  Zhou et al. (2019) UNet++. IEEE TAMI. arXiv:1807.10165
  Waymo Open Dataset (2021) Motion prediction with constant-velocity projection.
  Cao et al. (2021) Swin-Unet. arXiv:2105.05537
===================================================================================
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional, Dict, List, Tuple
from config_loader import CFG

# ── physical constants ────────────────────────────────────────────────────────
_MU_DEFAULT    = 0.7   # tyre–surface friction coefficient (DeepRacer urethane on carpet)
_CAR_HALF_W    = 0.09  # DeepRacer 1/18 scale: ~0.18m wide; half-width for clearance subtraction
                       # REF: Heinzmann & Zelinsky (2003) -- safety envelope uses body dimensions
_G             = 9.81  # m s⁻²
_SAFETY        = 1.20  # extra stopping margin factor
_DT_BOT_PROJ   = 1.50  # seconds ahead to project bot position (object permanence)

# ── sector-to-angle mapping for the 16-sector Swin mask ──────────────────────
# Sector 0 = directly ahead, sectors 1–7 clockwise, 8 = directly behind, 9–15 counter-cw.
_SECTOR_ANGLES_RAD = np.array(
    [i * (2 * math.pi / 16) for i in range(16)], dtype=np.float32
)

# ── helper functions ──────────────────────────────────────────────────────────

def _v_perp(speed: float, heading_rad: float, barrier_angle_rad: float) -> float:
    """Speed component perpendicular to the barrier surface on current heading.
    v_⊥ = v · |sin(heading − barrier_normal)|
    At v_⊥=0 the car skims the surface; the constraint is v_⊥ ≤ 0 at contact.
    """
    return float(abs(speed * math.sin(heading_rad - barrier_angle_rad)))


def _req_decel(v_perp_val: float, dist: float, mu: float = _MU_DEFAULT) -> float:
    """Minimum deceleration (m/s²) so that v_⊥ → 0 by dist metres.
    From kinematics:  a_min = v_⊥² / (2d)
    Physical cap: mu·g (maximum available friction deceleration).
    """
    d = max(dist, 0.01)
    return float(min(v_perp_val ** 2 / (2.0 * d), mu * _G))


def _brake_urgency(req_d: float, mu: float = _MU_DEFAULT) -> float:
    """Urgency ∈ [0,1]:  0 = no braking needed, 1 = at physical friction limit."""
    return float(np.clip(req_d / (mu * _G + 1e-9), 0.0, 1.0))


def _compliance_grad(req_d: float, actual_decel: float,
                     is_braking: bool, urgency: float) -> float:
    """Continuous compliance ∈ [0,1].
    If urgency < threshold: full credit (no braking needed).
    If braking: credit proportional to decel quality vs required.
    If not braking despite urgency: credit proportional to 1 − urgency.
    """
    if urgency < 0.12:
        return 1.0
    if is_braking and actual_decel > 0.0:
        ratio = float(np.clip(actual_decel / (req_d + 1e-9), 0.0, 1.0))
        return float(0.25 + 0.75 * ratio)
    return float(np.clip(1.0 - urgency, 0.05, 0.88))


def _stopping_dist(speed: float, mu: float = _MU_DEFAULT) -> float:
    """Minimum stopping distance at current speed under friction deceleration."""
    return float((speed ** 2) / (2.0 * mu * _G + 1e-9)) * _SAFETY


# ── CLASS 0: CENTER LINE field (weak attractor for race-line harmonisation) ──

class CenterLineField:
    """Weak Gaussian attractor toward the center line.
    Used only as a reference for race-line harmonisation, not for braking.
    magnitude ∝ Gaussian of lateral displacement from centerline.
    """
    CLASS_NAME = "center_line"

    def __init__(self, sigma_fraction: float = 0.8):
        self.sigma = sigma_fraction  # in units of track half-width

    def step(self,
             car_lat_offset: float,   # signed metres from centreline (+ve = left)
             track_half_w: float,
             speed: float,
             **_kw) -> Dict:
        hw    = max(track_half_w, 0.1)
        frac  = car_lat_offset / hw
        mag   = float(math.exp(-0.5 * (frac / self.sigma) ** 2))
        return {
            "v_perp_impact":       0.0,
            "required_decel":      0.0,
            "brake_urgency":       0.0,
            "compliance_gradient": 1.0,
            "field_magnitude":     mag,
            "class":               self.CLASS_NAME,
        }


# ── CLASS 1: CURB field (anisotropic short-range repulsor) ────────────────────

class CurbField:
    """Curb repulsion: anisotropic.
    The curb is only dangerous when approached at an angle that brings the car
    laterally into it — i.e. high v_⊥.  A car clipping a curb at near-zero
    lateral angle is fine (cornering technique).  We therefore gate urgency on
    the lateral approach angle:
        curb_urgency = v_⊥ / v_⊥_max  ×  angle_weight
    where angle_weight = |sin(approach_angle)|  (1 = full broadside, 0 = tangent).

    Swin integration: sector mask votes for curb proximity per sector.
    """
    CLASS_NAME = "curb"
    _MAX_CURB_DIST = 0.6  # metres — beyond this, no curb effect

    def __init__(self, mu: float = _MU_DEFAULT):
        self.mu = mu
        self._cg_buf: List[float] = []

    def step(self,
             speed: float,
             heading_rad: float,
             curb_dist: float,         # metres to nearest curb (0 = touching)
             curb_angle_rad: float,    # angle of curb normal relative to car heading
             is_braking: bool = False,
             actual_decel: float = 0.0,
             swin_curb_prob: float = 0.0,   # Swin sector probability this is a curb
             **_kw) -> Dict:
        # Effective dist — Swin votes pull it closer if confidence is high
        eff_dist = float(np.clip(
            curb_dist * (1.0 - 0.3 * swin_curb_prob),
            0.01, self._MAX_CURB_DIST + 1.0
        ))
        if eff_dist > self._MAX_CURB_DIST:
            cg = 1.0
            self._cg_buf.append(cg)
            return {"v_perp_impact": 0.0, "required_decel": 0.0,
                    "brake_urgency": 0.0, "compliance_gradient": 1.0,
                    "field_magnitude": 0.0, "class": self.CLASS_NAME}

        vp     = _v_perp(speed, heading_rad, curb_angle_rad)
        angle_w = abs(math.sin(heading_rad - curb_angle_rad))  # anisotropy
        req_d  = _req_decel(vp, eff_dist, self.mu)
        urg    = _brake_urgency(req_d, self.mu) * float(np.clip(angle_w, 0.0, 1.0))
        cg     = _compliance_grad(req_d, actual_decel, is_braking, urg)
        mag    = float(np.clip(1.0 - eff_dist / self._MAX_CURB_DIST, 0.0, 1.0))

        self._cg_buf.append(cg)
        return {
            "v_perp_impact":       vp,
            "required_decel":      req_d,
            "brake_urgency":       urg,
            "compliance_gradient": cg,
            "field_magnitude":     mag * angle_w,
            "class":               self.CLASS_NAME,
        }

    @property
    def mean_cg(self) -> float:
        return float(np.mean(self._cg_buf)) if self._cg_buf else 1.0

    def reset(self):
        self._cg_buf.clear()


# ── CLASS 2: OBSTACLE (static) field — full object-permanence dimension+angle

class ObstacleField:
    """Static obstacle braking field.
    Object-permanence aware: the obstacle's angular span (visible_angle_deg)
    scales the urgency — a large object at angle blocks more of the path.

    v_⊥ constraint: the car must reach v_⊥ ≤ 0 before impact.
    Required decel = v_⊥² / (2 × d_impact)
    where d_impact = distance × |cos(approach_angle)| (projected stopping depth).

    Swin integration: obstacle sector mask → refine distance + approach angle.
    """
    CLASS_NAME = "obstacle"

    def __init__(self, mu: float = _MU_DEFAULT):
        self.mu = mu
        self._cg_buf: List[float] = []

    def step(self,
             speed: float,
             heading_rad: float,
             obs_dist: float,          # metres to nearest obstacle
             obs_angle_rad: float,     # angle to obstacle centre from car heading
             obs_visible_angle_rad: float = 0.3,  # angular span (object permanence)
             obs_dim_w: float = 0.20,  # estimated width (metres)
             obs_dim_h: float = 0.20,  # estimated height/depth
             is_braking: bool = False,
             actual_decel: float = 0.0,
             swin_obs_prob: float = 0.0,   # Swin mask probability this sector is obstacle
             **_kw) -> Dict:
        # Angular span scales urgency (object-permanence dimension):
        # a wider angular span = more of the car's path is blocked.
        angle_span_weight = float(np.clip(obs_visible_angle_rad / (math.pi / 4), 0.0, 1.0))

        # Object dimension penalty: car half-width vs obstacle half-width
        clearance_needed = max(obs_dim_w / 2.0 + 0.10, 0.15)   # 0.10 = car half-width
        eff_dist = max(obs_dist - clearance_needed, 0.01)

        # Swin confidence boosts urgency
        if swin_obs_prob > 0.6:
            eff_dist = max(eff_dist * (1.0 - 0.25 * swin_obs_prob), 0.01)

        # Impact: approach angle projection → actual stopping depth
        approach_cos = abs(math.cos(heading_rad - obs_angle_rad))
        d_proj = max(eff_dist * approach_cos, 0.01)

        vp     = _v_perp(speed, heading_rad, obs_angle_rad)
        req_d  = _req_decel(vp, d_proj, self.mu)
        urg    = _brake_urgency(req_d, self.mu) * float(
                     np.clip(0.4 + 0.6 * angle_span_weight, 0.0, 1.0))
        cg     = _compliance_grad(req_d, actual_decel, is_braking, urg)
        mag    = float(np.clip(angle_span_weight * (1.0 - eff_dist / max(obs_dist + 0.5, 0.5)),
                               0.0, 1.0))

        self._cg_buf.append(cg)
        return {
            "v_perp_impact":            vp,
            "required_decel":           req_d,
            "brake_urgency":            urg,
            "compliance_gradient":      cg,
            "field_magnitude":          mag,
            "class":                    self.CLASS_NAME,
            "angle_span_weight":        angle_span_weight,
            "obs_eff_dist":             eff_dist,
        }

    @property
    def mean_cg(self) -> float:
        return float(np.mean(self._cg_buf)) if self._cg_buf else 1.0

    def reset(self):
        self._cg_buf.clear()


# ── CLASS 3: BOT field — dynamic, projection-aware, race-line aware ───────────

class BotField:
    """Bot (other DeepRacer) braking field.
    Dynamic — projects bot position T=1.5 s ahead (constant-velocity model;
    Waymo Motion Prediction 2021).
    Accounts for:
      1. Bot's own race line projection → predicted lateral position
      2. Bot's heading angle relative to own car → closing angle → v_⊥
      3. Bot's speed → closing speed = |own_speed - bot_speed| (vector)

    v_⊥ constraint:
      v_⊥_impact = closing_speed × |sin(relative_heading)|
      required_decel = v_⊥_impact² / (2 × projected_dist)

    Swin integration: bot sector probability refines distance estimates.
    """
    CLASS_NAME = "bot"

    def __init__(self, mu: float = _MU_DEFAULT, projection_dt: float = _DT_BOT_PROJ):
        self.mu = mu
        self.dt = projection_dt
        self._cg_buf: List[float] = []

    def _project_bot(self, bot_x: float, bot_y: float,
                     bot_heading_rad: float, bot_speed: float) -> Tuple[float, float]:
        """Constant-velocity projection. REF: Waymo Motion Prediction (2021)."""
        dx = bot_speed * math.cos(bot_heading_rad) * self.dt
        dy = bot_speed * math.sin(bot_heading_rad) * self.dt
        return (bot_x + dx, bot_y + dy)

    def step(self,
             speed: float,
             heading_rad: float,
             car_x: float, car_y: float,
             bot_x: float, bot_y: float,
             bot_heading_rad: float,
             bot_speed: float,
             bot_visible_angle_rad: float = 0.2,
             is_braking: bool = False,
             actual_decel: float = 0.0,
             swin_bot_prob: float = 0.0,
             **_kw) -> Dict:
        # Project bot ahead by self.dt seconds
        proj_x, proj_y = self._project_bot(bot_x, bot_y, bot_heading_rad, bot_speed)

        # Vector from own car to projected bot position
        dx = proj_x - car_x
        dy = proj_y - car_y
        proj_dist = float(math.sqrt(dx ** 2 + dy ** 2) + 1e-6)
        proj_angle = float(math.atan2(dy, dx))

        # Swin confidence tightens the effective distance
        if swin_bot_prob > 0.5:
            proj_dist = max(proj_dist * (1.0 - 0.2 * swin_bot_prob), 0.05)

        # Closing speed in the direction of impact
        own_vx = speed * math.cos(heading_rad)
        own_vy = speed * math.sin(heading_rad)
        bot_vx = bot_speed * math.cos(bot_heading_rad)
        bot_vy = bot_speed * math.sin(bot_heading_rad)
        rel_vx = own_vx - bot_vx
        rel_vy = own_vy - bot_vy
        closing_speed = float(math.sqrt(rel_vx ** 2 + rel_vy ** 2))

        # Relative heading for v_⊥ computation
        rel_heading = float(math.atan2(rel_vy, rel_vx))
        vp = _v_perp(closing_speed, rel_heading, proj_angle)

        # Object-permanence: angular span of bot
        angle_span_w = float(np.clip(
            bot_visible_angle_rad / (math.pi / 8), 0.0, 1.0))

        req_d = _req_decel(vp, proj_dist, self.mu)
        urg   = _brake_urgency(req_d, self.mu) * float(
                    np.clip(0.3 + 0.7 * angle_span_w, 0.0, 1.0))
        cg    = _compliance_grad(req_d, actual_decel, is_braking, urg)
        mag   = float(np.clip(angle_span_w * urg, 0.0, 1.0))

        self._cg_buf.append(cg)
        return {
            "v_perp_impact":         vp,
            "required_decel":        req_d,
            "brake_urgency":         urg,
            "compliance_gradient":   cg,
            "field_magnitude":       mag,
            "class":                 self.CLASS_NAME,
            "projected_bot_x":       proj_x,
            "projected_bot_y":       proj_y,
            "closing_speed":         closing_speed,
            "angle_span_w":          angle_span_w,
        }

    @property
    def mean_cg(self) -> float:
        return float(np.mean(self._cg_buf)) if self._cg_buf else 1.0

    def reset(self):
        self._cg_buf.clear()


# ── COMBINED BRAKE FIELD ─────────────────────────────────────────────────────

class CombinedBrakeField:
    """CombinedBrakeField — aggregates four per-class vector fields.

    Combination policy (all four fields run every step):
      φ_combined = max(φ_curb, φ_obs, φ_bot)  [braking relevance]
      compliance_combined = weighted harmonic mean of per-class compliance gradients,
                            weights proportional to each field's urgency.
      If all urgencies are zero: compliance = 1.0 (nothing to comply with).

    Race-line harmonisation:
      .race_line_safe_speed(wp_idx, car_speed, race_speeds, ...) returns the
      maximum speed at this waypoint that satisfies ALL active brake field
      constraints.  race_line_engine can use this to floor-clamp its speed
      profile, ensuring brake field and race line give consistent advice.

    Swin integration:
      Pass raw obs_flat_np (38464-dim) → SwinUNetPP.forward() →
      64-dim fused:  [:16] mask | [16:32] dist | [32:] cam.
      Per-sector classification:
        class = argmax([mask[s], dist[s] normwise, cam corr]) per sector.
      Fallback: if SwinUNetPP unavailable, uses DeepRacer params dict directly.

    Usage in run.py:
        _bf = CombinedBrakeField(waypoints)
        _bf.set_swin(swin_pp)   # SwinUNetPP instance from utransformer.py
        ...per step...
        out = _bf.step(
            wp_idx=closest_wp,
            speed=rp['speed'], heading_rad=math.radians(rp['heading']),
            car_x=rp['x'], car_y=rp['y'],
            is_braking=bool(_braking_intent), actual_decel=_computed_decel,
            obs_flat_np=obs_raw,         # full 38464-dim flat obs
            rp=rp,                        # DeepRacer params dict (fallback)
        )
    """

    # Sector-to-class assignment from SwinUNetPP output
    # Mask logits: [:16], dist: [16:32], cam: [32:]
    # We classify sector s as:
    #   center_line  → distance from centreline is low (from rp, not Swin)
    #   curb         → dist[s] high + mask[s] low-ish  (near but low confidence obstacle)
    #   obstacle     → mask[s] high + dist[s] medium    (clear obstacle presence)
    #   bot          → mask[s] moderate + dist varies + bot info from rp
    _CLS_CENTERLINE = 0
    _CLS_CURB       = 1
    _CLS_OBSTACLE   = 2
    _CLS_BOT        = 3

    def __init__(self, waypoints=None, mu: float = _MU_DEFAULT):
        _cfg    = CFG.get("brake_field", {})
        self.mu = _cfg.get("mu", mu)
        self.waypoints = (np.array([[w[0], w[1]] for w in waypoints], dtype=np.float64)
                          if waypoints is not None else None)
        # Per-class fields
        self._cl_field   = CenterLineField()
        self._curb_field = CurbField(self.mu)
        self._obs_field  = ObstacleField(self.mu)
        self._bot_field  = BotField(self.mu)
        # SwinUNetPP reference (optional, set via set_swin())
        self._swin = None
        # Episode accumulators
        self._step_count    = 0
        self._cg_weighted   = []   # per-step weighted compliance
        self._in_field_steps = 0
        self._corner_indices: Optional[List[int]] = None

    def set_swin(self, swin_pp) -> None:
        """Attach a SwinUNetPP instance. Must be called once after construction."""
        self._swin = swin_pp

    def set_waypoints(self, waypoints) -> None:
        self.waypoints = np.array([[w[0], w[1]] for w in waypoints], dtype=np.float64)
        self._corner_indices = None

    def reset(self) -> None:
        self._step_count     = 0
        self._cg_weighted.clear()
        self._in_field_steps = 0
        self._corner_indices = None
        self._curb_field.reset()
        self._obs_field.reset()
        self._bot_field.reset()

    # ── Swin sector parsing ───────────────────────────────────────────────────

    def _parse_swin(self, obs_flat_np: Optional[np.ndarray]) -> Dict:
        """Run SwinUNetPP on raw obs and return per-class sector probabilities.
        Returns dict with keys: mask16, dist16, sector_class16, front_clear.
        Falls back to zeros if Swin unavailable.
        """
        null = {"mask16": np.zeros(16, np.float32),
                "dist16": np.ones(16, np.float32),   # 1 = far
                "sector_class16": np.zeros(16, dtype=int),
                "front_clear": 0.5}
        if self._swin is None or obs_flat_np is None:
            return null
        try:
            import torch
            t   = torch.tensor(obs_flat_np.astype(np.float32)).unsqueeze(0)
            out = self._swin(t)[0].detach().numpy()   # (64,)
            mask = 1.0 / (1.0 + np.exp(-out[:16].astype(np.float64)))  # sigmoid
            dist = np.clip(out[16:32].astype(np.float64), 0.0, 1.0)    # normalised dist
            # Sector class: curb=near(dist>0.6) & low obstacle, obstacle=mask>0.5, else clear
            sector_cls = np.zeros(16, dtype=int)
            for s in range(16):
                if mask[s] > 0.55:
                    sector_cls[s] = self._CLS_OBSTACLE
                elif dist[s] > 0.65 and mask[s] < 0.4:
                    sector_cls[s] = self._CLS_CURB
                # center_line and bot assigned externally via rp
            front_clear = float(1.0 - np.max(mask[0:3]))  # sectors 0,1,2 = forward arc
            return {"mask16": mask, "dist16": dist,
                    "sector_class16": sector_cls, "front_clear": front_clear}
        except Exception:
            return null

    # ── Corner potential (same as v1.2.0, for backward compatibility) ─────────

    def _find_corners(self, thresh: float = 0.05) -> None:
        if self.waypoints is None or len(self.waypoints) < 3:
            self._corner_indices = []
            return
        wps = self.waypoints
        self._corner_indices = []
        for i in range(1, len(wps) - 1):
            v1 = wps[i] - wps[i - 1]
            v2 = wps[i + 1] - wps[i]
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            norm  = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            if cross / norm > thresh:
                self._corner_indices.append(i)

    def _corner_potential(self, wp_idx: int, speed: float) -> float:
        if self._corner_indices is None:
            self._find_corners()
        if not self._corner_indices or self.waypoints is None:
            return 0.0
        d_brake = _stopping_dist(speed, self.mu)
        n       = len(self.waypoints)
        min_d   = float("inf")
        for ci in self._corner_indices:
            hops = (ci - wp_idx) % n
            if hops <= 10:
                seg  = np.linalg.norm(self.waypoints[(ci + 1) % n] - self.waypoints[ci])
                min_d = min(min_d, hops * max(float(seg), 0.1))
        return float(np.clip(1.0 - min_d / max(d_brake, 0.01), 0.0, 1.0))                if min_d < float("inf") else 0.0

    # ── Race-line harmonisation ───────────────────────────────────────────────

    def race_line_safe_speed(
        self,
        wp_idx: int,
        car_speed: float,
        race_speeds: Optional[np.ndarray],
        heading_rad: float,
        barrier_angle_rad: float,
        obs_dist: float,
        curb_dist: float,
        bot_dist: float,
    ) -> float:
        """Maximum speed at wp_idx consistent with all brake field constraints.
        race_line_engine should call this and clamp its speed profile.
        Ensures brake field and race line give CONSISTENT advice.

        Return: safe_speed ≥ 0.5 m/s (never stalls the car outright).
        """
        # For each class: max speed so stopping dist ≤ object dist
        speeds = [car_speed]

        def _max_v(dist):
            # v_max s.t. stopping_dist(v) ≤ dist
            # v² / (2·mu·g) ≤ dist  →  v ≤ sqrt(2·mu·g·dist)
            return float(max(math.sqrt(2.0 * self.mu * _G * max(dist, 0.05)), 0.5))

        if obs_dist < 5.0:
            speeds.append(_max_v(obs_dist))
        if curb_dist < 0.8:
            # curbs allow more speed (can graze them)
            speeds.append(_max_v(curb_dist * 1.5))
        if bot_dist < 3.0:
            speeds.append(_max_v(bot_dist * 0.8))

        if race_speeds is not None and len(race_speeds) > 0:
            n = len(race_speeds)
            speeds.append(float(race_speeds[wp_idx % n]))

        return float(max(min(speeds), 0.5))

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(
        self,
        # Track / kinematics
        wp_idx:           int,
        speed:            float,
        heading_rad:      float,
        car_x:            float         = 0.0,
        car_y:            float         = 0.0,
        # Braking state
        is_braking:       bool          = False,
        actual_decel:     float         = 0.0,
        # Barrier geometry (from DeepRacer params / lidar)
        barrier_dist:     float         = 5.0,
        barrier_angle:    float         = 0.0,
        # Curb geometry
        curb_dist:        float         = 5.0,
        curb_angle:       float         = 0.0,
        # Obstacle geometry (static)
        obs_dist:         float         = 5.0,
        obs_angle:        float         = 0.0,
        obs_visible_deg:  float         = 20.0,
        obs_dim_w:        float         = 0.20,
        obs_dim_h:        float         = 0.20,
        # Bot geometry (dynamic)
        bot_x:            float         = 0.0,
        bot_y:            float         = 0.0,
        bot_heading:      float         = 0.0,   # radians
        bot_speed:        float         = 0.0,
        bot_visible_deg:  float         = 15.0,
        # Center line offset
        car_lat_offset:   float         = 0.0,
        track_half_w:     float         = 0.30,
        # SwinUNetPP raw obs (optional)
        obs_flat_np:      Optional[np.ndarray] = None,
        # AdaptiveRewardShaper curb urgency multiplier [1.0, 3.0]
        # Feeds per-WP EMA v_perp danger signal into curb field distance scaling.
        # REF: Khatib (1986) APF -- repulsive potential scales with observed danger.
        curb_urgency_mul: float                = 1.0,
    ) -> Dict:
        """Run all four per-class vector fields and combine.

        Returns a dict with per-class results + combined scalars.
        All keys used by run.py:
          "brake_potential"        float [0,1]
          "compliance_gradient"    float [0,1]  ← continuous, used by harmonized_metrics
          "v_perp"                 float [m/s]
          "urgency"                float [0,1]
          "reward_multiplier"      float [0.2, 2.0]
          "in_brake_field"         bool
          "race_line_safe_speed"   float [m/s]
          "swin_front_clear"       float [0,1]
          + per-class sub-dicts: "curb", "obstacle", "bot", "center_line"
        """
        try:
            swin = self._parse_swin(obs_flat_np)

            # ── Curb field ────────────────────────────────────────────────────
            # Effective curb dist: use barrier_dist if curb_dist not provided
            # Subtract car half-width for physics-accurate clearance
            # Without this, "0.30m to curb" means car edge is 0.30-0.09=0.21m from barrier
            # REF: Heinzmann & Zelinsky (2003) -- safety envelope must account for body width
            _raw_curb = min(curb_dist, barrier_dist) if curb_dist < 4.0 else barrier_dist
            eff_curb_dist = max(0.05, _raw_curb - _CAR_HALF_W)

            # Apply curb_urgency_mul from AdaptiveRewardShaper (per-WP EMA v_perp scaling)
            # curb_urgency_mul > 1.0 shrinks effective distance -> field activates earlier
            # REF: Khatib (1986) APF -- repulsive field magnitude = k / d^2; scaling d is equivalent
            _curb_urgency_scaled = float(curb_urgency_mul)
            if _curb_urgency_scaled > 1.0:
                eff_curb_dist = max(0.05, eff_curb_dist / _curb_urgency_scaled)
            swin_curb_prob = float(np.mean([
                swin["mask16"][s] for s in range(16)
                if swin["sector_class16"][s] == self._CLS_CURB
            ] or [0.0]))
            r_curb = self._curb_field.step(
                speed=speed, heading_rad=heading_rad,
                curb_dist=eff_curb_dist, curb_angle_rad=curb_angle,
                is_braking=is_braking, actual_decel=actual_decel,
                swin_curb_prob=swin_curb_prob,
            )

            # ── Obstacle field ────────────────────────────────────────────────
            swin_obs_prob = float(np.mean([
                swin["mask16"][s] for s in range(16)
                if swin["sector_class16"][s] == self._CLS_OBSTACLE
            ] or [0.0]))
            # Apply car half-width to obstacle distance too
            _eff_obs_dist = max(0.05, float(obs_dist) - _CAR_HALF_W)
            r_obs = self._obs_field.step(
                speed=speed, heading_rad=heading_rad,
                obs_dist=_eff_obs_dist, obs_angle_rad=obs_angle,
                obs_visible_angle_rad=math.radians(max(obs_visible_deg, 5.0)),
                obs_dim_w=obs_dim_w, obs_dim_h=obs_dim_h,
                is_braking=is_braking, actual_decel=actual_decel,
                swin_obs_prob=swin_obs_prob,
            )

            # ── Bot field ─────────────────────────────────────────────────────
            has_bot  = (bot_speed > 0.1) or (abs(bot_x - car_x) + abs(bot_y - car_y) > 0.05)
            if has_bot:
                swin_bot_prob = float(np.max(swin["mask16"]))  # bots trigger max mask
                r_bot = self._bot_field.step(
                    speed=speed, heading_rad=heading_rad,
                    car_x=car_x, car_y=car_y,
                    bot_x=bot_x, bot_y=bot_y,
                    bot_heading_rad=bot_heading,
                    bot_speed=bot_speed,
                    bot_visible_angle_rad=math.radians(max(bot_visible_deg, 5.0)),
                    is_braking=is_braking, actual_decel=actual_decel,
                    swin_bot_prob=swin_bot_prob,
                )
            else:
                r_bot = {"v_perp_impact": 0.0, "required_decel": 0.0,
                         "brake_urgency": 0.0, "compliance_gradient": 1.0,
                         "field_magnitude": 0.0, "class": BotField.CLASS_NAME}

            # ── Center line field ─────────────────────────────────────────────
            r_cl = self._cl_field.step(
                car_lat_offset=car_lat_offset,
                track_half_w=track_half_w,
                speed=speed,
            )

            # ── Corner potential (track geometry aware) ───────────────────────
            phi_corner = self._corner_potential(wp_idx, speed)

            # ── Combined urgency = max across safety-relevant classes ─────────
            urgencies  = [r_curb["brake_urgency"], r_obs["brake_urgency"],
                          r_bot["brake_urgency"]]
            max_urg    = float(max(urgencies))
            in_field   = max_urg > 0.12

            # Dominant v_perp = worst case across classes
            max_vp = float(max(r_curb["v_perp_impact"],
                               r_obs["v_perp_impact"],
                               r_bot["v_perp_impact"]))

            # Weighted compliance harmonic mean (urgency as weight).
            # Zero-urgency fields get weight = 0.05 (small but present).
            def _wt(r):
                return max(r["brake_urgency"], 0.05)
            w_c, w_o, w_b = _wt(r_curb), _wt(r_obs), _wt(r_bot)
            w_tot = w_c + w_o + w_b
            combined_cg = float(
                (w_c * r_curb["compliance_gradient"]
                 + w_o * r_obs["compliance_gradient"]
                 + w_b * r_bot["compliance_gradient"]) / w_tot
            )
            combined_cg = float(np.clip(combined_cg, 0.0, 1.0))

            # Brake potential = max of per-class magnitudes + corner potential
            brake_potential = float(np.clip(
                max(r_curb["field_magnitude"], r_obs["field_magnitude"],
                    r_bot["field_magnitude"], phi_corner),
                0.0, 1.0
            ))

            # Reward multiplier [0.2, 2.0] — penalise non-compliance, reward good braking
            if in_field:
                rw_mult = float(np.clip(0.2 + 1.8 * combined_cg, 0.2, 2.0))
            else:
                rw_mult = 1.0

            # Race-line safe speed: max speed consistent with all fields
            rl_safe_spd = self.race_line_safe_speed(
                wp_idx=wp_idx, car_speed=speed, race_speeds=None,
                heading_rad=heading_rad, barrier_angle_rad=barrier_angle,
                obs_dist=obs_dist, curb_dist=eff_curb_dist,
                bot_dist=float(math.sqrt((bot_x - car_x)**2 + (bot_y - car_y)**2) + 1e-6)
                         if has_bot else 10.0,
            )

            self._step_count     += 1
            self._cg_weighted.append(combined_cg)
            if in_field:
                self._in_field_steps += 1

            return {
                # ── combined scalars (legacy-compatible keys) ─────────────────
                "brake_potential":       brake_potential,
                "potential":             phi_corner,
                "in_brake_field":        in_field,
                "compliance_gradient":   combined_cg,
                "compliance":            bool(in_field and is_braking),
                "v_perp":                max_vp,
                "urgency":               max_urg,
                "reward_multiplier":     rw_mult,
                "swin_front_clear":      swin["front_clear"],
                "race_line_safe_speed":  rl_safe_spd,
                # ── per-class sub-dicts ───────────────────────────────────────
                "center_line":           r_cl,
                "curb":                  r_curb,
                "obstacle":              r_obs,
                "bot":                   r_bot,
                # ── v_perp per class (for ep_step_log) ───────────────────────
                "v_perp_curb":           r_curb["v_perp_impact"],
                "v_perp_obs":            r_obs["v_perp_impact"],
                "v_perp_bot":            r_bot["v_perp_impact"],
                "urgency_curb":          r_curb["brake_urgency"],
                "urgency_obs":           r_obs["brake_urgency"],
                "urgency_bot":           r_bot["brake_urgency"],
                "cg_curb":               r_curb["compliance_gradient"],
                "cg_obs":                r_obs["compliance_gradient"],
                "cg_bot":                r_bot["compliance_gradient"],
            }

        except Exception as _e:
            return {
                "brake_potential": 0.0, "potential": 0.0, "in_brake_field": False,
                "compliance_gradient": 1.0, "compliance": False, "v_perp": 0.0,
                "urgency": 0.0, "reward_multiplier": 1.0, "swin_front_clear": 0.5,
                "race_line_safe_speed": speed,
                "center_line": {}, "curb": {}, "obstacle": {}, "bot": {},
                "v_perp_curb": 0.0, "v_perp_obs": 0.0, "v_perp_bot": 0.0,
                "urgency_curb": 0.0, "urgency_obs": 0.0, "urgency_bot": 0.0,
                "cg_curb": 1.0, "cg_obs": 1.0, "cg_bot": 1.0,
                "_error": str(_e),
            }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def compliance_rate(self) -> float:
        if self._step_count == 0:
            return 0.0
        return self._in_field_steps / max(self._step_count, 1)

    @property
    def mean_compliance_gradient(self) -> float:
        if not self._cg_weighted:
            return 1.0
        return float(np.clip(np.mean(self._cg_weighted), 0.0, 1.0))


# ── BACKWARD COMPATIBILITY: BrakeField alias ──────────────────────────────────
# v1.3.0: CombinedBrakeField is the canonical class.
# run.py imports BrakeField — keep alias so no import changes needed.
BrakeField = CombinedBrakeField


# ── MODULE-LEVEL HELPERS (used by run.py directly) ───────────────────────────

def braking_distance(speed, mu=_MU_DEFAULT, g=_G, safety=_SAFETY):
    """Keep v1.2.0 API: raw braking distance."""
    return float((speed ** 2 / (2.0 * mu * g + 1e-8)) * safety + speed * 0.08)


def v_perp_impact(speed, heading_rad, barrier_angle_rad):
    """Keep v1.2.0 API."""
    return _v_perp(speed, heading_rad, barrier_angle_rad)


def required_decel_to_zero_perp(speed, heading_rad, barrier_angle_rad,
                                  dist_to_barrier, mu=_MU_DEFAULT, g=_G):
    """Keep v1.2.0 API."""
    vp = _v_perp(speed, heading_rad, barrier_angle_rad)
    return _req_decel(vp, dist_to_barrier, mu)


def brake_compliance_gradient(speed, heading_rad, barrier_angle_rad,
                                dist_to_barrier, is_braking, actual_decel,
                                mu=_MU_DEFAULT, g=_G):
    """Keep v1.2.0 API."""
    vp    = _v_perp(speed, heading_rad, barrier_angle_rad)
    req_d = _req_decel(vp, dist_to_barrier, mu)
    urg   = _brake_urgency(req_d, mu)
    return _compliance_grad(req_d, actual_decel, is_braking, urg)
