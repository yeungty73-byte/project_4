import math
import numpy as np


def lookahead_curvature_scan(waypoints, closest, max_lookahead=15, step=2):
# REF: Gonzalez, R. et al. (2020). Advanced speed planning for autonomous vehicles. IEEE Trans. ITS.
    """Scan ahead for the sharpest upcoming corner.
    Returns: (max_curv, max_curv_wp_idx, safe_speed_at_max, distance_to_max)
    """
    if not waypoints or len(waypoints) < 5:
        return 0.0, 0, 4.0, 999.0
    n = len(waypoints)
    idx = closest[1] if len(closest) > 1 else 0
    max_curv = 0.0
    max_curv_idx = idx
    cum_dist = 0.0
    dist_to_max = 0.0
    prev_wp = waypoints[idx % n]
    for i in range(1, max_lookahead + 1, step):
        wi = (idx + i) % n
        wp = waypoints[wi]
        seg_dist = math.sqrt((wp[0]-prev_wp[0])**2 + (wp[1]-prev_wp[1])**2)
        cum_dist += seg_dist
        prev_wp = wp
        # 3-point curvature at this lookahead
        p0 = waypoints[(wi - 2) % n]
        p1 = waypoints[wi]
        p2 = waypoints[(wi + 2) % n]
        ax, ay = p0[0]-p1[0], p0[1]-p1[1]
        bx, by = p2[0]-p1[0], p2[1]-p1[1]
        cross = abs(ax*by - ay*bx)
        d01 = (ax**2+ay**2)**0.5 + 1e-8
        d12 = (bx**2+by**2)**0.5 + 1e-8
        cx, cy = p2[0]-p0[0], p2[1]-p0[1]
        d02 = (cx**2+cy**2)**0.5 + 1e-8
        curv = 2.0 * cross / (d01 * d12 * d02 + 1e-8)
        if curv > max_curv:
            max_curv = curv
            max_curv_idx = wi
            dist_to_max = cum_dist
    safe_speed = min(4.0, max(1.0, 0.8 * (1.0 / (max_curv + 1e-6))**0.5))
    return max_curv, max_curv_idx, safe_speed, dist_to_max


def compute_braking_reward(current_speed, safe_speed_ahead, dist_to_corner,
                          decel_rate=0.3, dt=0.1):
    """F1-style braking reward: reward proactive trail braking.
    Positive rewards for good speed management, no punishment.
    Returns: braking_reward in [0, 2]
    """
    if current_speed <= safe_speed_ahead:
        return 1.0  # already at or below safe speed - good
    speed_diff = current_speed - safe_speed_ahead
    steps_needed = speed_diff / max(decel_rate, 0.05)
    avg_speed_during_brake = (current_speed + safe_speed_ahead) / 2.0
    brake_dist_needed = avg_speed_during_brake * steps_needed * dt
    if dist_to_corner <= 0:
        # Past the corner apex - reward if speed is close to safe
        return max(0.0, 1.0 - speed_diff)
    ratio = brake_dist_needed / max(dist_to_corner, 0.01)
    if ratio <= 0.5:
        return 1.5  # plenty of room, good early positioning
    elif ratio <= 0.8:
        return 2.0  # trail braking zone - F1 optimal!
    elif ratio <= 1.0:
        return 1.0  # tight but manageable
    elif ratio <= 1.3:
        return 0.5  # cutting it close but still ok
    else:
        return 0.2  # too fast but still positive (no punishment)


def compute_turn_alignment_reward(heading, waypoints, closest, lookahead=5):
    """F1-style racing line reward: reward for proper turn-in angle.
    Returns alignment_reward in [0, 2]
    """
    if not waypoints or len(waypoints) < 3:
        return 1.0
    n = len(waypoints)
    idx = closest[1] if len(closest) > 1 else 0
    target_wp = waypoints[(idx + lookahead) % n]
    current_wp = waypoints[idx % n]
    dx = target_wp[0] - current_wp[0]
    dy = target_wp[1] - current_wp[1]
    target_heading = math.atan2(dy, dx)
    heading_rad = math.radians(heading)
    angle_diff = abs(heading_rad - target_heading)
    angle_diff = min(angle_diff, 2*math.pi - angle_diff)
    # Curvature at current point
    p0 = waypoints[(idx - 2) % n]
    p1 = waypoints[idx % n]
    p2 = waypoints[(idx + 2) % n]
    ax, ay = p0[0]-p1[0], p0[1]-p1[1]
    bx, by = p2[0]-p1[0], p2[1]-p1[1]
    cross = abs(ax*by - ay*bx)
    d01 = (ax**2+ay**2)**0.5 + 1e-8
    d12 = (bx**2+by**2)**0.5 + 1e-8
    cx, cy = p2[0]-p0[0], p2[1]-p0[1]
    d02 = (cx**2+cy**2)**0.5 + 1e-8
    local_curv = 2.0 * cross / (d01 * d12 * d02 + 1e-8)
    # In corners, alignment is MORE important
    curv_weight = min(2.0, 1.0 + local_curv * 5.0)
    # Max reward when perfectly aligned, scales down with angle diff
    alignment = max(0.0, 1.0 - angle_diff / (math.pi / 2))
    return alignment * curv_weight


def compute_racing_line_reward(current_speed, heading, waypoints, closest):
    """Combined corner reward: braking + alignment + apex speed.
    REF: Tian et al. (2024) "Balanced reward-inspired RL for autonomous racing"
    REF: Ng, Harada & Russell (1999) "Policy invariance under reward transformations"
    REF: PMC12708685 (2025) "Reward design for generalizable DRL"
    All positive rewards, no punishment.
    Returns: total_corner_reward in [0, 5]
    """
    max_curv, max_curv_idx, safe_speed, dist_to_max = lookahead_curvature_scan(
        waypoints, closest)
    braking_r = compute_braking_reward(current_speed, safe_speed, dist_to_max)
    align_r = compute_turn_alignment_reward(heading, waypoints, closest)
    # Apex speed bonus: reward carrying max safe speed through corners
    if max_curv > 0.05:  # in a corner zone
        speed_ratio = current_speed / max(safe_speed, 0.5)
        if 0.8 <= speed_ratio <= 1.1:
            apex_bonus = 1.5  # carrying optimal speed through corner!
        elif 0.6 <= speed_ratio <= 1.2:
            apex_bonus = 0.8
        else:
            apex_bonus = 0.3
    else:
        apex_bonus = 0.5  # on straight, mild bonus
    return braking_r + align_r + apex_bonus


def get_stuck_antecedent_bonus(stuck_tracker, waypoints, closest, lookback=5):
    """For waypoints approaching a known stuck zone, boost reward for
    correct approach behavior (slower speed, proper steering angle).
    Returns: (is_approaching_stuck, bonus_multiplier, stuck_wp)
    """
    if not stuck_tracker or not hasattr(stuck_tracker, 'stats'):
        return False, 1.0, -1
    n = len(waypoints) if waypoints else 0
    idx = closest[1] if len(closest) > 1 else 0
    # Check if any of the next lookback waypoints are stuck zones
    for ahead in range(1, lookback + 1):
        future_wp = (idx + ahead) % n if n > 0 else ahead
        cluster = stuck_tracker._cluster(future_wp) if hasattr(stuck_tracker, '_cluster') else future_wp
        stat = stuck_tracker.stats.get(cluster)
        if stat and stat.total_episodes >= 3 and stat.breakout_rate < 0.4:
            # This is a known stuck zone ahead
            proximity = 1.0 - (ahead / lookback)  # closer = stronger
            bonus = 1.0 + proximity * (1.0 - stat.breakout_rate) * 2.0
            return True, bonus, future_wp
    return False, 1.0, -1


def curvature_radius(waypoints, current_wp, lookahead=3):
    """Estimate turn radius at current_wp using Garlick & Middleditch (2022) method."""
    import numpy as np
    n = len(waypoints)
    idx = int(current_wp) % n
    prev_idx = (idx - lookahead) % n
    next_idx = (idx + lookahead) % n
    p1 = np.array(waypoints[prev_idx][:2])
    p2 = np.array(waypoints[idx][:2])
    p3 = np.array(waypoints[next_idx][:2])
    d1 = np.linalg.norm(p2 - p1)
    d2 = np.linalg.norm(p3 - p2)
    d3 = np.linalg.norm(p3 - p1)
    area = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])) / 2.0
    if area < 1e-6:
        return float('inf')  # straight line
    return (d1 * d2 * d3) / (4.0 * area)


def optimal_speed(radius, C=9.81):
    """Compute optimal speed from radius using Gonzalez (2020): v = sqrt(C * r)."""
    import math
    return math.sqrt(C * max(radius, 0.01))

    
# =============================================================================
# Integrated from research_modules.py (phased out)
# =============================================================================


# REF: Garlick, J., & Middleditch, A. (2022). Real-time optimal racing
#      line computation. Computer Graphics Forum, 41(8), 293-304.
class LineOfSightReward:
    """Line-of-sight reward: bonus for heading toward upcoming waypoints."""
    def __init__(self, lookahead=5, weight=0.3):
        self.la = lookahead
        self.w = weight

    def compute(self, x, y, hdg, wpts, ci):
        if not wpts or len(wpts) < 2:
            return 0.0
        n = len(wpts)
        bonus = 0.0
        for k in range(1, self.la + 1):
            wx, wy = wpts[(ci + k) % n][:2]
            dx, dy = wx - x, wy - y
            dist = max(math.hypot(dx, dy), 1e-6)
            ta = math.atan2(dy, dx)
            ad = abs(math.atan2(math.sin(ta - hdg), math.cos(ta - hdg)))
            bonus += math.exp(-ad) * math.exp(-dist / 5.0) / k
        return self.w * bonus / self.la


# REF: Yang, S., Li, Z., & Wang, H. (2023). Corner classification for
#      autonomous racing. In IEEE COMPSAC (pp. 1121-1126).
class CornerAnalyzer:
    """Corner classification and curvature-based speed targets."""
    THRESH = [200, 50, 15]

    def classify(self, r):
        for i, t in enumerate(self.THRESH):
            if r > t:
                return i
        return 3

    def speed_target(self, r, vmax=4.0):
        return [vmax, vmax * 0.8, vmax * 0.55, vmax * 0.35][self.classify(r)]

    def corner_reward(self, v, r):
        t = self.speed_target(r)
        return max(0.0, 1.0 - abs(v - t) / max(t, 0.1))

    def curvature_at(self, wpts, ci, w=3):
        """REF: Coulom, R. (2002). Reinforcement learning using neural
        networks [Doctoral dissertation]. Institut National Polytechnique
        de Grenoble."""
        return curvature_radius(wpts, ci, w=w)

    def optimal_speed_at(self, curv, C=1.5, vmin=0.5, vmax=4.0):
        """REF: Gonzalez, R. et al. (2020). Advanced speed planning."""
        return optimal_speed(curv, C=C)


# REF: Yang, S. et al. (2023). Reward shaping for overtake in
#      multi-agent racing. In REUNS Workshop.
class OvertakeAnalyzer:
    """Overtake reward shaping for safe passing."""
    def __init__(self, safe=1.5, bonus=2.0):
        self.safe = safe
        self.bonus = bonus

    def compute(self, own_prog, bot_prog, lidar_min):
        return (
            self.bonus * max(0.0, own_prog - bot_prog)
            + (-1.0 if lidar_min < self.safe else 0.0)
        )

def optimal_speed(radius, C=1.5):
    if radius == float('inf') or radius > 1000:
        return 4.0  # max speed on straight
    return min(4.0, math.sqrt(C * max(radius, 0.1)))


# ============================================================
# Precomputed Racing Line State Map
# ============================================================
# For each waypoint, precompute the optimal (speed, lateral_offset, heading)
# so the reward is simply: how close are you to the optimal state?
# On straights: optimal speed = max → full throttle rewarded maximally.
# On corners: optimal speed = sqrt(C*r), optimal position = apex cut.
# This replaces all penalty-based sub-rewards with a single positive
# proximity-to-optimal reward.

def build_racing_line_map(waypoints, track_width, v_max=4.0, C=1.5, mu=0.7, chassis_w=0.20, chassis_l=0.25):
    """Precompute per-waypoint optimal state for time-trial racing line.
    
    Returns list of dicts, one per waypoint:
        {speed, heading, lateral_offset, curvature, is_straight}
    
    lateral_offset: fraction of half-track-width from center.
        Positive = left of center, negative = right.
        On corners, cut toward apex (inside of turn).
    """
    n = len(waypoints)
    if n < 5:
        return [{"speed": v_max, "heading": 0.0, "lateral_offset": 0.0,
                 "curvature": 0.0, "is_straight": True} for _ in range(max(n, 1))]
    
    race_map = []
    hw = track_width / 2.0  # half width
    
    for i in range(n):
        # 3-point curvature at waypoint i
        p0 = waypoints[(i - 2) % n]
        p1 = waypoints[i]
        p2 = waypoints[(i + 2) % n]
        
        ax, ay = p0[0] - p1[0], p0[1] - p1[1]
        bx, by = p2[0] - p1[0], p2[1] - p1[1]
        cross = ax * by - ay * bx  # signed cross product
        d01 = math.sqrt(ax**2 + ay**2) + 1e-8
        d12 = math.sqrt(bx**2 + by**2) + 1e-8
        cx, cy = p2[0] - p0[0], p2[1] - p0[1]
        d02 = math.sqrt(cx**2 + cy**2) + 1e-8
        kappa = 2.0 * abs(cross) / (d01 * d12 * d02 + 1e-8)  # unsigned curvature
        signed_kappa = 2.0 * cross / (d01 * d12 * d02 + 1e-8)  # + = left turn
        
        # Optimal speed: on straights go max, on corners v=sqrt(C*r)
        radius = 1.0 / max(kappa, 1e-6)
        is_straight = kappa < 0.02  # threshold for "straight"
        if is_straight:
            opt_speed = v_max  # GAS TO THE MAX
        else:
            opt_speed = min(v_max, math.sqrt(C * max(radius, 0.1)))
        
        # Optimal heading: tangent direction
        p_prev = waypoints[(i - 1) % n]
        p_next = waypoints[(i + 1) % n]
        opt_heading = math.degrees(math.atan2(
            p_next[1] - p_prev[1], p_next[0] - p_prev[0]))
        
        # Optimal lateral offset: on straights stay center,
        # on corners cut toward inside (apex strategy)
        if is_straight:
            lat_offset = 0.0  # center of track
        else:
            # Cut inside: if turning left (positive kappa), go right (-)
            # Magnitude scales with curvature, max 0.6 of half-width
            cut_amount = min(0.6, kappa * 15.0)  # aggressive apex cut
            lat_offset = -math.copysign(cut_amount, signed_kappa)
        
        race_map.append({
            "speed": opt_speed,
            "heading": opt_heading,
            "lateral_offset": lat_offset,
            "curvature": kappa,
            "signed_curvature": signed_kappa,
            "is_straight": is_straight,
        })
    
    # Smooth the speed profile: brake BEFORE corners, not AT them
    # Work backwards from each corner to create a braking zone
    for i in range(n):
        for lookback in range(1, 8):
            j = (i - lookback) % n
            # If waypoint j is faster than what's needed to slow to i
            max_entry_speed = math.sqrt(
                race_map[i]["speed"]**2 + 2 * 3.0 * lookback * 0.5  # 3.0 m/s^2 decel
            )
            if race_map[j]["speed"] > max_entry_speed:
                race_map[j]["speed"] = max_entry_speed
                race_map[j]["is_straight"] = False  # braking zone
    
    return race_map


def racing_line_reward(race_map, waypoint_idx, speed, dist_from_center,
                       heading, track_width, is_left_of_center,
                       _waypoints=None, _closest=None):
    """Single unified reward: proximity to optimal racing line state.
    
    Returns reward in [0, ~10+] — all positive, no penalties.
    On straights at max speed: reward is HUGE (up to 10+).
    Off the optimal line: reward decays via Gaussian but never goes negative.
    """
    n = len(race_map)
    if n == 0:
        return 1.0
    
    wp = waypoint_idx % n
    opt = race_map[wp]
    hw = track_width / 2.0
    
    # === 1. Speed proximity reward ===
    opt_speed = opt["speed"]
    speed_err = abs(speed - opt_speed) / max(opt_speed, 0.5)
    
    if opt["is_straight"] and speed >= opt_speed * 0.9:
        # ON A STRAIGHT GOING FAST: reward to oblivion
        # Exponential bonus for being at or above optimal speed
        straight_bonus = 5.0 * math.exp(min(speed / max(opt_speed, 0.5), 2.0) - 1.0)
        speed_reward = straight_bonus
    else:
        # Gaussian reward centered on optimal speed
        speed_reward = 3.0 * math.exp(-0.5 * (speed_err / 0.3) ** 2)
    
    # === 2. Lateral position proximity reward ===
    # Convert actual position to same frame as optimal offset
    actual_offset = (dist_from_center / max(hw, 0.1)) * (1 if is_left_of_center else -1)
    opt_offset = opt["lateral_offset"]
    lat_err = abs(actual_offset - opt_offset)
    lat_sigma = 0.4 + 0.20 / max(track_width, 0.01)
    lat_reward = 2.0 * math.exp(-0.5 * (lat_err / lat_sigma) ** 2)
    
    # === 3. Heading alignment reward ===
    opt_heading = opt["heading"]
    hdg_diff = abs(heading - opt_heading)
    if hdg_diff > 180:
        hdg_diff = 360 - hdg_diff
    hdg_reward = 2.0 * math.exp(-0.5 * (hdg_diff / 15.0) ** 2)
    
    # === 4. Progress multiplier: reward being further along ===
    # (handled separately in run.py via progress reward)
    
    total = speed_reward + lat_reward + hdg_reward
    return total

