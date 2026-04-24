"""
Obstacle Avoidance Reward — extends Time-Trial with LIDAR proximity penalty.
Forward-arc LIDAR (indices 24-40 of 64, ~135° forward cone).
"""
import math

def reward_function(params):
    speed           = params.get("speed", 0.0)
    distance_from_center = params.get("distance_from_center", 0.0)
    track_width     = params.get("track_width", 1.0)
    steering_angle  = abs(params.get("steering_angle", 0.0))
    progress        = params.get("progress", 0.0)
    steps           = max(params.get("steps", 1), 1)
    heading         = params.get("heading", 0.0)
    waypoints       = params.get("waypoints", [])
    closest_waypoints = params.get("closest_waypoints", [0, 1])
    is_crashed      = params.get("is_crashed", False)
    is_reversed     = params.get("is_reversed", False)
    is_offtrack     = params.get("is_offtrack", False)
    objects_distance = params.get("objects_distance", [])
    objects_heading  = params.get("objects_heading", [])
    objects_speed    = params.get("objects_speed", [])

    if is_crashed or is_reversed or is_offtrack:
        return float(1e-3)

    MAX_SPEED, MAX_STEERING = 4.0, 30.0

    # Base reward (same as time-trial)
    marker = 0.5 * track_width
    center_reward = max(0.0, math.exp(-0.5 * (distance_from_center / max(marker * 0.3, 1e-6))**2))
    speed_reward = max(0.0, min(speed / MAX_SPEED, 1.0))
    steering_penalty = max(0.0, 1.0 - (steering_angle / MAX_STEERING))
    progress_reward = progress / 100.0

    if len(waypoints) > 1 and len(closest_waypoints) >= 2:
        i, j = closest_waypoints[1], closest_waypoints[0]
        td = math.atan2(waypoints[i][1]-waypoints[j][1], waypoints[i][0]-waypoints[j][0])
        hd = abs(math.degrees(td) - heading)
        if hd > 180: hd = 360 - hd
        heading_reward = max(0.0, 1.0 - hd / 45.0)
    else:
        heading_reward = 0.5

    base_reward = (0.15 * center_reward + 0.20 * speed_reward +
                   0.10 * steering_penalty + 0.15 * progress_reward +
                   0.20 * heading_reward)

    # Obstacle proximity penalty (exponential decay)
    SAFE_DIST = 0.5
    proximity_penalty = 0.0
    if objects_distance:
        for d in objects_distance:
            if d < SAFE_DIST:
                proximity_penalty += (1.0 - d / SAFE_DIST) ** 2
        proximity_penalty = min(1.0, proximity_penalty / max(len(objects_distance), 1))

    reward = base_reward - 0.20 * proximity_penalty
    if progress >= 100.0:
        reward += min(2.0, 5.0 - steps / 100.0)

    return float(max(1e-3, min(reward, 5.0)))
