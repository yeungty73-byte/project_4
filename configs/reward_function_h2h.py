"""
Head-to-Bot Reward — extends Time-Trial with 360° LIDAR proximity awareness.
Bots can approach from any direction, so use full LIDAR scan.
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
    objects_speed    = params.get("objects_speed", [])

    if is_crashed or is_reversed or is_offtrack:
        return float(1e-3)

    MAX_SPEED, MAX_STEERING = 4.0, 30.0

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

    base_reward = (0.15 * center_reward + 0.25 * speed_reward +
                   0.05 * steering_penalty + 0.15 * progress_reward +
                   0.15 * heading_reward)

    # 360° proximity awareness (bots approach from any angle)
    SAFE_DIST = 0.4
    proximity_penalty = 0.0
    if objects_distance:
        for d in objects_distance:
            if d < SAFE_DIST:
                proximity_penalty += (1.0 - d / SAFE_DIST) ** 2
        proximity_penalty = min(1.0, proximity_penalty / max(len(objects_distance), 1))

    # Overtake bonus: faster than nearby bots = good
    overtake_bonus = 0.0
    if objects_speed and objects_distance:
        for d, s in zip(objects_distance, objects_speed):
            if d < 0.8 and speed > s:
                overtake_bonus += 0.1 * (speed - s) / MAX_SPEED
        overtake_bonus = min(0.3, overtake_bonus)

    # Survival bonus (still racing = still winning)
    survival = 0.05

    reward = base_reward - 0.15 * proximity_penalty + overtake_bonus + survival
    if progress >= 100.0:
        reward += min(2.0, 5.0 - steps / 100.0)

    return float(max(1e-3, min(reward, 5.0)))
