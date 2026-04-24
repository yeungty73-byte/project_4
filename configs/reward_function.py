"""
Unified reward function for DeepRacer P4.
Works across time-trial, obstacle-avoidance, and head-to-bot.
Params reference: https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-reference.html
"""

def reward_function(params):
    # ---- unpack ----
    track_width      = params["track_width"]
    distance_center  = params["distance_from_center"]
    all_wheels_on    = params["all_wheels_on_track"]
    speed            = params["speed"]
    steering_abs     = abs(params["steering_angle"])
    progress         = params["progress"]
    steps            = params["steps"]
    is_offtrack      = params["is_offtrack"]
    is_crashed       = params["is_crashed"]
    is_reversed      = params["is_reversed"]

    # ---- terminal penalties ----
    if is_offtrack or is_crashed:
        return 1e-3

    # ---- centerline reward (Gaussian) ----
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    if distance_center <= marker_1:
        center_reward = 1.0
    elif distance_center <= marker_2:
        center_reward = 0.5
    elif distance_center <= marker_3:
        center_reward = 0.1
    else:
        center_reward = 1e-3

    # ---- speed reward (encourage fast driving) ----
    SPEED_THRESHOLD_LOW = 1.0
    SPEED_THRESHOLD_HIGH = 3.0
    if speed < SPEED_THRESHOLD_LOW:
        speed_reward = 0.5
    elif speed < SPEED_THRESHOLD_HIGH:
        speed_reward = 0.5 + 0.5 * (speed - SPEED_THRESHOLD_LOW) / (SPEED_THRESHOLD_HIGH - SPEED_THRESHOLD_LOW)
    else:
        speed_reward = 1.0

    # ---- steering penalty (high speed + high steer = bad) ----
    if steering_abs > 15.0:
        steer_penalty = 0.8
    else:
        steer_penalty = 1.0

    # ---- progress bonus (scaled by efficiency) ----
    if steps > 0:
        progress_bonus = (progress / 100.0) * 2.0
    else:
        progress_bonus = 0.0

    # ---- lap completion bonus ----
    if progress >= 99.9:
        progress_bonus += 10.0

    # ---- heading alignment (smooth cornering) ----
    heading_reward = 1.0 - (steering_abs / 30.0)
    heading_reward = max(heading_reward, 0.0)

    # ---- obstacle / bot awareness via LIDAR (if present) ----
    # The reward function doesn't directly receive LIDAR,
    # but closest_objects and objects_* params exist for obstacles
    obstacle_bonus = 0.0
    if "objects_distance" in params and params.get("objects_distance"):
        closest_dist = min(params["objects_distance"]) if params["objects_distance"] else 999
        if closest_dist < 0.5:
            obstacle_bonus = -0.5  # too close
        elif closest_dist < 1.0:
            obstacle_bonus = 0.0   # caution zone
        else:
            obstacle_bonus = 0.2   # safe margin

    # ---- combine ----
    reward = (
        center_reward * 1.5
        + speed_reward * 1.0
        + steer_penalty * 0.5
        + heading_reward * 0.5
        + progress_bonus
        + obstacle_bonus
    )

    return float(max(reward, 1e-3))
