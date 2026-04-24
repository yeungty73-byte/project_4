# project_4 v205: continuous-action reward for TD3/SAC/PPO ensemble
# Shapes: centerline + progress + speed + off-track/crash penalty + steering smoothness
def reward_function(params):
    # --- Core params ---
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    progress = params.get('progress', 0.0)
    steps = max(1, params.get('steps', 1))
    speed = params.get('speed', 0.0)
    steering = abs(params.get('steering_angle', 0.0))
    all_wheels_on_track = params.get('all_wheels_on_track', True)
    is_offtrack = params.get('is_offtrack', False)
    is_crashed = params.get('is_crashed', False)

    # --- Terminal penalties ---
    if is_offtrack or is_crashed or not all_wheels_on_track:
        return 1e-3

    # --- Centerline marker reward (AWS template) ---
    m1 = 0.1 * track_width
    m2 = 0.25 * track_width
    m3 = 0.5 * track_width
    if distance_from_center <= m1:
        center_r = 1.0
    elif distance_from_center <= m2:
        center_r = 0.5
    elif distance_from_center <= m3:
        center_r = 0.1
    else:
        center_r = 1e-3

    # --- Progress-per-step (encourages finishing laps fast) ---
    prog_r = (progress / steps) * 0.5

    # --- Speed bonus (continuous action space: speed in [0.5, 4.0]) ---
    speed_r = max(0.0, (speed - 1.0) / 3.0) * 0.3  # 0 at 1 m/s, 0.3 at 4 m/s

    # --- Steering smoothness penalty (>20 deg gets docked) ---
    steer_p = 0.15 if steering > 20.0 else 0.0

    reward = center_r + prog_r + speed_r - steer_p
    return float(max(1e-3, reward))
