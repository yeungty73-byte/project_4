def reward_function(params):
    """Object Avoidance: TT base + LIDAR proximity penalty + obstacle avoidance bonus.
    Informed by Knox et al. (arXiv:2104.13906) sanity checks."""
    import math

    distance_from_center = params["distance_from_center"]
    track_width = params["track_width"]
    speed = params["speed"]
    steering_angle = params["steering_angle"]
    progress = params["progress"]
    steps = params["steps"]
    heading = params["heading"]
    closest_waypoints = params["closest_waypoints"]
    waypoints = params["waypoints"]
    is_crashed = params.get("is_crashed", False)
    is_offtrack = params.get("is_offtrack", False)
    is_reversed = params.get("is_reversed", False)
    objects_distance = params.get("objects_distance", [])
    closest_objects = params.get("closest_objects", [0, 0])

    if is_crashed or is_offtrack or is_reversed:
        return float(1e-3)

    half_w = track_width * 0.5
    center = math.exp(-0.5 * (distance_from_center / max(half_w * 0.3, 0.01)) ** 2)
    spd = min(speed / 4.0, 1.0)
    prog = progress / 100.0
    steer = 1.0 - abs(steering_angle) / 30.0

    nw = waypoints[closest_waypoints[1]]
    pw = waypoints[closest_waypoints[0]]
    td = math.degrees(math.atan2(nw[1]-pw[1], nw[0]-pw[0]))
    dd = abs(heading - td)
    if dd > 180: dd = 360 - dd
    hdg = 1.0 - dd / 180.0

    obs_pen = 0.0
    if objects_distance:
        md = min(objects_distance)
        if md < 0.5: obs_pen = 0.4 * (1.0 - md / 0.5)
        elif md < 0.8: obs_pen = 0.1 * (1.0 - md / 0.8)

    avoid_bonus = 0.0
    if objects_distance and len(closest_objects) >= 2:
        bi = closest_objects[1]
        if bi < len(objects_distance) and objects_distance[bi] < 1.0:
            avoid_bonus = 0.05

    reward = 0.25*center + 0.15*spd + 0.15*prog + 0.10*steer + 0.10*hdg + avoid_bonus - obs_pen
    if progress >= 99.0: reward += 2.0
    if steps > 0: reward += 0.05 * min(progress / (steps * 0.01), 1.0)

    return float(max(1e-3, min(reward, 5.0)))
