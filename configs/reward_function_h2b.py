def reward_function(params):
    """Head-to-Bot: TT base + 360 bot proximity + overtake + survival.
    Game-theoretic framing: bots are fixed-policy, so single-agent MDP."""
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
    objects_speed = params.get("objects_speed", [])
    closest_objects = params.get("closest_objects", [0, 0])

    if is_crashed or is_offtrack or is_reversed:
        return float(1e-3)

    half_w = track_width * 0.5
    center = math.exp(-0.5 * (distance_from_center / max(half_w * 0.25, 0.01)) ** 2)
    spd = min(speed / 4.0, 1.0)
    prog = progress / 100.0
    steer = 1.0 - abs(steering_angle) / 30.0

    nw = waypoints[closest_waypoints[1]]
    pw = waypoints[closest_waypoints[0]]
    td = math.degrees(math.atan2(nw[1]-pw[1], nw[0]-pw[0]))
    dd = abs(heading - td)
    if dd > 180: dd = 360 - dd
    hdg = 1.0 - dd / 180.0

    spd_adv = 0.0
    if objects_speed and closest_objects[0] < len(objects_speed):
        if speed > objects_speed[closest_objects[0]]:
            spd_adv = 0.1

    bot_pen = 0.0
    if objects_distance:
        for d in objects_distance:
            if d < 0.3: bot_pen += 0.3 * (1.0 - d / 0.3)
            elif d < 0.6: bot_pen += 0.1 * (1.0 - d / 0.6)
    bot_pen = min(bot_pen, 0.5)

    overtake = 0.0
    if objects_distance and len(closest_objects) >= 2:
        bi = closest_objects[1]
        if bi < len(objects_distance) and objects_distance[bi] < 1.0:
            overtake = 0.08

    survival = min(steps / 1000.0, 0.1)

    reward = (0.20*center + 0.15*spd + spd_adv + 0.20*prog + 
              0.08*steer + 0.10*hdg + overtake + survival - bot_pen)
    if progress >= 99.0: reward += 2.5

    return float(max(1e-3, min(reward, 5.0)))
