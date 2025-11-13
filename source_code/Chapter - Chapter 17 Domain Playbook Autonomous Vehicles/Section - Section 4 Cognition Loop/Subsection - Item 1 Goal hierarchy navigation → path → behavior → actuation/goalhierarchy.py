import math, time

# Simple scene representation
scene = {'obstacles':[ (5,5), (6,5) ], 'agents':[ (8,3) ], 'speed_limit':10}

# Mission-level navigation goal
G_nav = {'target': (10,0), 'preferred_lane': 1}

def plan_path(start, target, scene):
    # straight-line path with coarse collision check
    path = [start, target]
    for ox,oy in scene['obstacles']:
        # distance from line segment; reject if too close
        dx,dy = target[0]-start[0], target[1]-start[1]
        if dx==0 and dy==0: continue
        t = max(0, min(1, ((ox-start[0])*dx + (oy-start[1])*dy)/(dx*dx+dy*dy)))
        px,py = start[0]+t*dx, start[1]+t*dy
        if math.hypot(px-ox, py-oy) < 1.0: 
            return None  # collision risk
    return path

def select_behavior(path, scene):
    # rule-based arbitration: if obstacles close, yield; else follow
    if path is None: 
        return 'abort'
    for ax,ay in scene['agents']:
        # simple proximity rule
        if math.hypot(path[-1][0]-ax, path[-1][1]-ay) < 3.0:
            return 'yield'
    return 'follow'

def actuate(behavior, path, current_speed):
    # map behavior to throttle/steer setpoints (very simple)
    if behavior=='abort':
        return {'throttle': -1.0, 'steer':0.0}  # brake hard
    if behavior=='yield':
        return {'throttle': max(-0.5, current_speed-1.0), 'steer':0.0}
    # follow: move towards last waypoint at speed limit
    target = path[-1]
    steer = math.atan2(target[1], target[0])
    throttle = min(1.0, 0.5 + 0.1*(10-current_speed))
    return {'throttle': throttle, 'steer': steer}

# runtime loop
start = (0,0); speed = 5.0
path = plan_path(start, G_nav['target'], scene)
behavior = select_behavior(path, scene)
cmd = actuate(behavior, path, speed)
print("path:", path, "behavior:", behavior, "cmd:", cmd)