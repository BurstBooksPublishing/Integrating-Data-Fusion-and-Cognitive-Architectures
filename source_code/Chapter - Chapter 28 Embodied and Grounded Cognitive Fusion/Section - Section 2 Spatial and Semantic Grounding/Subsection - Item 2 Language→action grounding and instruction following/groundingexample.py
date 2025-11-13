import math, re

# Scene graph: list of entities with labels, positions (x,y), and affordances.
scene_graph = [
    {"id":"o1","label":"red cup","pos":(1.2,0.5),"affordances":["grasp"]},
    {"id":"o2","label":"blue cup","pos":(2.0,0.6),"affordances":["grasp"]},
    {"id":"o3","label":"table","pos":(1.5,0.0),"affordances":["support"]}
]

def parse_instruction(instr):
    # naive parse: extract color and noun (e.g., "red cup")
    m = re.search(r"(red|blue|green)?\s*(cup|bottle|plate)", instr)
    return (m.group(1)+" "+m.group(2)).strip() if m else None

def euclid(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def find_best_match(frame, scene):
    # score by exact label match then distance to robot origin (0,0)
    candidates = [e for e in scene if frame in e["label"]]
    if not candidates: return None
    # tie-break: prefer nearer object
    return min(candidates, key=lambda e: euclid(e["pos"], (0.0,0.0)))

def compute_pick_pose(obj):
    x,y = obj["pos"]
    # simple offset above object for top-down pick
    return (x, y, 0.15)  # z = 0.15m above surface

def plan_from_instruction(instr, scene):
    frame = parse_instruction(instr)
    if not frame: raise ValueError("Could not parse intent")
    obj = find_best_match(frame, scene)
    if not obj: raise LookupError("No matching object in scene")
    if "grasp" not in obj["affordances"]: raise RuntimeError("Object not graspable")
    pick_pose = compute_pick_pose(obj)
    # return action primitive dict
    return {"action":"pick","target_id":obj["id"],"pose":pick_pose}

# Example usage
instr = "Pick up the red cup on the table"
action = plan_from_instruction(instr, scene_graph)
print(action)  # {'action':'pick','target_id':'o1','pose':(1.2,0.5,0.15)}