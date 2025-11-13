import math, time, random

# Config thresholds and budgets
LAMBDA = 0.5           # cost weight
BANDWIDTH_BUDGET = 10  # MB/s
MAP_STALE_THRESH = 0.7
DWELL_SECONDS = 2

# Actions with estimated IG and cost
ACTIONS = {
    "increase_lidar_fps": {"IG": 0.8, "cost": 2.0, "bw": 1.0},
    "heavy_seg_model":    {"IG": 1.2, "cost": 4.0, "bw": 0.5},
    "request_map_refresh":{"IG": 1.0, "cost": 6.0, "bw": 8.0},
    "do_nothing":         {"IG": 0.0, "cost": 0.0, "bw": 0.0},
}

# simple staleness state
s = 0.2  # initial staleness
last_action_time = 0

def utility(a, H_conf):
    # expected IG scaled by hypothesis confidence minus cost
    return ACTIONS[a]["IG"] * H_conf - LAMBDA * ACTIONS[a]["cost"]

def choose_action(H_conf, available_bw):
    # budget-aware selection with safety veto (example)
    best, best_u = "do_nothing", -1e9
    for a in ACTIONS:
        if ACTIONS[a]["bw"] > available_bw: continue  # budget veto
        u = utility(a, H_conf)
        if u > best_u: best, best_u = a, u
    return best

# loop: receive hypothesis confidence from L3
for step in range(5):
    H_conf = random.uniform(0.4, 0.95)  # L3 intent posterior
    available_bw = BANDWIDTH_BUDGET - random.uniform(0,5) # dynamic
    if time.time() - last_action_time < DWELL_SECONDS:
        action = "do_nothing"  # dwell to avoid thrash
    else:
        action = choose_action(H_conf, available_bw)
    # apply map staleness rules
    if s > MAP_STALE_THRESH and available_bw >= ACTIONS["request_map_refresh"]["bw"]:
        action = "request_map_refresh"
    # execute and update staleness (simplified)
    s = s * math.exp(0.1) - 0.3 * (ACTIONS[action]["IG"]/1.2)
    s = max(0.0, min(1.0, s))
    last_action_time = time.time()
    print(f"step{step}: H_conf={H_conf:.2f} action={action} staleness={s:.2f}")
    time.sleep(0.5)  # simulated timestep