import time

# stream descriptors: id, importance (0..1), last_update (epoch), tau (sec)
streams = [
    {"id": "cam_front", "imp": 0.9, "last": time.time()-0.2, "tau": 0.5},
    {"id": "radar", "imp": 0.7, "last": time.time()-1.2, "tau": 1.0},
    {"id": "lidar", "imp": 0.8, "last": time.time()-0.6, "tau": 0.8},
]

# actions with cost (compute units) and added latency (sec) and utility multiplier
ACTIONS = {
    "full":    {"cost": 3.0, "lat": 0.05, "util": 1.0},
    "reduced": {"cost": 1.2, "lat": 0.02, "util": 0.7},
    "skip":    {"cost": 0.0, "lat": 0.0,  "util": 0.0},
}

C_budget = 4.0  # available compute units

def schedule(streams, budget):
    now = time.time()
    # compute urgency and feasible actions per stream
    candidates = []
    for s in streams:
        staleness = now - s["last"]
        for a_name, a in ACTIONS.items():
            if staleness + a["lat"] <= s["tau"]:
                util = s["imp"] * a["util"]  # simple utility model
                # score = utility per cost (avoid divide by zero)
                score = util / (a["cost"] + 1e-6)
                candidates.append((score, s["id"], a_name, a["cost"], util))
    # greedy select highest score until budget exhausted
    selection = {}
    candidates.sort(reverse=True)
    used = 0.0
    for score, sid, aname, cost, util in candidates:
        if sid in selection:
            continue
        if used + cost <= budget:
            selection[sid] = aname
            used += cost
    # assign skip to unselected streams
    for s in streams:
        selection.setdefault(s["id"], "skip")
    return selection

print(schedule(streams, C_budget))  # example run