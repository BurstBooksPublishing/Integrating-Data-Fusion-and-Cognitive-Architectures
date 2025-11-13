#!/usr/bin/env python3
import numpy as np, time

# simulated sensor detection: (id, prob_detect, x,y,z,grasp_score)
def get_detections():
    return [
        ("glasses", 0.7, 1.2, 0.5, 0.9, 0.6),
        ("mug",     0.9, 0.8, 0.3, 0.0, 0.8),
    ]

# short-term belief store (id -> (p_exist, pose, last_seen))
belief = {}

def bayesian_update(obj_id, p_detect, pose):
    p_prior = belief.get(obj_id, (0.2, None, 0.0))[0]
    # simple Bayesian update on existence
    p_post = (p_detect * p_prior) / (p_detect * p_prior + (1 - p_detect) * (1 - p_prior) + 1e-6)
    belief[obj_id] = (p_post, pose, time.time())

def score_action(obj_id, p_exist, grasp_score, distance):
    # Expected utility: reward scaled by success minus time cost and risk
    reward = 1.0
    time_cost = distance / 0.5  # assume 0.5 m/s
    risk = max(0.0, 0.5 - grasp_score)  # penalty for low graspability
    return reward * p_exist * grasp_score - 0.1 * time_cost - 0.5 * risk

def loop_once():
    dets = get_detections()
    for d in dets:
        oid, pd, x,y,z, g = d
        bayesian_update(oid, pd, (x,y,z))
    # planner selects best object to fetch
    candidates = []
    for oid,(p,pose,ts) in belief.items():
        x,y,z = pose
        dist = np.hypot(x, y)
        # assume stored grasp_score from last detection (here simplified)
        grasp_score = next((d[5] for d in dets if d[0]==oid), 0.2)
        u = score_action(oid, p, grasp_score, dist)
        candidates.append((u, oid, p, pose))
    if not candidates:
        print("No belief; scheduling reminder/search.")
        return
    best = max(candidates, key=lambda x:x[0])
    if best[0] > 0.0:
        print(f"Dispatch fetch: {best[1]} (p={best[2]:.2f})")
    else:
        print("Utility too low; send reminder instead.")

if __name__=="__main__":
    loop_once()  # single cycle demonstration