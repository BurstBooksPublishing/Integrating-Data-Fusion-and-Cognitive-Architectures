import math, json, time

# Dummy rule engine: returns compliance score and violations list.
def rule_engine(action, situation):
    # enforce pedestrian safety: any motion with TTC<3s is non-compliant
    ttc = situation.get("ttc", math.inf)
    if action == "stop":
        return 1.0, []
    if ttc < 3.0:
        return 0.0, ["pedestrian_TTC_breach"]
    return 1.0, []

# Dummy learned policy: returns priority score and model confidence
def learned_policy(action, situation):
    # simple heuristic: prefer proceed if low density and high speed
    density = situation.get("traffic_density", 0.5)
    score = 1.0 - density if action == "proceed" else 0.2
    confidence = 0.8 if situation.get("sensor_confidence",1.0)>0.7 else 0.4
    return max(0.0, min(1.0, score)), confidence

def arbitrate(actions, situation, w_r=0.6, w_l=0.4, gamma=0.9):
    trace = {"time": time.time(), "candidates": []}
    legal = []
    for a in actions:
        R_score, violations = rule_engine(a, situation)
        L_score, C = learned_policy(a, situation)
        U = w_r*R_score + w_l*L_score*C
        entry = {"action": a, "R": R_score, "L": L_score, "C": C, "U": U, "violations": violations}
        trace["candidates"].append(entry)
        if R_score >= gamma:
            legal.append((U, a))
    if not legal:
        # safe fallback: stop; include reason in trace
        trace["selected"] = {"action":"stop","reason":"no_legal_candidate"}
    else:
        selected = max(legal)[1]
        trace["selected"] = {"action":selected, "reason":"max_utility_within_legal"}
    print(json.dumps(trace, indent=2))  # auditable log
    return trace["selected"]["action"]

# Demo situation
situation = {"ttc":2.1, "traffic_density":0.2, "sensor_confidence":0.9}
actions = ["proceed","stop","slow_yield"]
arbitrate(actions, situation)