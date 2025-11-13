import numpy as np, json, time

# Example candidates: dicts with throughput, safety, energy estimates and provenance.
candidates = [
    {"id":"A","throughput":120.0,"safety":0.92,"energy":45.0},
    {"id":"B","throughput":150.0,"safety":0.85,"energy":60.0},
    {"id":"C","throughput":100.0,"safety":0.97,"energy":30.0},
]

S_min = 0.9           # safety threshold (hard)
E_max = 50.0          # preferred energy budget (soft)
w_T, w_E = 1.0, 0.02  # throughput weight, energy penalty

def select_action(cands):
    safe = [c for c in cands if c["safety"] >= S_min]  # safety filter
    if not safe:
        # fallback: highest safety candidate, trigger operator alert
        best = max(cands, key=lambda x: x["safety"])
        decision_reason = "no_safe_candidate_fallback_to_max_safety"
    else:
        # utility with soft energy penalty; prefer meeting E_max when possible
        for c in safe:
            c["utility"] = w_T*c["throughput"] - w_E*max(0.0, c["energy"]-E_max)
        best = max(safe, key=lambda x: x["utility"])
        decision_reason = "constrained_selection"
    # provenance record
    record = {
        "timestamp": time.time(), "decision": best["id"],
        "metrics": best, "reason": decision_reason
    }
    print(json.dumps(record))  # emit to telemetry/logging
    return best

action = select_action(candidates)
# action would be sent to policy executor with a safety interlock.