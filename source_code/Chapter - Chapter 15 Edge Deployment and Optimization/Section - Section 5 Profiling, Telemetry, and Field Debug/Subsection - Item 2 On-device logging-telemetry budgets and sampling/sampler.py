import random
from typing import List, Dict

# streams: list of dicts with keys: name, size(bytes), weight(utility)
streams = [
    {"name":"track_summary", "size":200, "weight":10.0},
    {"name":"full_track",     "size":5000, "weight":4.0},
    {"name":"model_logits",   "size":1200, "weight":6.0},
    {"name":"frame",          "size":150000, "weight":1.0},
]

B = 20000  # bytes per period budget

def allocate_probabilities(streams: List[Dict], B: float, eps=1e-6):
    # compute r_i = w_i / s_i
    r = [s["weight"]/s["size"] for s in streams]
    # if unconstrained full-log fit budget, set p_i=1
    full_cost = sum(s["size"] for s in streams)
    if full_cost <= B:
        return {s["name"]: 1.0 for s in streams}
    # bisection on alpha so sum min(1, alpha*r_i)*s_i = B
    lo, hi = 0.0, max((B / s["size"])*2 for s in streams) + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cost = 0.0
        for s_i, ri in zip(streams, r):
            p = min(1.0, mid * ri)
            cost += p * s_i["size"]
        if cost > B:
            hi = mid
        else:
            lo = mid
        if hi - lo < eps:
            break
    alpha = lo
    return {s["name"]: min(1.0, alpha*(s["weight"]/s["size"])) for s in streams}

probs = allocate_probabilities(streams, B)
print("Sampling probabilities:", probs)

def should_log(stream_name: str, probs: Dict[str, float]) -> bool:
    return random.random() < probs[stream_name]  # probabilistic sample

# runtime example: decide to log one event per stream
for s in streams:
    if should_log(s["name"], probs):
        # here we would serialize and emit, respecting redaction and encryption
        print("LOG:", s["name"])