import json, time, copy
# simple scorer and rule engine (placeholders)
def neural_score(evidence): return sum(evidence)/len(evidence)  # normalized
def apply_rules(evidence):
    # returns dict of rule->count (inline comments explain)
    hits = {"r_close_proximity": int(evidence[0]>0.8),
            "r_high_speed": int(evidence[1]>0.7)}
    return hits

def compute_sensitivity(E, scorer, delta=1e-3):
    base = scorer(E)
    s = []
    for i in range(len(E)):
        Ep = E.copy(); Ep[i] += delta
        s.append((scorer(Ep)-base)/delta)
    return base, s

# example evidence vector and instrumentation
evidence = [0.85, 0.6, 0.2]  # sensor-derived features
rule_hits = apply_rules(evidence)
score, slices = compute_sensitivity(evidence, neural_score)

trace = {
  "trace_id": f"t-{int(time.time()*1e3)}",
  "timestamp": time.time(),
  "evidence": evidence,
  "rule_hits": rule_hits,            # symbolic provenance
  "score": score,
  "sensitivity_slices": slices
}

print(json.dumps(trace))  # send to logging/trace store