import math, statistics, json, sys

# Example thresholds (set by owners)
THRESHOLDS = {
    "accuracy": 0.88,          # required mean task accuracy
    "p99_latency_ms": 200.0,   # p99 end-to-end latency
    "robust_pass_rate": 0.95,  # stress-suite pass fraction
    "safety_invariants": True  # safety checks boolean
}

def p99(samples):
    return statistics.quantiles(samples, n=100)[98]  # 99th percentile

def evaluate(metrics):
    # metrics: dict with keys "accuracy","latency_ms_samples","robust_pass_rate","safety_ok"
    acc_ok = metrics["accuracy"] >= THRESHOLDS["accuracy"]
    lat_ok = p99(metrics["latency_ms_samples"]) <= THRESHOLDS["p99_latency_ms"]
    rob_ok = metrics["robust_pass_rate"] >= THRESHOLDS["robust_pass_rate"]
    safe_ok = bool(metrics["safety_ok"]) == THRESHOLDS["safety_invariants"]
    return {"gate_pass": acc_ok and lat_ok and rob_ok and safe_ok,
            "details": {"accuracy_ok": acc_ok, "latency_ok": lat_ok,
                        "robust_ok": rob_ok, "safety_ok": safe_ok}}

if __name__ == "__main__":
    metrics = json.load(sys.stdin)  # load metrics JSON from stdin
    result = evaluate(metrics)
    print(json.dumps(result, indent=2))