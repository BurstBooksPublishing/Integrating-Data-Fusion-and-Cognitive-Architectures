#!/usr/bin/env python3
import time, json, urllib.request
# Services with health URLs and latency SLOs (s)
SERVICES = [
    {"name":"preproc","url":"http://localhost:8001/health","slo":0.05},
    {"name":"tracker","url":"http://localhost:8002/health","slo":0.06},
    {"name":"situation","url":"http://localhost:8003/health","slo":0.08},
]
THRESHOLD_WINDOW = 30.0  # seconds
VIOLATION_RATIO = 0.2    # extend if >20% checks violate SLO

def probe(service):
    t0 = time.time()
    try:
        with urllib.request.urlopen(service["url"], timeout=1.0) as r:
            _ = r.read()  # simple liveness check
        latency = time.time() - t0
        ok = latency <= service["slo"]
    except Exception:
        latency = float("inf")
        ok = False
    return ok, latency

def evaluate():
    violations = 0
    total = 0
    latencies = {}
    for s in SERVICES:
        ok, lat = probe(s)
        total += 1
        if not ok: violations += 1
        latencies[s["name"]] = lat
    if violations / max(1,total) > VIOLATION_RATIO:
        print("EXTEND_TRIGGER", json.dumps(latencies))  # operator/action hook
    else:
        print("OK", json.dumps(latencies))

if __name__ == "__main__":
    evaluate()  # run once; integrate into scheduler for periodic checks