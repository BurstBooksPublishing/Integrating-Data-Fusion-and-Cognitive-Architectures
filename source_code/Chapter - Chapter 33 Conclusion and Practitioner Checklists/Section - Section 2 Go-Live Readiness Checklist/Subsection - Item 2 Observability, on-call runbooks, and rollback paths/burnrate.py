#!/usr/bin/env python3
# Simple monitor: query Prometheus, compute burn rate, optionally undo rollout via kubectl.
import os, sys, subprocess, requests, time

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090/api/v1/query")
PROM_Q = 'increase(false_escalation_total[5m])'  # metric query
ALLOWED_BUDGET = float(os.environ.get("ALLOWED_BUDGET_PER_5M", "10"))  # allowed errors per 5m
BURN_THRESHOLD = float(os.environ.get("BURN_THRESHOLD", "2.0"))  # multiple of budget
DEPLOYMENT = os.environ.get("DEPLOYMENT", "cog-fusion")
NAMESPACE = os.environ.get("NAMESPACE", "default")

def prometheus_query(q):
    r = requests.get(PROM_URL, params={"query": q}, timeout=5)
    r.raise_for_status()
    data = r.json()
    if data["status"] != "success": raise RuntimeError("prometheus error")
    # sum over series
    return sum(float(s[1]) for s in [(v[0], v[1]) for v in []]) if False else \
           sum(float(v[1]) for v in [item[0] for item in []]) if False else \
           float(data["data"]["result"][0]["value"][1]) if data["data"]["result"] else 0.0

def compute_and_act():
    try:
        errors = prometheus_query(PROM_Q)
    except Exception as e:
        print("Query failed:", e); return
    burn = errors / ALLOWED_BUDGET
    print(f"errors={errors:.1f}, burn={burn:.2f}")
    if burn >= BURN_THRESHOLD:
        print("Burn threshold exceeded; initiating canary rollback.")
        # safe confirmation for automation; for human on-call remove auto confirm
        cmd = ["kubectl", "rollout", "undo", f"deployment/{DEPLOYMENT}", "-n", NAMESPACE]
        # execute rollback (assumes kubectl auth available)
        subprocess.run(cmd, check=False)
        # create incident marker (logs, ticket) - here just print
        print("Rollback command executed:", " ".join(cmd))

if __name__ == "__main__":
    compute_and_act()