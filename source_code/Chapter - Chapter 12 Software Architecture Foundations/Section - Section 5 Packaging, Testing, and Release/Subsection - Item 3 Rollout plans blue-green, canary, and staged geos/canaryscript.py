#!/usr/bin/env python3
import requests, time, sys
PROM = "http://prometheus:9090/api/v1/query"  # metrics endpoint
CONTROL_API = "http://traffic-controller.local/v1/weights"  # control plane

def query_prom(query):
    r = requests.get(PROM, params={"query": query}, timeout=5)
    r.raise_for_status()
    return float(r.json()["data"]["result"][0]["value"][1])

def set_weight(version, weight):
    # simple REST call to set traffic weight; real control plane will auth/sign.
    r = requests.post(CONTROL_API, json={"version": version, "weight": weight}, timeout=5)
    r.raise_for_status()

def main():
    canary_version = "v2"
    control_version = "v1"
    set_weight(canary_version, 0.01)  # start at 1% traffic
    for minute in range(30):  # dwell for 30 minutes
        time.sleep(60)
        # KPI: scenario_false_escalation_rate and NEES consistency probe
        try:
            f_rate = query_prom('scenario_false_escalation_rate{version="%s"}' % canary_version)
            nees = query_prom('nees_consistency{version="%s"}' % canary_version)
        except Exception as e:
            print("metric poll failed:", e); set_weight(canary_version, 0.0); sys.exit(2)
        # simple thresholds; real rules use statistical tests and hold counters
        if f_rate > 0.02 or nees > 1.5:  # thresholds tuned in HIL
            set_weight(canary_version, 0.0)  # immediate rollback
            print("Rollback: thresholds exceeded"); sys.exit(1)
    # promote to full traffic after clean dwell
    set_weight(canary_version, 1.0); set_weight(control_version, 0.0)
    print("Promoted canary to production")

if __name__ == "__main__": main()