#!/usr/bin/env python3
"""
Simple recert scheduler: loads registry entries, computes age/trust,
runs tests, and flags deprecation or recertification.
"""
import json, time, math, subprocess, logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

# Load registry (example JSON list) -- replace with DB/API in production.
with open("registry.json") as f:
    registry = json.load(f)

def trust_decay(T0, lam, days):
    return T0 * math.exp(-lam * days)

def run_test_suite(entry):
    # placeholder: call CI job or local tests; return True if passed
    cmd = entry.get("smoke_cmd") or "pytest -q"
    try:
        subprocess.check_call(cmd, shell=True)  # run tests
        return True
    except subprocess.CalledProcessError:
        return False

now = datetime.utcnow()
for entry in registry:
    last_cert = datetime.fromisoformat(entry["last_certified"])
    days = (now - last_cert).days + (now - last_cert).seconds/86400.0
    T0 = entry.get("initial_trust", 1.0)
    lam = entry.get("decay_rate", 0.01)
    T = trust_decay(T0, lam, days)

    logging.info("Entry %s: days=%0.2f, trust=%0.3f", entry["name"], days, T)

    if T <= entry.get("recert_threshold", 0.8):
        logging.info("Trigger recert for %s", entry["name"])
        passed = run_test_suite(entry)
        if passed:
            entry["last_certified"] = now.isoformat()
            logging.info("Re-certified %s", entry["name"])
        else:
            # escalate: move to shadow/canary and notify owners
            entry["state"] = "shadow"
            logging.warning("Failed recert -> moved to shadow: %s", entry["name"])
    elif days >= entry.get("deprecation_max_age_days", 180):
        entry["state"] = "deprecated"
        logging.warning("Deprecated %s due to age", entry["name"])

# Persist changes
with open("registry.json","w") as f:
    json.dump(registry, f, indent=2)