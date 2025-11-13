#!/usr/bin/env python3
import subprocess, requests, sys, os

def run(cmd):                                  # run shell step, raise on fail
    print("+", cmd); subprocess.run(cmd, shell=True, check=True)

# 1. Unit
run("pytest tests/unit --junitxml=reports/unit.xml")   # unit xUnit report

# 2. Property (Hypothesis)
run("pytest tests/property --maxfail=1 --hypothesis-show-statistics")

# 3. Integration (start docker compose, run contract tests)
run("docker-compose -f ci/docker-compose.yml up -d --build")
run("pytest tests/integration --junitxml=reports/integ.xml")
run("docker-compose -f ci/docker-compose.yml down")

# 4. System (scenario runner with golden-trace comparison)
run("python tools/run_scenarios.py --seed-file ci/seeds.json --out reports/sys.json")
# quick check: fail if system-level recall below threshold (example)
import json
r=json.load(open("reports/sys.json"))
if r['scenario_recall'] < 0.90:                  # abort threshold
    print("System gate failed"); sys.exit(2)

# 5. HIL: trigger test controller; wait for HIL bench verdict
resp=requests.post("http://hil-controller.local/start", json={"artifact":"reports/sys.json"})
job=resp.json()["job_id"]
status=requests.get(f"http://hil-controller.local/status/{job}").json()
if status["verdict"]!="PASS":
    print("HIL gate failed"); sys.exit(3)
print("All gates passed")