#!/usr/bin/env python3
"""
Process test result logs and decide quarantine; compute rerun economics.
Input: JSON list of records with fields: name, run_id, outcome ('pass'/'fail'), timestamp.
"""
import json, math, statistics
from collections import defaultdict

# policy parameters
FLAKE_Q_THRESHOLD = 0.2  # quarantine if flake rate exceeds this
RETRY_BUDGET = 3         # retries per flaky test
COST_PER_RUN = 0.05      # cost in compute units per test run

def load_results(path):
    with open(path) as f:
        return json.load(f)

def compute_flake_rate(events):
    # events: list of outcomes in time order
    # flake if test has both fail and pass within window
    failures = 0
    flaky_instances = 0
    for i,out in enumerate(events):
        if out == 'fail':
            failures += 1
            # consider next k runs (here immediate next) for flake detection
            if i+1 < len(events) and events[i+1] == 'pass':
                flaky_instances += 1
    return (flaky_instances / failures) if failures else 0.0

def analyze(records):
    by_test = defaultdict(list)
    for r in sorted(records, key=lambda x: (x['name'], x['timestamp'])):
        by_test[r['name']].append(r['outcome'])
    report = {}
    N = len(by_test)
    total_expected_extra = 0.0
    for name, events in by_test.items():
        f = compute_flake_rate(events)
        expected_extra = (f / (1 - f)) if f < 1 else math.inf
        total_expected_extra += expected_extra
        report[name] = {'flake_rate': f, 'expected_extra_runs': expected_extra}
    # compute rerun cost using equation (scaled by cost per run)
    C_rerun = N * COST_PER_RUN * (statistics.mean([r['flake_rate'] for r in report.values()]) \
                / (1 - statistics.mean([r['flake_rate'] for r in report.values()])))
    # decide quarantine list
    quarantine = [n for n,v in report.items() if v['flake_rate'] > FLAKE_Q_THRESHOLD]
    return {'per_test': report, 'quarantine': quarantine, 'estimated_rerun_cost': C_rerun}

if __name__ == '__main__':
    import sys
    recs = load_results(sys.argv[1])  # pass path to JSON file
    out = analyze(recs)
    print(json.dumps(out, indent=2))