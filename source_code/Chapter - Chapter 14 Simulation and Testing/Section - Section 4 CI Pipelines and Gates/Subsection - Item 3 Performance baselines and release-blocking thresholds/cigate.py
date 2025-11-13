#!/usr/bin/env python3
"""
Simple CI gate: load metrics JSON, compare to baseline datastore,
exit non-zero on release-blocking condition.
"""
import sys, json, statistics, pathlib

# Config (normally injected via CI env)
BASELINE_DB = pathlib.Path("baseline_db.json")  # historical summaries
METRICS_FILE = pathlib.Path("metrics.json")     # current run artifact
K_SIGMA = 3.0                                    # conservative flag

def load_json(p): return json.loads(p.read_text())

def check_metric(name, obs, summary):
    # summary: {'mean':..., 'std':..., 'p99':..., 'direction': 'lower'|'higher'}
    mu, sigma, p99 = summary['mean'], summary['std'], summary['p99']
    direction = summary.get('direction', 'lower')
    # sigma rule
    if sigma > 0:
        z = (obs - mu) / sigma
    else:
        z = float('inf') if obs != mu else 0.0
    # percentile rule
    if direction == 'lower':
        sigma_block = (obs - mu) <= -K_SIGMA * sigma
        perc_block = obs <= summary['p01']  # left tail block
    else:
        sigma_block = (obs - mu) >= K_SIGMA * sigma
        perc_block = obs >= p99
    return {'z': z, 'sigma_block': sigma_block, 'perc_block': perc_block}

def main():
    if not (BASELINE_DB.exists() and METRICS_FILE.exists()):
        print("Missing artifacts", file=sys.stderr); sys.exit(2)
    base = load_json(BASELINE_DB)
    metrics = load_json(METRICS_FILE)
    violations = []
    for name, obs in metrics.items():
        if name not in base: continue
        res = check_metric(name, obs, base[name])
        if res['sigma_block'] or res['perc_block']:
            violations.append((name, obs, res))
    if violations:
        for v in violations:
            print(f"BLOCK: {v[0]} obs={v[1]} z={v[2]['z']:.2f}")
        sys.exit(1)  # release-blocking
    print("OK: all baselines satisfied")
    sys.exit(0)

if __name__ == "__main__":
    main()