#!/usr/bin/env python3
import json, hashlib, os, sys, numpy as np
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Load predictions: list of dicts with keys: 'scenario_id','label','score'
# For demo, replace with real loader.
data = json.load(open('preds.json'))  # [{...}, ...]

# Parameters (record these in the card)
SEED = 42
B = 2000
ALPHA = 0.05
METRIC = 'roc_auc'  # supported: 'precision','recall','roc_auc'
np.random.seed(SEED)

# Index by scenario to preserve clusters
clusters = defaultdict(list)
for d in data:
    clusters[d['scenario_id']].append(d)
scenario_ids = list(clusters.keys())
n_scen = len(scenario_ids)

def compute_metric(rows):
    y = np.array([r['label'] for r in rows])
    s = np.array([r['score'] for r in rows])
    if METRIC == 'precision':
        return precision_score(y, s > 0.5)
    if METRIC == 'recall':
        return recall_score(y, s > 0.5)
    return roc_auc_score(y, s)

# Point estimate on full dataset
all_rows = [r for cluster in clusters.values() for r in cluster]
theta_hat = compute_metric(all_rows)

# Stratified bootstrap: sample scenario IDs with replacement
boot_thetas = np.empty(B)
for b in range(B):
    sampled_ids = np.random.choice(scenario_ids, size=n_scen, replace=True)
    sampled_rows = [r for sid in sampled_ids for r in clusters[sid]]
    boot_thetas[b] = compute_metric(sampled_rows)

# Percentile CI
lower, upper = np.quantile(boot_thetas, [ALPHA/2, 1-ALPHA/2])

# Metrics card
card = {
    "metric_name": METRIC,
    "point_estimate": float(theta_hat),
    "ci_lower": float(lower),
    "ci_upper": float(upper),
    "ci_method": "stratified_bootstrap_percentile",
    "confidence": 1-ALPHA,
    "B": B,
    "seed": SEED,
    "n_scenarios": n_scen,
    "provenance": {
        "script": os.path.basename(__file__),
        "commit_hash": os.getenv('COMMIT_HASH','unknown'),
        "python": sys.version.splitlines()[0]
    }
}
print(json.dumps(card, indent=2))
# Write card for artifact registry
open('metrics_card.json','w').write(json.dumps(card))