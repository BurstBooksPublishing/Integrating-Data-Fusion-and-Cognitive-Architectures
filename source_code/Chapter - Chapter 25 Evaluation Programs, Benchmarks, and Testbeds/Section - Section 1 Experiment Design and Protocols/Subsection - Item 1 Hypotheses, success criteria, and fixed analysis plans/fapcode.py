#!/usr/bin/env python3
# Requires: numpy, pandas, scipy
import json, hashlib, time
import numpy as np
import pandas as pd
from scipy.stats import permutation_test

SEED = 42
np.random.seed(SEED)

def median_ttd(arr): return float(np.median(arr))

def bootstrap_ci(data, statfunc, n=5000, alpha=0.05):
    n = int(n)
    boots = np.random.choice(data, size=(n, len(data)), replace=True)
    stats = np.apply_along_axis(statfunc, 1, boots)
    lo = np.quantile(stats, alpha/2)
    hi = np.quantile(stats, 1-alpha/2)
    return lo, hi

# Load pre-recorded, versioned artifacts (paths pre-registered)
baseline = pd.read_csv("baseline_ttd.csv")["ttd"].to_numpy()
candidate = pd.read_csv("candidate_ttd.csv")["ttd"].to_numpy()

# Primary metric
m_base = median_ttd(baseline)
m_cand = median_ttd(candidate)
ci_lo, ci_hi = bootstrap_ci(candidate, median_ttd, n=2000)

# Permutation test (two-sided)
res = permutation_test((baseline, candidate),
                       statistic=lambda x, y: np.median(x)-np.median(y),
                       permutation_type='independent', n_resamples=10000,
                       random_state=SEED)
p_value = res.pvalue

# Pack results and sign (simple hash for artifact integrity)
result = {
  "seed": SEED,
  "m_base": m_base, "m_cand": m_cand,
  "ci_candidate": [ci_lo, ci_hi],
  "p_value": float(p_value),
  "timestamp": time.time()
}
result_bytes = json.dumps(result, sort_keys=True).encode('utf-8')
result["sha256"] = hashlib.sha256(result_bytes).hexdigest()

# Write artifact (pre-registered path)
with open("fap_result.json","w") as f:
    json.dump(result, f, indent=2)
# Brief exit code conveys pass/fail per pre-registered success criterion
SUCCESS = (p_value < 0.05) and (m_cand < (m_base - 0.5))  # pre-registered rule
exit(0 if SUCCESS else 2)