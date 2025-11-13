import numpy as np
from sklearn.utils import resample

# rows: dicts with keys 'alert', 'accepted', 'override', 'success', 'treated'
# 'treated' indicates actual exposure to alert (for baseline matching)
rows = ...  # load telemetry (list of dicts) -- e.g., from parquet/logs

# point estimates
N = sum(1 for r in rows if r['alert'])
A = sum(1 for r in rows if r['alert'] and r['accepted'])
O = sum(1 for r in rows if r['alert'] and r['accepted'] and r['override'])
S_assisted = sum(1 for r in rows if r['alert'] and r['accepted'] and r['success'])
S_baseline = sum(1 for r in rows if not r['treated'] and r['success'])

accept_rate = A / N
override_rate = O / A if A else 0.0
uplift = (S_assisted / A - S_baseline / max(1, sum(1 for r in rows if not r['treated'])))  # absolute uplift

# bootstrap CI for uplift
def uplift_stat(sample):
    A = sum(1 for r in sample if r['alert'] and r['accepted'])
    S_ass = sum(1 for r in sample if r['alert'] and r['accepted'] and r['success'])
    base_n = sum(1 for r in sample if not r['treated'])
    S_base = sum(1 for r in sample if not r['treated'] and r['success'])
    return (S_ass / A if A else 0) - (S_base / base_n if base_n else 0)

boots = [uplift_stat(resample(rows, replace=True, n_samples=len(rows))) for _ in range(2000)]
ci_lower, ci_upper = np.percentile(boots, [2.5, 97.5])
# print or push to telemetry