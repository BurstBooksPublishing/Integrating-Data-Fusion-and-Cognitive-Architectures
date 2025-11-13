import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Load dataframes: gt and inf with columns ['id','onset','offset','participants','context']
# participants: set/list; context: string label
# small helper: similarity of participant sets (Jaccard)
def jaccard(a,b): return len(a & b) / len(a | b) if (a|b) else 1.0

# Build cost matrix where feasible pairs have cost = latency_seconds, infeasible set large cost
def build_cost(gt, inf, max_dt=5.0, min_part_sim=0.5):
    n,m = len(gt), len(inf)
    C = np.full((n,m), 1e6)
    for i,g in gt.iterrows():
        for j,f in inf.iterrows():
            dt = abs(f.onset - g.onset)
            ps = jaccard(set(g.participants), set(f.participants))
            if dt<=max_dt and ps>=min_part_sim:
                C[i,j] = dt  # prefer lower latency
    return C

# Example usage
gt = pd.DataFrame([...])     # populate with ground-truth rows
inf = pd.DataFrame([...])    # populate with inferred rows
C = build_cost(gt, inf, max_dt=10.0, min_part_sim=0.6)
row_ind, col_ind = linear_sum_assignment(C)
matches = [(i,j) for i,j in zip(row_ind, col_ind) if C[i,j]<1e5]

# Compute metrics
tp = 0; latencies = []; falses_context = 0
for i,j in matches:
    g = gt.iloc[i]; f = inf.iloc[j]
    tp += 1
    latencies.append(f.onset - g.onset)
    if f.context != g.context: falses_context += 1

precision = tp / len(inf) if len(inf)>0 else 0.0
recall = tp / len(gt) if len(gt)>0 else 0.0
fcr = falses_context / len(inf) if len(inf)>0 else 0.0
median_latency = np.median(latencies) if latencies else np.nan
p95_latency = np.percentile(latencies,95) if latencies else np.nan

print(precision, recall, fcr, median_latency, p95_latency)  # inline output for CI/CD