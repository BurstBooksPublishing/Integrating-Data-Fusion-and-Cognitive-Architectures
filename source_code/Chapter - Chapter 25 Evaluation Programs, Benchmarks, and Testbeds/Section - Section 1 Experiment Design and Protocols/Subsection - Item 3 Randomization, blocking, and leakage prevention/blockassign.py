# Minimal runnable example; install pandas,numpy.
import hashlib, json, pandas as pd, numpy as np

RS = np.random.RandomState(20251107)                     # fixed seed registry
df = pd.read_csv("captures.csv")                         # cols: scenario,entity,timestamp,...
# define blocks by scenario template and day
df['block'] = df['scenario'].astype(str) + ":" + pd.to_datetime(df['timestamp']).dt.date.astype(str)

# block-level fold assignment (k-fold) while preventing entity overlap
k = 5
blocks = df['block'].unique()
RS.shuffle(blocks)
fold_assign = {b: int(i % k) for i,b in enumerate(blocks)}  # randomized round-robin within blocks

df['fold'] = df['block'].map(fold_assign)

# leakage check: ensure disjoint entity sets across folds
def leakage_detect(dframe):
    fold_entities = {f: set(dframe.loc[dframe['fold']==f,'entity']) for f in dframe['fold'].unique()}
    overlaps = {}
    for i in fold_entities:
        for j in fold_entities:
            if i
\section{Section 2: Benchmarks and Datasets}
\subsection{Item 1:  Public vs private corpora; licensing, bias, and drift audits}
Building on the experimental protocols and scenario libraries just discussed, dataset choice and stewardship complete the evaluation lifecycle by linking test design to legal, ethical, and operational controls. The following frames the system context, practical processes, an illustrative example, and operational applications for corpus selection, licensing, bias audits, and drift monitoring.

Concept: system and lifecycle context
- Evaluation datasets sit at three lifecycle stages: development (model training/tuning), validation (benchmarks, golden traces), and operations (monitoring, online updates).
- Public corpora enable reproducibility but may impose redistribution or use limits. Private corpora protect sensitive telemetry and PHI but complicate third party verification and benchmarking.
- Assurance requires metadata and provenance as first class artifacts so every decision can be traced to a captured record, license, and curator.

Process: governance, bias, and drift audits
1. Licensing and provenance
   - Capture license, contributor, capture timestamp, geo-fence, and export-control tags in a dataset card.
   - Enforce compatibility checks before combining corpora; treat non-commercial, share-alike, and government-restricted licenses as blockers for redistribution.
2. Bias audits (static)
   - Define explicit protected or operational subgroups (e.g., sensor model, region, time-of-day, demographic proxies).
   - Compute subgroup performance and per-subgroup calibration (ECE) to surface asymmetric errors.
3. Drift audits (dynamic)
   - Monitor covariate distributions p_0(x) (reference) vs p_t(x) (current). Use divergence measures or classifier two-sample tests to detect shift.
   - Distinguish:
     a) covariate shift: p(x) changes, p(y|x) stable;
     b) label shift: p(y) changes;
     c) concept drift: p(y|x) changes.
4. Thresholds and gates
   - Define alarm thresholds on statistical divergence and on subgroup delta-performance; tie them to retrain/rollback/alert actions.

Example: compact math for drift scoring
A symmetric divergence helps quantify covariate shift. For discrete binning or histogrammed features, use Jensen–Shannon divergence:
\begin{equation}[H]\label{eq:js}
\mathrm{JS}(P\|Q) \;=\; \tfrac{1}{2}\mathrm{KL}(P\|M) + \tfrac{1}{2}\mathrm{KL}(Q\|M),\quad M=\tfrac{1}{2}(P+Q).
\end{equation}
JS is bounded and interpretable. Set action when $\mathrm{JS}(P\|Q)>\tau_{\mathrm{cov}}$.

Code: small audit script computing license checks, subgroup metrics, and JS drift
\begin{lstlisting}[language=Python,caption={Dataset audit: license compatibility, subgroup metrics, JS drift},label={lst:dataset_audit}]
import json, pandas as pd, numpy as np
from scipy.stats import entropy
from sklearn.metrics import precision_score, recall_score

# load metadata and labels/features (simple CSVs) -- replace paths
meta = pd.read_csv("dataset_metadata.csv")      # contains license, capture_time, region
data = pd.read_parquet("features.parquet")      # contains features and label, subgroup_col

# simple license whitelist check
permissive = {"MIT","Apache-2.0","BSD-3-Clause"}
meta["license_ok"] = meta["license"].apply(lambda L: L in permissive)

# subgroup performance: compute precision/recall per region
group = data.groupby("region")
perf = {}
for name, g in group:
    y_true = g["label"].values
    y_pred = g["pred"].values
    perf[name] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": len(g),
    }

# JS divergence between reference and current histograms for a chosen feature
def js_divergence(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p /= p.sum(); q /= q.sum()
    m = 0.5*(p+q)
    return 0.5*(entropy(p, m) + entropy(q, m))

# example histogram bins
ref = np.load("ref_hist_feature1.npy")   # reference histogram
curr = np.histogram(data["feature1"], bins=len(ref), range=(0,1))[0]
js = js_divergence(ref+1e-12, curr+1e-12)

# summary output
out = {"license_violations": int((~meta["license_ok"]).sum()), "js_feature1": float(js), "perf_by_region": perf}
print(json.dumps(out, indent=2))