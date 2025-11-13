import json, math
import numpy as np
from sklearn.metrics import precision_score, recall_score

def ece_score(probs, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for i in range(n_bins):
        idx = (probs >= bins[i]) & (probs < bins[i+1])
        if idx.sum() == 0:
            continue
        acc = labels[idx].mean()
        conf = probs[idx].mean()
        ece += (idx.sum() / N) * abs(acc - conf)
    return float(ece)

def compute_metrics(decision_id, probs, labels, latency_ms, subgroup_ids):
    # basic performance
    y_pred = (probs >= 0.5).astype(int)
    perf = float(precision_score(labels, y_pred))
    lat = min(latency_ms / 1000.0, 10.0)  # normalized seconds (capped)
    cal = ece_score(probs, labels, n_bins=15)
    # safety proxy: fraction of high-confidence wrong decisions
    safety_violation = float(((probs > 0.9) & (y_pred != labels)).mean())
    # fairness: per-subgroup TPR gap
    subgroup_metrics = {}
    tprs = []
    for sg in np.unique(subgroup_ids):
        mask = subgroup_ids == sg
        if mask.sum() == 0:
            continue
        tp = ((y_pred == 1) & (labels == 1) & mask).sum()
        fn = ((y_pred == 0) & (labels == 1) & mask).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        subgroup_metrics[str(int(sg))] = {"tpr": float(tpr)}
        tprs.append(tpr)
    fair_gap = float(max(tprs) - min(tprs)) if tprs else 0.0
    metrics = {
        "id": decision_id,
        "perf": perf, "lat": lat, "cal": cal,
        "safety_violation": safety_violation, "fair_gap": fair_gap,
        "subgroups": subgroup_metrics
    }
    # composite (example weights)
    w = np.array([0.35, 0.15, 0.15, 0.25, 0.10])  # perf,lat,cal,safety,fair
    m = np.array([perf, 1.0 - lat/10.0, 1.0 - cal, 1.0 - safety_violation, 1.0 - fair_gap])
    metrics["composite"] = float(np.dot(w, m))
    # alerting rules
    metrics["alerts"] = {
        "safety": safety_violation > 0.01,  # configurable
        "fairness": fair_gap > 0.05
    }
    return metrics

# example write to registry (file or telemetry)
# save_json would be replaced by telemetry call in production
def save_metrics(metrics, path="metrics_registry.json"):
    with open(path, "a") as fh:
        fh.write(json.dumps(metrics) + "\n")
# end of listing