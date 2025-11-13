import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

# load a table of detections: features, label_id, scene_id (provenance)
df = pd.read_csv("detections.csv")  # columns: feature_0..N, label_id, scene_id

X = df.filter(regex="^feature").values  # sensor-derived features
y = df["label_id"].values               # ground-truth identity labels
groups = df["scene_id"].values          # group by scene to avoid leakage

gkf = GroupKFold(n_splits=5)
metrics = []
for train_idx, test_idx in gkf.split(X, y, groups):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    # simple scorer illustrating pipeline; replace with association model
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(yte, ypred, average="macro")
    metrics.append({"precision": p, "recall": r, "f1": f})
# report median metrics to reduce effect of skewed scenes
import numpy as np
med = {k: np.median([m[k] for m in metrics]) for k in metrics[0]}
print("Group-holdout median metrics:", med)  # faithful, leakage-minimized estimate