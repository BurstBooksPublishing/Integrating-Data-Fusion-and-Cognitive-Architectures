import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def ece(probs, labels, n_bins=10, sample_weight=None):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    N = np.sum(sample_weight) if sample_weight is not None else len(labels)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if not np.any(mask): continue
        w = None if sample_weight is None else sample_weight[mask]
        conf = np.average(probs[mask], weights=w)
        acc = np.average(labels[mask], weights=w)
        ece += (np.sum(w) if w is not None else mask.sum())/N * abs(acc - conf)
    return ece

# --- inputs: X_train,y_train from source; X_dev,y_dev for dev test; X_target for deployment sampling
clf = LogisticRegression(max_iter=200)  # base scorer for sepsis risk
clf.fit(X_train, y_train)
# Platt scaling calibration
calibrator = CalibratedClassifierCV(base_estimator=clf, cv=StratifiedKFold(3), method='sigmoid')
calibrator.fit(X_train, y_train)  # fitted on source
probs_dev = calibrator.predict_proba(X_dev)[:,1]

# per-group ECE
groups = np.unique(group_ids)  # e.g., age_bin labels aligned to X_dev
ece_by_group = {g: ece(probs_dev[group_ids==g], y_dev[group_ids==g]) for g in groups}
overall_ece = ece(probs_dev, y_dev)

# importance weights via domain classifier (source=0, target=1)
X_pool = np.vstack([X_train, X_target])
y_dom = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_target))])
dom_clf = LogisticRegression(max_iter=200).fit(X_pool, y_dom)
p_target = dom_clf.predict_proba(X_dev)[:,1]
w = p_target / (1.0 - p_target + 1e-12)  # density ratio estimate
iw_ece_value = ece(probs_dev, y_dev, sample_weight=w)