import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# synthesize pool and initial labeled seed
X_pool, y_pool = make_classification(n_samples=1000, n_features=20, n_classes=2)
scaler = StandardScaler().fit(X_pool)
X_pool = scaler.transform(X_pool)
rng = np.random.default_rng(0)

# seed labeled set
seed_idx = rng.choice(len(X_pool), size=20, replace=False)
X_seed, y_seed = X_pool[seed_idx], y_pool[seed_idx]

clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.partial_fit(X_seed, y_seed, classes=np.array([0,1]))  # initial model

unlabeled_idx = np.setdiff1d(np.arange(len(X_pool)), seed_idx)

def margin_scores(model, X):
    probs = model.decision_function(X)            # signed distance
    return np.abs(probs)                          # margin magnitude

# simulate active learning rounds
for round in range(50):
    X_unl = X_pool[unlabeled_idx]
    margins = margin_scores(clf, X_unl)
    # select lowest margin (most uncertain)
    query_pos = np.argmin(margins)
    q_idx = unlabeled_idx[query_pos]
    # simulated human label (oracle)
    y_label = y_pool[q_idx]
    # relevance weight could be provided by operator; use 1.0 here
    sample_weight = np.array([1.0])
    clf.partial_fit(X_pool[[q_idx]], [y_label], sample_weight=sample_weight)
    # remove from unlabeled
    unlabeled_idx = np.delete(unlabeled_idx, query_pos)
# model now updated with relevance feedback