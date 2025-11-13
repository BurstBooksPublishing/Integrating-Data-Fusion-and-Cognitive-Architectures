import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

def entropy_uncertainty(probs):
    # probs: (N, C) predictive probabilities
    return entropy(probs.T).clip(min=1e-12)  # per-sample entropy

def ensemble_impact(X, ensemble):
    # ensemble: list of regressors predicting downstream utility
    preds = np.column_stack([m.predict(X) for m in ensemble])
    return np.std(preds, axis=1)  # proxy for expected impact

def normalize(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / (rng if rng>0 else 1.0)

def select_batch(X_pool, clf_proba, utility_ensemble, k=10,
                 alpha=1.0, beta=1.0, diversity_radius=0.2):
    # Step 1: uncertainty
    u = entropy_uncertainty(clf_proba)
    # Step 2: impact proxy
    i = ensemble_impact(X_pool, utility_ensemble)
    # Step 3: normalized multiplicative score (eq. above)
    A = (normalize(u)**alpha) * (normalize(i)**beta)
    # Step 4: greedy diversity selection with distance suppression
    selected_idx = []
    remaining = np.arange(len(A))
    scores = A.copy()
    dists = pairwise_distances(X_pool)  # precompute
    while len(selected_idx) < k and remaining.size:
        idx = remaining[np.argmax(scores[remaining])]
        selected_idx.append(int(idx))
        # suppress neighbors within radius
        neighbors = np.where(dists[idx] < diversity_radius)[0]
        scores[neighbors] *= 0.2  # soft suppression
        remaining = np.setdiff1d(remaining, [idx])
    return np.array(selected_idx, dtype=int)

# Minimal runnable demo
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=200, n_features=8, n_classes=2)
    clf = LogisticRegression(max_iter=1000).fit(X[:50], y[:50])  # small labeled set
    clf_proba = clf.predict_proba(X[50:])                         # pool probs
    # build ensemble for utility proxy (trained on small random targets here)
    ensemble = [RandomForestRegressor(n_estimators=10).fit(X[:50], np.random.rand(50))
                for _ in range(5)]
    sel = select_batch(X[50:], clf_proba, ensemble, k=8)
    print("selected pool indices:", sel)  # indices into X[50:]