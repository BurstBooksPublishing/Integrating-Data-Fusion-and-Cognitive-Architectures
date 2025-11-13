import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
# synthetic simulator (provides perfect labels) -- replace with sim API
def simulate_batch(n):
    X = np.random.randn(n, 5)  # features
    y = (X[:,0] + 0.5*X[:,1] > 0).astype(int)  # ground truth rule
    return X, y
# weak labeler: noisy threshold heuristic
def weak_labeler(X):
    y = (X[:,0] > 0.2).astype(int)
    return y, np.full(len(y), 0.7)  # label and confidence
# simulated human oracle with cost
def human_oracle(X):
    _, y = simulate_batch(len(X))  # simulate expert response
    return y
# bootstrap
X_synth, y_synth = simulate_batch(200)            # synthetic dataset
X_pool, _ = simulate_batch(1000)                  # unlabeled pool
X_weak, y_weak = X_pool[:200], weak_labeler(X_pool[:200])[0]
# initial training set
X_train = np.vstack([X_synth, X_weak])
y_train = np.concatenate([y_synth, y_weak])
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
budget = 50  # human labeling budget
for t in range(budget):
    probs = model.predict_proba(X_pool)[:,1]
    # acquisition: highest predictive entropy (uncertainty sampling)
    ent = -probs*np.log(np.clip(probs,1e-8,1)) - (1-probs)*np.log(np.clip(1-probs,1e-8,1))
    idx = np.argmax(ent)
    x_query = X_pool[idx:idx+1]
    y_human = human_oracle(x_query)               # costly label
    # incorporate human label, retrain periodically
    X_train = np.vstack([X_train, x_query])
    y_train = np.concatenate([y_train, y_human])
    if (t+1) % 10 == 0:
        model.fit(X_train, y_train)               # retrain
# final evaluation (log-loss as calibration proxy)
X_test, y_test = simulate_batch(200)
print("final loss:", log_loss(y_test, model.predict_proba(X_test)))