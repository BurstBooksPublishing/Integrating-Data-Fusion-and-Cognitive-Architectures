import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
# simulate baseline data
np.random.seed(0)
n = 2000
X_base = np.random.normal(size=(n,4))            # sensor features
y_base = (X_base[:,0] + X_base[:,1] > 0).astype(int)  # scenario label
# train detector on baseline
clf = LogisticRegression().fit(X_base, y_base)
# create drifted stream: mean shift on feature0
shift = 0.8
X_drift = X_base.copy()
X_drift[:,0] += shift
# inject rare events (positive cases) sparsely
rare_rate = 0.005
rare_mask = (np.random.rand(n) < rare_rate)
y_stream = y_base.copy()
y_stream[rare_mask] = 1
# adversarial perturbation: nudge negatives toward positives
adv_mask = (np.random.rand(n) < 0.01)
X_adv = X_drift.copy()
X_adv[adv_mask,0] += 2.0  # constrained semantic perturbation
# run detector on adversarial stream and score
y_pred = clf.predict(X_adv)
p,r,f,_ = precision_recall_fscore_support(y_stream, y_pred, average='binary', zero_division=0)
# compute simple latency metric: first detection index for each true positive
latencies = []
for i in np.where(y_stream==1)[0]:
    if y_pred[i]==1:
        latencies.append(0)  # immediate in this batch example
    else:
        latencies.append(1)  # proxy for delayed detection
print(f"Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, AvgLatency={np.mean(latencies):.2f}")