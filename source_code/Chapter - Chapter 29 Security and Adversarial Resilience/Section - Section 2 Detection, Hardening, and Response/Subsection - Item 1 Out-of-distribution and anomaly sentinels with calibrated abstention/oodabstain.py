import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression

# X_ref: in-distribution embeddings (n_ref x d)
# X_val: validation embeddings with labels y_val (0=in,1=ood)
# X_query: embeddings to score
cov = EmpiricalCovariance().fit(X_ref)            # fit covariance on ref set
mu = X_ref.mean(axis=0)                           # centroid
def mahalanobis_batch(X):
    delta = X - mu
    m = np.einsum('ij,jk,ik->i', delta, cov.precision_, delta)  # squared
    return np.sqrt(np.maximum(m, 0.0))

s_ref = mahalanobis_batch(X_ref)
s_val = mahalanobis_batch(X_val)
# Calibrate scores -> probability of OOD (Platt scaling)
platt = LogisticRegression().fit(s_val.reshape(-1,1), y_val)
# At runtime
s_q = mahalanobis_batch(X_query)
p_ood = platt.predict_proba(s_q.reshape(-1,1))[:,1]   # calibrated prob
tau = 0.1                                             # policy threshold
actions = np.where(p_ood>tau, 'ABSTAIN', 'PROCESS')   # gate
# publish per-sample sentinel messages (pseudo)
for score, p, action in zip(s_q, p_ood, actions):
    msg = {'score': float(score), 'p_abstain': float(p), 'action': action}
    # send to cognition: e.g., rclpy publisher or message bus
    print(msg)  # replace with actual publish