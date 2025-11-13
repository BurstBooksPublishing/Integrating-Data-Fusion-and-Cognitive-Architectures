import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# synthetic weak sources and features
np.random.seed(0)
n, m = 500, 3
X = np.random.randn(n, 5)                      # features for policy model
true_y = (X[:,0] + 0.5*X[:,1] > 0).astype(int) # hidden truth
# weak sources with different noise rates
A = np.empty((n, m), dtype=int)
flip_rates = [0.2, 0.35, 0.4]
for j,r in enumerate(flip_rates):
    A[:,j] = true_y ^ (np.random.rand(n) < r)  # noisy views

# incorporate sparse operator corrections (indices)
correct_idx = np.random.choice(n, size=30, replace=False)
A[correct_idx, 0] = true_y[correct_idx]        # operator corrected source 0

# EM to estimate P(a|y) confusion matrices and priors
eps = 1e-9
pi = np.array([0.5, 0.5])                      # prior for y=0,1
# initialize source likelihoods uniformly
P = np.full((m,2,2), 0.5)                      # P(a=j_val | y=k)
# EM loop
for it in range(50):
    # E-step: compute posterior p(y=1|a) for each sample using (1)
    log_prods0 = np.log(pi[0]+eps) + np.sum(np.log(P[np.arange(m), A.T, 0]+eps), axis=0)
    log_prods1 = np.log(pi[1]+eps) + np.sum(np.log(P[np.arange(m), A.T, 1]+eps), axis=0)
    maxlog = np.maximum(log_prods0, log_prods1)
    p1 = np.exp(log_prods1-maxlog) / (np.exp(log_prods0-maxlog)+np.exp(log_prods1-maxlog))
    # enforce operator corrections as near-certain
    p1[correct_idx] = true_y[correct_idx]
    # M-step: update priors and confusion matrices
    pi[1] = p1.mean(); pi[0] = 1-pi[1]
    for j in range(m):
        for yval in (0,1):
            mask = (A[:,j] == 0)
            # count P(a=0|y)
            P[j,0,yval] = ((1-p1)[mask].sum() if yval==0 else p1[mask].sum()) / ( (1-p1 if yval==0 else p1).sum() + eps)
            P[j,1,yval] = 1 - P[j,0,yval]

# compute calibrated posterior for training labels
post = p1

# downstream classifier training (policy model)
clf = LogisticRegression(solver='lbfgs')
clf.fit(X, (post>0.5).astype(int))             # use MAP labels for example

# safe policy update: compute expected risk delta before commit
probs = clf.predict_proba(X)[:,1]
risk_before = log_loss(true_y, 0.5*np.ones_like(true_y))  # baseline
risk_after = log_loss(true_y, probs)
# Gate: only allow policy switch if risk decreases by margin
if risk_after < risk_before - 0.01:
    # commit policy (atomic swap with rollback capability)
    policy_weights = clf.coef_.copy()           # commit snapshot
else:
    # skip commit, schedule further human review
    policy_weights = None
print("risk_before, risk_after, commit:", risk_before, risk_after, policy_weights is not None)