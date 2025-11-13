import numpy as np
from sklearn.linear_model import Ridge
np.random.seed(0)
# synthetic contexts and behavior policy (softmax logits)
N, d = 2000, 5
X = np.random.normal(size=(N,d))
logits_b = X[:,0] * 0.5  # simple behavior dependence
pb = 1/(1+np.exp(-logits_b))                     # prob of action 1
a = (np.random.rand(N) < pb).astype(int)         # logged actions
# reward model depends on context and action
true_theta = np.array([0.6, -0.2, 0.0, 0.0, 0.0])
r = (X @ true_theta) + 0.8*a + 0.1*np.random.randn(N)
# candidate policy: deterministic threshold on first feature
def pi_prob(x): return 1.0 if x[0]>0 else 0.0
pi = np.array([pi_prob(x) for x in X])
# importance weights (avoid division by zero with small eps)
eps = 1e-6
w = (pi*(1/pb) + (1-pi)*(1/(1-pb))).astype(float)
# IPS estimate
V_ips = np.mean(w * r)
# DR estimate: fit q(x,a) with linear model on features concatenated with action
Xa = np.hstack([X, a.reshape(-1,1)])
model = Ridge(alpha=1.0).fit(Xa, r)
# predict q(x,pi) and q(x,a)
q_pi = model.predict(np.hstack([X, pi.reshape(-1,1)]))
q_a = model.predict(Xa)
V_dr = np.mean(q_pi - w * (q_a - r))
# bootstrap LCB for DR
B = 200
boot = []
for _ in range(B):
    idx = np.random.randint(0, N, N)
    boot.append(np.mean(q_pi[idx] - w[idx] * (q_a[idx] - r[idx])))
lb = np.percentile(boot, 5)  # 95% bootstrap LCB
print("V_IPS", V_ips, "V_DR", V_dr, "LCB95_DR", lb)