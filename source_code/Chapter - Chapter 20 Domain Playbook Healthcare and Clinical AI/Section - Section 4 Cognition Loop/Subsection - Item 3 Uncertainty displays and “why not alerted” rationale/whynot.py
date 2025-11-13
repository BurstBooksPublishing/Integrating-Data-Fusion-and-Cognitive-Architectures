import numpy as np
from scipy.special import expit, logit

# simple logistic model params (example only)
beta = np.array([-3.0, 0.8, 0.5])  # intercept, heart_rate, lactate
threshold = 0.2  # policy alert threshold

def posterior(features):
    z = beta[0] + np.dot(beta[1:], features)  # linear logit
    return expit(z)  # posterior probability

def minimal_delta(features, k, tau=threshold):
    # compute delta required on feature k per eq. (2)
    other = beta[0] + np.dot(beta[1:], features) - beta[1+k-1]*features[k]  # adjust index
    # if beta_k is near zero, return None (non-identifiable)
    bk = beta[1+k]
    if abs(bk) < 1e-6:
        return None
    return (logit(tau) - other) / bk

# example patient features: [heart_rate, lactate]
f = np.array([100.0, 1.8])
p = posterior(f)
alert = p >= threshold

if not alert:
    # compute counterfactuals for each input dimension
    deltas = [minimal_delta(f, k, threshold) for k in range(len(f))]
    rationale = {
        "posterior_prob": float(p),
        "calibration_band": [max(0, float(p)-0.05), min(1, float(p)+0.05)],  # simple CI proxy
        "top_evidence": [("heart_rate", f[0], 0.8), ("lactate", f[1], 0.5)],  # (name,value,weight)
        "suppressions": [],  # list active suppression policies if any
        "counterfactual_deltas": deltas
    }
else:
    rationale = {"alert": True, "posterior_prob": float(p)}

print(rationale)  # send to UI/back-end (structured JSON)