import numpy as np

# Example hypotheses and actions
hypotheses = ["friendly","hostile","ambiguous"]
actions = ["approach","hold","evade","request_human"]

# Posterior probabilities from L2 (sum to 1)
P = np.array([0.5, 0.2, 0.3])  # P(h | E)

# Base utility table U[h,a] (rows=hypotheses, cols=actions)
U = np.array([
    [ 10,  2, -5,  0],  # friendly
    [-50, 1,  20,  5],  # hostile
    [ -5, 3,  5,  2],   # ambiguous
])

# User model: weights per action (mission_oriented vs safety_oriented)
user_role = "mission"  # swap to "safety" for different pref
user_weights = {"mission": np.array([1.2,1.0,0.8,0.9]),
                "safety":  np.array([0.8,1.0,1.3,1.1])}
w = user_weights[user_role]

def expected_utility(P, U, w):
    U_adj = U * w  # user-weighted utilities
    return P @ U_adj  # vector of EU per action

def risk_metric(P, actions_risk):
    return P @ actions_risk  # expected risk scalar per action set

# Example action risk profile
actions_risk = np.array([0.1, 0.02, 0.2, 0.01])

EU = expected_utility(P, U, w)
# enforce meta-control safety threshold
r_max = 0.15 if user_role=="mission" else 0.05
risks = risk_metric(P, np.diag(actions_risk))  # per-action expected risks
# select feasible actions then argmax EU
feasible_idx = np.where(actions_risk <= r_max)[0]
if feasible_idx.size == 0:
    choice = "request_human"  # safe fallback
else:
    choice = actions[feasible_idx[np.argmax(EU[feasible_idx])]]

print("EU:", EU, "choice:", choice)  # simple audit output