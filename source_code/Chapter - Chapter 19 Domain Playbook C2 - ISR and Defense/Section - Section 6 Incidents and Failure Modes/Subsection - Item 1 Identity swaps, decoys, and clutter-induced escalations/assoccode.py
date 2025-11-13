import numpy as np
from scipy.spatial.distance import mahalanobis

def assoc_score(track_state, track_cov, meas, appearance_sim):
    # kinematic score via Mahalanobis distance
    dM = mahalanobis(meas, track_state, np.linalg.inv(track_cov))
    S_k = np.exp(-0.5 * dM**2)
    # appearance score already normalized in [0,1]
    S_a = np.clip(appearance_sim, 0.0, 1.0)
    # weighted fusion
    return 0.6*S_k + 0.4*S_a

def escalation_decision(best_score, lambda_clutter, tau0=0.7, alpha=0.2):
    # raise threshold with clutter density to reduce false escalation
    tau = tau0 + alpha * lambda_clutter
    # return True to escalate, False to hold and request revisit
    return best_score >= tau

# Example usage with simple track list and one measurement
tracks = [ {'state': np.array([10.0,5.0]), 'cov': np.diag([2.0,2.0])},
           {'state': np.array([10.5,5.2]), 'cov': np.diag([2.5,2.5])} ]
meas = np.array([10.3,5.1])
appearance_sim = 0.65  # embedding similarity to track A
lambda_clutter = 3.0   # estimated false alarms per scan

scores = [assoc_score(t['state'], t['cov'], meas, appearance_sim) for t in tracks]
best = max(scores)
if escalation_decision(best, lambda_clutter):
    print("Escalate: sufficient confidence")
else:
    print("Hold: request revisit or higher-fidelity sensing")  # safe fallback