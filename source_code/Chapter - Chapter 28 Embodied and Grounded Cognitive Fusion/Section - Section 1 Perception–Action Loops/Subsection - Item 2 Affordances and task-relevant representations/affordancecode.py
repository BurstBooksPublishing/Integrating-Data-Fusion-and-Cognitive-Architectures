import numpy as np

# Example fused track (pose, visibility, size, covariance)
track = {'pose': np.array([0.6, 0.2, 0.0]),  # x,y,z (m)
         'visibility': 0.8,                    # fraction observed
         'size': 0.05,                         # diameter (m)
         'cov_trace': 0.01}                    # uncertainty magnitude

def logistic(x): return 1/(1+np.exp(-x))

def affordance_score(track, gripper_reach=0.8):
    # simple feature combination: closer, visible, right size, low uncertainty
    dist = np.linalg.norm(track['pose'][:2])                # planar distance
    feat = np.array([
        -dist,                     # closer is better
        track['visibility'],       # more visible better
        -abs(track['size']-0.06),  # size closer to 6cm better
        -track['cov_trace']        # lower uncertainty better
    ])
    weights = np.array([1.5, 1.0, 1.2, 2.0])                 # tuned factors
    score = logistic(feat.dot(weights))
    # reachability gating
    reachable = (dist <= gripper_reach)
    return float(score) if reachable else 0.0

# decision threshold and safety check
tau = 0.6
score = affordance_score(track)
if score >= tau:
    # emit action request (here just printed)
    print(f"ACT: request_grasp(score={score:.2f})")  # planner receives request
else:
    print(f"HOLD: affordance={score:.2f} below {tau}")  # defer or probe further