import numpy as np

# Simple Kalman update for linear measurement H, R noise
def kalman_update(x, P, z, H, R):
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_up = x + K @ y
    P_up = (np.eye(len(x)) - K @ H) @ P
    return x_up, P_up

# Prior states for two tracks (2D pos, 2x2 cov)
X_prior = [np.array([0.,0.]), np.array([1.,0.])]
P_prior = [0.1*np.eye(2), 0.1*np.eye(2)]

# Measurements (position), measurement model H and R
Z = [np.array([0.1, 0.0]), np.array([0.9, 0.0])]
H = np.eye(2); R = 0.05*np.eye(2)

# Example joint hypotheses: list of assignments per track (index into Z or 0 for miss), with weights
# e.g., H0: track1->z0, track2->z1 ; H1: track1->z1, track2->z0  etc.
hypotheses = [([0,1], 0.45), ([1,0], 0.40), ([0,0], 0.15)]  # format: (assignments, weight)

# Normalize weights
weights = np.array([w for (_,w) in hypotheses])
weights /= weights.sum()

# Compute marginals beta_{ij}
num_tracks = len(X_prior); num_meas = len(Z)
beta = np.zeros((num_tracks, num_meas+1))  # last column is miss prob
for (assignments, _) , w in zip(hypotheses, weights):
    for i, a in enumerate(assignments):
        if a == 0:
            beta[i,-1] += w  # miss
        else:
            beta[i,a-1] += w  # measurement indices are 1-based in assignments

# JPDA weighted update per track
X_post = []
P_post = []
for i in range(num_tracks):
    x_mix = np.zeros_like(X_prior[i])
    P_mix = np.zeros_like(P_prior[i])
    # contributions from each measurement
    for j in range(num_meas):
        if beta[i,j] > 0:
            x_up, P_up = kalman_update(X_prior[i], P_prior[i], Z[j], H, R)
            x_mix += beta[i,j] * x_up
            P_mix += beta[i,j] * (P_up + np.outer(x_up, x_up))
    # miss contribution (no update)
    x_miss = X_prior[i]; P_miss = P_prior[i]
    x_mix += beta[i,-1] * x_miss
    P_mix += beta[i,-1] * (P_miss + np.outer(x_miss, x_miss))
    # convert second moment to covariance
    P_final = P_mix - np.outer(x_mix, x_mix)
    X_post.append(x_mix); P_post.append(P_final)

# Outputs: X_post, P_post, beta
print("betas:", beta); print("posteriors:", X_post)