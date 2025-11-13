import numpy as np
from scipy.optimize import linear_sum_assignment

# Simple KF for constant velocity in 2D; dt assumed 1 for brevity.
class KalmanTrack:
    def __init__(self, x, P):
        self.x = x         # state [px, py, vx, vy]
        self.P = P
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        q = 1e-2
        self.Q = q * np.eye(4)
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z, R):
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P

# Example tracks and measurements
tracks = [KalmanTrack(np.array([0.,0.,1.,0.]), np.eye(4)),
          KalmanTrack(np.array([10.,0.,-1.,0.]), np.eye(4)*2)]
measurements = [np.array([0.9, 0.2]), np.array([9.0, -0.1]), np.array([50.,50.])] # one clutter
R = np.eye(2)*0.5

# Predict
for t in tracks:
    t.predict()

# Build cost with Mahalanobis gating; large cost if gated out.
cost = np.full((len(tracks), len(measurements)), 1e6)
for i,t in enumerate(tracks):
    for j,z in enumerate(measurements):
        S = t.H @ t.P @ t.H.T + R
        y = z - (t.H @ t.x)
        d2 = float(y.T @ np.linalg.inv(S) @ y)
        gate_thresh = 9.21 # chi2 2-dof 0.99
        if d2 <= gate_thresh:
            cost[i,j] = d2

# Assignment
row_ind, col_ind = linear_sum_assignment(cost)
for i,j in zip(row_ind, col_ind):
    if cost[i,j] < 1e5:
        tracks[i].update(measurements[j], R)  # matched update
# unmatched measurements -> spawn; unmatched tracks -> miss handling