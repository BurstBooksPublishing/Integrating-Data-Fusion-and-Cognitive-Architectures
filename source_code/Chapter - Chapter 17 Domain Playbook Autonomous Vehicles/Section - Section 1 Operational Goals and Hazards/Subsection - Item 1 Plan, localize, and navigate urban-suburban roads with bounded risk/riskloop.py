#!/usr/bin/env python3
import numpy as np

# Simple EKF update for pose (2D) with odom+GNSS (toy example).
def ekf_predict(x, P, u, Q):
    A = np.eye(3)  # state transition approx
    x = A @ x + u
    P = A @ P @ A.T + Q
    return x, P

def ekf_update(x, P, z, R):
    H = np.eye(3)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ (z - H @ x)
    P = (np.eye(3) - K @ H) @ P
    return x, P

# Risk scoring: approximate P_coll by convolving ego pose covariance with occupancy prob at point.
def point_collision_probability(x_mean, P, occ_prob):
    # Use Mahalanobis distance heuristic mapping to probability
    d2 = np.sum((x_mean[:2])**2)  # distance to cell center assumed at origin in this toy
    sigma = np.sqrt(np.trace(P[:2,:2]) + 1e-6)
    # map distance and spread to collision likelihood
    p = occ_prob * np.exp(-0.5 * d2 / (sigma**2))
    return float(min(max(p, 0.0), 1.0))

# Loop: fuse, predict, plan-check
x = np.array([0.0, 0.0, 0.0])      # pose x,y,yaw
P = np.eye(3) * 0.5
Q = np.eye(3) * 0.01
R = np.eye(3) * 1.0
R_budget = 0.02  # allowable cumulative collision risk

for t in range(1, 6):  # 5-step horizon
    u = np.array([0.5, 0.0, 0.0])     # commanded motion
    x, P = ekf_predict(x, P, u, Q)
    # occasional GNSS measurement (simulate)
    if t % 2 == 0:
        z = x + np.random.randn(3) * 0.5
        x, P = ekf_update(x, P, z, R)
    # occupancy probability ahead (higher in occlusion)
    occ_prob = 0.2 if t < 3 else 0.05
    p_coll = point_collision_probability(x, P, occ_prob)
    # accumulate risk and decide fallback
    if t == 1: cum_risk = 0.0
    cum_risk += p_coll
    print(f"t={t}, p_coll={p_coll:.4f}, cum_risk={cum_risk:.4f}")
    if cum_risk > R_budget:
        print("Risk budget exceeded: trigger minimum-risk maneuver (stop).")
        break
# End of loop