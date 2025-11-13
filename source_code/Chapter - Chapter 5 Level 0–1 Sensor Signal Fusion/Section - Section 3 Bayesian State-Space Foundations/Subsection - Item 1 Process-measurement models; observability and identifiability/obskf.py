import numpy as np

def observability_matrix(F, H):
    n = F.shape[0]
    O = []
    M = np.eye(n)
    for i in range(n):
        O.append(H @ M)       # H F^i
        M = M @ F
    return np.vstack(O)

# simple constant-velocity model
dt = 0.1
F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
H = np.array([[1,0,0,0],[0,1,0,0]])  # position measurements only

O = observability_matrix(F, H)
rank = np.linalg.matrix_rank(O)
if rank < F.shape[0]:
    print("ALERT: system has unobservable modes; rank=", rank)

# simulate and run KF (compact)
Q = 0.01 * np.eye(4); R = 0.1 * np.eye(2)
x = np.zeros(4); P = np.eye(4)
for k in range(50):
    # predict (no control)
    x = F @ x
    P = F @ P @ F.T + Q
    # generate noisy measurement (simulate)
    z = H @ x + np.random.multivariate_normal(np.zeros(2), R)
    # Kalman update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ (z - H @ x)
    P = (np.eye(4) - K @ H) @ P
    # diagnostic: monitor NEES (normalized estimation error squared)
    # here we lack ground truth; use innovation statistic instead
    inn = z - H @ x
    if inn.T @ np.linalg.inv(S) @ inn > 16:   # simple threshold
        print(f"High innovation at step {k}; check sensor/model")