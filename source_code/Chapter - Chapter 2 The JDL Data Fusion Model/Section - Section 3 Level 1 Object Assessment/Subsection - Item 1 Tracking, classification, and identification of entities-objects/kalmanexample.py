import numpy as np

# Simple 4D constant-velocity KF for position (x,y) and velocity (vx,vy).
F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
H = np.array([[1,0,0,0],[0,1,0,0]])
Q = np.eye(4)*0.01
R = np.eye(2)*0.1

def predict(x,P):
    x = F @ x
    P = F @ P @ F.T + Q
    return x,P

def update(x,P,z,class_p,track_class_p):
    # KF update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    y = z - H @ x
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H) @ P
    # Fuse class probabilities (Bayesian multiplicative update, renormalize)
    # class_p and track_class_p are dicts of class->prob
    fused = {}
    for c in track_class_p:
        fused[c] = class_p.get(c,1e-6) * track_class_p[c]
    # normalize
    s = sum(fused.values())
    for c in fused: fused[c] /= s
    return x,P,fused

# Initialize a track
x = np.array([0.,0.,0.,0.])
P = np.eye(4)
track_class_p = {'ped':0.6,'veh':0.4}

# Incoming measurement and classifier softmax
z = np.array([1.02,0.95])
class_p = {'ped':0.8,'veh':0.2}

x,P = predict(x,P)
# Simple gate: Mahalanobis distance
S = H @ P @ H.T + R
d2 = (z - H @ x).T @ np.linalg.inv(S) @ (z - H @ x)
if d2 < 9.21:  # chi2 2-dof 99% ~9.21
    x,P,track_class_p = update(x,P,z,class_p,track_class_p)
# x,P,track_class_p now updated
print('state',x,'class_p',track_class_p)