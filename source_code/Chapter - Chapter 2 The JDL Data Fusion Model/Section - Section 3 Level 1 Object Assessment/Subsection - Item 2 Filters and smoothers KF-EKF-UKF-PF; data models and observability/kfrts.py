import numpy as np

def kalman_predict(x,P,F,Q):
    x_pred = F @ x                      # predict state
    P_pred = F @ P @ F.T + Q            # predict covariance
    return x_pred, P_pred

def kalman_update(x_pred,P_pred,z,H,R):
    S = H @ P_pred @ H.T + R            # innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S) # Kalman gain
    y = z - H @ x_pred                  # innovation
    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_upd, P_upd, y, S

def rts_smoother(xs, Ps, F, Q):
    # xs, Ps: lists of filtered states/covs for k=0..N
    N = len(xs)-1
    xs_s = xs.copy()
    Ps_s = Ps.copy()
    for k in range(N-1, -1, -1):
        P_pred = F @ Ps[k] @ F.T + Q
        C = Ps[k] @ F.T @ np.linalg.inv(P_pred)  # smoother gain
        xs_s[k] = xs[k] + C @ (xs_s[k+1] - F @ xs[k])
        Ps_s[k] = Ps[k] + C @ (Ps_s[k+1] - P_pred) @ C.T
    return xs_s, Ps_s

# Example usage: 4D constant-velocity, 2D position measurement
dt = 1.0
F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
H = np.array([[1,0,0,0],[0,1,0,0]])
Q = np.eye(4)*0.1
R = np.eye(2)*1.0
x = np.zeros(4); P = np.eye(4)*10

zs = [np.array([i+np.random.randn()*0.5, i*0.5+np.random.randn()*0.5]) for i in range(20)]
xs, Ps = [], []
for z in zs:
    x,P = kalman_predict(x,P,F,Q)
    x,P,_,_ = kalman_update(x,P,z,H,R)
    xs.append(x.copy()); Ps.append(P.copy())
xs_sm, Ps_sm = rts_smoother(xs, Ps, F, Q)  # fixed-lag smoothing