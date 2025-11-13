import numpy as np

# Simple constant-velocity KF for 2D position
class Track:
    def __init__(self, id, x0, P0):
        self.id = id
        self.x = x0           # state [px, py, vx, vy]
        self.P = P0
        self.Q = np.diag([0.1,0.1,1.0,1.0]) # process noise
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
    def predict(self, dt):
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    def update(self, z, R):
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P

# Condition aggregator: EWMA + z-score flagging
class ConditionMonitor:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ewma_vib = None
        self.ewma_tmp = None
        self.history = []
    def ingest(self, vib, tmp):
        if self.ewma_vib is None:
            self.ewma_vib = vib; self.ewma_tmp = tmp
        else:
            self.ewma_vib = self.alpha*vib + (1-self.alpha)*self.ewma_vib
            self.ewma_tmp = self.alpha*tmp + (1-self.alpha)*self.ewma_tmp
        self.history.append((vib,tmp))
    def health_score(self):
        # combine normalized departures; lower is healthier
        vib_arr = np.array([h[0] for h in self.history])
        tmp_arr = np.array([h[1] for h in self.history])
        vib_z = (self.ewma_vib - vib_arr.mean())/(vib_arr.std()+1e-6)
        tmp_z = (self.ewma_tmp - tmp_arr.mean())/(tmp_arr.std()+1e-6)
        return 1.0 / (1.0 + max(0, vib_z) + max(0, tmp_z))  # heuristic