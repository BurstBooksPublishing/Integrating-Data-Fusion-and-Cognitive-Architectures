import numpy as np
class OcclusionHypothesis:
    def __init__(self, state, cov, weight=0.2):
        self.x = np.asarray(state)        # [x,y,vx,vy]
        self.P = np.asarray(cov)          # covariance
        self.w = float(weight)            # existence prob

    def predict(self, dt, Q):
        F = np.eye(4)
        F[0,2] = F[1,3] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def miss_update(self, decay=0.9, birth=0.01):
        # reduce weight when no supporting measurement; allow small birth
        self.w = decay * self.w + birth*(1-self.w)
        # inflate uncertainty
        self.P *= 1.25

    def detect_update(self, z, R):
        # simple Kalman update for demonstration
        H = np.array([[1,0,0,0],[0,1,0,0]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        # boost existence on detection
        self.w = min(1.0, 0.8*self.w + 0.2)

    def should_caution(self, speed, tau=0.35):
        # trigger caution if existence weight and relative speed imply risk
        stopping_margin = (speed**2) / (2*3.0)  # simple braking model, a=3 m/s2
        dist = np.linalg.norm(self.x[:2])
        return (self.w > tau) and (dist < stopping_margin*1.5)

# Example usage
Q = np.diag([0.1,0.1,0.5,0.5])
R = np.diag([0.6,0.6])
h = OcclusionHypothesis([10,0,0,0], np.eye(4)*1.0)
h.predict(0.1, Q)
h.miss_update()
if h.should_caution(speed=8.0):
    print("engage cautious policy")