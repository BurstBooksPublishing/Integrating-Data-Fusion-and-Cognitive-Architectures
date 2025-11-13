import numpy as np

class Anchor:
    def __init__(self, aid, x, P, desc, first_seen):
        self.id = aid
        self.x = np.array(x)        # pose mean
        self.P = np.array(P)        # covariance
        self.desc = np.array(desc)  # descriptor vector
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.confidence = 0.5       # epistemic confidence [0,1]
    def mahalanobis(self, z, H, R):
        S = H @ self.P @ H.T + R
        e = z - H @ self.x
        return float(e.T @ np.linalg.inv(S) @ e)
    def appearance_sim(self, obs_desc):
        a = self.desc / (np.linalg.norm(self.desc)+1e-9)
        b = obs_desc / (np.linalg.norm(obs_desc)+1e-9)
        return float(np.clip(a @ b, 0.0, 1.0))  # cosine in [0,1]
    def associate_score(self, z, H, R, obs_desc, beta=10.0, tau=0.6):
        d2 = self.mahalanobis(z,H,R)
        s = self.appearance_sim(obs_desc)
        spatial = np.exp(-0.5*d2)
        sem = 1.0/(1.0+np.exp(-beta*(s-tau)))   # logistic
        return spatial * sem
    def kalman_update(self, z, H, R, obs_desc, t):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innov = z - H @ self.x
        self.x = self.x + K @ innov
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
        self.desc = 0.9*self.desc + 0.1*obs_desc   # simple descriptor smoothing
        self.last_seen = t
        # update confidence heuristics
        self.confidence = min(1.0, self.confidence + 0.05*(np.exp(-0.5*(innov@innov))))
# Usage: create anchors, compute associate_score for candidates, pick best, call kalman_update.