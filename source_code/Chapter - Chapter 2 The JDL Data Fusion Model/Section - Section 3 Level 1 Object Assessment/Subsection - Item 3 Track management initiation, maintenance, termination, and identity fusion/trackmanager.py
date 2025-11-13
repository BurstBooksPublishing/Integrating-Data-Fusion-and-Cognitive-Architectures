import numpy as np

# Simple 2D constant-velocity KF and track manager
class Track:
    def __init__(self, meas, meas_cov, id_prior=1e-3):
        # state: [x,y,vx,vy]
        self.x = np.array([meas[0], meas[1], 0.0, 0.0])
        self.P = np.diag([10.,10.,100.,100.])
        self.exist_score = np.log(id_prior/(1-id_prior))  # logit
        self.id_post = {}  # identity posterior dict
        self.last_update = 0

    def predict(self, dt):
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        Q = np.eye(4)*0.1
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

    def gating_maha(self, meas, R):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        S = H.dot(self.P).dot(H.T) + R
        y = meas - H.dot(self.x)
        maha = float(y.T.dot(np.linalg.inv(S)).dot(y))
        return maha

    def update(self, meas, R, detect_llr=5.0):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        y = meas - H.dot(self.x)
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4)-K.dot(H)).dot(self.P)
        # existence score update by simple LLR surrogate
        self.exist_score += detect_llr  # add detection evidence
        self.last_update = 0

    def miss(self, miss_penalty=1.0):
        self.exist_score -= miss_penalty
        self.last_update += 1

    def promote(self, thresh=2.0):
        p = 1/(1+np.exp(-self.exist_score))
        return p > 1/(1+np.exp(-thresh))