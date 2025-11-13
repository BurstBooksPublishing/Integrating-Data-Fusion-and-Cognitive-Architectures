import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanTrack:
    def __init__(self, x0, P0, id):
        self.x = x0        # 6-DoF pose vector (x,y,z,roll,pitch,yaw)
        self.P = P0
        self.id = id
        self.age = 0

    def predict(self, Q):
        # simple constant-position model
        self.P = self.P + Q
        self.age += 1

    def update(self, z, R, H):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P
        self.age = 0

class TrackManager:
    def __init__(self, gate_thresh=9.21):  # chi2 0.99 for 2D
        self.tracks = []
        self.next_id = 1
        self.gate = gate_thresh

    def associate(self, detections, H, R):
        if not self.tracks:
            return [], list(range(len(detections))), []
        cost = np.full((len(self.tracks), len(detections)), np.inf)
        for i,t in enumerate(self.tracks):
            for j,z in enumerate(detections):
                S = H @ t.P @ H.T + R
                d2 = (z - H @ t.x).T @ np.linalg.inv(S) @ (z - H @ t.x)
                if d2 <= self.gate:
                    cost[i,j] = d2  # lower cost better
        row,col = linear_sum_assignment(cost)
        matches, unmatched_t, unmatched_d = [], [], []
        assigned_d = set()
        for r,c in zip(row,col):
            if np.isfinite(cost[r,c]):
                matches.append((r,c)); assigned_d.add(c)
            else:
                unmatched_t.append(r)
        for j in range(len(detections)):
            if j not in assigned_d: unmatched_d.append(j)
        return matches, unmatched_d, unmatched_t

    def step(self, detections, z_cov):
        H = np.eye(6)      # identity observation for 6-DoF
        R = z_cov
        # predict all
        Q = np.eye(6)*1e-3
        for t in self.tracks: t.predict(Q)
        matches, unassoc_d, unassoc_t = self.associate(detections, H, R)
        for (i,j) in matches:
            self.tracks[i].update(detections[j], R, H)
        for j in unassoc_d:
            # spawn new track
            x0 = detections[j]; P0 = np.eye(6)*0.1
            self.tracks.append(KalmanTrack(x0,P0,self.next_id)); self.next_id+=1
        # age-out stale tracks
        self.tracks = [t for t in self.tracks if t.age < 10]

# Example usage
if __name__ == "__main__":
    tm = TrackManager()
    detections = [np.zeros(6), np.array([1.0,0,0,0,0,0])]  # two detections
    cov = np.eye(6)*0.05
    tm.step(detections, cov)
    for t in tm.tracks: print("Track",t.id,"pose",t.x[:3])