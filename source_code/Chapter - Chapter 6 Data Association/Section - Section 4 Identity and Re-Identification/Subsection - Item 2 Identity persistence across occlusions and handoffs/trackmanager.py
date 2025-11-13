import time, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis

# Simple track object
class Track:
    def __init__(self, state, cov, embedding, track_id, sensor_id):
        self.state = state                # kinematic state vector
        self.cov = cov                    # covariance matrix
        self.embedding = embedding        # re-id vector
        self.id = track_id
        self.sensor_id = sensor_id
        self.last_seen = time.time()
        self.missed = 0                   # consecutive missed frames

# Cosine similarity
def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return (a @ b) / (na*nb + 1e-9)

# Mahalanobis distance wrapper
def mahalanobis_dist(x, z, cov):
    VI = np.linalg.inv(cov)
    return mahalanobis(x, z, VI)

# Association cost matrix combining kinematics, appearance, handoff penalty
def association_cost(tracks, detections, w_kin=1.0, w_app=2.0, handoff_penalty=1.5, kappa=0.1):
    m, n = len(tracks), len(detections)
    C = np.full((m,n), 1e6)
    for i,t in enumerate(tracks):
        for j,d in enumerate(detections):
            dM = mahalanobis_dist(t.state, d['state'], t.cov)
            sim = cos_sim(t.embedding, d['embedding'])
            dt = time.time() - t.last_seen
            sim_eff = sim * np.exp(-kappa*dt)         # staleness decay
            pen = handoff_penalty if t.sensor_id != d['sensor_id'] else 0.0
            score = -w_kin * (dM**2)/2.0 - w_app*(sim_eff) - pen
            C[i,j] = -score                            # convert to cost
    return C

# Tracker update step (simplified)
def update_tracks(tracks, detections, next_id):
    if len(tracks)==0:
        # create new tracks
        for d in detections:
            tracks.append(Track(d['state'], d['cov'], d['embedding'], next_id(), d['sensor_id']))
        return tracks
    C = association_cost(tracks, detections)
    row,col = linear_sum_assignment(C)
    assigned_det = set()
    for i,j in zip(row,col):
        if C[i,j] < 1000:                    # threshold to accept assignment
            t = tracks[i]
            d = detections[j]
            t.state = d['state']             # replace with measurement or run KF predict+update
            t.cov = d['cov']
            t.embedding = 0.5*(t.embedding + d['embedding'])  # simple fusion
            t.last_seen = time.time()
            t.missed = 0
            t.sensor_id = d['sensor_id']
            assigned_det.add(j)
    # create tracks for unassigned detections
    for j,d in enumerate(detections):
        if j not in assigned_det:
            tracks.append(Track(d['state'], d['cov'], d['embedding'], next_id(), d['sensor_id']))
    # age unassigned tracks into ghosts and retire if missed too long
    for t in tracks[:]:
        if time.time() - t.last_seen > 0.2:    # assume 0.2s frame gap
            t.missed += 1
        if t.missed > 15:                      # retire after threshold
            tracks.remove(t)
    return tracks