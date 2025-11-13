import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

def mahalanobis_nll(z, x, P, H, R):
    # innovation and covariance
    y = z - H @ x
    S = H @ P @ H.T + R
    invS = np.linalg.inv(S)
    d2 = float(y.T @ invS @ y)                         # Mahalanobis
    nll = 0.5 * d2 + 0.5 * np.log(np.linalg.det(S))    # ignore constant term
    return nll

# synthetic tracks/detections and embeddings
np.random.seed(0)
T = 5; D = 6; dim = 2
tracks_x = [np.random.randn(dim) for _ in range(T)]
tracks_P = [np.eye(dim)*0.5 for _ in range(T)]
H = np.eye(dim); R = np.eye(dim)*0.2
dets_z = [x + 0.1*np.random.randn(dim) for x in tracks_x] + [np.random.randn(dim)]  # one spurious

# synthetic embeddings for appearance similarity
track_emb = np.random.randn(T,128)
det_emb = np.random.randn(D,128)

# train a tiny calibrator on synthetic match/non-match pairs
# label true pairs where embedding distance small
pairs = []
labels = []
for i in range(T):
    for j in range(D):
        sim = np.dot(track_emb[i], det_emb[j]) / (np.linalg.norm(track_emb[i])*np.linalg.norm(det_emb[j]))
        pairs.append([sim])
        labels.append(int(j < T and np.allclose(dets_z[j], tracks_x[i], atol=0.3)))  # proxy label
clf = LogisticRegression().fit(pairs, labels)            # Platt-style logistic calibrator

# build cost matrix (rows=tracks, cols=detections)
cost = np.full((T, D), 1e6)
eps = 1e-9
for i in range(T):
    for j in range(D):
        nll = mahalanobis_nll(dets_z[j], tracks_x[i], tracks_P[i], H, R)
        sim = np.dot(track_emb[i], det_emb[j]) / (np.linalg.norm(track_emb[i])*np.linalg.norm(det_emb[j]))
        p_app = float(clf.predict_proba([[sim]])[0,1])
        combined = nll - 2.0 * np.log(p_app + eps)   # lambda=2.0 chosen by policy
        cost[i,j] = combined

row_ind, col_ind = linear_sum_assignment(cost)  # solves min-cost assignment
assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
print("assignments:", assignments)