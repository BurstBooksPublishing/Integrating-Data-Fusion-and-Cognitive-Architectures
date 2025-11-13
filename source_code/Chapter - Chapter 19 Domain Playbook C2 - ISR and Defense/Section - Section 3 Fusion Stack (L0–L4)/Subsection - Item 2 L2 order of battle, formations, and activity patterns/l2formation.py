import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import mahalanobis

def formation_hypotheses(tracks, eps=500, min_samples=3):
    # tracks: list of dicts {'id':, 'pos':(x,y), 'cov':2x2, 'emitters':[...] , 't':timestamp}
    X = np.array([t['pos'] for t in tracks])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    hyps = []
    for label in set(db.labels_):
        if label == -1: continue
        idx = np.where(db.labels_ == label)[0]
        members = [tracks[i] for i in idx]
        # compute centroid and aggregate score (sum Mahalanobis distances)
        mu = X[idx].mean(axis=0)
        score = 0.0
        for m in members:
            VI = np.linalg.inv(m['cov'])            # inverse covariance
            d2 = (m['pos'] - mu) @ VI @ (m['pos'] - mu)
            score += d2
        # convert to likelihood-like score
        likelihood = np.exp(-0.5 * score)
        # rudimentary role heuristic using emitter presence and geometry
        emitters = set(e for m in members for e in m.get('emitters',[]))
        role = 'unknown'
        if 'radar_cmd' in emitters: role = 'command_element'
        elif len(members) >= 8: role = 'battalion_proxy'
        hyps.append({
            'members': [m['id'] for m in members],
            'centroid': mu.tolist(),
            'score': float(likelihood),
            'role': role,
            'provenance': {'method':'dbscan+mahalanobis','member_count':len(members)}
        })
    return hyps

# Example usage: feed L1 track list and inspect hypotheses.