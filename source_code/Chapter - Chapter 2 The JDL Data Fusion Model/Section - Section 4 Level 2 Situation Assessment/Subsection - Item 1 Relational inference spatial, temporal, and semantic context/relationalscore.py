import numpy as np
import networkx as nx

# Example tracks (pose x,y), covariances, velocity vectors
tracks = {
    'tveh': {'pos': np.array([10.0, 5.0]), 'vel': np.array([-2.0, 0.0]), 'cov': np.eye(2)*0.5},
    'tped': {'pos': np.array([8.0, 5.5]), 'vel': np.array([0.5, 0.0]), 'cov': np.eye(2)*0.2}
}

def mahalanobis(p1, p2, S):
    d = p1 - p2
    return float(d.T @ np.linalg.inv(S) @ d)

def approaching_score(a, b, time_window=2.0):
    # spatial likelihood (lower d^2 -> higher score)
    S = a['cov'] + b['cov']  # combined uncertainty
    d2 = mahalanobis(a['pos'], b['pos'], S)
    spatial = np.exp(-0.5 * d2)  # Gaussian-like
    # relative radial speed (positive means closing)
    rel_vel = np.dot((b['vel'] - a['vel']), (b['pos'] - a['pos'])) 
    temporal = 1.0 if rel_vel > 0.1 else 0.1  # simple sign test
    return spatial * temporal

G = nx.DiGraph()
for tid, info in tracks.items():
    G.add_node(tid, **info)

score = approaching_score(tracks['tveh'], tracks['tped'])
G.add_edge('tveh', 'tped', rel='approaching', score=score)
print(f"Approach score: {score:.3f}")  # use for downstream thresholding