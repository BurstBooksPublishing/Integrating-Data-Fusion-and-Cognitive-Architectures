import numpy as np
import networkx as nx
from math import cos
from sklearn.cluster import DBSCAN

def pair_affinity(pos_i,pos_j,heading_i,heading_j,w=1.0,sigma=10.0):
    d = np.linalg.norm(pos_i-pos_j)
    delta_psi = abs((heading_i-heading_j + np.pi) % (2*np.pi) - np.pi)  # angular diff
    return w * np.exp(-0.5*(d/sigma)**2) * cos(delta_psi)

def build_affinity_matrix(positions,headings,weights=None,sigma=10.0):
    n = len(positions)
    A = np.zeros((n,n))
    if weights is None: weights = np.ones(n)
    for i in range(n):
        for j in range(i+1,n):
            a = pair_affinity(positions[i],positions[j],headings[i],headings[j],
                              w=weights[i]*weights[j], sigma=sigma)
            A[i,j] = A[j,i] = max(0.0,a)  # keep non-negative affinities
    return A

def detect_formations(positions,headings,ids,timestamp,
                      sigma=10.0,eps=0.3,min_samples=2,persist_seconds=5,
                      state_store=None):
    A = build_affinity_matrix(np.array(positions),np.array(headings),sigma=sigma)
    # convert affinities to distance for DBSCAN: higher affinity -> smaller distance
    maxA = max(A.max(),1e-6)
    D = 1.0 - (A / maxA)
    # cluster with DBSCAN on affinity-derived distances
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(D)
    clusters = {}
    for lbl,entity in zip(db.labels_, ids):
        if lbl == -1: continue
        clusters.setdefault(lbl, []).append(entity)
    # persistence check: required in system lifecycle; use state_store to track times
    results = []
    for lbl,members in clusters.items():
        key = tuple(sorted(members))
        state = state_store.setdefault(key, {'first_seen': timestamp, 'last_seen': timestamp})
        state['last_seen'] = timestamp
        duration = state['last_seen'] - state['first_seen']
        if duration >= persist_seconds:
            results.append({'members': members, 'formed_at': state['first_seen'],
                            'last_seen': state['last_seen']})
    return results