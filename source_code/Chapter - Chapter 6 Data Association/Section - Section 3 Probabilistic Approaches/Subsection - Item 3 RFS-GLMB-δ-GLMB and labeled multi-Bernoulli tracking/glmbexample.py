import itertools, heapq, math
import numpy as np
from scipy.stats import multivariate_normal as mvn

# Simple Gaussian single-target likelihood under linear H, R
def single_likelihood(mean, cov, z, H, R):
    S = H @ cov @ H.T + R
    zpred = H @ mean
    return float(mvn.pdf(z, mean=zpred, cov=S))

# Tracks: list of dicts with 'label','r','mean','cov'
tracks = [
    {'label':1,'r':0.8,'mean':np.array([0.,0.]),'cov':np.eye(2)},
    {'label':2,'r':0.6,'mean':np.array([5.,0.]),'cov':np.eye(2)*1.5},
]

# Measurements (2D)
Z = [np.array([0.2,-0.1]), np.array([5.1,0.3])]

# Sensor model
H = np.eye(2); R = np.eye(2)*0.1
Pd = 0.9; clutter_density = 1e-3

# Precompute likelihoods and missed-detection term
L = [[single_likelihood(t['mean'], t['cov'], z, H, R) for z in Z] for t in tracks]
miss = [ (1 - t['r']) + t['r']*(1-Pd) for t in tracks ]  # unassigned term per track

# Enumerate feasible assignments (small N example): each track -> {0..M} where 0 means missed
M = len(Z); N = len(tracks)
feasible = []
for assign in itertools.product(range(M+1), repeat=N):
    # no two tracks assign same measurement
    assigned = [a for a in assign if a>0]
    if len(assigned) != len(set(assigned)): continue
    # compute unnormalized weight: product over tracks of psi
    w = 1.0
    for i,a in enumerate(assign):
        if a==0:
            w *= miss[i]
        else:
            # include Pd, single-target likelihood and divide by clutter density
            w *= tracks[i]['r'] * Pd * L[i][a-1] / clutter_density
    feasible.append((w,assign))

# Return K-best hypotheses
K = 5
best = heapq.nlargest(K, feasible, key=lambda x: x[0])
for w,assign in best:
    print(f"weight={w:.3e}, assignment={assign}")
# The caller will perform per-hypothesis Kalman updates and marginalize back to LMB.