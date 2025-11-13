import numpy as np
from numpy.linalg import slogdet, inv
from math import log

def avg_accuracy(P): return P[:, -1].mean()  # final acc over domains

def forgetting(P):
    # P shape: (num_domains, num_checkpoints)
    return (P.max(axis=1) - P[:, -1]).mean()

def bwt(P):
    # P[i,i] is perf right after learning domain i (assumes checkpoint ordering aligns)
    N = P.shape[0]
    return np.mean(P[:, -1] - np.diag(P[:, :N]))  # diagonal as per-recorded after-train

def fwt(P, random_baseline):
    # initial column compared to random baseline
    return (P[:, 0] - random_baseline).mean()

def kl_gaussian(X, Y):
    # X, Y: (n_samples, dim) feature batches
    mu0 = X.mean(axis=0); mu1 = Y.mean(axis=0)
    S0 = np.cov(X, rowvar=False) + 1e-6*np.eye(X.shape[1])
    S1 = np.cov(Y, rowvar=False) + 1e-6*np.eye(Y.shape[1])
    sign0, logdet0 = slogdet(S0); sign1, logdet1 = slogdet(S1)
    k = X.shape[1]
    term = np.trace(inv(S1) @ S0) + (mu1-mu0) @ inv(S1) @ (mu1-mu0) - k
    return 0.5*(term + (logdet1 - logdet0))

# Synthetic quick-run
P = np.array([[0.8,0.78,0.75],[0.6,0.62,0.65]])  # two domains, three checkpoints
print("ACC", avg_accuracy(P))
print("Forgetting", forgetting(P))
print("BWT", bwt(P))
print("FWT", fwt(P, random_baseline=0.5))
# features X and Y would be real embeddings from sensors/models