import numpy as np
from numpy.linalg import pinv, eig

def mvdr_weights(R, a):
    # regularize to avoid singularity
    reg = 1e-6 * np.trace(R) / R.shape[0]
    Rr = R + reg * np.eye(R.shape[0])
    Rinv = pinv(Rr)
    w = Rinv @ a
    return w / (a.conj().T @ Rinv @ a)  # normalized MVDR

def cfar_ca(power, guard=2, bg=8, alpha=4.0):
    # cell-averaging CFAR: return boolean detections
    N = len(power)
    det = np.zeros(N, bool)
    for i in range(N):
        start = max(0, i - bg - guard)
        end = min(N, i + bg + guard + 1)
        # exclude guard cells around cell under test
        idx = list(range(start, max(0, i - guard))) + list(range(min(N, i + guard + 1), end))
        if len(idx) == 0:
            continue
        noise_level = np.mean(power[idx])
        threshold = alpha * noise_level
        det[i] = power[i] > threshold
    return det

# synthetic example
M, L, T = 8, 1, 200  # channels, sources, snapshots
theta = 20 * np.pi/180
# steering vector for ULA
a = np.exp(-1j * np.pi * np.arange(M) * np.sin(theta))
# simulate snapshots: desired + jammer + noise
np.random.seed(0)
s = np.exp(1j * 2*np.pi*0.05*np.arange(T))  # narrowband source
snapshots = np.outer(a, s) + 0.5*(np.random.randn(M, T)+1j*np.random.randn(M, T))
# covariance estimate
R = snapshots @ snapshots.conj().T / T
w = mvdr_weights(R, a)
beamformed = w.conj().T @ snapshots  # shape (T,)
power = np.abs(beamformed)**2
detections = cfar_ca(power, guard=2, bg=16, alpha=5.0)
# produce evidence records for Level-1
evidence = [{"t": int(t), "power": float(power[t]), "det": bool(detections[t])}
            for t in range(T)]