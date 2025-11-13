import numpy as np
from sklearn.metrics import f1_score

# synthetic example: replace with real OSPA and labels
def normalize(x, lo, hi): return np.clip((x-lo)/(hi-lo), 0, 1)

# compute normalized fusion metric (lower OSPA better)
ospa = 3.2                   # example OSPA distance
ospa_lo, ospa_hi = 0.0, 10.0
F_norm = 1.0 - normalize(ospa, ospa_lo, ospa_hi)  # higher better

# cognitive output: scenario labels and predictions
y_true = np.array([1,0,1,1,0])      # golden situation labels
y_pred = np.array([1,0,0,1,0])      # system decisions
C_raw = f1_score(y_true, y_pred)    # scenario F1

# latency normalization (seconds)
latency = 0.18
lat_lo, lat_hi = 0.05, 1.0
L_norm = normalize(latency, lat_lo, lat_hi)

# weights tuned by mission owners
alpha, beta, gamma = 0.4, 0.5, 0.1
U = alpha * F_norm + beta * C_raw - gamma * L_norm
print(f"F_norm={F_norm:.3f}, C={C_raw:.3f}, L_norm={L_norm:.3f}, U={U:.3f}")