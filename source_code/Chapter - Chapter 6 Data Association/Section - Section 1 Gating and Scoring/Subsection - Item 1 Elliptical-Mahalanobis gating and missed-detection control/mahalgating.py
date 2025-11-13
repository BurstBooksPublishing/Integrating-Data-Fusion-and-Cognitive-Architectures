import numpy as np
from scipy.stats import chi2
from scipy.special import gamma

def mahalanobis_d2(z, z_hat, S):
    nu = z - z_hat
    return float(nu.T @ np.linalg.inv(S) @ nu)

def gate_threshold(m, P_G):
    return chi2.ppf(P_G, df=m)  # gamma

def gate_volume(m, S, gamma_val):
    V_m = np.pi**(m/2) / gamma(m/2 + 1)   # unit-ball volume
    return V_m * np.sqrt(np.linalg.det(S)) * (gamma_val**(m/2))

# Example usage: 2D measurement, one track
m = 2
P_G = 0.997
gamma_val = gate_threshold(m, P_G)
S = np.array([[4.0, 0.5],[0.5, 3.0]])         # innovation covariance
z_hat = np.array([10.0, 0.2])                # predicted measurement
measurements = np.array([[12.1, -0.1],[9.9,0.3],[50.0,5.0]])  # candidates

# Gate test
gated = []
for z in measurements:
    d2 = mahalanobis_d2(z, z_hat, S)
    if d2 <= gamma_val:
        gated.append((z, d2))
# Gate volume and expected clutter
V_gate = gate_volume(m, S, gamma_val)
lambda_c = 1e-4   # clutter density per measurement-space unit
expected_false_alarms = lambda_c * V_gate

# Print results (brief)
print("gamma:", gamma_val)
print("gated measurements:", gated)
print("V_gate:", V_gate, "expected false alarms/scan:", expected_false_alarms)