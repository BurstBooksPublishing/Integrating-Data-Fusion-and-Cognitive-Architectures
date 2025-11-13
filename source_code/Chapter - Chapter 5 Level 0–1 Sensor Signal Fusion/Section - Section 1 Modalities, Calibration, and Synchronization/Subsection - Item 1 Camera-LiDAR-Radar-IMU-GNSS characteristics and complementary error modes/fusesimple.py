import numpy as np

# Measurements (x,y) and covariances (2x2) from GNSS and LiDAR
z1 = np.array([12.3, 7.8])         # GNSS (noisy absolute)
R1 = np.diag([3.0**2, 3.0**2])     # GNSS ~3 m std dev

z2 = np.array([12.1, 7.6])         # LiDAR-localized relative (registered)
R2 = np.diag([0.15**2, 0.15**2])   # LiDAR ~15 cm std dev

# Precision-weighted fusion (eq. inv_cov)
P_inv = np.linalg.inv(R1) + np.linalg.inv(R2)
P = np.linalg.inv(P_inv)
x_fused = P @ (np.linalg.inv(R1) @ z1 + np.linalg.inv(R2) @ z2)

# Consistency: Normalized Innovation Squared (NIS)
innovation = z2 - x_fused         # compare one sensor against fused
NIS = innovation.T @ np.linalg.inv(R2 + P) @ innovation

print("Fused position:", x_fused)
print("Fused covariance:\n", P)
print("NIS:", float(NIS))          # compare to chi2 df=2 threshold