import numpy as np
from scipy.stats import chi2

def nis_and_gate(z, z_pred, H, P_pred, R, pd=0.95):
    # innovation and its covariance
    nu = z - z_pred                      # measurement residual
    S = H @ P_pred @ H.T + R             # eq. (1)
    d2 = float(nu.T @ np.linalg.solve(S, nu))  # Mahalanobis distance
    gamma = chi2.ppf(pd, df=nu.size)     # gate threshold
    return d2, gamma, S, nu

def adaptive_inflation(nis_window, dof, alpha=0.05, inflate_factor=1.5):
    # sum NIS over window ~ chi2(df = dof * N)
    N = len(nis_window)
    s = np.sum(nis_window)
    lower = chi2.ppf(alpha/2, df=dof*N)
    upper = chi2.ppf(1-alpha/2, df=dof*N)
    if s > upper:
        return inflate_factor  # underestimation: inflate covariances
    if s < lower:
        return 1.0/inflate_factor  # overestimation: deflate (careful)
    return 1.0  # no change

# Example integration within filter cycle omitted for brevity.