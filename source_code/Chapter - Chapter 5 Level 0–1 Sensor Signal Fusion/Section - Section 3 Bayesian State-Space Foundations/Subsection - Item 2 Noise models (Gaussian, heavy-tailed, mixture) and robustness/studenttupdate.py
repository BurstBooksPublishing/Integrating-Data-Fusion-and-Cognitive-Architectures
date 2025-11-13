import numpy as np

def student_t_kalman_update(x_prior, P_prior, z, H, R, nu, max_iter=10, tol=1e-6):
    """Iterative Student-t measurement update. x_prior: (n,), P_prior: (n,n).
       z: (m,), H: (m,n), R: (m,m), nu: degrees of freedom scalar."""
    x = x_prior.copy()
    P = P_prior.copy()
    m = z.shape[0]
    # Initial innovation and covariance
    for _ in range(max_iter):
        r = z - H @ x                          # residual (m,)
        S = H @ P @ H.T + R                    # innovation cov (m,m)
        # Mahalanobis distance
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        maha = float(r.T @ S_inv @ r)          # scalar
        # Expected precision (scale-mixture weight)
        lam = (nu + m) / (nu + maha)           # weight per Eq. (2)
        # Scaled measurement covariance
        R_tilde = R / lam
        S_tilde = H @ P @ H.T + R_tilde
        try:
            S_tilde_inv = np.linalg.inv(S_tilde)
        except np.linalg.LinAlgError:
            S_tilde_inv = np.linalg.pinv(S_tilde)
        K = P @ H.T @ S_tilde_inv               # Kalman gain
        x_new = x + K @ (z - H @ x)
        P_new = P - K @ H @ P
        # Convergence check
        if np.linalg.norm(x_new - x) < tol:
            return x_new, P_new
        x, P = x_new, P_new
    return x, P
# Minimal test (1D measurement)
if __name__ == "__main__":
    x0 = np.array([0.,0.,0.])                # state
    P0 = np.eye(3)*1.0
    H = np.array([[1.,0.,0.]])
    z = np.array([20.0])                     # large outlier
    R = np.array([[1.0]])
    x_upd, P_upd = student_t_kalman_update(x0, P0, z, H, R, nu=4.0)
    print("x_upd:", x_upd)                   # robustly limited update