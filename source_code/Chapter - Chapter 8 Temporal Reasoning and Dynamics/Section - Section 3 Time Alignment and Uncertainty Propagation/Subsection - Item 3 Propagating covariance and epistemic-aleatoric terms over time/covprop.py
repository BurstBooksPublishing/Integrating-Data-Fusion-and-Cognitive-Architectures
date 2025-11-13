import numpy as np

def propagate_covariances(x, P_a, P_e, F, Q, G, Sigma_theta):
    # Predict aleatoric covariance (process noise added)
    P_a_pred = F @ P_a @ F.T + Q
    # Predict epistemic covariance (parameter-mapped uncertainty)
    P_e_pred = F @ P_e @ F.T + G @ Sigma_theta @ G.T
    # Total predicted covariance
    P_pred = P_a_pred + P_e_pred
    return P_pred, P_a_pred, P_e_pred

# Example numeric run
if __name__ == "__main__":
    n = 4
    x = np.zeros(n)
    P_a = np.eye(n)*0.1            # aleatoric posterior
    P_e = np.eye(n)*0.5            # epistemic posterior
    F = np.eye(n) + 0.01*np.random.randn(n,n)  # linearized dynamics
    Q = np.eye(n)*0.05
    G = 0.1*np.ones((n,2))        # map from 2 uncertain params
    Sigma_theta = np.diag([0.2,0.05])
    P, P_a_p, P_e_p = propagate_covariances(x,P_a,P_e,F,Q,G,Sigma_theta)
    print("trace total, alea, epi:", np.trace(P), np.trace(P_a_p), np.trace(P_e_p))