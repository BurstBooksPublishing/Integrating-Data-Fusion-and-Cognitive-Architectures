import numpy as np
from scipy.optimize import minimize

# synthetic example: load arrays shape (N, m)
y_real = np.load('y_real.npy')   # measured observations (golden traces)
y_sim  = np.load('y_sim.npy')    # simulator outputs for same scenarios

m = y_real.shape[1]

def loss(params):
    b = params[:m]                 # per-dimension bias
    log_alpha = params[-1]         # optimize log-scale for positivity
    alpha = np.exp(log_alpha)
    residuals = y_real - (y_sim + b)   # vector residuals
    # assume diagonal sim-covariance; learn scalar multiplier
    R = alpha * np.var(residuals, axis=0) + 1e-8
    # Mahalanobis-like robust loss (sum of squared normalized residuals)
    return np.sum((residuals**2) / R)

# init: zero bias, log_alpha=0
init = np.concatenate([np.zeros(m), [0.0]])
res = minimize(loss, init, method='L-BFGS-B')
bias_hat = res.x[:m]
alpha_hat = float(np.exp(res.x[-1]))
# persist calibration artifacts
np.save('bias_hat.npy', bias_hat)
print('bias', bias_hat, 'alpha', alpha_hat)