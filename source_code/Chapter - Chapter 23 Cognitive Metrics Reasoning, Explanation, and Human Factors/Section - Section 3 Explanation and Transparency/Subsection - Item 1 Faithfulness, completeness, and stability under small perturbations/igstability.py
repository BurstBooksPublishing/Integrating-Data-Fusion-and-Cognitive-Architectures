import numpy as np

def f(x):
    # simple fused scoring: weighted sum followed by sigmoid
    w = np.array([0.6, 0.3, 0.1])  # example fusion weights
    return 1/(1+np.exp(-w.dot(x)))

def integrated_gradients(x, x0=None, m=50):
    x0 = np.zeros_like(x) if x0 is None else x0
    alphas = np.linspace(0,1,m)
    grads = np.zeros_like(x)
    for a in alphas:
        xi = x0 + a*(x-x0)
        # analytic gradient of sigmoid(w·x) here
        s = f(xi); w = np.array([0.6,0.3,0.1])
        grad = s*(1-s)*w  # ∂f/∂x
        grads += grad
    avg_grad = grads / m
    return (x - x0) * avg_grad  # IG attributions

# sample input
x = np.array([0.9, 0.2, 0.1])
phi = integrated_gradients(x)          # attributions
residual = f(x) - f(np.zeros_like(x)) - phi.sum()  # completeness residual

# deletion test: zero-out top-k features and measure drop
order = np.argsort(-np.abs(phi))
x_del = x.copy(); x_del[order[0]] = 0
drop = f(x) - f(x_del)

# stability test: small Gaussian perturbations
eps = 1e-3
n_samples = 100
deltas = np.random.normal(scale=eps, size=(n_samples, x.size))
changes = []
for d in deltas:
    phi_d = integrated_gradients(x + d)
    changes.append(np.linalg.norm(phi_d - phi) / np.linalg.norm(d))
L_hat = np.max(changes)  # empirical Lipschitz estimate

print("phi", phi, "residual", residual, "drop", drop, "L_hat", L_hat)