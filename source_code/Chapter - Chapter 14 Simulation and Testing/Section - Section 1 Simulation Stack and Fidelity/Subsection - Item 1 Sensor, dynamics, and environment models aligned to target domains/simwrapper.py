import numpy as np

class SimpleSim:
    def __init__(self, dyn_params, sensor_params):
        self.f = dyn_params['f']            # state-transition function
        self.h = sensor_params['h']         # observation function
        self.Q = dyn_params['Q']            # process noise cov
        self.R = sensor_params['R']         # measurement noise cov

    def step(self, x, u):
        w = np.random.multivariate_normal(np.zeros(len(self.Q)), self.Q)
        x_next = self.f(x, u) + w
        v = np.random.multivariate_normal(np.zeros(len(self.R)), self.R)
        z = self.h(x_next) + v
        return x_next, z

def calibrate(sim, dataset, lr=1e-3, epochs=100):
    # simple gradient-free calibration adjusting R by matching variances
    for _ in range(epochs):
        errs = []
        for x, u, z_real in dataset:
            x_sim, z_sim = sim.step(x, u)
            errs.append((z_real - z_sim))
        errs = np.stack(errs)
        # update R toward empirical covariance (diagonal for simplicity)
        target_var = np.var(errs, axis=0)
        sim.R = 0.9*sim.R + 0.1*np.diag(target_var)  # exponential blend
    return sim