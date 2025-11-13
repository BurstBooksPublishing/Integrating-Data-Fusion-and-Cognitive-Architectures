import numpy as np

class SensorHealthManager:
    def __init__(self, T0=25.0):
        self.T0 = T0  # reference temp Celsius

    def thermal_variance(self, sigma02, alpha, T):
        # Eq. (thermal_variance) with floor at sigma0^2
        factor = max(0.0, 1.0 + alpha * (T - self.T0))
        return sigma02 * factor

    def availability(self, reliabilities):
        # Eq. (parallel_reliability)
        return 1.0 - np.prod([1.0 - r for r in reliabilities])

    def fused_measurement(self, zs, covs, reliabilities, temps, alphas):
        # scale covariances by temperature sensitivity
        adj_covs = [self.thermal_variance(c, a, T)
                    for c, a, T in zip(covs, alphas, temps)]
        # convert covs to weights for simple precision-weighted average
        precisions = [1.0 / c if c > 0 else 0.0 for c in adj_covs]
        weights = [p * r for p, r in zip(precisions, reliabilities)]
        total_w = sum(weights)
        if total_w == 0:
            raise RuntimeError("No usable sensor input")
        z_fused = sum(w * z for w, z in zip(weights, zs)) / total_w
        cov_fused = 1.0 / total_w
        return z_fused, cov_fused

# Example usage (scalar measurements)
if __name__ == "__main__":
    mgr = SensorHealthManager()
    zs = [10.2, 9.8, 10.5]              # measurements
    covs = [0.5, 0.7, 0.4]             # sigma^2 at T0
    reliabilities = [0.98, 0.95, 0.90]
    temps = [30.0, 40.0, 22.0]         # current temps
    alphas = [0.02, 0.05, 0.01]        # sensitivity per sensor
    z, cov = mgr.fused_measurement(zs, covs, reliabilities, temps, alphas)
    print("fused z, cov:", z, cov)
    print("availability:", mgr.availability(reliabilities))