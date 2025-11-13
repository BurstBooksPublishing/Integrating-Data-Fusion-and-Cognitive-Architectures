import numpy as np
from sklearn.neighbors import KernelDensity

class OnlineClutterKDE:
    def __init__(self, bandwidth=0.5, window=2000):
        self.bandwidth = bandwidth
        self.window = window
        self.buffer = np.empty((0,2))  # 2-D example: range, bearing
        self.kde = None

    def add_samples(self, samples):
        # samples: (N,2) array of pseudo-clutter measurements
        self.buffer = np.vstack((self.buffer, samples))
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window:]  # sliding window

    def fit(self):
        if len(self.buffer) >= 10:
            self.kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
            self.kde.fit(self.buffer)

    def density(self, points):
        # return per-unit-area density estimate; small epsilon floor
        if self.kde is None:
            return np.full(len(points), 1e-6)
        logd = self.kde.score_samples(points)
        return np.maximum(np.exp(logd), 1e-9)

def association_score(p_D, likelihood, clutter_density, eps=1e-9):
    # score consistent with eq. (2)
    return (p_D * likelihood) / (clutter_density + eps)

# Example usage
kde_model = OnlineClutterKDE(bandwidth=0.3)
kde_model.add_samples(np.random.randn(500,2))  # pseudo-clutter buffer
kde_model.fit()
z = np.array([[0.1, -0.2]])  # incoming measurement
c_z = kde_model.density(z)[0]
score = association_score(p_D=0.9, likelihood=0.05, clutter_density=c_z)
print("clutter density", c_z, "assoc score", score)