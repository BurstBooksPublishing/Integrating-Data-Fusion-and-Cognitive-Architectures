# executable Python3 example: numpy-only; simulates sensor updates
import numpy as np

# per-device calibration table (gain, noise_std)
calibrations = {'depth_cam_1': (1.02, 0.05), 'lidar_1': (0.99, 0.02)}

class OccupancyGrid:
    def __init__(self, shape, prior=0.0, decay=0.98):
        self.logodds = np.full(shape, prior)  # log-odds grid
        self.decay = decay

    def update_cell(self, idx, p_occ):
        # log-odds update with decay (eq. logodds)
        lz = np.log(p_occ/(1-p_occ))
        self.logodds[idx] = self.decay * self.logodds[idx] + lz

    def occupancy_prob(self):
        return 1.0 - 1.0/(1.0 + np.exp(self.logodds))

def classify_dynamic(vel, s_prev, alpha=0.8, gamma=4.0):
    # eq. dynprob: returns new dynamic score
    s = alpha*s_prev + (1-alpha)*(1.0 - np.exp(-gamma*abs(vel)))
    return s

def sensor_health_check(device, residuals, max_resid=0.2):
    # simple health metric: mean residual scaled by nominal noise
    gain, sigma = calibrations.get(device, (1.0, 0.1))
    score = 1.0 - min(1.0, np.mean(np.abs(residuals))/(max_resid*sigma))
    return score  # 1 healthy .. 0 unhealthy

# simulate
grid = OccupancyGrid((100,100))
dynamic_score = 0.0
# one timestep: detection with velocity and occupancy evidence
vel = 0.15  # m/s observed for object
dynamic_score = classify_dynamic(vel, dynamic_score)
# if dynamic_score low, treat as static evidence; else ephemeral
p_occ = 0.9 if dynamic_score < 0.4 else 0.6  # less trust if dynamic
grid.update_cell((50,50), p_occ)
# device residuals to check health
residuals = [0.01, 0.02, 0.03]  # example residuals
health = sensor_health_check('depth_cam_1', residuals)
if health < 0.5:
    print('Degrade stream: use fallback sensor')  # policy action
# output occupancy probability for debugging
print('occupancy at dock:', grid.occupancy_prob()[50,50])