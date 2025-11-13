import numpy as np

class MultimodalFusion:
    def __init__(self):
        # state: [x, y] position; simple 2D constant-state KF
        self.x = np.zeros((2,1))
        self.P = np.eye(2)*1.0

    def kf_update(self, H, z, R):
        # Kalman measurement update
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(2) - K @ H) @ self.P

    def process_rgbd(self, pos_xy, cov_xy):
        z = np.reshape(pos_xy,(2,1))
        H = np.eye(2)
        R = cov_xy
        self.kf_update(H, z, R)  # dense local pose with moderate trust

    def process_ble(self, beacon_pos, rssi, rssi_sigma):
        # convert RSSI to range and fuse as circle-constraint approximated by radial measurement
        # linearize: measure bearing from current state to beacon
        dx = self.x[0,0]-beacon_pos[0]; dy = self.x[1,0]-beacon_pos[1]
        r_hat = np.hypot(dx,dy) + 1e-6
        # predicted range from RSSI (assume external mapping function)
        range_meas = self.rssi_to_range(rssi)
        # H maps state to radial distance; approximate Jacobian H = [dx/r, dy/r]
        H = np.array([[dx/r_hat, dy/r_hat]])
        z = np.array([[range_meas]])
        R = np.array([[rssi_sigma**2]])
        self.kf_update(H, z, R)

    def rssi_to_range(self, rssi, rssi0=-50, n=2.0, d0=1.0):
        # invert path loss: coarse conversion; handle saturation
        return d0 * 10**((rssi0 - rssi) / (10*n))

    def process_microphone_doa(self, doa_angle, doa_sigma, distance_guess=1.0):
        # convert DOA to a bearing line measurement at guessed distance
        z = np.array([[self.x[0,0] + distance_guess*np.cos(doa_angle)],
                      [self.x[1,0] + distance_guess*np.sin(doa_angle)]])
        H = np.eye(2)
        R = np.eye(2)*(doa_sigma*distance_guess)**2
        self.kf_update(H, z, R)

    def process_tactile(self, contact_force, force_thresh=0.5):
        # tactile confirmation promotes object-held hypothesis; set high trust near contact
        if contact_force > force_thresh:
            # collapse position uncertainty (example action)
            self.P *= 0.1  # tighten belief about object under grasp
# end class