import numpy as np
from scipy.signal import correlate
# load arrays: imu_ts (s), imu_wz (rad/s); cam_ts (s), vo_yawrate (rad/s)
imu_ts, imu_wz = np.load('imu.npy')     # [N,2] columns: ts, wz
cam_ts, vo_wz  = np.load('vo.npy')      # [M,2] columns: ts, wz

# resample both to common grid for cross-correlation
t_min, t_max = max(imu_ts[0], cam_ts[0]), min(imu_ts[-1], cam_ts[-1])
fs = 200.0                               # target sampling frequency (Hz)
t_grid = np.arange(t_min, t_max, 1.0/fs)
imu_s  = np.interp(t_grid, imu_ts, imu_wz)
cam_s  = np.interp(t_grid, cam_ts,  vo_wz)

# zero-mean and window to reduce edge effects
imu_s -= imu_s.mean(); cam_s -= cam_s.mean()
corr = correlate(imu_s, cam_s, mode='full')
lags = np.arange(-len(t_grid)+1, len(t_grid)) / fs
lag_best = lags[np.argmax(corr)]
print(f"Estimated constant offset (imu - cam): {lag_best:.4f} s")