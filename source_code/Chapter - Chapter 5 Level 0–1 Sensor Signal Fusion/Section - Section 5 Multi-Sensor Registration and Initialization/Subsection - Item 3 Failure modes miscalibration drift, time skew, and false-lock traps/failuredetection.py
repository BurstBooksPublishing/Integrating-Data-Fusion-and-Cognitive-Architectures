import numpy as np
from scipy.signal import correlate
from scipy.linalg import inv

def nis(innovation, S):  # innovation vector and covariance
    return float(innovation.T @ inv(S) @ innovation)

def estimate_time_offset(sig_a_times, sig_a_vals, sig_b_times, sig_b_vals, max_lag_s=0.1, fs=100):
    # resample on common grid for cross-correlation (simple linear interp)
    t0 = max(sig_a_times[0], sig_b_times[0])
    t1 = min(sig_a_times[-1], sig_b_times[-1])
    t = np.linspace(t0, t1, int((t1-t0)*fs))
    a = np.interp(t, sig_a_times, sig_a_vals)
    b = np.interp(t, sig_b_times, sig_b_vals)
    corr = correlate(a - a.mean(), b - b.mean(), mode='full')
    lags = np.arange(-len(t)+1, len(t)) / fs
    # restrict to plausible lag window
    mask = np.abs(lags) <= max_lag_s
    best_lag = lags[mask][np.argmax(corr[mask])]
    return best_lag

# Example usage: per-update diagnostics in a fusion node
# innovation, S come from a Kalman update; sensor buffers provide time-series.
innovation = np.array([0.2, -0.05])  # example
S = np.array([[0.05, 0.0],[0.0, 0.02]])
current_nis = nis(innovation, S)
if current_nis > 7.8:  # chi2 threshold for 2-dim at 95%
    print("NIS high: possible miscalibration or outlier")

# time-offset estimation between camera reprojection error and lidar range residual
cam_t, cam_err = np.load('cam_err.npy').T  # times, scalar reproj error
lid_t, lid_err = np.load('lid_err.npy').T
tau = estimate_time_offset(cam_t, cam_err, lid_t, lid_err)
if abs(tau) > 0.01:
    print(f"Estimated clock skew {tau*1000:.1f} ms — flagging time-sync service")