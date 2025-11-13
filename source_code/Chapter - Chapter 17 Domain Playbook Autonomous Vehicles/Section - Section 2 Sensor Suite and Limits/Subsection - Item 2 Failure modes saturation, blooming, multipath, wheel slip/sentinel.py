import numpy as np
from scipy.stats import chi2

# thresholds (tuned in validation)
SAT_FRAC_THRESH = 0.01        # saturated pixels fraction
BLOOM_AC_THRESHOLD = 0.5     # spatial autocorr increase
MULTIPATH_SNR_DROP = 6.0     # dB drop indicative
NIS_ALPHA = 0.01             # false-alarm rate

def check_camera(frame):
    sat_frac = np.mean(frame >= 255)            # saturated pixels (8-bit)
    ac = spatial_autocorr(frame)                # brief helper
    flags = {}
    flags['saturation'] = sat_frac > SAT_FRAC_THRESH
    flags['blooming'] = (ac - ac_baseline()) > BLOOM_AC_THRESHOLD
    return flags

def check_gnss(obs, predicted_ranges, snr_history):
    residuals = obs.pseudoranges - predicted_ranges
    snr_drop = np.max(snr_history[-4:]) - np.mean(snr_history[-20:-4])
    mp_flag = (np.abs(residuals).mean() > 5.0) or (snr_drop > MULTIPATH_SNR_DROP)
    return {'multipath': mp_flag, 'residual_mean': residuals.mean()}

def check_wheel_slip(wheel_vel, imu_vel, S_cov):
    v_res = wheel_vel - imu_vel
    nis = float(v_res.T @ np.linalg.inv(S_cov) @ v_res)
    df = len(v_res)
    th = chi2.ppf(1.0 - NIS_ALPHA, df)  # threshold
    slip = nis > th
    return {'wheel_slip': slip, 'nis': nis, 'threshold': th}

def update_fusion_weights(weights, flags):
    # simple downweighting rule; production uses smoother adaptation
    for s, f in flags.items():
        if f:
            weights[s] *= 0.1  # aggressive downweight
    return weights

# Example run (simplified)
weights = {'camera':1.0,'lidar':1.0,'gnss':1.0,'odo':1.0}
flags_cam = check_camera(frame)                       # frame from sensor
flags_gnss = check_gnss(gnss_obs, predicted_ranges, snr_hist)
flags_odo = check_wheel_slip(odo_vel, imu_vel, S_cov)
# aggregate flags
combined = {**flags_cam, **flags_gnss, **flags_odo}
weights = update_fusion_weights(weights, {k:v for k,v in combined.items() if isinstance(v,bool)})
emit_diagnostic(combined, weights)                    # L0 diagnostic message