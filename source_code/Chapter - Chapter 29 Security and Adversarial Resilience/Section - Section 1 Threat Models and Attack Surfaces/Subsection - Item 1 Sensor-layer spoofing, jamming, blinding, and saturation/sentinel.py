import numpy as np
from scipy.stats import chi2

# params (would be loaded from config/service)
SNR_THRESH = 6.0           # dB threshold for acceptable SNR
NIS_P = 0.99               # chi-square p-value for inconsistency
QUORUM = 0.6               # required corroboration fraction

def compute_snr(signal_power, noise_power, jammer_power=0.0):
    return 10*np.log10(signal_power / (noise_power + jammer_power))  # dB

def nis(innovation, S):
    # innovation vector and innovation covariance S
    return float(innovation.T @ np.linalg.inv(S) @ innovation)

def sensor_check(sensor):
    snr_db = compute_snr(sensor['S'], sensor['N'], sensor.get('J',0.0))
    nis_val = nis(sensor['v'], sensor['Scov'])
    nis_thresh = chi2.ppf(NIS_P, df=len(sensor['v']))
    ok = (snr_db >= SNR_THRESH) and (nis_val <= nis_thresh)
    return {'id': sensor['id'], 'snr': snr_db, 'nis': nis_val, 'ok': ok}

def quorum_check(sensor_results):
    weights = np.array([r.get('weight',1.0) for r in sensor_results])
    ok_flags = np.array([r['ok'] for r in sensor_results], dtype=float)
    score = (weights * ok_flags).sum() / weights.sum()
    return score >= QUORUM, score

# Example usage: evaluate sensors and raise anomaly if necessary
# (In deployment, hook results into policy manager and signed telemetry)