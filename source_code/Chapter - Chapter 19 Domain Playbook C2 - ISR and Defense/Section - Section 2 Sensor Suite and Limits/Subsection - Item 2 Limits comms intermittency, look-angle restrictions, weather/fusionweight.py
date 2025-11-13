import math, random, time

# tunable thresholds from calibration
SNR_THRESH = 10.0  # dB
k_weather, alpha = 0.03, 0.8  # band constants (example)
p0, k_angle = 0.95, 2.0       # detection model params

def p_det(theta_rad):          # look-angle detectability
    return max(0.0, p0 * math.cos(theta_rad)**k_angle)

def attenuation(precip_mm_per_hr, path_km):  # simple rain model
    return k_weather * (precip_mm_per_hr**alpha) * path_km

def fusion_weight(link_up_prob, theta_rad, snr0_db, precip, path_km):
    A = attenuation(precip, path_km)
    snr_eff = snr0_db - A                    # weather penalty
    snr_score = 1.0 / (1.0 + math.exp(-(snr_eff - SNR_THRESH)/2.0))
    return link_up_prob * p_det(theta_rad) * snr_score

# example runtime update (would be in a node loop)
if __name__ == "__main__":
    # simulated inputs
    link_up_prob = 0.8
    theta = math.radians(30)
    snr0 = 15.0
    precip = 10.0
    path_km = 5.0
    w = fusion_weight(link_up_prob, theta, snr0, precip, path_km)
    print(f"fusion weight = {w:.3f}")  # used in association and tasking