import math
import numpy as np

def gaussian_likelihood(resid, sigma):
    return math.exp(-0.5*(resid/sigma)**2) / (math.sqrt(2*math.pi)*sigma)

def spoof_posterior(a_resid, s_resid, p_d=0.01):
    # Likelihoods under honest model: small residuals
    p_a_auth = gaussian_likelihood(a_resid, sigma=50.0)   # meters
    p_s_auth = gaussian_likelihood(s_resid, sigma=25.0)
    p_auth = p_a_auth * p_s_auth
    # Spoof model: AIS likely large residual; SAR consistent
    p_a_spoof = gaussian_likelihood(a_resid, sigma=500.0)
    p_s_spoof = gaussian_likelihood(s_resid, sigma=25.0)
    p_spoof = p_a_spoof * p_s_spoof
    # Bayes update (Equation for posterior)
    num = p_d * p_spoof
    den = (1.0 - p_d) * p_auth + num
    return num / den if den>0 else 0.0

# Example inputs: AIS vs SAR residuals (meters)
ais_resid = 120.0   # AIS reported position vs fused estimate
sar_resid = 10.0    # SAR sensor residual
post = spoof_posterior(ais_resid, sar_resid, p_d=0.02)
print(f"Spoof posterior: {post:.3f}")
# Decision rule
if post>0.5:
    print("Action: Hold fire, retask assets, escalate to human.")
elif post>0.1:
    print("Action: Lower automation confidence; request additional revisit.")
else:
    print("Action: Normal track propagation.")