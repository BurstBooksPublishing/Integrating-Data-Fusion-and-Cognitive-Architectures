import numpy as np
from math import erf
# helper: standard normal CDF
def Phi(x): return 0.5*(1+erf(x/np.sqrt(2)))
# propagate ego and pedestrian predicted along-path separation (m)
def predict_separation(d, v, dt): return d - v*dt
# approximate collision probability per Eq.(1)
def collision_prob(d, v, dt, sigma_along, sigma_time):
    denom = np.sqrt(sigma_along**2 + (v*sigma_time)**2)
    arg = (d - v*dt) / denom
    return 1 - Phi(arg)
# decision rule with cognitive veto/hysteresis
def decide_action(d, v, dt, sigma_along, sigma_time,
                  p_thresh=0.01, veto_confidence=0.6, hysteresis=0.005):
    p_coll = collision_prob(d, v, dt, sigma_along, sigma_time)
    # cognitive veto example: if provenance low, escalate stop
    provenance_conf = 0.7  # fused metadata (0..1) from L0/L2 checks
    # hysteresis to avoid flip-flop
    effective_thresh = p_thresh - hysteresis if provenance_conf < veto_confidence else p_thresh
    if p_coll > effective_thresh or provenance_conf < veto_confidence:
        return "STOP", p_coll, provenance_conf  # auditable decision
    return "PROCEED", p_coll, provenance_conf
# example call
action, p, prov = decide_action(d=12.0, v=8.0, dt=0.5,
                                sigma_along=0.9, sigma_time=0.8)
print(action, p, prov)  # stop or proceed with diagnostics