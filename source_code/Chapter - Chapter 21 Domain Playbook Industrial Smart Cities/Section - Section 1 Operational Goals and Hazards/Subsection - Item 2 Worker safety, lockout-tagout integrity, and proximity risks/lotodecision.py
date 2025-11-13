#!/usr/bin/env python3
import math, time, json, hashlib

# Gaussian CDF
def cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))

def fuse_distance(meas):
    # meas: list of (d, sigma)
    prec = sum(1/(s*s) for _,s in meas)
    d_f = sum(d/(s*s) for d,s in meas)/prec
    sigma_f = math.sqrt(1/prec)
    return d_f, sigma_f

def loto_failure_prob(plc_state, tamper_flag):
    # simple Bayesian heuristic; tune with field data
    base = 0.01 if plc_state=="locked" else 0.2
    if tamper_flag: base = min(1.0, base + 0.3)
    return base

def decision_loop():
    # simulate inputs; replace with real streams in integration
    lidar = (1.2, 0.1)        # (distance m, sigma m)
    camera = (1.5, 0.2)
    plc_state = "partial"     # "locked","partial","unlocked"
    tamper = False

    d_f, s_f = fuse_distance([lidar, camera])
    d_th = 1.5                # guard threshold (m)
    p_prox = cdf((d_th - d_f)/s_f)  # probability human inside threshold

    p_loto = loto_failure_prob(plc_state, tamper)
    R = 1 - (1-p_prox)*(1-p_loto)

    action = "allow"
    if R > 0.05: action = "inhibit"    # policy threshold
    elif R > 0.02: action = "alert"

    # produce audit trace
    trace = {
        "ts": time.time(),
        "fused_distance": d_f,
        "fused_sigma": s_f,
        "p_prox": p_prox,
        "p_loto_fail": p_loto,
        "risk_R": R,
        "action": action
    }
    trace['hash'] = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    print(json.dumps(trace))  # emit to telemetry/ledger

if __name__ == "__main__":
    decision_loop()