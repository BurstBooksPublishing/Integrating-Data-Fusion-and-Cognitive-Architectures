import json, numpy as np
from scipy.stats import chi2

# config (would come from schema registry)
schema = {"dim": 2, "rate_hz": 10, "max_latency_ms": 50}
sensor_id = "range_sensor_A"         # identifier

def adapter(raw_frame):
    # map raw to canonical vector (x,y)
    return np.array([raw_frame["r"]*np.cos(raw_frame["theta"]),
                     raw_frame["r"]*np.sin(raw_frame["theta"])])

def synth_scene(n=200):
    # synthetic ground-truth track (straight line) + noise
    t = np.linspace(0,1,n)
    truth = np.stack([0.5 + 1.0*t, 2.0 + 0.2*t], axis=1)
    noise = np.random.normal(scale=0.1, size=truth.shape)
    return truth + noise

def compute_innovations(synth, pred_func, cov_est):
    innovations = []
    for z in synth:
        mu = pred_func()                # dummy predictor; replace with track predictor
        innov = z - mu
        innovations.append(innov)
    innovations = np.array(innovations)
    cov = np.cov(innovations, rowvar=False)
    return innovations, cov

# warm-up run
synth = synth_scene(300)
pred_func = lambda: np.array([0.5, 2.0])   # stationary prior for probe
innovations, cov = compute_innovations(synth, pred_func, None)

# compute Mahalanobis distances and acceptance rate
inv_cov = np.linalg.pinv(cov + 1e-6*np.eye(cov.shape[0]))
d2 = np.sum(innovations @ inv_cov * innovations, axis=1)
alpha = 0.99
threshold = chi2.ppf(alpha, df=synth.shape[1])
accept_rate = np.mean(d2 < threshold)

print(f"sensor={sensor_id} accept_rate={accept_rate:.3f} threshold={threshold:.2f}")

# decision: must exceed a pre-defined accept fraction for promotion
if accept_rate > 0.92:
    print("PROMOTE: sensor meets statistical consistency")
else:
    print("HOLD: route to shadow mode and investigate")