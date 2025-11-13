import numpy as np
from scipy.stats import wasserstein_distance
import json, time

# Parameters
ref_window = 500            # reference window size
curr_window = 200           # current window size
threshold = 0.1             # alarm threshold (tuned per feature)
rng = np.random.default_rng(0)

def simulate_feature(t):
    # sensor bias drifts after t>2000
    base = np.sin(0.01*t)
    bias = 0.0 if t<2000 else 0.5*(1+0.5*np.sin(0.002*t))
    return base + bias + 0.1*rng.normal()

# streaming buffers
ref_buf = [simulate_feature(i) for i in range(ref_window)]
curr_buf = []

def audit_record(ts, metric, details):
    rec = {"ts": ts, "metric": metric, "details": details}
    print("AUDIT:", json.dumps(rec))       # replace with durable store

# main loop
for t in range(ref_window, 4000):
    x = simulate_feature(t)
    curr_buf.append(x)
    if len(curr_buf) >= curr_window:
        w = wasserstein_distance(ref_buf, curr_buf)
        if w > threshold:
            # alarm: create evidence bundle
            details = {"wasserstein": w, "ref_mean": np.mean(ref_buf),
                       "curr_mean": np.mean(curr_buf)}
            audit_record(time.time(), "feature_drift", details)
            # mitigation action: shift cognitive policy to safe mode (placeholder)
            # e.g., set \lstinline|policy_mode| to "conservative" and flag retraining
        # slide window
        ref_buf = ref_buf[int(curr_window/2):] + curr_buf
        curr_buf = []