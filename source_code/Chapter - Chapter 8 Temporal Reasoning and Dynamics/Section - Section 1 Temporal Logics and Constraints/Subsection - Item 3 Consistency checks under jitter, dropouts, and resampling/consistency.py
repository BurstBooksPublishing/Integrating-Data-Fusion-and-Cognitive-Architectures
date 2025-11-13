# Minimal example: run with Python 3, requires numpy.
import numpy as np
from collections import deque
chi2_thresh = 7.81  # 95% for 3 DOF, example

class FixedLagBuffer:
    def __init__(self, lag):
        self.lag = lag
        self.buf = deque()
    def ingest(self, t, value, sigma_t):
        self.buf.append((t, value, sigma_t))
    def emit_until(self, now):
        out = []
        while self.buf and self.buf[0][0] <= now - self.lag:
            out.append(self.buf.popleft())
        return out

def linear_resample(batch, t_query):
    # batch: list of (t, value, sigma_t) sorted by t
    ts = np.array([b[0] for b in batch])
    vals = np.array([b[1] for b in batch])
    if t_query <= ts[0]:
        return vals[0], batch[0][2]
    if t_query >= ts[-1]:
        return vals[-1], batch[-1][2]
    idx = np.searchsorted(ts, t_query)
    t0, v0, s0 = batch[idx-1]
    t1, v1, s1 = batch[idx]
    alpha = (t_query - t0) / (t1 - t0)
    v = (1-alpha)*v0 + alpha*v1
    # simple bound: interpolation uncertainty + timestamp var
    sigma = np.hypot((1-alpha)*s0 + alpha*s1, (alpha*(1-alpha)*(t1-t0)))
    return v, sigma

def nis_check(pred, meas, P_pred, R_meas):
    v = meas - pred
    S = P_pred + R_meas
    nis = float(v.T @ np.linalg.inv(S) @ v)
    return nis, nis <= chi2_thresh

# Simulated usage
buf = FixedLagBuffer(lag=0.1)
# ingest jittered samples
for t in [0.00, 0.05, 0.10, 0.18]:
    buf.ingest(t + np.random.normal(0,0.005), np.array([t*1.0]), 0.005)
emitted = buf.emit_until(now=0.25)
# resample to query time 0.12
v_resamp, sigma_t = linear_resample(emitted, 0.12)
# simple pred and cov
pred = np.array([0.12])
P_pred = np.array([[0.002]])
R_meas = np.array([[sigma_t**2 + 0.001]])
nis, ok = nis_check(pred, v_resamp, P_pred, R_meas)
print("NIS", nis, "consistent?", ok)
# interval precedence check
tA, dA = 0.10, 0.01
tB, dB, margin = 0.14, 0.01, 0.02
precedence_ok = (tA + dA + margin) <= (tB - dB)
print("Interval precedence ok?", precedence_ok)