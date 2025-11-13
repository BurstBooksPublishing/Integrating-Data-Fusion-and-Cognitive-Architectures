import numpy as np
from collections import deque
# simulate_stream() yields tuples (distance, hazard_score)
def simulate_stream():
    # demo generator: replace with real sensor/model
    rng = np.random.default_rng(0)
    for i in range(1,10000):
        d = 10.0 - 0.01*i + rng.normal(0,0.05)      # closing trend
        s = float(np.clip(1.0/(1.0+np.exp(d-5.0)),0,1)) # surrogate hazard score
        yield d, s

window = deque(maxlen=5)          # short temporal window for trend monitor
scores = []                       # history for Hoeffding
delta = 1e-3                      # desired false-violation prob
p_safe = 0.2                      # safe hazard probability threshold

def hoeffding_eps(n, delta):
    if n == 0: return 1.0
    return np.sqrt((np.log(2/delta)) / (2*n))

for t, (dist, score) in enumerate(simulate_stream(), start=1):
    window.append(dist)
    scores.append(score)
    # simple formal monitor: strictly decreasing distances over window
    trend_flag = (len(window) == window.maxlen) and all(
        window[i] < window[i-1] for i in range(1, len(window))
    )
    n = len(scores)
    hat_p = np.mean(scores)
    eps = hoeffding_eps(n, delta)
    upper_p = hat_p + eps                # high-confidence upper bound
    if trend_flag and upper_p > p_safe:
        print(f"t={t}: GUARD - trend+stat cert (upper_p={upper_p:.3f})")
        break  # in practice, trigger shield or safe maneuver