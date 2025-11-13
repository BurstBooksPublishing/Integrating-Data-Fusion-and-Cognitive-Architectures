import numpy as np

def globally(signal_bool, window):
    # signal_bool: boolean array, window: (a,b) indices relative to current time
    a, b = window
    # compute sliding-window global: True iff all values in [t+a,t+b] are True
    n = len(signal_bool)
    out = np.zeros(n, dtype=bool)
    for t in range(n):
        lo = max(0, t + a)
        hi = min(n, t + b + 1)
        out[t] = signal_bool[lo:hi].all()
    return out

def eventually(signal_bool, window):
    # True iff exists True in [t+a,t+b]
    a, b = window
    n = len(signal_bool)
    out = np.zeros(n, dtype=bool)
    for t in range(n):
        lo = max(0, t + a)
        hi = min(n, t + b + 1)
        out[t] = signal_bool[lo:hi].any()
    return out

# Example usage: distance signal and separation rule
dist = np.linspace(10,0,101)  # distance decreasing over 10s at 0.1s samples
r = 2.0
safe = dist > r
# check globally over next 50 samples (~5s)
violations = ~globally(safe, (0,50))
# emit alert times
alert_times = np.where(violations)[0]  # indices where safety violated
print("First violation index:", alert_times[0] if alert_times.size else "none")