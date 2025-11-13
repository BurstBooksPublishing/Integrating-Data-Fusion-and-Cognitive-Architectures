import numpy as np
from scipy.signal import periodogram
# synthetic data generation
np.random.seed(0)
T = 2000
t = np.arange(T)
season = 2.0*np.sin(2*np.pi*t/144)  # daily-like cycle
drift = 0.0005*t + np.cumsum(0.0001*np.random.randn(T))  # slow bias
noise = 0.2*np.random.randn(T)
emi = np.zeros(T)
emi[500:520] += 5.0*np.sin(2*np.pi*50*(t[500:520]/1000.))  # transient EMI
# redaction mask (policy): remove segments
redacted = np.zeros(T, dtype=bool)
redacted[1200:1250] = True
y = season + drift + noise + emi
y[redacted] = np.nan  # enforce redaction
# detection: short-window residual mean and spectral burst test
win = 64
res_mean = np.convolve(np.nan_to_num(y), np.ones(win)/win, mode='same')
f, Pxx = periodogram(np.nan_to_num(y), fs=1.0)
emi_flag = np.zeros(T, bool)
# simple energy threshold on short windows
for i in range(0, T-win, win//4):
    seg = y[i:i+win]
    if np.isnan(seg).all(): continue
    p = np.nanmean(seg**2)
    if p > 0.5: emi_flag[i:i+win] = True
drift_flag = np.abs(res_mean - np.nanmedian(res_mean)) > 0.3  # heuristic
# output diagnostics
print("EMI segments:", np.where(emi_flag)[0][:10])
print("Drift flags (count):", np.count_nonzero(drift_flag))
print("Redaction fraction:", redacted.mean())
# simple mitigation: interpolate redacted with seasonal+trend fit
A = np.vstack([np.sin(2*np.pi*t/144), np.cos(2*np.pi*t/144), t, np.ones(T)]).T
mask = ~np.isnan(y)
coeffs,_,_,_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
y_imputed = y.copy()
y_imputed[~mask] = A[~mask] @ coeffs  # impute with uncertainty note