import numpy as np
from scipy.stats import chi2

def nis_cusum_localize(nu_seq, S_seq, m, H=50, k=0.5, h=10.0):
    # nu_seq: (T, m) innovations; S_seq: (T, m, m) covariances
    T = nu_seq.shape[0]
    nis = np.empty(T)
    for t in range(T):
        nis[t] = nu_seq[t].T @ np.linalg.inv(S_seq[t]) @ nu_seq[t]  # per-step NIS
    # sliding cumulative statistic (eq. \eqref{eq:cum_nis})
    cum = np.convolve(nis, np.ones(H, dtype=float), mode='full')[:T]
    # chi2 threshold per window (alpha small)
    alpha = 1e-3
    chi_thr = chi2.ppf(1-alpha, df=m*H)
    # CUSUM on deviation from expected m (to detect drift)
    s = 0.0
    cusum = np.zeros(T)
    for t in range(T):
        w = nis[t] - m  # per-step deviation
        s = max(0.0, s + w - k)
        cusum[t] = s
    # localize segments where cusum exceeds h or windowed cum exceeds chi_thr
    segs = []
    in_seg = False
    start = 0
    for t in range(T):
        alarm = (cum[t] > chi_thr) or (cusum[t] > h)
        if alarm and not in_seg:
            in_seg = True; start = t
        if not alarm and in_seg:
            in_seg = False; segs.append((start, t-1))
    if in_seg: segs.append((start, T-1))
    return {'nis': nis, 'cum': cum, 'cusum': cusum, 'segments': segs}

# Example usage: compute with real residuals, then inspect segments for root cause analysis.