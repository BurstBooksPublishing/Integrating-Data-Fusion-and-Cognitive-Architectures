import numpy as np
import pandas as pd

def detection_delays(gt_times, det_times, max_match=2.0):
    # gt_times, det_times: 1D numpy arrays (seconds)
    det_times = np.sort(det_times)
    delays = []
    for t0 in np.sort(gt_times):
        # pick first detection >= t0 within max_match window
        idx = np.searchsorted(det_times, t0)
        if idx < len(det_times) and det_times[idx] - t0 <= max_match:
            delays.append(det_times[idx] - t0)
        else:
            delays.append(np.nan)  # missed (censor)
    return np.array(delays)

def time_to_stability(times, estimates, ref, eps=0.5, hold=2.0):
    # times: monotonic timestamps; estimates/ref: arrays (N,dim)
    N = len(times)
    stable_times = np.full(N, np.nan)
    # precompute per-sample norms
    err = np.linalg.norm(estimates - ref, axis=1)
    for i in range(N):
        # find earliest j >= i such that err[j:j+k] <= eps for required hold
        # determine window length in samples
        # assume uniform sampling or compute dynamic window by time
        t_start = times[i]
        # find j where subsequent span >= hold
        j = i
        while j < N:
            t_j = times[j]
            # find index k s.t. times[k] >= t_j + hold
            k = np.searchsorted(times, t_j + hold, side='left')
            if k > j and np.all(err[j:k] <= eps):
                stable_times[i] = t_j
                break
            j += 1
    return stable_times  # per-sample earliest stable start (nan if none)

def robustness_to_gaps(times, estimates, ref, gaps, eps=0.5, hold=2.0, delta=5.0):
    # gaps: iterable of gap lengths to test
    results = {}
    for g in gaps:
        recoveries = []
        # simulate gap starting at random valid onset
        for onset in np.linspace(times[0], times[-1]-g-1e-6, num=20):
            # zero out estimates during [onset, onset+g)
            est2 = estimates.copy()
            mask = (times >= onset) & (times < onset+g)
            est2[mask] = est2[np.maximum(np.where(~mask)[0].min(),0)]  # hold-last-sample as fallback
            st = time_to_stability(times, est2, ref, eps=eps, hold=hold)
            # find first stable time after gap end
            idx_after = np.searchsorted(times, onset+g)
            stable = st[idx_after:]
            recovered = np.any(~np.isnan(stable) & (stable - (onset+g) <= delta))
            recoveries.append(recovered)
        results[g] = np.mean(recoveries)
    return results