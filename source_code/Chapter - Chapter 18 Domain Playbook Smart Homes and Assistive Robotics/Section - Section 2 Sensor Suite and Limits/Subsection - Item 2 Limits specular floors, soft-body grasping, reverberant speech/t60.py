import numpy as np
from scipy.signal import fftconvolve

def estimate_t60(h, fs):
    # compute energy decay curve (EDC)
    e = h**2
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10*np.log10(edc/np.max(edc)+1e-12)
    # linear fit between -5 dB and -35 dB for decay slope
    lo, hi = -5, -35
    idx = np.where((edc_db<=lo)&(edc_db>=hi))[0]
    if len(idx)<2:
        return None
    t = np.arange(len(edc_db))/fs
    # least squares fit
    a, b = np.polyfit(t[idx], edc_db[idx], 1)
    # T60 from slope: slope (dB/s) => T60 = -60 / slope
    t60 = -60.0 / a
    return float(t60)

# example usage: h = measured impulse response array, fs = sampling rate
# t60 = estimate_t60(h, fs)  # None if insufficient decay region