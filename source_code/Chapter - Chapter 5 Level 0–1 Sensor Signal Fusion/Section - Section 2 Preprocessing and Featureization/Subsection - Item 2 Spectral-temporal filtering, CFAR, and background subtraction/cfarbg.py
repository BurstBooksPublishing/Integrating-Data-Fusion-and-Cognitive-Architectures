import numpy as np
from scipy.ndimage import median_filter

# Temporal median background model for a stack of grayscale frames
def bg_subtract_median(frames, kernel_size=5, threshold_scale=2.5):
    # frames: (T,H,W) uint8 or float array
    B = np.median(frames, axis=0)                    # background median
    var = np.var(frames, axis=0) + 1e-6              # local variance estimate
    diff = np.abs(frames[-1].astype(float) - B)      # latest frame residual
    tau = threshold_scale * np.sqrt(var)             # adaptive threshold map
    mask = diff > tau                                 # foreground mask
    return B.astype(frames.dtype), mask.astype(np.uint8)

# CA-CFAR for 1D power series
def ca_cfar(power, num_ref=24, num_guard=4, pfa=1e-6):
    N = 2 * num_ref
    # analytic alpha for exponential noise model (eq. cf)
    alpha = N * (pfa**(-1.0/N) - 1.0)
    L = len(power)
    out = np.zeros(L, dtype=np.uint8)
    half_ref = num_ref + num_guard
    for i in range(L):
        i_start = max(0, i - half_ref)
        i_end   = min(L, i + half_ref + 1)
        # exclude guard cells and CUT
        ref_cells = np.concatenate([
            power[i_start: max(0, i - num_guard)],
            power[min(L, i + num_guard + 1): i_end]
        ])
        if ref_cells.size < 1:
            continue
        noise_est = np.mean(ref_cells)
        threshold = alpha * noise_est
        out[i] = 1 if power[i] > threshold else 0
    return out

# Demo usage (synthetic)
if __name__ == "__main__":
    # synth video stack: background + transient object
    H,W = 64,64
    T = 7
    frames = np.random.normal(10,2,(T,H,W)).astype(np.float32)     # scene noise
    frames[-1,20:30,25:35] += 40                                    # transient
    B, mask = bg_subtract_median(frames, kernel_size=5)             # bg-subtract
    # synth 1D radar profile with target
    rng = np.linspace(0,1,256)
    power = np.random.exponential(scale=1.0, size=rng.size)
    power[120:123] += 15.0                                          # target bump
    detections = ca_cfar(power, num_ref=10, num_guard=2, pfa=1e-4)
    # outputs: B (background), mask (foreground), detections (1D)