import numpy as np

def ca_cfar(signal, num_train=24, num_guard=4, pfa=1e-6):
    # signal: 1D numpy array (power)
    N = num_train
    # precompute threshold multiplier from eq. (1)
    alpha = N * (pfa**(-1.0/N) - 1.0)
    L = len(signal)
    detections = []
    half_train = N // 2
    step = 1
    for i in range(half_train + num_guard, L - (half_train + num_guard)):
        # training region indices (excluding guard cells)
        start1 = i - num_guard - half_train
        end1 = i - num_guard
        start2 = i + num_guard + 1
        end2 = i + num_guard + 1 + half_train
        noise_cells = np.concatenate((signal[start1:end1], signal[start2:end2]))
        noise_est = np.mean(noise_cells)  # CA estimate
        threshold = alpha * noise_est
        if signal[i] > threshold:
            snr_est = signal[i] / (noise_est + 1e-12)
            detections.append({
                'index': i,
                'power': float(signal[i]),
                'noise_est': float(noise_est),
                'threshold': float(threshold),
                'snr_est': float(snr_est)
            })
    return detections

# Example usage: profile with injected target
if __name__ == '__main__':
    np.random.seed(0)
    bg = np.random.exponential(scale=1.0, size=1024)  # synthetic clutter
    bg[500] += 50.0  # injected target
    dets = ca_cfar(bg)
    print(dets[:3])  # brief output