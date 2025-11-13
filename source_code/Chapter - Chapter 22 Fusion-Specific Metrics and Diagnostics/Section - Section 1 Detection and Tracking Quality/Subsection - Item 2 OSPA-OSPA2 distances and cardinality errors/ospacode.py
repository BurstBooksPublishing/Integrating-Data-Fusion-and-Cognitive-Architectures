import numpy as np
from scipy.optimize import linear_sum_assignment

def ospa_distance(X, Y, p=2, c=10.0):
    # X, Y: arrays shape (n, dim), (m, dim)
    n, m = len(X), len(Y)
    if n == 0 and m == 0: return 0.0
    N = max(n, m)
    # cost matrix with ghost padding
    C = np.full((N, N), c**p)
    if n: C[:n, :m] = np.minimum(c, np.linalg.norm(
        X[:, None, :] - Y[None, :, :], axis=2))**p
    row_ind, col_ind = linear_sum_assignment(C)
    total = C[row_ind, col_ind].sum()
    return (total / N)**(1.0 / p)

# Simple multi-frame proxy: average per-frame OSPA
def frame_avg_ospa(frames_truth, frames_est, p=2, c=10.0):
    # frames_*: list of arrays per frame
    osps = [ospa_distance(gt, est, p=p, c=c) 
            for gt, est in zip(frames_truth, frames_est)]
    return float(np.mean(osps))

# Small runnable demo
if __name__ == "__main__":
    gt = np.array([[0.0,0.0],[5.0,0.0]])             # two truth points
    est = np.array([[0.5,0.1]])                     # one estimate
    print("OSPA single-frame:", ospa_distance(gt, est, p=2, c=5.0))
    # multi-frame example (3 frames)
    frames_truth = [gt, gt+0.1, gt+0.2]
    frames_est   = [est, est+0.1, np.empty((0,2))]  # last frame missing
    print("Frame-avg OSPA proxy:", frame_avg_ospa(frames_truth, frames_est))