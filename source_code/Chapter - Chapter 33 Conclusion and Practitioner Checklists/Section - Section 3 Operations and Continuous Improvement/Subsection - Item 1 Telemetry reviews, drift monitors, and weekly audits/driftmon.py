#!/usr/bin/env python3
import numpy as np
import logging
from scipy.stats import chisquare

logging.basicConfig(level=logging.INFO)

def psi(expected, actual, eps=1e-8):
    # population stability index between histograms
    expected = np.asarray(expected, dtype=float) + eps
    actual = np.asarray(actual, dtype=float) + eps
    expected /= expected.sum()
    actual /= actual.sum()
    return np.sum((expected - actual) * np.log(expected / actual))

def nis_test(innovations, cov_trace, dof, alpha=0.01):
    # normalized innovation squared aggregate vs chi-square
    nis = np.sum((innovations**2) / (cov_trace + 1e-9))
    pval = 1.0 - chisquare([nis], f_exp=[dof])[1]  # wrapper to get p-like value
    return nis, pval

# Example usage with synthetic data
if __name__ == "__main__":
    # baseline and recent feature histograms (same bins)
    baseline = np.array([500, 300, 200])
    recent = np.array([400, 350, 250])
    score = psi(baseline, recent)
    logging.info("PSI=%.4f", score)
    if score > 0.2:
        logging.warning("PSI breach: investigate feature drift")  # create ticket

    # innovations from a Kalman filter residual stream
    innovations = np.random.normal(0, 1.0, size=50)
    cov_trace = np.full_like(innovations, 1.0)  # expected innovation variance
    nis, p = nis_test(innovations, cov_trace, dof=50)
    logging.info("NIS=%.2f p_approx=%.3f", nis, p)
    if nis > 70:  # example threshold for 50 dof
        logging.error("Filter inconsistency: check process/measurement noise tuning")