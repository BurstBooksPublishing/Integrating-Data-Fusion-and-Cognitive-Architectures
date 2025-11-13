#!/usr/bin/env python3
import time, yaml, logging
import numpy as np
from collections import deque

# config load (paths, thresholds)
cfg = yaml.safe_load(open("calib_monitor_cfg.yaml"))  # contains thresholds, windows
TH_NIS = cfg["nis_threshold"]; WIN = cfg["window_size"]
SHOCK_THRESH = cfg["shock_threshold"]

# rolling buffers
nis_buf = deque(maxlen=WIN); temp_buf = deque(maxlen=WIN)
shock_buf = deque(maxlen=WIN)

def compute_nis(residual, S):
    return float(residual.T @ np.linalg.inv(S) @ residual)

def detect_shock(imu_accel):
    # simple energy metric; real systems use bandpass and event detection
    return np.linalg.norm(imu_accel) > SHOCK_THRESH

def trigger_self_calibration():
    logging.info("Triggering self-calibration job") 
    # call constrained calibration routine (not expanded here)
    # must record inputs, outputs, and signatures for traceability

while True:
    # read telemetry: residual, S, temperature, imu (placeholders)
    residual = np.load("last_residual.npy")           # e.g., reprojection residual
    S = np.load("last_innov_cov.npy")                 # innovation covariance
    temp = float(np.loadtxt("temp_sensor.txt"))
    imu = np.load("imu_accel.txt")                    # 3-vector

    nis = compute_nis(residual, S)
    nis_buf.append(nis); temp_buf.append(temp)
    shock = detect_shock(imu); shock_buf.append(shock)

    # statistical tests
    if np.mean(nis_buf) > TH_NIS:
        logging.warning("Elevated NIS mean: %f", np.mean(nis_buf))
        trigger_self_calibration()

    # shock-driven inflation
    if shock and np.mean(nis_buf) > 0.5*TH_NIS:
        logging.warning("Shock plus elevated NIS; forcing conservative recal")
        trigger_self_calibration()

    time.sleep(cfg["poll_interval"])  # keep loop responsive