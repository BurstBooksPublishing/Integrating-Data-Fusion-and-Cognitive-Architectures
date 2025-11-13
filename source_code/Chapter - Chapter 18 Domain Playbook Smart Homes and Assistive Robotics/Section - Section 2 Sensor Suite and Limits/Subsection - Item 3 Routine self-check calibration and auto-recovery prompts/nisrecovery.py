#!/usr/bin/env python3
# simple NIS monitor and auto-recovery; requires numpy
import time, logging
import numpy as np
from scipy.stats import chi2

logging.basicConfig(level=logging.INFO)

CHI_ALPHA = 0.99
WINDOW = 5              # sliding window length
RECOVERY_MAX = 2        # automated retries
PROMPT_COOLDOWN = 60.0  # seconds

class SensorMonitor:
    def __init__(self, dim):
        self.dim = dim
        self.thresh = chi2.ppf(CHI_ALPHA, df=dim)
        self.nis_window = []
        self.retries = 0
        self.last_prompt = 0.0

    def compute_nis(self, v, S):
        # v: innovation vector, S: innovation covariance (dim x dim)
        try:
            nis = float(v.T @ np.linalg.inv(S) @ v)
        except np.linalg.LinAlgError:
            nis = float('inf')
        return nis

    def handle_measurement(self, v, S):
        nis = self.compute_nis(v, S)
        self.nis_window.append(nis)
        if len(self.nis_window) > WINDOW:
            self.nis_window.pop(0)
        avg_nis = np.mean(self.nis_window)
        logging.info("NIS=%.2f avg=%.2f thresh=%.2f", nis, avg_nis, self.thresh)

        if avg_nis <= self.thresh:
            self.retries = 0
            return "OK"

        # attempt automated recovery
        if self.retries < RECOVERY_MAX:
            self.retries += 1
            self.automated_recovery()
            return "RECOVERY_ATTEMPT"
        # otherwise escalate to user prompt with cooldown
        now = time.time()
        if now - self.last_prompt > PROMPT_COOLDOWN:
            self.last_prompt = now
            self.user_prompt()
            return "USER_PROMPT"
        return "DEGRADED"

    def automated_recovery(self):
        logging.warning("Attempting automated recalibration (retry %d)", self.retries)
        # Placeholder: insert real recalibration logic here
        time.sleep(0.5)  # simulate work

    def user_prompt(self):
        logging.error("Issuing user prompt: manual calibration or consent required")
        # Replace with cognitive layer call, notification, or voice UI invoke

def simulate():
    mon = SensorMonitor(dim=3)
    rng = np.random.default_rng(0)
    for t in range(60):
        # simulate nominal then drifting innovations
        if t < 20:
            v = rng.normal(0, 0.1, size=(3,1))
            S = np.eye(3)*0.01
        else:
            v = rng.normal(0.5, 0.5, size=(3,1))  # drift
            S = np.eye(3)*0.05
        state = mon.handle_measurement(v, S)
        time.sleep(0.1)

if __name__ == "__main__":
    simulate()