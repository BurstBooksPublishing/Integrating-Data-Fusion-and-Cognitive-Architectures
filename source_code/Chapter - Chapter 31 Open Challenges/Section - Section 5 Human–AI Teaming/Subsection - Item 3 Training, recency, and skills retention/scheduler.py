import time, math, requests
import numpy as np

# Configurable weights and thresholds
alpha, beta, gamma = 0.6, 0.3, 0.1
S_THRESHOLD = 0.7
TAU = 30*24*3600  # retention time constant (30 days)

def retention_since(ts, tau=TAU):
    # ts: last training epoch seconds
    return math.exp(-(time.time()-ts)/tau)

def get_model_drift():
    # placeholder: compute from validation and feature monitoring
    return float(np.loadtxt("drift_metric.txt"))  # 0..1

def mission_exposure():
    return 0.2  # normalized exposure score

def trigger_rehearsal(op_id):
    # call to scheduler/service that logs and schedules the drill
    payload = {"op": op_id, "reason": "recency+drift", "time": int(time.time())}
    requests.post("https://ops.example/api/schedule_rehearsal", json=payload)  # auth omitted

def evaluate_and_act(op_id, last_training_ts):
    R_H = retention_since(last_training_ts)
    D = get_model_drift()
    M = mission_exposure()
    S = alpha*(1-R_H) + beta*D + gamma*M
    if S >= S_THRESHOLD:
        trigger_rehearsal(op_id)  # schedule targeted drill

# Example loop for multiple operators
ops = {"alice": 1698000000, "bob": 1695400000}
for op, ts in ops.items():
    evaluate_and_act(op, ts)