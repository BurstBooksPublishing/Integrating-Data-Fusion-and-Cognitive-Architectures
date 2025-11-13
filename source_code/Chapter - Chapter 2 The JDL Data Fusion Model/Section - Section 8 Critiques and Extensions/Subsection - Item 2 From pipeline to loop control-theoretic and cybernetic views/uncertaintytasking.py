import numpy as np, time, logging
# Simple 1D KF for illustration
A = np.array([[1.0]])
H = np.array([[1.0]])
Q = np.array([[1e-3]])
R_base = 0.1
x = np.zeros((1,1)); P = np.eye(1)
rate = 10.0  # Hz initial
last_change = time.time()
HYSTERESIS_S = 2.0  # seconds minimum dwell

def predict():
    global x,P
    x = A @ x
    P = A @ P @ A.T + Q

def update(z, R):
    global x,P
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x[:] = x + K @ (z - H @ x)
    P[:] = (np.eye(1) - K @ H) @ P

def schedule_rate(traceP):
    # rates candidates in Hz and their costs (simple)
    candidates = [2.0, 5.0, 10.0, 20.0]
    costs = {2.0:1,5.0:2,10.0:4,20.0:8}
    # pick smallest cost meeting uncertainty target
    if traceP > 0.5: return 20.0
    if traceP > 0.2: return 10.0
    if traceP > 0.05: return 5.0
    return 2.0

while True:
    t0 = time.time()
    predict()
    # synthetic measurement with variable R by rate (higher rate -> lower R)
    R = R_base / np.sqrt(rate)
    z = (H @ x)[0,0] + np.sqrt(R)*np.random.randn()
    update(np.array([[z]]), np.array([[R]]))
    traceP = float(np.trace(P))
    # decision with hysteresis
    new_rate = schedule_rate(traceP)
    if new_rate != rate and (time.time()-last_change) > HYSTERESIS_S:
        logging.info("Rate change %s -> %s; traceP=%.3f", rate, new_rate, traceP)
        rate = new_rate; last_change = time.time()
    # log belief and diagnostics for assurance
    logging.debug("t=%.3f x=%.3f traceP=%.6f rate=%.1f", time.time(), x[0,0], traceP, rate)
    time.sleep(1.0/max(rate,1.0))  # loop timing regulated by rate