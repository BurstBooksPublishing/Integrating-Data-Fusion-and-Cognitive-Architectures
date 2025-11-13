import numpy as np
# simple 3-state linear model: HR, SpO2, HR_trend
dt = 1.0
F = np.array([[1,0,dt],[0,1,0],[0,0,1]])      # constant-SpO2, simple trend
H_hr = np.array([[1,0,0]])                   # heart-rate sensor
H_spo2 = np.array([[0,1,0]])                 # spo2 sensor
Q = np.diag([0.1,0.05,0.01])                 # process noise
R_hr = np.array([[4.0]])                     # HR sensor var (bpm^2)
R_spo2 = np.array([[0.5]])                   # SpO2 sensor var

x = np.array([70.0, 98.0, 0.0])              # initial state
P = np.eye(3)

def predict(x,P,dt):
    x = F.dot(x)
    P = F.dot(P).dot(F.T) + Q
    return x,P

def update(x,P,y,H,R,nis_threshold=7.8):
    # compute innovation and NIS
    nu = y - H.dot(x)
    S = H.dot(P).dot(H.T) + R
    nis = float(nu.T.dot(np.linalg.inv(S)).dot(nu))
    if nis > nis_threshold:
        # gating: skip update and inflate covariance (soft fail)
        P = P + 2.0*np.eye(P.shape[0])
        return x,P,nis,False
    K = P.dot(H.T).dot(np.linalg.inv(S))
    x = x + K.dot(nu)
    P = (np.eye(P.shape[0]) - K.dot(H)).dot(P)
    return x,P,nis,True

# example loop: asynchronous arrivals
measurements = [("hr",71.0,1.0),("spo2",97.5,2.0),("hr",120.0,3.0)] # (type,val,t)
t_prev = 0.0
for typ,val,t in measurements:
    # propagate to event time
    steps = int((t - t_prev)/dt)
    for _ in range(steps):
        x,P = predict(x,P,dt)
    t_prev = t
    if typ=="hr":
        x,P,nis,ok = update(x,P,np.array([val]),H_hr,R_hr)
    else:
        x,P,nis,ok = update(x,P,np.array([val]),H_spo2,R_spo2)
    print(f"time {t:.1f}s type {typ} nis {nis:.2f} update_ok {ok} state {x}")