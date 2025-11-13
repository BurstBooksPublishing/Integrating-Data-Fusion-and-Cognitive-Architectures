import numpy as np
class FixedLagKF:
    def __init__(self, F, Q, H_map, R_map, L):
        self.F, self.Q = F, Q                       # dynamics and process noise
        self.H_map, self.R_map = H_map, R_map       # per-sensor H, R functions
        self.L = L                                  # lag in steps
        self.buf = []                               # buffer of dicts: {'t', 'x', 'P'}
    def predict_step(self, x,P):
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x,P
    def update(self, x,P, z, H, R):
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - H @ x)
        P = (np.eye(len(x)) - K @ H) @ P
        return x,P
    def push_time(self, t, x_prior, P_prior):
        # append new time index prediction into buffer
        self.buf.append({'t':t, 'x':x_prior.copy(), 'P':P_prior.copy()})
        # trim older than lag
        while len(self.buf) > self.L + 1:
            self.buf.pop(0)
    def insert_measurement(self, t_meas, sensor, z):
        # find index in buffer with matching time (assumes exact or nearest)
        idx = next((i for i,v in enumerate(self.buf) if v['t']==t_meas), None)
        if idx is None:
            return False  # OOSM too old or future; handle separately
        H = self.H_map[sensor](self.buf[idx]['x'])     # allow H depend on state
        R = self.R_map[sensor]()
        # apply update at idx
        x,P = self.update(self.buf[idx]['x'], self.buf[idx]['P'], z, H, R)
        self.buf[idx]['x'], self.buf[idx]['P'] = x,P
        # replay forward to maintain consistency inside window
        for j in range(idx+1, len(self.buf)):
            x,P = self.predict_step(self.buf[j-1]['x'], self.buf[j-1]['P'])
            # if there was a sensor update at j, keep it after predict
            self.buf[j]['x'], self.buf[j]['P'] = x,P
        return True
    def get_smoothed(self):
        # run a backward RTS-smoother on buffer (linear version)
        n = len(self.buf)
        xs = [v['x'].copy() for v in self.buf]
        Ps = [v['P'].copy() for v in self.buf]
        # compute P_pred for each forward link
        P_pred = [None]*(n-1)
        for k in range(n-1):
            P_pred[k] = self.F @ Ps[k] @ self.F.T + self.Q
        for k in range(n-2, -1, -1):
            A = Ps[k] @ self.F.T @ np.linalg.inv(P_pred[k])
            xs[k] = xs[k] + A @ (xs[k+1]- self.F @ xs[k])
            Ps[k] = Ps[k] + A @ (Ps[k+1]-P_pred[k]) @ A.T
        return [{'t':v['t'],'x':xs[i],'P':Ps[i]} for i,v in enumerate(self.buf)]
# usage requires concrete F,Q,H_map,R_map and time-step driven pushes/measurements