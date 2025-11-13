import numpy as np
# simple linear KF; small buffer for past states
class BufferedKF:
    def __init__(self,F,Q,H,R,init_x,init_P,buffer_len=50):
        self.F,self.Q,self.H,self.R = F,Q,H,R
        self.buffer_len = buffer_len
        # circular buffer of (t, x, P)
        self.buf = []
        self.cur_t = 0.0
        self.x = init_x; self.P = init_P

    def predict_to(self,dt):
        # simple predict step for dt steps of constant F
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def step(self,dt):
        self.predict_to(dt); self.cur_t += dt
        # push state snapshot
        self.buf.append((self.cur_t, self.x.copy(), self.P.copy()))
        if len(self.buf) > self.buffer_len: self.buf.pop(0)

    def apply_measurement(self,t_meas,z):
        # find latest buffer entry at or before t_meas
        idx = max([i for i,(t,_,_) in enumerate(self.buf) if t<=t_meas], default=None)
        if idx is None:
            # too old or no history: drop or do approximate update at current time
            print("Measurement outside buffer; drop or approximate")
            return
        # extract snapshot, reprocess forward
        t0, x0, P0 = self.buf[idx]
        # update at t_meas
        S = self.H @ P0 @ self.H.T + self.R
        K = P0 @ self.H.T @ np.linalg.inv(S)
        x0_upd = x0 + K @ (z - self.H @ x0)
        P0_upd = (np.eye(len(self.x)) - K @ self.H) @ P0
        # replace and repropagate subsequent entries
        self.buf[idx] = (t0, x0_upd.copy(), P0_upd.copy())
        for j in range(idx+1, len(self.buf)):
            dt = self.buf[j][0] - self.buf[j-1][0]
            x_pred = self.F @ self.buf[j-1][1]
            P_pred = self.F @ self.buf[j-1][2] @ self.F.T + self.Q
            self.buf[j] = (self.buf[j][0], x_pred.copy(), P_pred.copy())
        # set current to last buffer state
        self.cur_t, self.x, self.P = self.buf[-1]
# --- usage omitted for brevity; integrate in runtime loop ---