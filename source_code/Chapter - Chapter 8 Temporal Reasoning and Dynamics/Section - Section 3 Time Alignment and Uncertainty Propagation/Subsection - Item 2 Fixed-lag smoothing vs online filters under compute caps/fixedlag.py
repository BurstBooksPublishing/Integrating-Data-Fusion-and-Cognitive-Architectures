import numpy as np
class StreamingFixedLag:
    def __init__(self, F, H, Q, R, P0, x0, max_lag, cost_per_lag, budget):
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x, self.P = x0.copy(), P0.copy()
        self.buffer = []                          # store tuples (x_k, P_k, F_k)
        self.max_lag = max_lag
        self.cost_per_lag = cost_per_lag
        self.budget = budget

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

    def step(self, z):
        self.predict()
        self.update(z)
        # push forward-pass result into buffer
        self.buffer.append((self.x.copy(), self.P.copy(), self.F.copy()))
        # compute allowable lag given budget
        allowed_lag = min(self.max_lag, int(self.budget // self.cost_per_lag))
        # if enough slack, run RTS backward pass over last allowed_lag steps
        if allowed_lag > 0 and len(self.buffer) > 1:
            self._rts_backward(allowed_lag)
            # trim buffer to keep only recent window
            self.buffer = self.buffer[-(allowed_lag+1):]

    def _rts_backward(self, L):
        # simple RTS on stored buffer entries
        for i in range(len(self.buffer)-2, len(self.buffer)-2-L, -1):
            xk, Pk, Fk = self.buffer[i]
            xkp1, Pkp1, _ = self.buffer[i+1]
            A = Pk @ Fk.T @ np.linalg.inv(Fk @ Pk @ Fk.T + self.Q)
            # update smoothed state in-place in buffer
            self.buffer[i] = (xk + A @ (xkp1 - Fk @ xk), Pk, Fk)