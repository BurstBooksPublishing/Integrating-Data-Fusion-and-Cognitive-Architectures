from dataclasses import dataclass, asdict
from typing import List, Dict
import json
import numpy as np
import time

@dataclass
class Track:
    track_id: str
    t: float
    x: List[float]         # state vector
    P: List[List[float]]   # covariance matrix
    source: str
    life_cycle: str
    provenance: Dict

    def predict(self, F: np.ndarray, Q: np.ndarray, u: np.ndarray = None):
        x_vec = np.asarray(self.x)
        P_mat = np.asarray(self.P)
        u = np.zeros(F.shape[0]) if u is None else u
        x_pred = F @ x_vec + u
        P_pred = F @ P_mat @ F.T + Q
        self.x = x_pred.tolist()
        self.P = P_pred.tolist()
        self.t = time.time()
        return self

    def validate(self):
        n = len(self.x)
        P = np.asarray(self.P)
        assert P.shape == (n, n), "covariance shape mismatch"
        assert np.allclose(P, P.T, atol=1e-6), "covariance not symmetric"
        return True

    def to_json(self):
        self.validate()
        return json.dumps(asdict(self))

# Example usage
if __name__ == "__main__":
    F = np.eye(4)  # simple identity motion model
    Q = 1e-2 * np.eye(4)
    tr = Track("T123", time.time(), [0,0,0,0], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
               "radar_front", "confirmed", {"sensor_ts": 163})
    tr.predict(F, Q)                     # apply predict step
    print(tr.to_json())                  # serialize for transport