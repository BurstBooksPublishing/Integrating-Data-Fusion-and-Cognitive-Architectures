import numpy as np, time
# Simple candidate seed + N-scan confirmation manager
class CandidateSeed:
    def __init__(self, state, cov, first_t, window_N, req_k):
        self.state = state            # state vector
        self.cov = cov                # covariance
        self.first_t = first_t
        self.N = window_N
        self.k = req_k
        self.hits = []                # timestamps of supporting detections
    def update_with_detection(self, meas, t):
        # short KF predict-update would go here; simplified as replace
        self.state = meas
        self.hits.append(t)
        # keep only last N timestamps
        if len(self.hits) > self.N:
            self.hits = self.hits[-self.N:]
    def is_confirmed(self, now):
        # require k hits within sliding N-scan window (time-based)
        return len(self.hits) >= self.k

class SeedManager:
    def __init__(self, gate_r, window_N=3, req_k=2, seed_timeout=1.0):
        self.gate_r = gate_r
        self.N = window_N; self.k = req_k
        self.timeout = seed_timeout
        self.candidates = []
    def process_frame(self, detections, tstamp):
        # detections: list of (pos, score)
        for pos, score in detections:
            matched = False
            for c in self.candidates:
                # simple Euclidean gating
                if np.linalg.norm(c.state - pos) <= self.gate_r:
                    c.update_with_detection(pos, tstamp); matched = True; break
            if not matched and score > 0.5:
                # create new seed from high-confidence detection
                self.candidates.append(CandidateSeed(np.array(pos), np.eye(2)*1.0,
                                                     tstamp, self.N, self.k))
        # confirm/promote and prune expired
        confirmed = []
        now = tstamp
        remaining = []
        for c in self.candidates:
            if c.is_confirmed(now):
                confirmed.append(c)  # hand off to tracker
            elif now - c.first_t > self.timeout:
                pass  # expire candidate, log reason
            else:
                remaining.append(c)
        self.candidates = remaining
        return confirmed
# Example usage:
# mgr = SeedManager(gate_r=2.0); tracks = mgr.process_frame(dets, time.time())