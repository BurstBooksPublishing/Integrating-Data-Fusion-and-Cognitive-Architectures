import numpy as np
from datetime import datetime, timedelta

class EventTemplate:
    def __init__(self, roles, dur_bounds, sigma_t=2.0, lam=1.0):
        self.roles = roles               # list of role names
        self.dur_bounds = dur_bounds     # (min_s, max_s)
        self.sigma_t = sigma_t
        self.lam = lam

    def temporal_score(self, intervals):
        # intervals: dict role->(t0,t1) as datetimes
        t0 = max(i[0] for i in intervals.values())
        t1 = min(i[1] for i in intervals.values())
        if t1 < t0:                      # no overlap
            gap = (t0 - t1).total_seconds()
            return np.exp(- (gap**2) / (2*self.sigma_t**2))
        dur = (t1 - t0).total_seconds()
        if dur < self.dur_bounds[0] or dur > self.dur_bounds[1]:
            return 0.0
        return 1.0                        # ideal overlap and duration

    def evidence_logp(self, evidence_logps):
        # evidence_logps: dict role->log P(attributes|role)
        return sum(evidence_logps.values())

    def score(self, intervals, evidence_logps):
        logE = self.evidence_logp(evidence_logps)
        T = self.temporal_score(intervals)
        if T <= 0:
            return -np.inf
        return logE + self.lam * np.log(T)

# Example usage (toy datetimes and log-probs)
now = datetime.utcnow()
ints = {'leader':(now - timedelta(seconds=10), now + timedelta(seconds=50)),
        'f1':(now - timedelta(seconds=5), now + timedelta(seconds=45))}
e_logp = {'leader': -1.2, 'f1': -0.8}
tmpl = EventTemplate(['leader','f1'], (20,300), sigma_t=5.0, lam=1.0)
s = tmpl.score(ints, e_logp)
print("score", s)                    # promote if above threshold