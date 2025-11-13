import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta

class SepsisFusion:
    def __init__(self, prior=0.01, window_minutes=60, dwell_seconds=600):
        self.prior_odds = prior/(1-prior)
        self.window = timedelta(minutes=window_minutes)
        self.dwell = timedelta(seconds=dwell_seconds)
        self.buffers = defaultdict(deque)   # per-stream (vitals/labs/notes)
        self.last_alert = None
        self.provenance = []                # list of evidence tuples

    def _lr_from_p(self, p, prevalence=0.01):
        # convert probability p to likelihood ratio given prevalence
        prior = prevalence
        prior_odds = prior/(1-prior)
        odds = p/(1-p) if 0=1 else 1e-6)
        return odds/prior_odds

    def ingest(self, stream, timestamp, score, meta):
        # score: per-stream probability; meta: provenance dict
        self.buffers[stream].append((timestamp, score, meta))
        self.provenance.append((timestamp, stream, meta))
        self._expire_buffers(timestamp)

    def _expire_buffers(self, now):
        cutoff = now - self.window
        for q in self.buffers.values():
            while q and q[0][0] < cutoff:
                q.popleft()

    def compute_posterior(self, now):
        # aggregate latest per-stream probability (example: max recent)
        lrs = []
        for stream, q in self.buffers.items():
            if not q: continue
            # pick most recent score as representative
            _, p, _ = q[-1]
            lrs.append(self._lr_from_p(p))
        if not lrs:
            return 0.0
        post_odds = self.prior_odds * np.prod(lrs)
        p = post_odds / (1 + post_odds)
        return p

    def check_alert(self, now, threshold=0.5):
        p = self.compute_posterior(now)
        if p >= threshold:
            if self.last_alert and now - self.last_alert < self.dwell:
                return False, p  # hysteresis: suppress repeated alerts
            self.last_alert = now
            return True, p
        return False, p

# Example usage: ingesting synthetic streams
sf = SepsisFusion()
now = datetime.utcnow()
sf.ingest('vitals', now - timedelta(minutes=5), 0.35, {'src':'bedside_mon','hr':120})
sf.ingest('labs', now - timedelta(minutes=30), 0.7, {'src':'lab','lactate':3.1})
sf.ingest('notes', now - timedelta(minutes=2), 0.6, {'src':'nurse_note','text_id':42})
alert, prob = sf.check_alert(now)
print('alert', alert, 'p', prob)