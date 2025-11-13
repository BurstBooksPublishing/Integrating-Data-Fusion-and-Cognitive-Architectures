import bisect, time
from collections import deque

class FixedLagStore:
    def __init__(self, lag_seconds):
        self.lag = lag_seconds
        self.times = []               # sorted timestamps
        self.items = []               # parallel payloads
    def insert(self, timestamp, payload):
        # insert OOSM into sorted arrays
        i = bisect.bisect_left(self.times, timestamp)
        if i < len(self.times) and self.times[i] == timestamp:
            self.items[i] = payload   # idempotent replace
        else:
            self.times.insert(i, timestamp)
            self.items.insert(i, payload)
        self._evict_old()
    def query_range(self, t0, t1):
        i0 = bisect.bisect_left(self.times, t0)
        i1 = bisect.bisect_right(self.times, t1)
        return list(zip(self.times[i0:i1], self.items[i0:i1]))
    def _evict_old(self):
        cutoff = time.time() - self.lag
        idx = bisect.bisect_right(self.times, cutoff)
        if idx:
            # evict while preserving minimal audit token elsewhere
            del self.times[:idx]; del self.items[:idx]

# Usage: ingest sensors, allow smoother to reconcile within lag.