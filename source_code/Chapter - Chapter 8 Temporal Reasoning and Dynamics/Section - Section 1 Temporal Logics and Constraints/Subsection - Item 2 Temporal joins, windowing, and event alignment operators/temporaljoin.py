import heapq, time
from collections import defaultdict, deque

class EventTimeJoiner:
    def __init__(self, tolerance, allowed_lateness):
        self.tol = tolerance                      # max time diff to match
        self.lateness = allowed_lateness         # allowed late arrival
        self.buf_left = defaultdict(deque)       # keyed buffers
        self.buf_right = defaultdict(deque)
        self.tmax = float("-inf")                # max event-time seen

    def _update_tmax(self, t):
        if t > self.tmax: self.tmax = t

    def ingest_left(self, key, t, payload):
        self._update_tmax(t)
        self.buf_left[key].append((t,payload))
        return self._emit_ready(key)

    def ingest_right(self, key, t, payload):
        self._update_tmax(t)
        self.buf_right[key].append((t,payload))
        return self._emit_ready(key)

    def watermark(self):
        return self.tmax - self.lateness

    def _emit_ready(self, key):
        out = []
        wm = self.watermark()
        # process events older than watermark in either buffer
        while self.buf_left[key] and self.buf_left[key][0][0] <= wm:
            tl, pl = self.buf_left[key].popleft()
            # find matches in right buffer within tolerance
            matches = [ (tr,pr) for tr,pr in self.buf_right[key]
                        if abs(tr - tl) <= self.tol ]
            if matches:
                for tr,pr in matches:
                    out.append(("match", key, tl, pl, tr, pr))
            else:
                out.append(("left_unmatched", key, tl, pl))
        while self.buf_right[key] and self.buf_right[key][0][0] <= wm:
            tr, pr = self.buf_right[key].popleft()
            matches = [ (tl,pl) for tl,pl in self.buf_left[key]
                        if abs(tr - tl) <= self.tol ]
            if matches:
                for tl,pl in matches:
                    out.append(("match", key, tl, pl, tr, pr))
            else:
                out.append(("right_unmatched", key, tr, pr))
        return out

# Simple usage demo
if __name__ == "__main__":
    j = EventTimeJoiner(tolerance=0.5, allowed_lateness=1.0)
    streamL = [("A",1.0,{"x":1}), ("A",2.5,{"x":2})]
    streamR = [("A",1.2,{"sym":"enter"}), ("A",4.0,{"sym":"exit"})]
    for k,t,p in streamL:
        for out in j.ingest_left(k,t,p):
            print(out)
    for k,t,p in streamR:
        for out in j.ingest_right(k,t,p):
            print(out)
    # advance watermark by ingesting a far-future tick
    j._update_tmax(10.0)
    for key in ["A"]:
        for out in j._emit_ready(key):
            print(out)