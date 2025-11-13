import time, math, heapq

LAG_SECONDS = 2.0            # fixed-lag smoothing window
LAMBDA = 0.5                 # staleness decay rate
PROMOTE_THRESH = 0.6         # promotion threshold

class FixedLagBuffer:
    def __init__(self, lag=LAG_SECONDS):
        self.lag = lag
        self.buffer = []      # min-heap by timestamp

    def add(self, track_msg):
        # track_msg: dict with 't_capture' and 'posterior'
        heapq.heappush(self.buffer, (track_msg['t_capture'], track_msg))

    def flush_ready(self, now=None):
        now = now or time.time()
        ready = []
        # pop items older than now - lag
        cutoff = now - self.lag
        while self.buffer and self.buffer[0][0] <= cutoff:
            ready.append(heapq.heappop(self.buffer)[1])
        return ready

def staleness_weight(delta_t, lam=LAMBDA):
    return math.exp(-lam * max(0.0, delta_t))

def evaluate_and_promote(track_msg, now=None):
    now = now or time.time()
    delta = now - track_msg['t_capture']
    score = track_msg['posterior'] * staleness_weight(delta)
    # return boolean decision and score for telemetry
    return (score > PROMOTE_THRESH, score)

# Example usage:
buf = FixedLagBuffer()
buf.add({'t_capture': time.time()-3.0, 'posterior': 0.8})  # older than lag
ready = buf.flush_ready()
for msg in ready:
    promote, score = evaluate_and_promote(msg)
    # brief comment: send to L2 if promote else demote/retain
    print('promote', promote, 'score', score)