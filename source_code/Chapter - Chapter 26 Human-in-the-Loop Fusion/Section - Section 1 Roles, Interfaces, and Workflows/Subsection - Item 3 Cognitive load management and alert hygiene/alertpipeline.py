import time
from collections import defaultdict, deque

TOKEN_RATE = 1.0        # tokens per second allowed to operator
BUCKET_SIZE = 5         # burst capacity
HYSTERESIS_SEC = 10.0   # dwell before emitting repeated alerts

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate, self.capacity = rate, capacity
        self.tokens = capacity
        self.last = time.time()
    def consume(self, n=1):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last)*self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

bucket = TokenBucket(TOKEN_RATE, BUCKET_SIZE)
last_emit = defaultdict(lambda: 0.0)
cluster_buffer = defaultdict(list)

def score_and_cluster(event):
    # event: dict with 'track_id','time','score','evidence'
    tid = event['track_id']
    cluster_buffer[tid].append(event)
    # simple clustering window: emit when score max exceeds threshold
    max_score = max(e['score'] for e in cluster_buffer[tid])
    if max_score < 0.7: return None
    if time.time() - last_emit[tid] < HYSTERESIS_SEC: return None
    # produce meta-alert with aggregated rationale
    evidence = sorted(cluster_buffer[tid], key=lambda e: e['score'], reverse=True)[:3]
    meta = {'track_id': tid, 'score': max_score, 'rationale': [e['evidence'] for e in evidence]}
    cluster_buffer[tid].clear()
    return meta

def emit_meta_alert(meta):
    if bucket.consume():
        last_emit[meta['track_id']] = time.time()
        # send to operator UI; placeholder print
        print("ALERT", meta['track_id'], meta['score'], meta['rationale'])
    else:
        # log and store for deferred review
        print("RATE-LIMITED", meta['track_id'], meta['score'])

# Example run: ingest stream
# for evt in incoming_stream: ...