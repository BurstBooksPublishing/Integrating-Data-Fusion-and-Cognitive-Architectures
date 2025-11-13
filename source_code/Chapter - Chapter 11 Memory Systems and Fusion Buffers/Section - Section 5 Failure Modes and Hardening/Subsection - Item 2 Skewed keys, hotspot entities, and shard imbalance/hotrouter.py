import hashlib, time, threading
from collections import Counter, defaultdict

# Simple sliding-window counter (not producion-grade CMS) -- demo only.
class SlidingCounter:
    def __init__(self, window_sec=5, step=1):
        self.window = window_sec
        self.step = step
        self.buckets = [Counter() for _ in range(window_sec//step)]
        self.idx = 0
        self.lock = threading.Lock()
    def tick(self):
        with self.lock:
            self.idx = (self.idx + 1) % len(self.buckets)
            self.buckets[self.idx].clear()
    def add(self, key, n=1):
        with self.lock:
            self.buckets[self.idx][key] += n
    def estimate(self, key):
        with self.lock:
            return sum(b.get(key,0) for b in self.buckets)

# Rendezvous hashing for stable placement
def rendezvous_hash(key, nodes):
    best, best_score = None, None
    for n in nodes:
        h = int(hashlib.md5((str(n)+key).encode()).hexdigest(),16)
        if best_score is None or h > best_score:
            best, best_score = n, h
    return best

# Demo pipeline
nodes = ['shardA','shardB','shardC']
hot_handlers = {}            # dedicated handlers for hot keys
counter = SlidingCounter(window_sec=10, step=1)

def route_update(key, payload):
    counter.add(key)
    rate = counter.estimate(key) / 10.0   # per-second approx
    if rate > 5.0:                       # hotspot threshold
        # create or reuse dedicated hot handler
        handler = hot_handlers.setdefault(key, f"hot-{key}")
        # send to hot handler (placeholder)
        print(f"ROUTE-> {handler}: {key} @{rate:.1f}/s")
    else:
        target = rendezvous_hash(key, nodes)
        print(f"ROUTE-> {target}: {key} @{rate:.1f}/s")

# Example ingestion
for i in range(60):
    route_update('entity-42', {})      # hot stream
    route_update(f'entity-{i%10}', {}) # many cold streams
    if i%1==0: counter.tick()
    time.sleep(0.05)