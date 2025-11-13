import time, random, asyncio
from collections import OrderedDict

class BoundedFusionBuffer:
    def __init__(self, max_items=10000, replay_rate=100):  # capacity and rate cap
        self.store = OrderedDict()              # key -> (value, ts)
        self.max_items = max_items
        self.replay_rate = replay_rate          # items/sec global cap
        self.last_replay = 0.0
    def ingest(self, key, value):
        ts = time.time()
        self.store.pop(key, None)               # remove old reference for LRU
        self.store[key] = (value, ts)
        # LRU eviction
        while len(self.store) > self.max_items:
            self.store.popitem(last=False)     # evict oldest
    def checkpoint(self):
        # simple snapshot; real system writes to durable log
        return [(k,v,ts) for k,(v,ts) in self.store.items()]
    async def replay(self, consumer, jitter_ms=100):
        # rate-limited, jittered replay with idempotent consumer call
        items = list(self.store.items())
        for idx,(key,(value,ts)) in enumerate(items):
            # global token bucket simple approximation
            now = time.time()
            elapsed = max(now - self.last_replay, 1e-6)
            allowed = self.replay_rate * elapsed
            if allowed < 1.0:
                # sleep with jitter to avoid synchronized spikes
                await asyncio.sleep((1.0/ self.replay_rate) + random.random()*jitter_ms/1000.0)
            # idempotent send: consumer should ignore duplicates using key+ts
            await consumer(key, value, ts)
            self.last_replay = time.time()
# example consumer
async def consumer(key, value, ts):
    # idempotent processing stub
    await asyncio.sleep(0)  # non-blocking placeholder
# quick run
async def main():
    b = BoundedFusionBuffer(max_items=1000, replay_rate=200)
    for i in range(1500):
        b.ingest(f"track{i}", {"state":i})
    await b.replay(consumer)
asyncio.run(main())