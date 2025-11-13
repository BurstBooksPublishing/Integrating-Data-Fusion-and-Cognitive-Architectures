import asyncio, time
from collections import Counter, deque

class HotMaterializer:
    def __init__(self, compute_fn, max_items=500, refresh_sec=5):
        self.compute_fn = compute_fn               # heavy join/compute function
        self.store = {}                            # key -> (value, version, ts)
        self.freq = Counter()                      # access frequency
        self.hot_queue = deque()                   # hot keys ordered
        self.max_items = max_items
        self.refresh_sec = refresh_sec
        self.lock = asyncio.Lock()
        asyncio.create_task(self._periodic_refresh())

    async def get(self, key, write_token=None):
        # read-your-writes: prefer local version if write_token present
        async with self.lock:
            self.freq[key] += 1
            if key in self.store:
                val, ver, ts = self.store[key]
                return val                       # hot-path hit
        # cold-path compute (async) and optionally pin
        val, ver = await self.compute_fn(key)
        await self._maybe_pin(key, val, ver)
        return val

    async def _maybe_pin(self, key, val, ver):
        async with self.lock:
            if len(self.store) < self.max_items or self._is_heavy(key):
                self.store[key] = (val, ver, time.time())
                self._refresh_order(key)

    def _is_heavy(self, key):
        return self.freq[key] >= 5                # simple threshold

    def _refresh_order(self, key):
        if key in self.hot_queue:
            self.hot_queue.remove(key)
        self.hot_queue.appendleft(key)

    async def _periodic_refresh(self):
        while True:
            await asyncio.sleep(self.refresh_sec)
            async with self.lock:
                keys = list(self.hot_queue)[:self.max_items]
            # refresh outside lock to allow reads
            for k in keys:
                val, ver = await self.compute_fn(k)   # recompute to reduce staleness
                async with self.lock:
                    self.store[k] = (val, ver, time.time())