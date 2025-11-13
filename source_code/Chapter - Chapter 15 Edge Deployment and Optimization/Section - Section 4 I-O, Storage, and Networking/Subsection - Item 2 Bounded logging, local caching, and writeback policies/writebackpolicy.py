import threading, time, json, queue, os, math, requests
from collections import deque

class BoundedLogger:
    def __init__(self, capacity=2048, batch_size=64, disk_spill_dir='/tmp/spill'):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque()            # in-memory ring buffer
        self.lock = threading.Lock()
        self.spill_dir = disk_spill_dir
        os.makedirs(self.spill_dir, exist_ok=True)
        self.stop = False
        self.write_thread = threading.Thread(target=self._writeback_loop, daemon=True)
        self.write_thread.start()

    def log(self, record, priority=0):
        # priority: 0 (low) .. 10 (highest) - keep high priority on disk until ack
        item = {'ts': time.time(), 'priority': priority, 'payload': record}
        with self.lock:
            if len(self.buffer) >= self.capacity:
                # evict policy: drop lowest priority oldest
                idx = next((i for i,v in enumerate(self.buffer) if v['priority']<=priority), None)
                if idx is None:
                    # all buffered items higher priority; spill to disk
                    self._spill_to_disk(item)
                    return
                else:
                    del self.buffer[idx]
            self.buffer.append(item)

    def _spill_to_disk(self, item):
        fname = os.path.join(self.spill_dir, f"{int(time.time()*1000)}.json")
        with open(fname, 'w') as f:
            json.dump(item, f)

    def _collect_batch(self):
        with self.lock:
            batch = []
            while self.buffer and len(batch) < self.batch_size:
                batch.append(self.buffer.popleft())
            return batch

    def _send_batch(self, batch):
        # simulate network send; replace with real endpoint and error handling
        try:
            payload = json.dumps(batch)
            resp = requests.post("https://example.com/edge/write", data=payload, timeout=2)
            resp.raise_for_status()
            return True
        except Exception:
            return False

    def _writeback_loop(self):
        backoff_base = 0.5
        fails = 0
        while not self.stop:
            batch = self._collect_batch()
            if not batch:
                # try to flush any spilled files
                self._flush_spill()
                time.sleep(0.1)
                continue
            ok = self._send_batch(batch)
            if ok:
                fails = 0
            else:
                # requeue batch head-of-line for retry; exponential backoff
                with self.lock:
                    for item in reversed(batch):
                        self.buffer.appendleft(item)
                fails += 1
                sleep = min(30, backoff_base * (2 ** (fails-1)))
                time.sleep(sleep)

    def _flush_spill(self):
        files = sorted(os.listdir(self.spill_dir))
        for fn in files[:8]:  # bounded per-iteration flush to avoid I/O spikes
            path = os.path.join(self.spill_dir, fn)
            with open(path) as f:
                item = json.load(f)
            ok = self._send_batch([item])
            if ok:
                os.remove(path)
            else:
                break  # stop on failure to preserve order

    def shutdown(self):
        self.stop = True
        self.write_thread.join(timeout=5)
# Usage: logger = BoundedLogger(); logger.log({'track':123}, priority=8)