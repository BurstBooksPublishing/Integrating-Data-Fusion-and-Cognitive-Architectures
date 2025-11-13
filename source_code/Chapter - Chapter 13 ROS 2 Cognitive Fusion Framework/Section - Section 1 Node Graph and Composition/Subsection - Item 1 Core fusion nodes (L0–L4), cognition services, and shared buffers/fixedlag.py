import time, threading
from collections import deque
from dataclasses import dataclass

@dataclass
class Entry:
    ts: float
    payload: dict

class FixedLagBuffer:
    def __init__(self, window_s):
        self.window = window_s
        self.buf = deque()
        self.lock = threading.Lock()
    def push(self, entry: Entry):
        with self.lock:
            self.buf.append(entry)
            self._evict_now()
    def _evict_now(self):
        cutoff = time.time() - self.window
        while self.buf and self.buf[0].ts < cutoff:
            self.buf.popleft()
    def snapshot(self):
        with self.lock:
            return list(self.buf)  # shallow copy for consumer

# Producer thread simulating L0 detections
def producer(buf: FixedLagBuffer, rate_hz=20):
    period = 1.0 / rate_hz
    i = 0
    while i < 200:
        e = Entry(ts=time.time(), payload={'id': i, 'feature': [i%5]})
        buf.push(e)  # L0 writes
        time.sleep(period); i += 1

# Consumer thread simulating L2 cognition access
def consumer(buf: FixedLagBuffer, poll_hz=5):
    period = 1.0 / poll_hz
    for _ in range(40):
        snap = buf.snapshot()  # read working memory
        # simple reasoning: count entries and last ts
        if snap:
            print(f"consumer: entries={len(snap)} last={snap[-1].ts:.3f}")
        time.sleep(period)

if __name__ == "__main__":
    buf = FixedLagBuffer(window_s=3.0)  # 3-second working memory
    t1 = threading.Thread(target=producer, args=(buf,))
    t2 = threading.Thread(target=consumer, args=(buf,))
    t1.start(); t2.start()
    t1.join(); t2.join()