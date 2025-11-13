#!/usr/bin/env python3
import time, random, statistics, threading, queue

RATE_HZ = 20                       # nominal frame rate
SLO_MS = 200                        # end-to-end SLO in ms
RUN_SEC = 10

q = queue.Queue(maxsize=100)

def producer():
    period = 1.0 / RATE_HZ
    end = time.monotonic() + RUN_SEC
    while time.monotonic() < end:
        ts = time.monotonic_ns()     # inject timestamp (ns)
        # simulate sensor jitter and occasional latency spike
        jitter = random.gauss(0, 0.001) + (0.01 if random.random()<0.01 else 0)
        time.sleep(max(0, period + jitter))
        q.put(ts)

def consumer(latencies):
    while True:
        try:
            ts = q.get(timeout=1.0)
        except Exception:
            return
        recv = time.monotonic_ns()
        # simulate processing: fusion + cognition stages with variance
        proc = 0.005 + random.expovariate(200)   # seconds
        time.sleep(proc)
        done = time.monotonic_ns()
        lat_ms = (done - ts) / 1e6
        latencies.append(lat_ms)
        q.task_done()

latencies = []
t1 = threading.Thread(target=producer, daemon=True)
t2 = threading.Thread(target=consumer, args=(latencies,), daemon=True)
t1.start(); t2.start()
t1.join(); q.join()  # wait until queue empty

# diagnostics
if latencies:
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95*len(latencies))-1]
    worst = max(latencies)
    print(f"p50={p50:.1f}ms p95={p95:.1f}ms worst={worst:.1f}ms")
    assert worst <= SLO_MS, "SLO violated: take mitigation actions"