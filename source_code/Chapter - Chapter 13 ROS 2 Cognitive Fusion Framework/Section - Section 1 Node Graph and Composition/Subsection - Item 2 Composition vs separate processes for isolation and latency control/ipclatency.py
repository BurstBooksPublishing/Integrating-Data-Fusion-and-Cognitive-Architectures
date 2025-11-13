#!/usr/bin/env python3
# Measure one-way transfer latency for two IPC modes.
import time, multiprocessing as mp, numpy as np
from multiprocessing import shared_memory

N = 1000               # samples
ARR_SHAPE = (64, 64)   # simulated feature/frame
DTYPE = np.float32

def producer_queue(q):
    a = np.random.rand(*ARR_SHAPE).astype(DTYPE)
    for _ in range(N):
        t0 = time.monotonic()
        q.put((t0, a))          # pickle serialization
    q.put(None)

def consumer_queue(q):
    lat = []
    while True:
        item = q.get()
        if item is None: break
        t0, a = item
        lat.append(time.monotonic() - t0)
    print("Queue: mean ms", 1e3*np.mean(lat), "p95 ms", 1e3*np.percentile(lat,95))

def producer_shared(name, shape, dtype):
    shm = shared_memory.SharedMemory(name=name, create=False)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    for _ in range(N):
        # write simulated frame in-place
        arr[:] = np.random.rand(*shape).astype(dtype)
        # publish timestamp via Pipe for fairness
        time.sleep(0)  # yield

def consumer_shared(name, shape, dtype, pipe):
    shm = shared_memory.SharedMemory(name=name, create=False)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    lat = []
    for _ in range(N):
        t0 = time.monotonic()
        # read in-place (no copy)
        v = arr.copy()   # realistic consumer may avoid even this copy
        lat.append(time.monotonic() - t0)
    print("SharedMem: mean ms", 1e3*np.mean(lat), "p95 ms", 1e3*np.percentile(lat,95))

if __name__ == "__main__":
    # Queue mode
    q = mp.Queue(maxsize=4)
    p = mp.Process(target=producer_queue, args=(q,))
    c = mp.Process(target=consumer_queue, args=(q,))
    p.start(); c.start(); p.join(); c.join()

    # Shared memory mode
    shm = shared_memory.SharedMemory(create=True, size=np.prod(ARR_SHAPE)*4)
    name = shm.name
    # initialize view
    arr = np.ndarray(ARR_SHAPE, dtype=DTYPE, buffer=shm.buf)
    p2 = mp.Process(target=producer_shared, args=(name, ARR_SHAPE, DTYPE))
    c2 = mp.Process(target=consumer_shared, args=(name, ARR_SHAPE, DTYPE, None))
    p2.start(); c2.start(); p2.join(); c2.join()
    shm.close(); shm.unlink()