#!/usr/bin/env python3
# pip install psutil
import os, time, multiprocessing as mp, psutil, ctypes
SCHED_FIFO = 1

def try_set_realtime(pid, prio=10):
    try:
        # syscall wrapper for sched_setscheduler
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        class SchedParam(ctypes.Structure):
            _fields_ = [('sched_priority', ctypes.c_int)]
        param = SchedParam(prio)
        res = libc.sched_setscheduler(pid, SCHED_FIFO, ctypes.byref(param))
        if res != 0:
            raise OSError(ctypes.get_errno(), "sched_setscheduler")
        return True
    except Exception as e:
        print(f"RT set failed for pid {pid}: {e}")
        return False

def worker(name, core, C_ms, T_ms, q):
    p = psutil.Process()
    p.cpu_affinity([core])            # pin to core
    try_set_realtime(0, prio=20)      # 0 -> calling thread
    deadline = T_ms / 1000.0
    next_t = time.monotonic()
    missed = 0
    while True:
        next_t += deadline
        start = time.monotonic()
        # simulate work (replace with real operator call)
        time.sleep(C_ms / 1000.0)
        elapsed = time.monotonic() - start
        if time.monotonic() > next_t:
            missed += 1
        # report a simple telemetry tuple
        q.put((name, elapsed, missed))
        # jitter guard
        sleep = max(0.0, next_t - time.monotonic())
        time.sleep(sleep)

if __name__ == "__main__":
    q = mp.Queue()
    # define tasks: (name, core, C_ms, T_ms)
    tasks = [("camera", 0, 12, 33), ("associate", 0, 8, 50), ("scenario", 1, 50, 200)]
    procs = []
    for t in tasks:
        p = mp.Process(target=worker, args=(*t, q), daemon=True)
        p.start()
        procs.append(p)
    # telemetry loop
    try:
        while True:
            name, elapsed, missed = q.get()
            print(f"{name}: exec={elapsed*1000:.1f}ms missed={missed}")
    except KeyboardInterrupt:
        for p in procs: p.terminate()