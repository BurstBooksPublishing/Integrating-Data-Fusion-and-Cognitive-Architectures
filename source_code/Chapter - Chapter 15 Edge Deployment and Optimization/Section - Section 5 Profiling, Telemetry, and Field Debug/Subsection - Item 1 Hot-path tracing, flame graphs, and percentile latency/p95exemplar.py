import time, threading, traceback
from collections import deque
from functools import wraps

WINDOW_SIZE = 1000                  # bounded memory for latencies
SAMPLE_INTERVAL = 0.01              # seconds between stack samples (if used)
EXEMPLAR_FILE = "folded_exemplars.txt"

latencies = deque(maxlen=WINDOW_SIZE)
lock = threading.Lock()

def empirical_percentile(data, q):
    # simple exact percentile per Eq. (1)
    if not data:
        return float('inf')
    s = sorted(data)
    idx = min(len(s)-1, max(0, int((q*len(s))-1e-9)))
    return s[max(0, idx)]

def capture_folded_stack():
    # capture current thread stacks and fold names; single-line exemplar
    stacks = []
    for thread_id, frame in sys._current_frames().items():
        stack = traceback.extract_stack(frame)
        names = [f"{frame.name}" for frame in stack if frame.name]
        if names:
            stacks.append(";".join(names[::-1]))  # root-first
    # write one exemplar per capture (count=1)
    if stacks:
        with open(EXEMPLAR_FILE, "a") as f:
            for st in stacks:
                f.write(f"{st} 1\n")

def exemplar_on_tail(q_threshold=0.95):
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (time.time() - start) * 1000.0  # ms
                with lock:
                    latencies.append(elapsed)
                    p = empirical_percentile(latencies, q_threshold)
                if elapsed > p:
                    # Tail case: capture folded stack exemplar
                    import sys
                    capture_folded_stack()
        return wrapped
    return decorator

# Usage: decorate critical pipeline functions
@exemplar_on_tail(0.95)
def process_frame(frame):
    # placeholder for fusion + cognition work
    time.sleep(0.005)  # simulate work
    return {"tracks": [], "decisions": []}