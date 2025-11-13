from collections import deque
import time, math

# sliding window config
WINDOW_SEC = 30
QUEUE_GROWTH_THRESHOLD = 5  # items/sec
BACKPRESSURE_DURATION = 5.0  # sec

# sample inputs: tuples (proc_time_sec, completion_ts, queue_len, stage)
samples = deque()  # (proc_time, ts, queue_len)

def record_sample(proc_time, queue_len, ts=None):
    ts = ts or time.time()
    samples.append((proc_time, ts, queue_len))
    # evict old samples
    cutoff = ts - WINDOW_SEC
    while samples and samples[0][1] < cutoff:
        samples.popleft()

def percentile(data, p):
    if not data: return float('nan')
    k = (len(data)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c: return sorted(data)[int(k)]
    d = sorted(data)
    return d[f] + (d[c]-d[f])*(k-f)

def compute_kpis():
    if not samples: return {}
    proc_times = [s[0] for s in samples]
    ts0, tsn = samples[0][1], samples[-1][1]
    duration = max(1e-6, tsn - ts0)
    throughput = len(samples) / duration
    qlens = [s[2] for s in samples]
    # queue growth estimate (simple linear fit)
    t_rel = [s[1]-ts0 for s in samples]
    n = len(t_rel)
    mean_t = sum(t_rel)/n
    mean_q = sum(qlens)/n
    num = sum((t_rel[i]-mean_t)*(qlens[i]-mean_q) for i in range(n))
    den = sum((t_rel[i]-mean_t)**2 for i in range(n)) or 1.0
    slope = num/den  # items/sec
    return {
        'p50': percentile(proc_times,50),
        'p95': percentile(proc_times,95),
        'p99': percentile(proc_times,99),
        'throughput': throughput,
        'queue_growth_slope': slope
    }

# backpressure detector using compute_kpis()
bp_start = None
def detect_backpressure():
    global bp_start
    k = compute_kpis()
    if k.get('queue_growth_slope',0) > QUEUE_GROWTH_THRESHOLD:
        if bp_start is None: bp_start = time.time()
        if time.time() - bp_start >= BACKPRESSURE_DURATION:
            return True
    else:
        bp_start = None
    return False

# Example usage: ingest synthetic stream and evaluate periodically.