import time, statistics

# sliding-window buffers for each stage latency (ms)
window = {'neural': [], 'retrieval': [], 'symbolic': []}
WINDOW_SIZE = 500  # samples

# SLOs (ms)
SLOS = {'p99_total': 300, 'p99_neural': 150, 'p99_symbolic': 150}

def record(stage, latency_ms):
    buf = window[stage]
    buf.append(latency_ms)
    if len(buf) > WINDOW_SIZE: buf.pop(0)

def p99(buf): return statistics.quantiles(buf, n=100)[-1] if len(buf)>0 else 0

def check_and_act():
    p99s = {s: p99(window[s]) for s in window}
    total_p99 = sum(p99s.values())  # conservative serial sum
    # trigger fallback if total p99 violates SLA
    if total_p99 > SLOS['p99_total']:
        # action: prefer deterministic symbolic-only path or cached response
        trigger_fallback(mode='symbolic_cached')  # comment: fast, auditable
    else:
        allow_full_pipeline()

def trigger_fallback(mode):
    # lightweight gate to change routing policy for next decisions
    print(f"Fallback triggered: {mode}")
    # code to flip runtime routing, notify operator, and log evidence