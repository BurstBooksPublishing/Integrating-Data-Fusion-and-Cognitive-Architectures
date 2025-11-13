#!/usr/bin/env python3
import time, uuid, random, json, statistics
from collections import defaultdict, deque

# Simple telemetry stores (replace with real exporters in production)
counters = defaultdict(int)
hist_buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf")]
hist_counts = [0]*len(hist_buckets)
trace_log = []                  # append-only trace store (compact)
exemplar_log = []               # sparse linkage records

def start_span(name):
    span = {"span_id": uuid.uuid4().hex, "name": name, "ts": time.time(), "events": []}
    trace_log.append(span)
    return span

def end_span(span):
    span["end_ts"] = time.time()

def record_counter(name, n=1, labels=None):
    key = (name, tuple(sorted((labels or {}).items())))
    counters[key] += n

def record_histogram(name, value, labels=None, exemplar_span=None):
    # bucket update
    for i,b in enumerate(hist_buckets):
        if value <= b:
            hist_counts[i] += 1
            break
    # sparse exemplar attach if provided (stores minimal pointer)
    if exemplar_span:
        exemplar_log.append({
            "metric": name,
            "value": value,
            "labels": labels or {},
            "trace_id": exemplar_span["span_id"],
            "ts": time.time()
        })

# Simulated control loop: decisions and occasional policy switches
lat_window = deque(maxlen=100)
for step in range(200):
    # simulate processing
    span = start_span("decision_evaluate")
    proc_latency = random.expovariate(5)  # seconds
    reward = random.gauss(0.5, 0.2)
    time.sleep(min(proc_latency, 0.01))   # fast sim
    span["events"].append({"latency": proc_latency, "reward": reward})
    end_span(span)

    # metrics
    record_counter("decisions_total", 1, labels={"policy":"default"})
    record_histogram("decision_latency_s", proc_latency, labels={"policy":"default"})

    # policy switch happens rarely; produce exemplar for high latency or low reward
    if random.random() < 0.03 or proc_latency > 0.4 or reward < 0.0:
        sw = start_span("policy_switch")
        sw["events"].append({"reason":"latency_or_reward"})
        end_span(sw)
        record_counter("policy_switches_total", 1, labels={"from":"default","to":"alt"})
        record_histogram("policy_switch_latency_s", proc_latency, labels={"from":"default","to":"alt"},
                         exemplar_span=sw)

    lat_window.append(proc_latency)

# emit compact telemetry snapshot
print(json.dumps({
    "counters": {f"{k[0]}|{dict(k[1])}": v for k,v in counters.items()},
    "histogram_buckets": list(zip(hist_buckets, hist_counts)),
    "traces": [{"span_id":t["span_id"], "name":t["name"]} for t in trace_log],
    "exemplars": exemplar_log[:10]
}, indent=2))