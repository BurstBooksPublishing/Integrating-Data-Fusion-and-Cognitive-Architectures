import json, math, statistics, datetime
from collections import deque, Counter

WINDOW_SECONDS = 3600  # 1-hour rolling window
maxlen = 10000

# rolling buffers
events = deque()  # (ts, is_near_miss, operator_action, severity)
def ingest_line(line):
    rec = json.loads(line)
    ts = datetime.datetime.fromisoformat(rec["ts"])
    events.append((ts, rec["near_miss"], rec["operator_action"], rec["severity"]))
    # evict old
    cutoff = ts - datetime.timedelta(seconds=WINDOW_SECONDS)
    while events and events[0][0] < cutoff:
        events.popleft()

def compute_metrics():
    N = len(events)
    if N == 0:
        return {}
    near_miss_count = sum(1 for e in events if e[1])
    override_count = sum(1 for e in events if e[2] == "override")
    exposure_hours = WINDOW_SECONDS/3600.0
    r = near_miss_count / exposure_hours
    override_rate = override_count / max(1, N)
    avg_severity = statistics.mean(e[3] for e in events) if N else 0
    # EWMA smoothing for trend (alpha chosen for short memory)
    return {"near_miss_rate": r, "override_rate": override_rate,
            "avg_severity": avg_severity, "count": N}

# review rule: if override_rate exceeds 15% or near_miss_rate increases by 20% vs baseline
BASELINE = {"near_miss_rate": 2.0, "override_rate": 0.10}  # example baselines
def check_review(metrics):
    if metrics.get("override_rate",0) > 0.15:
        return "policy_review:high_override"
    if metrics.get("near_miss_rate",0) > 1.2 * BASELINE["near_miss_rate"]:
        return "policy_review:near_miss_increase"
    return "ok"
# Example: ingest a line and evaluate (comment: in production, hook to telemetry stream)
# ingest_line('{"ts":"2025-11-07T12:00:00", "near_miss": true, "operator_action":"accept","severity":3}')