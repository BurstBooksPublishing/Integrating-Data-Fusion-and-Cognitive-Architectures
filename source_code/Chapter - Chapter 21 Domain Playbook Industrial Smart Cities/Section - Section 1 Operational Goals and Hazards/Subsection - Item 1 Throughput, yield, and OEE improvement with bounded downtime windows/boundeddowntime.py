import math, random, datetime
# Simulated fused anomalies: (asset_id, anomaly_score [0..1], predicted_downtime_hours)
assets = [
    ("pumpA", 0.92, 4.0),
    ("motorB", 0.55, 2.0),
    ("conveyorC", 0.72, 1.5),
    ("valveD", 0.30, 0.5),
]
D_max = 6.0  # available planned downtime hours in horizon
# Priority = score * expected_OEE_impact proxy; here we use score*downtime to reflect urgency/value
candidates = [(a, s, d, s*d) for (a, s, d) in assets]
candidates.sort(key=lambda x: x[3], reverse=True)  # high priority first
schedule = []
used = 0.0
for asset, score, dur, priority in candidates:
    # simple safety check placeholder; replace with L2/L3 rule checks and operator approval
    safe = (score > 0.4)  # example rule: only consider >0.4
    if not safe: 
        continue
    if used + dur <= D_max:
        start = datetime.datetime.now() + datetime.timedelta(hours=used)  # naive sequencing
        schedule.append({"asset":asset, "start":start.isoformat(), "dur_h":dur, "score":score})
        used += dur
# Output schedule (traceable) and remaining budget
print("Scheduled:", schedule)
print("Remaining downtime budget (h):", round(D_max-used,2))