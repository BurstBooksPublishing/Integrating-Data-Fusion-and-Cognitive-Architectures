import json, numpy as np
from collections import defaultdict
# load event log (JSON lines): each event has capture_ts, emit_ts, label, pred, conf, modalities
events = [json.loads(line) for line in open("events.jl")]
# accuracy and PR
y_true = np.array([e["label"] for e in events])
y_pred = np.array([e["pred"] for e in events])
acc = (y_true == y_pred).mean()
# latency p95
latencies = np.array([e["emit_ts"] - e["capture_ts"] for e in events])
p95 = np.percentile(latencies,95)
# ECE (10 bins)
confs = np.array([e["conf"] for e in events])
bins = np.linspace(0,1,11)
ece = 0.0
n = len(events)
for i in range(10):
    idx = (confs >= bins[i]) & (confs < bins[i+1])
    if idx.sum()==0: continue
    acc_bin = (y_true[idx] == y_pred[idx]).mean()
    conf_bin = confs[idx].mean()
    ece += (idx.sum()/n) * abs(acc_bin - conf_bin)
# robustness via modality dropout simulation
def simulate_dropout(events, drop_mod):
    # returns accuracy under drop_mod set (e.g., {"radar"})
    y_sim = []
    for e in events:
        m = {k:v for k,v in e["modalities"].items() if k not in drop_mod}
        # simple heuristic: if camera missing, fallback to pred_prob*0.8 (simulate degrade)
        pred = e["pred"]
        if "camera" in drop_mod and e["pred_conf"]<0.6: pred = "unknown"
        y_sim.append(pred)
    return (np.array(y_sim) == y_true).mean()
robustness = { "no_drop": acc,
               "drop_camera": simulate_dropout(events, {"camera"}),
               "drop_radar": simulate_dropout(events, {"radar"}) }
print(acc, p95, ece, robustness)  # telemetry for dashboards