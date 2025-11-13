from prometheus_client import start_http_server, Gauge, Summary  # metrics export
import time, random, collections, threading
import numpy as np

# Prometheus metrics (simple gauges for demo)
ACC = Gauge('fusion_accuracy', 'sliding-window accuracy')
LAT_P95 = Gauge('fusion_latency_p95_ms', 'p95 latency in ms')
ROB_MIN = Gauge('fusion_robustness_min', 'min score across scenarios')
SAF_VIOL_RATE = Gauge('safety_violation_rate', 'safety violations per window')

window = collections.deque(maxlen=1000)  # simple sliding window store

def ingest_event():
    # simulate streamed fused output with label/oracle tags
    while True:
        label = random.choice([True, False])              # ground truth
        pred = random.random() > 0.2                      # prediction
        latency = max(1, np.random.exponential(30))      # ms
        scenario = random.choice(['rain','clear','dense'])
        safety_violation = (random.random() < 0.005)     # rare
        window.append((label, pred, latency, scenario, safety_violation))
        time.sleep(0.01)

def compute_metrics():
    while True:
        data = list(window)
        if data:
            labels, preds, lat, scen, viol = zip(*data)
            accuracy = sum(l==p for l,p in zip(labels,preds)) / len(data)
            p95 = np.percentile(lat,95)
            # robustness: min accuracy per scenario
            scen_scores = {}
            for s in set(scen):
                idx = [i for i,x in enumerate(scen) if x==s]
                scen_scores[s] = sum(labels[i]==preds[i] for i in idx)/len(idx)
            rob_min = min(scen_scores.values())
            viol_rate = sum(viol)/len(data)
            # export
            ACC.set(accuracy)
            LAT_P95.set(p95)
            ROB_MIN.set(rob_min)
            SAF_VIOL_RATE.set(viol_rate)
        time.sleep(1)

if __name__ == '__main__':
    start_http_server(8000)          # Prometheus scrape endpoint
    threading.Thread(target=ingest_event, daemon=True).start()
    compute_metrics()                # blocking