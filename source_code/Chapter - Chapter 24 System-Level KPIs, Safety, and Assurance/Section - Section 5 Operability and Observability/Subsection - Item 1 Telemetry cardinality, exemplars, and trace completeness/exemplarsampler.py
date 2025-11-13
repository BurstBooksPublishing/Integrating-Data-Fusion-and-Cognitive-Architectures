import random, uuid, time
K = 50  # max exemplars per decision class (budget)

class ExemplarReservoir:
    def __init__(self, k=K):
        self.k = k
        self.reservoir = []  # list of (weight, exemplar)

    def consider(self, exemplar, weight=1.0):
        # reservoir sampling with weights (simple priority here)
        if len(self.reservoir) < self.k:
            self.reservoir.append((weight, exemplar))
        else:
            # replace with probability proportional to weight
            total = sum(w for w,_ in self.reservoir) + weight
            if random.random() * total < weight:
                idx = random.randrange(self.k)
                self.reservoir[idx] = (weight, exemplar)

def emit_decision(decision_class, payload, sampler_map):
    trace_id = str(uuid.uuid4())  # stable correlation header
    ts = time.time()
    header = {"trace_id": trace_id, "decision_class": decision_class, "ts": ts}
    # attach header to telemetry (metric/event) - lightweight
    send_metric(decision_class, header)  # placeholder emitter
    # decide exemplar retention based on policies
    sampler = sampler_map.setdefault(decision_class, ExemplarReservoir())
    score = payload.get("uncertainty", 0.0) + payload.get("anomaly_score", 0.0)
    sampler.consider({"header": header, "payload": payload}, weight=score)
    # return trace header for downstream spans
    return header

def send_metric(name, header):
    # real system would push to metrics backend and traces system
    print(f"metric:{name} header:{header}")  # minimal example