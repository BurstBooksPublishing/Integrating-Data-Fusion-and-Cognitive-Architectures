import numpy as np
import json
# toy candidate features: rows=candidates, cols=[detection_score, false_alarm_penalty]
candidates = np.array([[0.8, 0.1],[0.6,0.05],[0.4,0.02]])

def slider_to_weights(s, alpha=8.0, eps=1e-6):
    v = np.log(np.array([s+eps, 1.0-s+eps]))  # interpretable basis
    w = np.exp(alpha * v)
    return w / w.sum()

def estimated_precision(w, candidates):
    # simple proxy: weighted detection / (weighted detection + avg penalty)
    det = candidates[:,0].max()
    pen = (w[1] * candidates[:,1].mean())
    return det / (det + pen + 1e-9)

def select_with_guardrail(s, candidates, p_min=0.7):
    w = slider_to_weights(s)
    prec = estimated_precision(w, candidates)
    if prec < p_min:
        # guardrail: switch to conservative preset weights and mark audit flag
        w = np.array([0.3,0.7])  # conservative favoring penalty reduction
        audit = "guardrail_triggered"
    else:
        audit = "ok"
    utilities = candidates.dot(w)
    choice = int(np.argmax(utilities))
    # provenance record
    record = {"slider": float(s), "weights": w.tolist(), "precision": float(prec),
              "choice": choice, "audit": audit}
    print(json.dumps(record))  # replace with secure telemetry write
    return choice

# Example usage
for s in [0.9, 0.6, 0.2]:
    print("chosen candidate:", select_with_guardrail(s, candidates))