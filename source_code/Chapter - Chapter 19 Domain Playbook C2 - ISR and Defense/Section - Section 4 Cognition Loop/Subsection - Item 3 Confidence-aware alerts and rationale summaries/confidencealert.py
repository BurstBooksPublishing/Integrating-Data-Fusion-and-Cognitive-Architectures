import math, json, time

def sigmoid(x): return 1.0/(1.0+math.exp(-x))

def calibrate_platt(s, a=1.0, b=0.0):
    # simple Platt scaling parameters a,b learned offline
    return sigmoid(a*s + b)

def make_alert(entity_id, raw_score, sigma_e, provenance,
               U_TP=10.0, U_FP=-1.0, platt_a=1.2, platt_b=-0.3,
               lcb_k=1.0, dwell_seconds=30):
    p = calibrate_platt(raw_score, a=platt_a, b=platt_b)   # calibrated prob
    p_lcb = max(0.0, p - lcb_k*sigma_e)                    # conservative bound
    expected = p*U_TP + (1-p)*U_FP                        # eq (1)
    expected_lcb = p_lcb*U_TP + (1-p_lcb)*U_FP
    # decide severity
    if expected_lcb > 0:
        level = "CRITICAL"       # actionable, human approval may follow
    elif expected > 0:
        level = "ADVISORY"       # informative, watch-and-wait
    else:
        level = "INFO"           # log only
    # assemble rationale
    rationale = {
        "entity": entity_id,
        "calibrated_p": round(p,3),
        "p_lcb": round(p_lcb,3),
        "sigma_epistemic": round(sigma_e,3),
        "decision_metric": round(expected_lcb,3) if level!="INFO" else round(expected,3),
        "top_evidence": provenance[:5],   # top-k evidence items
        "counterfactual_hint": "removing AIS raises p to ~0.78"  # computed elsewhere
    }
    alert = {
        "timestamp": time.time(),
        "level": level,
        "rationale": rationale,
        "action": "hold-fire" if level!="CRITICAL" else "require-approval"
    }
    return json.dumps(alert)   # transportable alert payload

# Example usage (standalone)
if __name__ == "__main__":
    prov = [
      {"sensor":"AIS","ts":"2025-11-07T12:01Z","confidence":0.9},
      {"sensor":"SAR","ts":"2025-11-07T11:59Z","confidence":0.7},
      {"sensor":"ESM","ts":"2025-11-07T11:58Z","signature_match":False}
    ]
    print(make_alert("track_123", raw_score=0.45, sigma_e=0.18, provenance=prov))