import json, time
from typing import Dict

# Compute assurance and decide action; then persist rationale.
def compute_assurance(p_c: float, u_e: float, C_e: float, weights=(0.5,0.3,0.2)) -> float:
    w1,w2,w3 = weights
    return w1*p_c + w2*(1.0 - u_e) + w3*C_e

def create_rationale(decision_id: str, inputs: Dict, metrics: Dict, S: float, threshold: float):
    rationale = {
        "decision_id": decision_id,
        "timestamp": time.time(),
        "inputs": inputs,               # summaries or pointers to evidence (signed)
        "metrics": metrics,             # p_c, u_e, C_e, weights
        "assurance_score": S,
        "threshold": threshold,
        "action": "AUTOMATE" if S >= threshold else "HUMAN_REVIEW"
    }
    return rationale

def persist_rationale(rationale: Dict, path="rationales.log"):
    with open(path, "a") as f: f.write(json.dumps(rationale) + "\n")  # append as audit trail

# Example usage
if __name__ == "__main__":
    inputs = {"track_id": "T123", "evidence_ptrs": ["s3://evidence/evt1", "s3://evidence/evt2"]}
    p_c, u_e, C_e = 0.72, 0.18, 0.95
    weights = (0.5, 0.3, 0.2); tau = 0.80
    S = compute_assurance(p_c, u_e, C_e, weights)
    metrics = {"p_c": p_c, "u_e": u_e, "C_e": C_e, "weights": weights}
    rationale = create_rationale("dec-0001", inputs, metrics, S, tau)
    persist_rationale(rationale)   # audit record for post-hoc inspection
    print("Assurance", S, "Action:", rationale["action"])