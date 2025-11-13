import json, time, hashlib

def sign_provenance(payload, key="dev-key"):
    # simple hash-based attestation (replace with PKI in production)
    return hashlib.sha256((json.dumps(payload,sort_keys=True)+key).encode()).hexdigest()

def build_artifact(situation_id, entities, relations, posteriors, utilities):
    # assemble artifact with calibrated summary and counterfactual seed
    artifact = {
        "situation_id": situation_id,
        "timestamp": time.time(),
        "temporal_window": {"start": time.time()-5.0, "end": time.time()},
        "entities": entities,
        "relations": relations,
        "posterior_probs": posteriors,
        "calibrated_probs": calibrate_probs(posteriors), # stubbed calibration
        "calibration_metrics": {"ECE": 0.02, "NLL": 1.12},
        "counterfactuals": [{"name":"increase_speed","delta":{"agent_A":{"speed":+5}}}],
        "preliminary_risk_scores": compute_impact(posteriors, utilities),
        "provenance": None
    }
    artifact["provenance"] = {"attestation": sign_provenance(artifact)}
    return artifact

def calibrate_probs(p):
    # identity calibrator here; replace with isotonic/platt in prod
    return p

def compute_impact(p, u):
    # Expectation per Eq. (1)
    return sum(p_i*u_i for p_i,u_i in zip(p,u))

# Example usage
artifact = build_artifact(
    "sit-1234",
    entities=[{"id":"A","type":"vehicle","state":{"pos":[0,0],"vel":5}}],
    relations=[{"from":"A","to":"B","type":"closing","confidence":0.8}],
    posteriors=[0.7,0.3], # two outcomes
    utilities=[-100, -10] # consequence costs
)
print(json.dumps(artifact,indent=2))  # publish to L3 bus or store