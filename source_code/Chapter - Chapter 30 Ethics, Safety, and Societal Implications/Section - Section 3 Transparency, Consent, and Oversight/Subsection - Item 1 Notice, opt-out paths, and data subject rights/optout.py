# simple opt-out propagation and remediation demo
from collections import deque
import time, json, hmac, hashlib

# sample graph: adjacency list {artifact_id: [downstream_ids]}
graph = {
    "raw_1": ["track_42","agg_7"],
    "track_42": ["scenario_3"],
    "agg_7": [],
    "scenario_3": []
}
provenance = {}  # artifact_id -> provenance metadata

def sign_attestation(artifact_id, action, key=b"secret"):
    payload = f"{artifact_id}|{action}|{int(time.time())}".encode()
    sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return {"payload": payload.decode(), "sig": sig}

def propagate_opt_out(seeds):
    # BFS to compute closure; returns list of affected artifacts
    q = deque(seeds)
    affected = set(seeds)
    while q:
        a = q.popleft()
        for b in graph.get(a, []):
            if b not in affected:
                affected.add(b); q.append(b)
    return list(affected)

def remediate(artifact_id, action):
    # action in {"erase","anonymize","restrict","flag_for_review","export"}
    # update provenance and emit signed attestation
    att = sign_attestation(artifact_id, action)
    provenance.setdefault(artifact_id, []).append({"action": action, "attestation": att})
    # implement local store ops (simulated)
    if action == "erase":
        # remove payload (simulate)
        pass
    elif action == "anonymize":
        # redact PII fields
        pass
    # return attestation for audit trail
    return att

# Usage: subject requests opt-out rooted at raw_1
seeds = ["raw_1"]
affected = propagate_opt_out(seeds)
results = {a: remediate(a, "anonymize") for a in affected}
print(json.dumps({"affected": affected, "results": results}, indent=2))