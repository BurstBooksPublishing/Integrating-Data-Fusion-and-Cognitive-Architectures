import json, hmac, hashlib, time, os
KEY = b'shared_hmac_key'             # production: use KMS-backed key
STORE = 'trace_log.aof'              # append-only file for demo

def now_ts():
    return time.time()               # wall-clock; include monotonic in real systems

def emit_trace(trace_id, parent_id, level_tag, actor, payload_hash, confidence, rule_hits):
    entry = {
        "trace_id": trace_id,
        "parent_id": parent_id,
        "timestamp": now_ts(),
        "level": level_tag,           # L0..L4
        "actor": actor,
        "payload_hash": payload_hash, # pointer to blob (hash)
        "confidence": confidence,     # scalar or dict
        "rule_hits": rule_hits,       # list of {rule_id, score}
        "schema_v": "1.0"
    }
    raw = json.dumps(entry, separators=(",", ":"), sort_keys=True).encode()
    sig = hmac.new(KEY, raw, hashlib.sha256).hexdigest()  # tamper-evidence
    envelope = {"entry": entry, "hmac": sig}
    with open(STORE, "ab") as f:
        f.write(json.dumps(envelope).encode() + b"\n")   # append-only write
    return envelope

# Example usage (edge or node):
emit_trace("t123", None, "L2", "situation_service", "sha256:abcd...", 0.87,
           [{"rule_id":"r42","score":0.6}])