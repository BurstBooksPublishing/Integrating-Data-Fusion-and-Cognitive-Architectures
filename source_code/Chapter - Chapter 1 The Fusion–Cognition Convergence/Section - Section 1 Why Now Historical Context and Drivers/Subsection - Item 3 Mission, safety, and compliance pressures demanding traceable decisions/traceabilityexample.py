import json, hashlib, time, os

KEY = b'secret-signing-key'  # HMAC key; replace with secure key management

def sign_payload(payload_bytes):
    # simple HMAC-like signature (illustrative); use real KMS/HSM in production
    h = hashlib.sha256(KEY + payload_bytes).hexdigest()
    return h

def append_signed_trace(trace_obj, logfile="trace_log.jsonl"):
    trace_obj['ts'] = time.time()
    payload = json.dumps(trace_obj, sort_keys=True).encode('utf-8')
    trace_obj['sig'] = sign_payload(payload)
    # atomic append to newline-delimited JSON log
    with open(logfile, "ab") as f:
        f.write(json.dumps(trace_obj).encode('utf-8') + b"\n")

# Example usage: record a decision rationale
rationale = {
  "decision": "brake",
  "evidence": [
    {"type": "track", "id": "T123", "pos": [12.1, -3.4], "cov": [[0.2,0],[0,0.3]]},
    {"type": "classifier", "name": "pedestrian_net", "prob": 0.92}
  ],
  "rules": [{"rule_id": "R_stop_1", "weight": 0.8}],
  "models": {"fusion": "EKF_v2.1", "classifier": "ped_net_v3"},
  "operator": {"id": None, "override": False}
}
append_signed_trace(rationale)  # append signed trace entry