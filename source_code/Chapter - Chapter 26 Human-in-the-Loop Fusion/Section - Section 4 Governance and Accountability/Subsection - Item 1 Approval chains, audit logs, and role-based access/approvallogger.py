import hashlib, hmac, json, time
from math import ceil

# Shared secret per role in production replaced by per-user keys / PKI.
ROLE_KEYS = {"analyst": b"key1", "supervisor": b"key2", "auditor": b"key3"}

def hmac_sign(role, payload):
    key = ROLE_KEYS[role]
    return hmac.new(key, payload, hashlib.sha256).hexdigest()

def chain_hash(entry_json, prev_hash):
    s = (entry_json + (prev_hash or "")).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

class ApprovalLog:
    def __init__(self, path="audit.log"):
        self.path = path
        self.prev_hash = None

    def append(self, entry):
        entry_json = json.dumps(entry, sort_keys=True)
        entry["chain_hash"] = chain_hash(entry_json, self.prev_hash)
        self.prev_hash = entry["chain_hash"]
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")  # append-only file

def require_approvals(hypothesis_id, decision, approvals, n, phi=0.67):
    # approvals: list of tuples (role, timestamp, signature)
    m = ceil(phi * n)
    if len(approvals) < m:
        raise PermissionError("Insufficient approvals")
    # verify each HMAC locally (example uses role keys)
    for role, ts, sig in approvals:
        payload = f"{hypothesis_id}|{decision}|{ts}".encode()
        if hmac_sign(role, payload) != sig:
            raise ValueError("Bad signature")
    # append combined audit entry
    log.append({"time": time.time(), "hypothesis": hypothesis_id,
                "decision": decision, "approvals": approvals})

# Usage example
log = ApprovalLog()
ts = int(time.time())
# each approver computes signature independently; here computed inline.
sig1 = hmac_sign("analyst", f"hyp42|promote|{ts}".encode())
sig2 = hmac_sign("supervisor", f"hyp42|promote|{ts}".encode())
require_approvals("hyp42", "promote", [("analyst", ts, sig1), ("supervisor", ts, sig2)], n=3)