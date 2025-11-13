import time, json, hmac, hashlib
from typing import Dict

def create_audit_record(processed_local: int, processed_cloud: int,
                        events_requiring_consent: int, auditable_events: int,
                        policy_version: str, key: bytes) -> str:
    # build payload
    payload: Dict = {
        "ts": int(time.time()),                   # epoch
        "processed_local": processed_local,
        "processed_cloud": processed_cloud,
        "events_requiring_consent": events_requiring_consent,
        "auditable_events": auditable_events,
        "policy_version": policy_version
    }
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    # HMAC signature for tamper evidence (use secure enclave key in production)
    sig = hmac.new(key, body, hashlib.sha256).hexdigest()
    record = {"payload": payload, "hmac": sig}
    return json.dumps(record)                    # store/send to audit ledger

def local_only_rate(processed_local: int, processed_cloud: int) -> float:
    total = processed_local + processed_cloud
    return processed_local / total if total > 0 else 1.0

# Example usage
if __name__ == "__main__":
    SECRET_KEY = b"dev_key_replace_in_prod"       # replace with device key
    rec = create_audit_record(18000, 2000, 20500, 20400, "v1.2.3", SECRET_KEY)
    print(rec)                                    # push to secure log/ledger
    print("L =", local_only_rate(18000, 2000))