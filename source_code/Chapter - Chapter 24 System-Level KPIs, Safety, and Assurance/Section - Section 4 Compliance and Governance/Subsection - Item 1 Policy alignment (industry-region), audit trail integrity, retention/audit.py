import json, time, hmac, hashlib, os

KMS_KEY = os.environ.get("AUDIT_HMAC_KEY", "dev-key-please-replace").encode()  # use KMS in prod
LEDGER_PATH = "audit_ledger.jsonl"

def sha256(b): return hashlib.sha256(b).digest()

def sign_mac(key, msg): return hmac.new(key, msg, hashlib.sha256).hexdigest()

def append_record(record):
    # record: dict with fields 'event', 'meta', 't_create'
    prev_hash = b"\x00"*32
    try:
        with open(LEDGER_PATH, "rb") as f:
            # read last line efficiently
            f.seek(0, 2)
            pos = f.tell()
            while pos:
                pos -= 1
                f.seek(pos)
                if f.read(1) == b"\n":
                    break
            last = f.readline()
            if last:
                prev = json.loads(last.decode())
                prev_hash = bytes.fromhex(prev["entry_hash"])
    except FileNotFoundError:
        pass
    blob = json.dumps(record, sort_keys=True).encode()
    entry_hash = hashlib.sha256(prev_hash + blob).hexdigest()
    mac = sign_mac(KMS_KEY, bytes.fromhex(entry_hash))
    entry = {"entry_hash": entry_hash, "mac": mac, "record": record}
    with open(LEDGER_PATH, "ab") as f:
        f.write((json.dumps(entry) + "\n").encode())
    return entry

def verify_ledger():
    prev_hash = b"\x00"*32
    with open(LEDGER_PATH, "rb") as f:
        for line in f:
            e = json.loads(line.decode())
            blob = json.dumps(e["record"], sort_keys=True).encode()
            recomputed = hashlib.sha256(prev_hash + blob).hexdigest()
            if recomputed != e["entry_hash"]:
                return False, "hash mismatch"
            expected_mac = sign_mac(KMS_KEY, bytes.fromhex(e["entry_hash"]))
            if expected_mac != e["mac"]:
                return False, "mac mismatch"
            prev_hash = bytes.fromhex(e["entry_hash"])
    return True, "ok"

# Example append
rec = {"event":"L2_situation_promotion","meta":{"policy_ids":["GDPR-DSAR"],"jurisdiction":"EU"},"t_create":int(time.time())}
append_record(rec)  # writes chained, signed entry