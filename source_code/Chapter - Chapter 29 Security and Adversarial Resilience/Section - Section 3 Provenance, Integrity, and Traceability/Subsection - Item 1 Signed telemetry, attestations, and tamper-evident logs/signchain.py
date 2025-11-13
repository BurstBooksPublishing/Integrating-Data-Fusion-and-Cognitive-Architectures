import json, time, base64, hashlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

def sha256_bytes(b): return hashlib.sha256(b).digest()

def sign_entry(payload, prev_hash, priv_key):
    entry = {
        "ts": time.time(),               # capture time
        "payload": payload,
        "prev": base64.b64encode(prev_hash).decode() if prev_hash else None
    }
    ser = json.dumps(entry, sort_keys=True).encode()
    H = sha256_bytes(ser + (prev_hash or b""))  # chain hash
    sig = priv_key.sign(H)
    entry.update({"hash": base64.b64encode(H).decode(),
                  "sig": base64.b64encode(sig).decode()})
    return entry, H

def verify_log(entries, pub_key):
    prev = None
    for e in entries:
        # reserialize and recompute hash
        ser = json.dumps({"ts": e["ts"], "payload": e["payload"], "prev": e["prev"]}, sort_keys=True).encode()
        H = sha256_bytes(ser + (base64.b64decode(e["prev"]) if e["prev"] else b""))
        if base64.b64encode(H).decode() != e["hash"]:
            return False, "hash_mismatch"
        try:
            pub_key.verify(base64.b64decode(e["sig"]), H)
        except Exception:
            return False, "signature_invalid"
        prev = H
    return True, "ok"

# demo keys (in production use KM provisioning)
priv = Ed25519PrivateKey.generate()
pub = priv.public_key()

# build log
log = []
prev = None
for payload in [{"range": 12.3}, {"range": 12.4}]:
    entry, prev = sign_entry(payload, prev, priv)
    log.append(entry)

print(verify_log(log, pub))  # (True, 'ok')