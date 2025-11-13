import hmac, hashlib, json, time
from typing import Tuple

# Simple in-memory key-store stub; replace with KMS/HSM in production.
class KeyStore:
    def __init__(self):
        self.current_key = b'initial_secret_32_bytes_long____'  # 32 bytes
        self.rotation_time = time.time() + 86400  # rotate in 1 day

    def get_key(self) -> bytes:
        # In real systems authenticate requester and fetch key handle.
        if time.time() > self.rotation_time:
            self.rotate()
        return self.current_key

    def rotate(self):
        # Rotate key securely via KMS API; here we derive new key for demo.
        self.current_key = hashlib.sha256(self.current_key + b'rotate').digest()
        self.rotation_time = time.time() + 86400

keystore = KeyStore()

# Append-only log entry creation with chained HMAC
def sign_entry(record: dict, prev_tag: bytes) -> Tuple[dict, bytes]:
    key = keystore.get_key()
    payload = json.dumps(record, sort_keys=True).encode()
    to_mac = prev_tag + payload
    tag = hmac.new(key, to_mac, hashlib.sha256).digest()  # 256-bit tag
    record['prev_tag'] = prev_tag.hex()
    record['tag'] = tag.hex()
    record['timestamp'] = int(time.time())
    return record, tag

# Verification routine
def verify_chain(entries: list) -> bool:
    prev_tag = bytes.fromhex(entries[0]['prev_tag'])
    for e in entries:
        payload = json.dumps({k:v for k,v in e.items() if k not in ('prev_tag','tag')}, sort_keys=True).encode()
        key = keystore.get_key()  # use historical keys from KMS in prod
        expected = hmac.new(key, prev_tag + payload, hashlib.sha256).digest()
        if expected.hex() != e['tag']:
            return False
        prev_tag = expected
    return True

# Example usage
log = []
prev = b'\x00'*32
record, prev = sign_entry({'sensor':'lidar','range':12.3}, prev)
log.append(record)