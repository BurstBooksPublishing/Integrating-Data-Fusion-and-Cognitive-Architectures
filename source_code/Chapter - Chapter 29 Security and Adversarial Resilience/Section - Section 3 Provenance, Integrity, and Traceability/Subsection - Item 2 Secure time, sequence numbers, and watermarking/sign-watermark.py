import hmac, hashlib, json, time
from base64 import b64encode, b64decode

SECRET = b"shared-secret"          # replace with secure key management
allowed_lateness = 0.5             # seconds

# persistent monotonic counter (simulate secure storage)
def load_counter(path="counter.txt"):
    try:
        return int(open(path).read().strip())
    except Exception:
        return 0
def store_counter(n, path="counter.txt"):
    open(path,"w").write(str(int(n)))

def sign_packet(payload, seq):
    payload_bytes = json.dumps(payload, sort_keys=True).encode()
    mac = hmac.new(SECRET, payload_bytes + b"::" + str(seq).encode(), hashlib.sha256).digest()
    return {"payload": payload, "seq": seq, "sig": b64encode(mac).decode()}

def verify_packet(packet, last_seq):
    mac = b64decode(packet["sig"])
    payload_bytes = json.dumps(packet["payload"], sort_keys=True).encode()
    expected = hmac.new(SECRET, payload_bytes + b"::" + str(packet["seq"]).encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        raise ValueError("signature mismatch")
    # monotonic check
    if packet["seq"] <= last_seq:
        raise ValueError("sequence replay or reorder")
    return True

# simple watermark state per partition
watermark_state = {}  # partition -> max_event_time

def ingest(packet, partition, last_seq):
    verify_packet(packet, last_seq)                      # verify auth + seq
    ev_time = packet["payload"]["event_time"]
    watermark_state[partition] = max(watermark_state.get(partition, -1), ev_time)
    # compute per-partition watermark with allowed lateness
    Wp = watermark_state[partition] - allowed_lateness
    return Wp

# produce a signed telemetry packet
if __name__ == "__main__":
    seq = load_counter()
    seq += 1
    payload = {"event_time": time.time(), "measure": [1.0,2.0]}  # capture
    packet = sign_packet(payload, seq)
    store_counter(seq)
    print("signed packet:", packet)
    # ingest example
    Wp = ingest(packet, partition="cam_front", last_seq=seq-1)
    print("partition watermark:", Wp)