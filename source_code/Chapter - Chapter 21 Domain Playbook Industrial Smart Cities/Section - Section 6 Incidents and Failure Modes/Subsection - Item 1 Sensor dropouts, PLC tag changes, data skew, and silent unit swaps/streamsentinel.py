#!/usr/bin/env python3
import time, json, math
import numpy as np

# Config
HEARTBEAT_TIMEOUT = 30.0  # seconds
RESIDUAL_WINDOW = 50      # samples
SWAP_Z_SCORE = 4.0

# In-memory state
last_ts = {}              # sensor_id -> last timestamp
schema_registry = {"temp": float, "vibration": float, "rpm": int}
buffers = {}              # sensor_id -> list of recent values

def on_message(msg_json):
    msg = json.loads(msg_json)
    sid = msg["sensor_id"]           # sensor identity
    t = msg["ts"]                    # epoch seconds
    tag = msg["tag"]                 # e.g., 'temp'
    val = msg["value"]

    # heartbeat detect
    prev = last_ts.get(sid)
    if prev and (t - prev) > HEARTBEAT_TIMEOUT:
        print(f"ALERT: heartbeat gap for {sid}, gap={t-prev:.1f}s")
        quarantine_stream(sid)
    last_ts[sid] = t

    # schema check
    expected = schema_registry.get(tag)
    if expected is None:
        print(f"ALERT: unknown tag {tag} from {sid}")
        record_event("schema_mismatch", sid, tag)
        return

    # buffer and simple corroboration with peer aggregate
    buf = buffers.setdefault(sid, [])
    buf.append(val)
    if len(buf) > RESIDUAL_WINDOW:
        buf.pop(0)

    # compute residual against peer mean (simple redundancy)
    peer_vals = collect_peer_values(tag, exclude=sid)
    if peer_vals:
        residual = val - np.mean(peer_vals)
        std = max(np.std(peer_vals), 1e-6)
        z = abs(residual)/std
        if z > SWAP_Z_SCORE:
            print(f"ALERT: residual z={z:.1f} for {sid} on {tag}; possible swap/calibration")
            record_event("silent_swap_suspect", sid, tag, z=z)
            quarantine_stream(sid)

def collect_peer_values(tag, exclude=None):
    vals = []
    for sid, buf in buffers.items():
        if sid == exclude: continue
        # assume we map sid->tag in real system; here approximate
        vals.extend(buf[-5:])
    return vals

def quarantine_stream(sid):
    # pragmatic containment: mark stream as degraded and require human ack
    print(f"QUARANTINE: {sid}")
    # log and reroute reads; in production, flip QoS and switch consensus rules

def record_event(kind, *args, **kw):
    # brief in-code logging; real system writes to tamper-evident store
    print("EVENT", kind, args, kw)

# Example usage
if __name__ == "__main__":
    # simulate messages
    on_message(json.dumps({"sensor_id":"S1","ts":time.time(),"tag":"temp","value":22.0}))
    time.sleep(0.1)
    on_message(json.dumps({"sensor_id":"S2","ts":time.time(),"tag":"temp","value":22.1}))
    # simulate drifted swap
    on_message(json.dumps({"sensor_id":"S1","ts":time.time(),"tag":"temp","value":40.0}))  # large residual