#!/usr/bin/env python3
import json, socket, time, hashlib
# load recorded trace (list of timestamped frames)
with open("golden_trace.json","r") as f:
    trace = json.load(f)  # [{'t':0.0,'payload':...}, ...]
# simple adversarial primitive: drop frames with probability p, add delay d
def apply_primitive(frame, p_drop=0.1, delay_sec=0.0):
    import random
    if random.random() < p_drop:
        return None   # drop
    frame['injected_delay'] = delay_sec
    return frame
# provenance stamp
def stamp(frame, run_id):
    h = hashlib.sha256(json.dumps(frame, sort_keys=True).encode()).hexdigest()
    frame['run_id'] = run_id
    frame['digest'] = h
    return frame
# publisher socket (local test endpoint)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dst = ("127.0.0.1", 9000)
run_id = "run-"+hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
# real-time replay loop
start = time.time()
for f in trace:
    # wait until scheduled time
    target = start + f['t']
    now = time.time()
    if target > now:
        time.sleep(target - now)
    f2 = apply_primitive(f.copy(), p_drop=0.2, delay_sec=0.05)  # adversarial params
    if f2 is None: 
        continue  # frame dropped
    if f2.get('injected_delay',0.0):
        time.sleep(f2['injected_delay'])  # simulate latency
    stamped = stamp(f2, run_id)
    sock.sendto(json.dumps(stamped).encode(), dst)  # publish
sock.close()