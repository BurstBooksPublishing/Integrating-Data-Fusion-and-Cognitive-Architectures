import time, collections, json
from statistics import mean

THRESHOLD = 100.0            # regulatory threshold (example)
QUORUM = 2                   # number of sensors needed
PERSIST_WINDOW = 5           # seconds for persistence
HYSTERESIS = 0.95            # release multiplier

# sliding buffer of (ts, sensor_id, value)
buffer = collections.deque()

def ingest(record):
    buffer.append((record['ts'], record['sensor'], record['value']))
    # drop old
    cutoff = time.time() - PERSIST_WINDOW
    while buffer and buffer[0][0] < cutoff:
        buffer.popleft()

def quorum_exceeded():
    latest = {}
    for ts, sid, val in buffer: latest[sid] = (ts, val)
    vals = [v for (_, v) in latest.values()]
    # quorum requires QUORUM distinct sensors above THRESHOLD
    return sum(1 for v in vals if v > THRESHOLD) >= QUORUM

# store active event state
active_event = None

def process_stream(record):
    global active_event
    ingest(record)
    if active_event is None and quorum_exceeded():
        active_event = {'t_detect': time.time(), 'evidence': list(buffer)}
        # dispatch mitigation (placeholder)
        dispatch_mitigation(active_event)
    elif active_event is not None:
        # check release: all sensors below hysteresis*THRESHOLD
        latest_vals = [v for (_, _, v) in buffer]
        if latest_vals and max(latest_vals) < HYSTERESIS * THRESHOLD:
            active_event['t_mitigate'] = time.time()
            log_event(active_event)    # compute TTM and store metrics
            active_event = None

def dispatch_mitigation(evt):
    # real system would call actuators, record provenance, and hand to cognitive planner
    print("Mitigation dispatched at", evt['t_detect'])

def log_event(evt):
    ttm = evt['t_mitigate'] - evt['t_detect']
    print(json.dumps({'t_detect':evt['t_detect'],'t_mitigate':evt['t_mitigate'],'ttm':ttm}))
    # write to telemetry, attach evidence graph, signed attestations