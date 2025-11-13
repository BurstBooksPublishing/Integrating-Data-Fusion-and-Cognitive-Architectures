import time, math
from collections import deque
from multiprocessing import Process, Manager, Lock

def sensor_ingest(buf, lock, dt=0.1, n=50):
    for i in range(n):
        t = time.time()
        frame = {'id': i, 't': t, 'obs': (math.sin(i/5), math.cos(i/7))}
        with lock:  # atomic write into buffer
            if len(buf) >= buf.maxlen: buf.popleft()
            buf.append(frame)
        time.sleep(dt)

def fusion_node(buf, lock, world, audit, poll=0.2):
    while True:
        with lock:
            if not buf: frames = []
            else: frames = list(buf)
        if frames:
            # simple "track" = average of observations
            xs = [f['obs'][0] for f in frames]
            ys = [f['obs'][1] for f in frames]
            t = frames[-1]['t']
            track = {'centroid': (sum(xs)/len(xs), sum(ys)/len(ys)),
                     't': t, 'cov': (0.1,)}
            # atomic update of shared world and audit append
            world['track'] = track
            audit.append({'t': t, 'producer': 'fusion', 'track': track})
        time.sleep(poll)

def cognition_service(world, audit, decision_log, poll=0.3):
    while True:
        track = world.get('track')
        if track:
            x,y = track['centroid']
            # simple rule: if centroid near origin, command "observe-close"
            if math.hypot(x,y) < 0.5:
                decision_log.append({'t': time.time(), 'cmd': 'observe-close'})
        time.sleep(poll)

if __name__ == '__main__':
    mgr = Manager()
    lock = Lock()
    buf = mgr.list()  # Manager-backed list used as proxy for deque
    # wrap with deque in main process for maxlen semantics
    local_buf = deque(maxlen=20)
    shared_world = mgr.dict()
    audit = mgr.list()
    decisions = mgr.list()

    # adaptors to expose deque-like behavior through manager (simple demo)
    # run ingest/fusion/cognition with direct objects for simplicity
    p1 = Process(target=sensor_ingest, args=(local_buf, lock))
    p2 = Process(target=fusion_node, args=(local_buf, lock, shared_world, audit))
    p3 = Process(target=cognition_service, args=(shared_world, audit, decisions))

    p1.start(); p2.start(); p3.start()
    p1.join(timeout=10)
    time.sleep(2)  # allow pipeline to process remaining frames
    for proc in (p2,p3): proc.terminate()
    print('Audit length:', len(audit), 'Decisions:', list(decisions))