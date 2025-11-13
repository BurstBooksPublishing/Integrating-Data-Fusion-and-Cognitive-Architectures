import numpy as np
from scipy.optimize import linear_sum_assignment

# Config (change for ablations)
np.random.seed(1)
frames = 100
n_targets = 5
scene_size = 100.0
v = 1.5                      # m/s
f = 10.0                     # Hz (frame rate)
tau = 0.05                   # seconds latency
lambda_c = 3.0               # expected clutter per frame
p_det = 0.95                 # detection probability
r_gate = 5.0                 # gating radius

dt = 1.0/f
# Generate true linear tracks (position, velocity)
pos0 = np.random.uniform(0, scene_size, (n_targets,2))
dirs = np.random.randn(n_targets,2); dirs /= np.linalg.norm(dirs,axis=1)[:,None]
vel = (dirs * v)             # constant velocity

# Simple latency buffer: hold detections for int(tau/dt) frames
latency_buffer = []
buffer_delay = int(round(tau/dt))

# Track persistence structure
next_track_id = 0
active_ids = {}  # persistent_id -> last_position

id_switches = 0
matches_total = 0

for t in range(frames):
    true_pos = pos0 + vel*(t*dt)
    detections = []
    # target detections
    for i in range(n_targets):
        if np.random.rand() < p_det:
            meas = true_pos[i] + np.random.randn(2)*0.5  # measurement noise
            detections.append({'pos':meas,'true_id':i})
    # clutter
    k = np.random.poisson(lambda_c)
    for _ in range(k):
        detections.append({'pos':np.random.uniform(0,scene_size,2),'true_id':None})
    latency_buffer.append(detections)
    # release detections only after buffer_delay
    if len(latency_buffer) <= buffer_delay:
        continue
    feed = latency_buffer.pop(0)
    # Association: cost matrix between active_ids and detections
    prev_ids = list(active_ids.keys())
    prev_pos = np.array([active_ids[i] for i in prev_ids]) if prev_ids else np.empty((0,2))
    det_pos = np.array([d['pos'] for d in feed]) if feed else np.empty((0,2))
    if prev_pos.size and det_pos.size:
        cost = np.linalg.norm(prev_pos[:,None,:]-det_pos[None,:,:],axis=2)
        # apply gating
        cost[cost>r_gate] = 1e6
        row,col = linear_sum_assignment(cost)
        assigned = {}
        for r,c in zip(row,col):
            if cost[r,c] < 1e5:
                pid = prev_ids[r]; assigned[pid]=feed[c]
                matches_total += 1
                # ID switch detection
                if 'assigned_true' in active_ids:
                    pass
        # update active positions and detect switches
        new_active = {}
        for pid,det in assigned.items():
            # find if detection's true_id was previously associated with different pid
            # simple heuristic: if same true_id mapped to different pid -> switch
            if det['true_id'] is not None:
                for other_pid,pos in active_ids.items():
                    if other_pid!=pid and np.linalg.norm(pos-det['pos'])<0.1:
                        id_switches += 1
            new_active[pid]=det['pos']
        # create new ids for unmatched detections
        unmatched_dets = [i for i in range(len(det_pos)) if i not in col]
        for idx in unmatched_dets:
            # spawn new persistent id
            new_active[next_track_id]=det_pos[idx]
            next_track_id += 1
        active_ids = new_active
    else:
        # initialize with detections
        for d in feed:
            active_ids[next_track_id]=d['pos']
            next_track_id += 1

print(f"Frames processed: {frames-buffer_delay}")
print(f"Matches total: {matches_total}")
print(f"ID switches (approx): {id_switches}")