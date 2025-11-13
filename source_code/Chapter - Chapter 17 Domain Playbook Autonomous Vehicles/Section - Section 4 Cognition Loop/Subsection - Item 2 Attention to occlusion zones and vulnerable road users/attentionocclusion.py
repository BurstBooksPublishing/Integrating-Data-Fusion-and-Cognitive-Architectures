# Minimal runnable example: pip install numpy
import numpy as np
from dataclasses import dataclass

@dataclass
class Track:
    id: int
    x: float; y: float; vx: float; vy: float
    cls: str  # 'pedestrian','cyclist','vehicle'
    cov: np.ndarray

# vulnerability mapping
VULN = {'pedestrian': 1.0, 'cyclist': 0.8, 'vehicle': 0.2}

def sample_occlusion_prob(occ_map, x, y):
    # occ_map: dict with resolution and numpy array; here we mock constant prob
    return occ_map.get('default', 0.0)

def predict_position(track, dt=1.0):
    return track.x + track.vx*dt, track.y + track.vy*dt

def attention_weights(tracks, occ_map, ego_path_fn, alpha=1.0, beta=5.0):
    logits = []
    for t in tracks:
        px, py = predict_position(t, dt=1.0)
        p_occ = sample_occlusion_prob(occ_map, px, py)         # [0,1]
        v = VULN.get(t.cls, 0.1)
        d = ego_path_fn(px, py)                               # lateral distance
        logits.append(beta * p_occ * v - alpha * d)
    logits = np.array(logits)
    # numerical stable softmax
    e = np.exp(logits - np.max(logits))
    weights = e / (e.sum() + 1e-12)
    return weights

def ego_path_distance_fn(px, py):
    # simple straight-line ego path at y=0
    return abs(py)

# demo
tracks = [Track(1, 10, 1.0, -0.5, 0.0, 'pedestrian', np.eye(2)),
          Track(2, 12, 2.0, 0.0, 0.0, 'vehicle', np.eye(2))]
occ_map = {'default': 0.7}  # high occlusion region
w = attention_weights(tracks, occ_map, ego_path_distance_fn)
for t, weight in zip(tracks, w):
    # simple sensor tasking: increase frame rate or ROI size proportional to weight
    fr_cmd = 30 + int(60 * weight)    # base 30 fps, up to +60
    roi_cmd = 100 + int(200 * weight)  # arbitrary ROI size units
    print(f"Track {t.id} ({t.cls}) weight={weight:.2f} -> fr={fr_cmd}, roi={roi_cmd}")