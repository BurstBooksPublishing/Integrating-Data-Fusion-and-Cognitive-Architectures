import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Track:
    id: int
    pos: np.ndarray   # [x,y]
    vel: np.ndarray   # [vx,vy]
    bbox: np.ndarray  # [w,h] (unused here)
    lane_id: int

def ttc(ego: Track, target: Track) -> float:
    r = target.pos - ego.pos
    v = target.vel - ego.vel
    vnorm2 = np.dot(v, v)
    if vnorm2 < 1e-6:  # relative stop
        return np.inf
    t = np.dot(r, v) / vnorm2
    return t if t > 0 else np.inf

def point_in_poly(pt: np.ndarray, poly: List[np.ndarray]) -> bool:
    # ray-casting algorithm for polygon membership (2D)
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0,y0 = poly[i]
        x1,y1 = poly[(i+1)%n]
        if ((y0>y) != (y1>y)) and (x < (x1-x0)*(y-y0)/(y1-y0)+x0):
            inside = not inside
    return inside

def run_oracle(ego: Track, tracks: List[Track], crosswalk_poly: List[np.ndarray],
               ttc_threshold: float=2.0) -> Dict:
    violations = []
    # TTC checks
    for tr in tracks:
        if tr.id == ego.id: continue
        t = ttc(ego, tr)
        if t < ttc_threshold:
            violations.append({'type':'TTC_violation','t':t,'track':tr.id})
    # Ontology/topology checks
    for tr in tracks:
        if point_in_poly(tr.pos, crosswalk_poly) and tr.lane_id != -1:
            # expect pedestrian lane_id == -1 (example rule)
            violations.append({'type':'crosswalk_role_mismatch','track':tr.id})
    label = 'PASS' if not violations else 'FAIL'
    return {'label':label,'violations':violations}

# Example run (simple synthetic)
ego = Track(0,np.array([0.0,0.0]),np.array([5.0,0.0]),np.array([2.0,1.0]),lane_id=1)
ped = Track(1,np.array([10.0,0.0]),np.array([-1.0,0.0]),np.array([0.5,0.5]),lane_id=2)
crosswalk = [np.array([9.5,-1.0]),np.array([9.5,1.0]),np.array([10.5,1.0]),np.array([10.5,-1.0])]
print(run_oracle(ego,[ped],crosswalk))  # emits TTC violation and role mismatch