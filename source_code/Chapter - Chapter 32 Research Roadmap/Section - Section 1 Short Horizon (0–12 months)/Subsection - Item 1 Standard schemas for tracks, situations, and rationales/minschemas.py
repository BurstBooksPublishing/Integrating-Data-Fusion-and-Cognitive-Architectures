from dataclasses import dataclass, asdict
import json, math, time

# Minimal 2D matrix inverse for covariance (assumes invertible)
def inv2x2(m):
    a,b,c,d = m[0][0], m[0][1], m[1][0], m[1][1]
    det = a*d - b*c
    if abs(det) < 1e-12: raise ValueError("Singular covariance")
    inv = [[ d/det, -b/det], [-c/det, a/det]]
    return inv

def maha2(resid, S):
    S_inv = inv2x2(S)
    # compute r^T S^{-1} r
    return resid[0]*(S_inv[0][0]*resid[0] + S_inv[0][1]*resid[1]) + \
           resid[1]*(S_inv[1][0]*resid[0] + S_inv[1][1]*resid[1])

@dataclass
class Track:
    track_id: str
    timestamp: float
    state: list  # [x,y]
    covariance: list  # [[sx,sxy],[sxy,sy]]
    sensors: list
    status: str
    provenance: dict

@dataclass
class Situation:
    situation_id: str
    start: float
    end: float
    entities: list  # list of entity descriptors
    confidence: float
    supporting_tracks: list

@dataclass
class Rationale:
    rationale_id: str
    claim: str
    supporting_evidence: list
    rule_hits: list
    score: float
    author: str
    timestamp: float

# Example usage: build track, compute Mahalanobis with a measurement.
trk = Track("T-100", time.time(), [10.0, 5.0], [[2.0,0.1],[0.1,1.5]], ["radar-A"], "active", {"creator":"fusion-node"})
measurement = [9.5, 5.4]
resid = [measurement[0]-trk.state[0], measurement[1]-trk.state[1]]
d2 = maha2(resid, trk.covariance)  # gating decision
rat = Rationale("R-1", "assoc_ok" if d2<9.21 else "assoc_reject", [trk.track_id], ["maha_gate"], 1.0/(1+d2), "auto", time.time())
print(json.dumps(asdict(trk)))   # serialized track
print("mah", d2, "rationale", json.dumps(asdict(rat)))