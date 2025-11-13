from dataclasses import dataclass
from typing import List

@dataclass
class Operator:
    name: str
    level: int            # 0..4
    latency_edge: float   # ms
    latency_cloud: float  # ms
    cost_edge: float      # $/hr
    cost_cloud: float     # $/hr
    user_ref_rate: float  # edits/hour
    privacy_sensitive: bool

def place_operators(ops: List[Operator], latency_slo_ms: float):
    placement = {}
    for o in ops:
        # hard privacy constraint: keep sensitive data on edge
        if o.privacy_sensitive:
            placement[o.name] = "edge"; continue
        # prefer cloud if cheaper and within latency SLO, but account for user edits
        cloud_ok = o.latency_cloud <= latency_slo_ms
        # promote to edge if many user refinements to reduce RTT
        if o.user_ref_rate > 1.0:  # >1 edit/hour heuristic
            placement[o.name] = "edge"
        elif cloud_ok and o.cost_cloud < o.cost_edge:
            placement[o.name] = "cloud"
        else:
            placement[o.name] = "edge"
    return placement

# Example usage (executable)
ops = [
    Operator("detection",0,5,50,0.5,0.2,0.1,True),
    Operator("tracker",1,10,80,0.6,0.25,0.2,False),
    Operator("situation",2,30,120,1.0,0.4,2.5,False),  # many refinements
]
print(place_operators(ops, latency_slo_ms=100))  # prints mapping