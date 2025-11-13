# Simple WM as dict of facts; rules as callables returning (match,action,utility)
from typing import Dict, List, Callable, Any
import random

WM: Dict[str, Any] = {
    "track_A": {"pos": (0,0), "vel": (1,0), "conf": 0.6},   # fused track
    "track_B": {"pos": (1,0), "vel": (1,0), "conf": 0.5},
    "latency_ms": 120
}

def rule_propose_merge(wm):
    a, b = wm.get("track_A"), wm.get("track_B")
    if a and b and abs(a["pos"][0]-b["pos"][0])<2.0:
        # match with an estimated utility and an action
        def action(wm):  # action merges tracks in WM
            merged_conf = max(a["conf"], b["conf"]) * 0.9
            wm["track_AB"] = {"pos": ((a["pos"][0]+b["pos"][0])/2,0),
                              "vel": a["vel"], "conf": merged_conf}
            wm.pop("track_A", None); wm.pop("track_B", None)
        utility = 0.4  # estimated benefit (mission-specific)
        return True, action, utility
    return False, None, 0.0

def rule_request_highres(wm):
    # prefer when confidence low and latency acceptable
    low_conf = any(t["conf"]<0.7 for k,t in wm.items() if k.startswith("track_"))
    if low_conf and wm["latency_ms"]<200:
        def action(wm):
            wm["highres_requested"]=True  # placeholder for sensor tasking
        utility = 0.8
        return True, action, utility
    return False, None, 0.0

productions: List[Callable] = [rule_propose_merge, rule_request_highres]

def decision_cycle(wm, productions):
    # match all rules
    conflict = []
    for p in productions:
        match, action, u = p(wm)
        if match:
            conflict.append((action,u))
    if not conflict: return
    # select max-utility action
    action, _ = max(conflict, key=lambda x: x[1])
    action(wm)  # apply
    # record simple outcome (chunking/logging stub)
    wm.setdefault("history", []).append(("cycle", conflict))

# run one cycle
decision_cycle(WM, productions)
print(WM)  # inspect WM changes