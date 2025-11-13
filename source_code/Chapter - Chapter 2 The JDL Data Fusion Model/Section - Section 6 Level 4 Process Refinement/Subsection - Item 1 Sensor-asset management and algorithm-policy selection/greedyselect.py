#!/usr/bin/env python3
# Simple greedy selector illustrating L4 selection loop.

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Action:
    name: str
    exp_info: float    # expected information gain / utility
    bandwidth: float   # MB/s
    energy: float      # joules/sec

# Candidate actions (sensor modes + algorithm variants)
candidates = [
    Action("cam_highres+NNv2", exp_info=8.0, bandwidth=5.0, energy=15.0),
    Action("cam_lowres+NNv1",  exp_info=4.0, bandwidth=1.5, energy=5.0),
    Action("radar_surface",    exp_info=6.5, bandwidth=2.0, energy=12.0),
    Action("lidar_dense",      exp_info=5.0, bandwidth=3.5, energy=20.0),
    Action("reid_model",       exp_info=3.0, bandwidth=0.5, energy=3.0),
]

BUDGET_BW = 8.0   # MB/s total available
BUDGET_EN = 30.0  # joules/sec total available

# Greedy by info per weighted cost (normalize bandwidth+energy)
def select_greedy(actions: List[Action], bw_budget: float, en_budget: float):
    # compute score = info / (alpha*bw + beta*energy)
    alpha, beta = 1.0, 0.05  # tuning: relative weight of bandwidth vs energy
    scored = sorted(actions,
                    key=lambda a: a.exp_info/(alpha*a.bandwidth + beta*a.energy),
                    reverse=True)
    chosen, bw_used, en_used = [], 0.0, 0.0
    for a in scored:
        if bw_used + a.bandwidth <= bw_budget and en_used + a.energy <= en_budget:
            chosen.append(a); bw_used += a.bandwidth; en_used += a.energy
    return chosen, bw_used, en_used

chosen, bw, en = select_greedy(candidates, BUDGET_BW, BUDGET_EN)
print("Selected actions:", [a.name for a in chosen])
print(f"Bandwidth used: {bw:.1f} / {BUDGET_BW}, Energy used: {en:.1f} / {BUDGET_EN}")
# In practice, chosen actions are then gated by safety checks and issued as commands.