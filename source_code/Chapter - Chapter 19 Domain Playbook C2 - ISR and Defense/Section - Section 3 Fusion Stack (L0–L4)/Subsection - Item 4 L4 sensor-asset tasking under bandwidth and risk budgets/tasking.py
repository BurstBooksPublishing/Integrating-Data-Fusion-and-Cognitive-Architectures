#!/usr/bin/env python3
# Simple scheduler: choose tasks by utility density, enforce budgets, emit audit.
from dataclasses import dataclass
from typing import List
import json

@dataclass
class Task:
    id: str
    util: float      # expected utility from L3/L2 fusion
    bandwidth: float # bandwidth cost (MB/s)
    risk: float      # risk cost (normalized 0..1)
    feasible: bool = True

def schedule(tasks: List[Task], B: float, R: float):
    # compute density and sort
    tasks = [t for t in tasks if t.feasible]
    tasks.sort(key=lambda t: t.util / (t.bandwidth + 1e-6), reverse=True)
    selected, b_used, r_used = [], 0.0, 0.0
    audit = {"selected": [], "rejected": []}
    for t in tasks:
        if b_used + t.bandwidth <= B and r_used + t.risk <= R:
            selected.append(t); b_used += t.bandwidth; r_used += t.risk
            audit["selected"].append({"id": t.id, "util": t.util,
                                      "bandwidth": t.bandwidth, "risk": t.risk})
        else:
            audit["rejected"].append({"id": t.id, "reason": "budget_exceeded"})
    # log provenance for assurance
    print(json.dumps({"bandwidth_budget": B, "risk_budget": R,
                      "bandwidth_used": b_used, "risk_used": r_used,
                      "audit": audit}, indent=2))
    return selected

if __name__ == "__main__":
    # sample tasks from L2/L3 fusion outputs
    tasks = [
        Task("T01", util=8.5, bandwidth=2.0, risk=0.05),
        Task("T02", util=6.0, bandwidth=1.5, risk=0.20),
        Task("T03", util=4.2, bandwidth=0.5, risk=0.01),
        Task("T04", util=9.0, bandwidth=3.5, risk=0.50),
    ]
    schedule(tasks, B=5.0, R=0.6)  # budgets set by higher-level policy