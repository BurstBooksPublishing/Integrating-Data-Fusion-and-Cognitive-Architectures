from typing import List, Dict

# Task definition: id, utility, cpu, bw, energy
tasks = [
    {"id":"capture","u":1.0,"cpu":0.1,"bw":0.2,"en":0.01},
    {"id":"detect","u":5.0,"cpu":2.0,"bw":0.5,"en":0.2},
    {"id":"track","u":3.0,"cpu":0.5,"bw":0.1,"en":0.05},
    {"id":"rationale","u":8.0,"cpu":4.0,"bw":1.0,"en":0.5},
]

# Available budgets
C, B, E = 4.0, 2.0, 0.6

def schedule(tasks: List[Dict], C: float, B: float, E: float):
    # compute score = utility / (cpu + bw + energy_weighted)
    energy_weight = 2.0
    for t in tasks:
        t["score"] = t["u"] / (t["cpu"] + t["bw"] + energy_weight * t["en"])
    # sort by descending score
    tasks_sorted = sorted(tasks, key=lambda x: x["score"], reverse=True)
    chosen = []
    for t in tasks_sorted:
        if t["cpu"] <= C and t["bw"] <= B and t["en"] <= E:
            chosen.append(t["id"])
            C -= t["cpu"]; B -= t["bw"]; E -= t["en"]
    return chosen

print(schedule(tasks, C, B, E))  # example output: ['rationale','detect','track']