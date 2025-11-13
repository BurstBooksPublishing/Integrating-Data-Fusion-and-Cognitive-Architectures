# Simple runtime allocator: tasks with multi-resource costs and utility
tasks = [                                          # id, util, comp, bw, energy, op
    {"id":"vis_hr",  "U":10.0, "C":[30, 50, 20], "W":0.5},  # HR camera
    {"id":"radar_ref","U":8.0,  "C":[20, 10, 10], "W":0.2},  # radar refine
    {"id":"llm_score","U":15.0, "C":[120,100,60], "W":1.0},  # LLM reasoning
]
budgets = {"comp":200, "bw":160, "energy":100, "op":2.0}

# compute utility-per-weight using weighted sum of normalized costs
def score(task, budgets):
    cost_norm = sum(c/b for c,b in zip(task["C"], [budgets["comp"], budgets["bw"], budgets["energy"]]))
    return task["U"] / (1e-6 + cost_norm + 5*task["W"])  # small op penalty

# greedy fractional allocation
alloc = {}
remaining = budgets.copy()
for t in sorted(tasks, key=lambda x: score(x, budgets), reverse=True):
    # determine max fraction allowed by each resource
    fracs = [
        remaining["comp"]/t["C"][0],
        remaining["bw"]/t["C"][1],
        remaining["energy"]/t["C"][2],
        remaining["op"]/t["W"]
    ]
    f = max(0.0, min(1.0, min(fracs)))  # fraction to allocate
    alloc[t["id"]] = f
    # subtract consumption
    remaining["comp"]  -= f*t["C"][0]
    remaining["bw"]    -= f*t["C"][1]
    remaining["energy"]-= f*t["C"][2]
    remaining["op"]    -= f*t["W"]
print(alloc)  # runtime decision map