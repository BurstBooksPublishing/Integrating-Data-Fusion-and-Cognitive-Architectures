import json
# sample input: situation, impact, resource budgets
situation = {"zone":"A","severity":0.8}
candidates = [  # actions with utility, cost (bandwidth, crew), safety flag
    {"id":"ptz_cam_A","utility":5.0,"cost":0.1,"crew":0,"safety":True},
    {"id":"slow_conveyor","utility":8.0,"cost":0.5,"crew":0,"safety":True},
    {"id":"dispatch_team","utility":12.0,"cost":1.2,"crew":2,"safety":False},
]
budgets = {"bandwidth":1.0,"crew":2}
# filter unsafe actions and compute value density
feasible = [a for a in candidates if a["safety"]]
for a in feasible:
    a["density"] = a["utility"] / (a["cost"] + 1e-6)
# greedy select by density under budgets
feasible.sort(key=lambda x: x["density"], reverse=True)
selected, used = [], {"bandwidth":0.0,"crew":0}
for a in feasible:
    if used["bandwidth"] + a["cost"] <= budgets["bandwidth"] and \
       used["crew"] + a["crew"] <= budgets["crew"]:
        selected.append(a)
        used["bandwidth"] += a["cost"]
        used["crew"] += a["crew"]
# emit simple work orders / tasking commands
work_orders = []
for a in selected:
    if a["id"].startswith("ptz"):
        work_orders.append({"type":"sensor_task","target":a["id"],"params":{"roi":"hotspot"}})
    else:
        work_orders.append({"type":"control","target":a["id"],"params":{"level":"reduced"}})
print(json.dumps({"situation":situation,"work_orders":work_orders}, indent=2))
# Note: replace greedy with solver for near-optimal allocation in production.