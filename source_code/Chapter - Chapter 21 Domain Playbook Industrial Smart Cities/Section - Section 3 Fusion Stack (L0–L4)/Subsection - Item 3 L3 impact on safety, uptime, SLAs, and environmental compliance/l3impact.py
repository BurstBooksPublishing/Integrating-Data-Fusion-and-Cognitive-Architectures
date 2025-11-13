import numpy as np, json, datetime
# simple repair-time model per asset (hours)
repair_dists = {"pumpA": lambda n: np.random.lognormal(mean=1.0, sigma=0.5, size=n)}
def expected_repair(asset, samples=10000):
    return float(np.mean(repair_dists[asset](samples)))  # Monte Carlo estimate
def score_impact(hypothesis, assets):
    p = hypothesis["prob"]                    # posterior prob from L2
    items=[]
    for a in assets:
        eT = expected_repair(a)               # E[T|H]
        expected_downtime = p * eT
        sla_risk = p * max(0.0, eT - hypothesis["sla_thresh"] + hypothesis["latency"])
        items.append({"asset":a,"exp_down":expected_downtime,"sla_risk":sla_risk})
    return items
# example input from fusion/cognition
H = {"id":"H42","prob":0.12,"sla_thresh":2.0,"latency":0.5}
assets = ["pumpA"]
impact = score_impact(H, assets)
# create work order with provenance
work_order = {"created":datetime.datetime.utcnow().isoformat()+"Z",
              "hypothesis_id":H["id"],
              "impact":impact,
              "rationale":"automated L3 scoring",
              "audit_tag":"trace_v1"}
print(json.dumps(work_order,indent=2))