import math, json

# unit prices (example)
UNIT = {"cpu_hr": 0.05, "mem_gb_hr": 0.01, "net_gb": 0.02, "storage_gb_day": 0.001}

def cost_per_scenario(telemetry, pricing=UNIT, shared_amort=0.0, alloc_weight=1):
    # telemetry: dict with usage keys; includes 'p_error' and 'utility'
    # compute resource cost
    rcost = (telemetry.get("cpu_hr",0)*pricing["cpu_hr"]
             + telemetry.get("mem_gb_hr",0)*pricing["mem_gb_hr"]
             + telemetry.get("net_gb",0)*pricing["net_gb"]
             + telemetry.get("storage_gb_day",0)*pricing["storage_gb_day"])
    # expected penalty for errors
    penalty = telemetry.get("p_error",0.0) * telemetry.get("loss_if_error",0.0)
    # amortized shared cost
    amort = shared_amort / max(1, alloc_weight)
    total = rcost + penalty + amort
    return total

def efficiency_scorecard(telemetry, pricing=UNIT, shared_amort=0.0, alloc_weight=1):
    c = cost_per_scenario(telemetry, pricing, shared_amort, alloc_weight)
    u = telemetry.get("utility", 0.0)
    # safety guard: force negative infinity if critical SLO violated
    if telemetry.get("slo_violation", False):
        return {"cost": c, "efficiency": -math.inf, "reason": "SLO_violation"}
    eff = (u - c) / c if c > 0 else math.inf
    return {"cost": c, "efficiency": eff, "p_error": telemetry.get("p_error",0.0)}
# Example usage (telemetry from pipeline)
tele = {"cpu_hr":0.2,"mem_gb_hr":0.5,"net_gb":0.1,"storage_gb_day":0.0,
        "p_error":0.01,"loss_if_error":100,"utility":10,"slo_violation":False}
print(json.dumps(efficiency_scorecard(tele), indent=2))