import math, random, json
random.seed(0)

# Minimal schema: keep only attributes required for fusion decision.
MIN_SCHEMA = {"timestamp","position","object_type","confidence"}  # allowed fields

# Residency rules: map dataset_tag->region
RESIDENCY = {"europe_sales":"eu", "testing":"us"}

# Per-actor exposure budgets (epsilon remaining)
epsilon_budget = {"analyst_A":1.0, "ops_console":0.25}

def minimize_record(record):
    # keep only fields in MIN_SCHEMA; add provenance note
    minimized = {k:record[k] for k in record if k in MIN_SCHEMA}
    minimized["_prov"]= {"removed": [k for k in record if k not in MIN_SCHEMA]}
    return minimized

def route_residency(record):
    tag = record.get("dataset_tag","testing")
    return RESIDENCY.get(tag,"us")

def laplace_noise(scale):
    # sample Laplace(0, scale)
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2*abs(u))

def expose_scalar(value, sensitivity, actor):
    eps = epsilon_budget.get(actor, 0.0)
    if eps <= 0:
        raise PermissionError("Exposure budget exhausted")
    scale = sensitivity/eps
    noise = laplace_noise(scale)
    # consume full budget for simplicity
    epsilon_budget[actor] = 0.0
    return value + noise

# Demo record
raw = {"timestamp":"2025-10-01T12:00:00Z","position":[12.3,45.6],
       "object_type":"vehicle","confidence":0.92,"owner_name":"Alice",
       "dataset_tag":"europe_sales"}

minrec = minimize_record(raw)           # PII minimization
region = route_residency(raw)           # residency decision
try:
    safe_conf = expose_scalar(minrec["confidence"], sensitivity=1.0, actor="analyst_A")
except PermissionError as e:
    safe_conf = None

print(json.dumps({"minimized":minrec,"region":region,"safe_confidence":safe_conf,
                  "budgets":epsilon_budget}, indent=2))