import json, hashlib, time, hmac

# simple provenance builder
def make_prov(dataset_id, sources, transforms, actors):
    return {
        "dataset_id": dataset_id,
        "timestamp": time.time(),
        "sources": sources,       # list of source dicts
        "transforms": transforms, # list of transform dicts
        "actors": actors          # list of actor dicts
    }

# constraint checks (example rules)
def run_checks(records):
    violations = 0
    total = len(records)
    for r in records:
        # rule: consent required for identifiable imagery
        if r.get("contains_identifiable") and not r.get("consent_flag"):
            violations += 1
        # rule: geo-restricted flag
        if r.get("geo") and r["geo"] in {"restricted_region"}:
            violations += 1
    return violations, total

# compute assurance score per Eq. (1)
def assurance_score(p_cov, c_rate, e_curr, weights=(0.4,0.4,0.2)):
    wp,wc,we = weights
    return wp*p_cov + wc*c_rate + we*e_curr

# sign manifest (HMAC for demo)
def sign_manifest(manifest_json, key=b"secret"):
    h = hmac.new(key, manifest_json.encode("utf-8"), hashlib.sha256)
    return h.hexdigest()

# example run
records = [{"contains_identifiable":True,"consent_flag":False,"geo":"open"},
           {"contains_identifiable":False,"consent_flag":False,"geo":"open"}]
prov = make_prov("maritime-2025", sources=[{"sensor":"SAR","fw":"v1.2"}],
                 transforms=[{"name":"ortho","params":{"res":0.5}}], actors=[{"team":"fusion"}])
violations, total = run_checks(records)
p_cov = 1.0  # assume all records have lineage for demo
c_rate = 1.0 - (violations/total)
e_curr = 1.0  # ethics review current
S = assurance_score(p_cov, c_rate, e_curr)
manifest = {"prov":prov, "records": total, "violations": violations, "assurance": S}
manifest_json = json.dumps(manifest, sort_keys=True)
sig = sign_manifest(manifest_json)
# simulate publish to artifact registry (placeholder function)
print("manifest_signature:", sig)  # registry would store manifest+signature