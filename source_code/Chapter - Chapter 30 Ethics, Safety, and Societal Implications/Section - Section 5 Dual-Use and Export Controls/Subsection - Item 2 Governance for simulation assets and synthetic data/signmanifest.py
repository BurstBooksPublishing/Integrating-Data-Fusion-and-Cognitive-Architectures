import json, hmac, hashlib, base64, time

# Example asset metadata (would come from CI pipeline)
manifest = {
    "asset_id": "maritime_sim_v1",
    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "domain": "maritime",
    "fidelity": 0.92,   # F_i
    "sensitivity": 0.75,# S_i
    "accessibility": 0.6,# A_i
    "generator_seed": 123456789,
    "dependencies": ["sensor_model_v3", "wave_model_v1"]
}

# Policy weights (alpha,beta,gamma) from governance config
alpha, beta, gamma = 0.5, 0.3, 0.2

def risk_score(m):
    return alpha*m["sensitivity"] + beta*m["fidelity"] + gamma*m["accessibility"]

# sign manifest with HMAC for tamper-evidence (replace with PKI in prod)
SECRET = b"supersecretkey"  # use KMS in production
payload = json.dumps(manifest, sort_keys=True).encode()
sig = hmac.new(SECRET, payload, hashlib.sha256).digest()
manifest["signature"] = base64.b64encode(sig).decode()
manifest["risk"] = risk_score(manifest)

# simple gate: require human review if risk > 0.7 or ECCN present
def access_allowed(user_attrs, manifest, threshold=0.7):
    if manifest["risk"] > threshold:
        return False  # escalate for review
    if user_attrs.get("region") not in ["US", "EU"] and manifest["domain"]=="maritime":
        return False  # geofence example
    return True

# Example usage
print(json.dumps(manifest, indent=2))
print("Access allowed for user in CN:", access_allowed({"region":"CN"}, manifest))