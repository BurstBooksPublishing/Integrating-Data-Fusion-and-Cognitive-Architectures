import jsonschema, json, requests, datetime
# load canonical schema and a candidate message
schema = json.load(open("canonical_track_schema.json"))
msg = json.load(open("candidate_track.json"))

# validate message (raises on failure)
jsonschema.validate(instance=msg, schema=schema)

# attach signed provenance and timestamp (simple example)
msg["provenance"]["validated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
# push to artifact registry (POST to in-house registry API)
resp = requests.post("https://registry.local/artifacts", json=msg, timeout=10)
resp.raise_for_status()  # fail loudly for CI gating
print("Registered artifact id:", resp.json()["artifact_id"])