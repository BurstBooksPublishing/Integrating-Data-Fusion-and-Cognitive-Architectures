import json, datetime, hashlib, statistics, hmac

# Example inputs (CI should load real values)
metric_name = "track_precision"               # human-friendly
mu_ref = 0.92                                 # baseline mean from golden trace
sigma_ref = 0.02                              # robust spread
m_run = 0.88                                  # measured on current run
owner = {"name":"Alice","role":"MetricOwner","email":"alice@example.com"}
dataset_id = "golden_trace:v3:seed=42"

# compute z and status (simple rule)
z = (m_run - mu_ref) / sigma_ref
if abs(z) <= 1:
    status = "GREEN"
elif abs(z) <= 2:
    status = "AMBER"
else:
    status = "RED"

card = {
  "id": f"card-{metric_name}-v1",
  "metric_name": metric_name,
  "definition": "Precision of confirmed tracks over reference window",
  "unit": "fraction",
  "evaluation_dataset": dataset_id,
  "baseline": {"mu_ref": mu_ref, "sigma_ref": sigma_ref},
  "measurement": {"value": m_run, "z_score": z, "status": status},
  "caveats": ["labels reviewed: 2025-06-01; known bias for occluded tracks"],
  "owner": owner,
  "provenance": {"run_id":"ci/2025/123", "timestamp": datetime.datetime.utcnow().isoformat()}
}

# sign-off (HMAC used here as illustrative signature)
secret = b"ci-signing-key"                     # CI key storage (rotate!)
card_bytes = json.dumps(card, sort_keys=True).encode()
card["owner"]["signature"] = hmac.new(secret, card_bytes, "sha256").hexdigest()

print(json.dumps(card, indent=2))              # persist artifact to registry