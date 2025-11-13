#!/usr/bin/env python3
import json, hmac, hashlib, datetime, os

KEY = b'supersecret-key'               # HMAC key (use KMS in production)
registry_dir = 'artifact_registry'     # local registry for demo
os.makedirs(registry_dir, exist_ok=True)

# Example metrics card with provenance and caveats
card = {
  "run_id": "run-20251107-001",
  "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
  "provenance": {"git": "git://repo/commit/abc123", "env": "ubuntu-22.04"},
  "metrics": {"accuracy": 0.912, "accuracy_ci": [0.901, 0.923], "ece": 0.03},
  "caveats": ["underrepresented rain scenarios", "sensor latency >150ms degrades recall"],
  "artifact_uri": "s3://bucket/artifacts/run-20251107-001.tar.gz"
}

card_bytes = json.dumps(card, sort_keys=True).encode('utf-8')
digest = hashlib.sha256(card_bytes).hexdigest()         # artifact digest
signature = hmac.new(KEY, digest.encode('utf-8'), hashlib.sha256).hexdigest()

record = {"card": card, "digest": digest, "signature": signature}
out_path = os.path.join(registry_dir, f"{card['run_id']}.json")
with open(out_path, "w") as f:
    json.dump(record, f, indent=2)
# End: registry now contains a signed, auditable record.