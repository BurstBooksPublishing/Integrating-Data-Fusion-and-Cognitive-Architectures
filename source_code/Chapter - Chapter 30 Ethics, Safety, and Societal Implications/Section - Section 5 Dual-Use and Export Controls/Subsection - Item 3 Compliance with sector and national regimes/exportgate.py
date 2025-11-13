#!/usr/bin/env python3
"""Check artifact metadata against policy and return CI exit code."""
import json
import sys

# Load artifact metadata and policy (CI provides paths).
artifact_path = sys.argv[1]  # artifact metadata JSON
policy_path = sys.argv[2]    # control policy JSON

with open(artifact_path) as f:
    meta = json.load(f)      # contains fields: id, capabilities, tags, jurisdictions

with open(policy_path) as f:
    policy = json.load(f)    # contains control_lists and rules

# Simple rule: if capability in control list, flag license_required.
def assess(meta, policy):
    caps = set(meta.get("capabilities", []))
    for ctrl, items in policy.get("control_lists", {}).items():
        if caps & set(items):
            return {"decision": "license_required", "reason": f"matched {ctrl}"}
    # check jurisdiction restrictions
    if meta.get("deployment_region") in policy.get("restricted_regions", []):
        return {"decision": "restricted_region", "reason": "region restricted"}
    return {"decision": "allowed", "reason": "no match"}

result = assess(meta, policy)
print(json.dumps(result))
# exit codes for CI: 0 allow, 2 require legal review, 1 block
if result["decision"] == "allowed":
    sys.exit(0)
if result["decision"] == "license_required":
    sys.exit(2)
sys.exit(1)