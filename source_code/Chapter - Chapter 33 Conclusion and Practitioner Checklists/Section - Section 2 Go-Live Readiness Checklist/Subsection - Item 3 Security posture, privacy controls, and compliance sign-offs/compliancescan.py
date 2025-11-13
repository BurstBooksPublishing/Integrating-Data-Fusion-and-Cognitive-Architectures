#!/usr/bin/env python3
import json, os, sys, subprocess
# Simple checks: KMS key policy, SBOM presence, log retention config
def check_file(path):
    return os.path.exists(path)

config = json.load(open("deployment_manifest.json"))  # includes artifacts, retention
# check KMS key exists (calls cloud CLI) -- replace with actual provider
key_ok = subprocess.call(["gcloud","kms","keys","describe",config["kms_key"]])==0
sbom_ok = check_file(config["sbom_path"])  # SBOM present
logs_ok = config.get("log_retention_days",0) >= 365  # retention policy
# output minimal compliance report
report = {"kms_ok":key_ok, "sbom_ok":sbom_ok, "log_retention_ok":logs_ok}
print(json.dumps(report,indent=2))
sys.exit(0 if all(report.values()) else 2)