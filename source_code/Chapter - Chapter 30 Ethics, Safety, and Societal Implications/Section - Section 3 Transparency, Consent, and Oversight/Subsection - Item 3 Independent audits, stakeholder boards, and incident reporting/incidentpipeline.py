import json, hashlib, hmac, time, requests
from datetime import datetime, timezone

SECRET = b'your_hmac_key'  # rotate and store securely

def hash_evidence(evidence: dict) -> str:
    payload = json.dumps(evidence, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()  # tamper-evident digest

def sign_report(report: dict) -> str:
    msg = json.dumps(report, sort_keys=True).encode()
    return hmac.new(SECRET, msg, hashlib.sha256).hexdigest()  # integrity tag

def assemble_and_submit(incident_id, evidence, severity_components):
    digest = hash_evidence(evidence)
    S = sum(severity_components.values()) / len(severity_components)  # simple score
    report = {
        "incident_id": incident_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evidence_digest": digest,
        "severity_score": S,
        "summary": "Automated triage",
    }
    report["signature"] = sign_report(report)
    # append to local append-only ledger
    with open("incident_ledger.log","a") as f:
        f.write(json.dumps(report)+"\n")
    # submit to stakeholder board endpoint (secure channel)
    resp = requests.post("https://oversight.example/submissions", json=report, timeout=10)
    resp.raise_for_status()
    return resp.json()
# end listing