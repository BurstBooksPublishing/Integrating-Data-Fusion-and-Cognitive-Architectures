#!/usr/bin/env python3
import json, hmac, hashlib, time, os
# secret key for HMAC (demo only)
SECRET = b"demo-key"

def redact(payload):
    # redact PII fields deterministically
    for k in ("user_email","device_id"):
        if k in payload: payload[k] = ""
    return payload

def severity_score(Ep, Es, Ef, wp=1.0, ws=3.0, wf=1.0):
    # implements equation (normalized weighted sum)
    num = wp*Ep + ws*Es + wf*Ef
    den = wp + ws + wf
    return num/den

def sign_blob(blob_bytes):
    return hmac.new(SECRET, blob_bytes, hashlib.sha256).hexdigest()

def publish_incident(incident, out_dir="registry"):
    incident = redact(dict(incident))                   # PII minimization
    incident["severity"] = severity_score(
        incident.get("E_p",0), incident.get("E_s",0), incident.get("E_f",0))
    incident["timestamp"] = time.time()
    payload = json.dumps(incident, sort_keys=True).encode("utf-8")
    signature = sign_blob(payload)                      # attestation
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"incident-{int(incident['timestamp'])}.json")
    with open(fname, "wb") as f:
        f.write(payload)
    with open(fname + ".sig", "w") as f:
        f.write(signature)
    # produce human-readable digest (brief)
    digest = f"- id: {os.path.basename(fname)} severity: {incident['severity']:.2f}\n"
    with open(os.path.join(out_dir, "digest.md"), "a") as f:
        f.write(digest)
    return fname, fname + ".sig"

if __name__ == "__main__":
    demo = {"summary":"Track swap during occlusion","E_p":0.4,"E_s":0.7,"E_f":0.0,
            "user_email":"ops@example.com","device_id":"dev-42","evidence":["/snap/1.bin"]}
    print(publish_incident(demo))