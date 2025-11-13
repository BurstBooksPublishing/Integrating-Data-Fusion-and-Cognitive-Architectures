import json, requests, pandas as pd, datetime as dt
# load telemetry and audit logs (CSV/JSON) -- replace paths with real sources
telemetry = pd.read_csv("telemetry.csv", parse_dates=["ts"])  # sensor/fusion events
rationales = pd.read_json("rationales.json")                 # L3/L2 rationale traces
audit = pd.read_csv("audit_log.csv", parse_dates=["ts"])     # approvals and operator actions

# incident timestamp (from alert)
incident_ts = pd.Timestamp("2025-10-12T14:23:05Z")
window = pd.Timedelta(seconds=60)
evidence = {
  "telemetry": telemetry[(telemetry.ts >= incident_ts - window) & 
                        (telemetry.ts <= incident_ts + window)].to_dict(orient="records"),
  "rationales": rationales[rationales.ts.between(incident_ts - window, incident_ts + window)].to_dict(orient="records"),
  "audit": audit[audit.ts.between(incident_ts - window, incident_ts + window)].to_dict(orient="records"),
  "snapshot_ts": incident_ts.isoformat()
}

# simple severity scoring (weights tuned by governance)
def score_severity(evidence):
    impact = max([e.get("impact",0) for e in evidence["telemetry"]] + [0])
    likelihood = 1.0 if any(a.get("action")=="override" for a in evidence["audit"]) else 0.3
    reproducibility = 1.0 if any(t.get("error_rate",0)>0.2 for t in evidence["telemetry"]) else 0.2
    alpha,beta,gamma = 0.5,0.3,0.2
    return alpha*impact + beta*likelihood + gamma*reproducibility

sev = score_severity(evidence)
capa = {
  "title": "Incident CAPA draft",
  "severity": float(sev),
  "actions": [
    {"action":"Model retrain","owner":"ML_TEAM","due":(dt.datetime.utcnow()+dt.timedelta(days=7)).isoformat()},
    {"action":"Add operator confirm dialog","owner":"UX_TEAM","due":(dt.datetime.utcnow()+dt.timedelta(days=3)).isoformat()}
  ],
  "evidence_bundle": evidence
}

# post CAPA to ticketing webhook (mock endpoint)
resp = requests.post("https://ticketing.example/api/create", json=capa, timeout=10)
print("ticket_response", resp.status_code, resp.text)