#!/usr/bin/env python3
import csv, datetime, collections, math, json
# Input CSV expected columns: start_time,detect_time,resolve_time,severity,runbook_id,automated
# Times in ISO8601 UTC.

def parse_iso(s): return datetime.datetime.fromisoformat(s.replace("Z","+00:00"))

incidents=[]
with open("incidents.csv") as f:
    r=csv.DictReader(f)
    for row in r:
        incidents.append({
            "start": parse_iso(row["start_time"]),
            "detect": parse_iso(row["detect_time"]),
            "resolve": parse_iso(row["resolve_time"]),
            "severity": int(row["severity"]),
            "runbook": row["runbook_id"],
            "automated": row["automated"].lower() in ("1","true","yes")
        })

N=len(incidents)
if N==0:
    raise SystemExit("No incidents found.")

# compute MTTD and MTTR (seconds)
mttd = sum((i["detect"]-i["start"]).total_seconds() for i in incidents)/N
mttr = sum((i["resolve"]-i["detect"]).total_seconds() for i in incidents)/N

# runbook statistics
rb=collections.defaultdict(lambda: {"count":0,"auto":0,"total_time":0.0})
for i in incidents:
    rb[i["runbook"]]["count"]+=1
    rb[i["runbook"]]["auto"]+=1 if i["automated"] else 0
    rb[i["runbook"]]["total_time"] += (i["resolve"]-i["detect"]).total_seconds()

# maturity score components
C = sum(1 for r in rb if rb[r]["count"]>0) / max(1,len(rb))  # simplistic checklist presence
A = sum(rb[r]["auto"] for r in rb) / sum(rb[r]["count"] for r in rb)  # automation fraction
E = 1.0  # assume evidence capture measured elsewhere; placeholder
w_c,w_a,w_e = 0.4,0.5,0.1
M = w_c*C + w_a*A + w_e*E

# prioritize candidate runbooks: high count, low automation, high avg MTTR
candidates = []
for r,s in rb.items():
    avg_mttr = s["total_time"]/s["count"]
    score = s["count"] * (1 - s["auto"]/s["count"]) * avg_mttr
    candidates.append((score,r,s["count"],s["auto"],avg_mttr))
candidates.sort(reverse=True)

output = {
    "MTTD_s": mttd, "MTTR_s": mttr, "maturity": M,
    "top_candidates":[{"runbook":c[1],"count":c[2],"automated":c[3],"avg_mttr_s":c[4]} for c in candidates[:5]]
}
print(json.dumps(output, indent=2))
# Integrate with alerting or ticketing systems to schedule automation work.