#!/usr/bin/env python3
# Read JSONL from sniff sidecar, align timestamps, produce timeline CSV.
import json, csv
from datetime import datetime
# simple per-node offset store (estimated externally).
node_offsets = {"camera_node": -0.032, "fusion_node": 0.005}  # seconds

def corrected_time(ev):
    # ev["ts"] is ISO8601; ev["node"] available.
    t = datetime.fromisoformat(ev["ts"]).timestamp()
    return t + node_offsets.get(ev["node"], 0.0)

def process(jsonl_path, out_csv):
    events = []
    with open(jsonl_path, "r") as fh:
        for line in fh:
            ev = json.loads(line)               # structured: ts,node,topic,seq,type,payload_digest
            ev["t_ref"] = corrected_time(ev)   # apply Eq. (1)
            events.append(ev)
    events.sort(key=lambda e: (e["t_ref"], e.get("seq",0)))
    with open(out_csv, "w", newline='') as csvf:
        w = csv.DictWriter(csvf, fieldnames=["t_ref","node","topic","seq","type","latency","payload_digest"])
        w.writeheader()
        for e in events:
            # latency: arrival - origin (if provided)
            latency = None
            if "origin_ts" in e:
                latency = corrected_time({"ts": e["origin_ts"], "node": e["node"]}) - e["t_ref"]
            w.writerow({"t_ref": e["t_ref"], "node": e["node"], "topic": e["topic"],
                        "seq": e.get("seq",""), "type": e.get("type",""),
                        "latency": latency, "payload_digest": e.get("payload_digest","")})
if __name__ == "__main__":
    process("sniff.jsonl","timeline.csv")