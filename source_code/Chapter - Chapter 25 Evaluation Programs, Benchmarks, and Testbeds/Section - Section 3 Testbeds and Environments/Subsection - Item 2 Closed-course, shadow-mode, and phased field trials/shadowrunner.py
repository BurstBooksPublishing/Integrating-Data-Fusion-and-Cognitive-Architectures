#!/usr/bin/env python3
import json, time, math
import numpy as np, pandas as pd

# Load telemetry: each row has timestamp, sensors, human_action
df = pd.read_json("golden_telemetry.json", lines=True)

def candidate_decision(frame):
    # Placeholder: run fusion->cognition (deterministic call in real use)
    # Here we simulate a decision score and action label.
    score = np.tanh(frame["confidence_feat"])  # mock scoring
    action = "brake" if score > 0.6 else "maintain"
    rationale = {"score": float(score), "rule_hits": ["prox_alert"] if score>0.6 else []}
    return action, rationale

records = []
for _, row in df.iterrows():
    action_sys, rationale = candidate_decision(row)
    action_human = row["human_action"]
    match = (action_sys == action_human)
    records.append({
        "t": row["timestamp"], "sys": action_sys, "human": action_human,
        "match": match, "rationale": rationale
    })

out = pd.DataFrame(records)
# metrics: agreement, confusion table
agree_rate = out["match"].mean()
confusion = pd.crosstab(out["human"], out["sys"], normalize="index")
out.to_json("shadow_comparisons.json", orient="records", lines=True)
print(f"Agreement: {agree_rate:.3f}\nConfusion:\n{confusion}")