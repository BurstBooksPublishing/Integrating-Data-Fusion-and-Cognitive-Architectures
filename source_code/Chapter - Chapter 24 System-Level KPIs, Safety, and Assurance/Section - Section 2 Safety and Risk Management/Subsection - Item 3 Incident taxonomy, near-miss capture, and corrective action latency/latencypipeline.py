import pandas as pd
import numpy as np
from datetime import timedelta

# load telemetry/incident table (CSV columns: id, detect_ts, corr_ts, raw_label, severity)
df = pd.read_csv("incidents.csv", parse_dates=["detect_ts","corr_ts"])

# compute latency (seconds) per eq. (1)
df["latency_s"] = (df["corr_ts"] - df["detect_ts"]).dt.total_seconds().clip(lower=0)

# simple taxonomy enrichment: map raw_label to cause_class via dictionary
cause_map = {"camera_drop":"sensor","trk_loss":"algorithm","link_loss":"comms","operator_override":"human"}
df["cause_class"] = df["raw_label"].map(cause_map).fillna("unknown")

# rolling KPIs: MTCA and percentiles over last 90 days
window = df["detect_ts"].max() - pd.Timedelta(days=90)
recent = df[df["detect_ts"] >= window]
mtca = recent["latency_s"].mean()  # MTCA
p95 = recent["latency_s"].quantile(0.95)

# EMA for trend detection (alpha tuned to days->weight)
df = df.sort_values("detect_ts")
df["latency_ema"] = df["latency_s"].ewm(span=20).mean()

# SLA check and generate escalation record
SLA_P95_S = 600  # 10 minutes
if p95 > SLA_P95_S:
    # simple escalation action: write to alerts table (could call API)
    alert = {"time": pd.Timestamp.now(), "metric":"p95_latency_s", "value": p95}
    pd.DataFrame([alert]).to_csv("escalation_alerts.csv", mode="a", index=False)