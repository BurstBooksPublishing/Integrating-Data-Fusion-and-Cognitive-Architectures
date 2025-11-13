import pandas as pd, numpy as np
# logs: one row per detection/association event
logs = pd.read_csv("fusion_logs.csv", parse_dates=["ts_detection","ts_cue"])
# compute per-track purity (fraction of correct associations)
track_purity = logs.groupby("track_id").apply(
    lambda df: df["is_correct"].mean()).rename("purity").reset_index()
# compute latency-to-cue per event (seconds)
logs["latency_sec"] = (logs["ts_cue"] - logs["ts_detection"]).dt.total_seconds()
# join track-level purity back to mission events
events = logs.drop_duplicates("event_id").merge(track_purity, on="track_id")
# simple A/B uplift: mean mission success difference
agg = events.groupby("treatment")["mission_success"].mean()
uplift = agg.loc[1] - agg.loc[0]  # Equation (1) empirical
# bootstrap CI
rng = np.random.default_rng(42)
boots = []
for _ in range(1000):
    sample = events.sample(frac=1, replace=True, random_state=rng)
    m = sample.groupby("treatment")["mission_success"].mean()
    boots.append(m.loc[1] - m.loc[0])
ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
print(f"Uplift={uplift:.4f}, 95% CI=({ci_low:.4f},{ci_high:.4f})")