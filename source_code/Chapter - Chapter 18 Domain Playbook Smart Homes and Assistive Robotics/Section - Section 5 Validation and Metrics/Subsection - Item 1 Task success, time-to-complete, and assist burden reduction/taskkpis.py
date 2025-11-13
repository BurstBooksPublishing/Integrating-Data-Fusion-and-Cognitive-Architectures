import pandas as pd
import numpy as np

# load CSV with columns: timestamp, task_id, user_id, event_type, success (bool), intervention (bool)
df = pd.read_csv("events.csv", parse_dates=["timestamp"])

# compute per-task records
start = df[df.event_type == "task_start"].set_index("task_id")
end = df[df.event_type == "task_end"].set_index("task_id")
tasks = start.join(end, lsuffix="_start", rsuffix="_end", how="inner")
tasks["TTC"] = (tasks["timestamp_end"] - tasks["timestamp_start"]).dt.total_seconds()
tasks["success"] = tasks["success_end"]  # success label at end event

# assist burden: count interventions per task
interv = df[df.intervention].groupby("task_id").size().rename("n_interventions")
tasks = tasks.join(interv, how="left").fillna({"n_interventions":0})

# compute KPIs
SR = tasks["success"].mean()
TTC_p50, TTC_p95, TTC_p99 = tasks["TTC"].quantile([0.5, 0.95, 0.99])
AB = tasks["n_interventions"].sum() / len(tasks)  # Eq. (1)

# composite assistance score (Eq. 2)
w1, w2, w3 = 0.5, 0.3, 0.2
Tref = 30.0
CAS = w1*SR + w2*np.exp(-tasks["TTC"].mean()/Tref) + w3*(1-AB)

print(f"SR={SR:.3f}, TTC_p95={TTC_p95:.1f}s, AB={AB:.2f}, CAS={CAS:.3f}")