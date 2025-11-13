import numpy as np, pandas as pd
# load event log with columns: time, mode (auto/manual), infraction, comfort_acc, event_gt, event_detected, weight
df = pd.read_csv("trace_log.csv", parse_dates=["time"])
# disengagement rate per 1000 driving-hours
drive_hours = (df.time.max() - df.time.min()).total_seconds()/3600.0
disengagements = (df.mode != "auto").sum()
disengagement_rate = disengagements / drive_hours * 1000.0
# infraction rate per 1000 km (km in metadata)
km = df.attrs.get("km_driven", 1.0)
infraction_rate = df.infraction.sum() / km * 1000.0
# comfort: 90th percentile of absolute jerk proxy (here comfort_acc)
comfort_p90 = np.percentile(np.abs(df.comfort_acc.dropna()), 90)
# importance-weighted recall and bootstrap CI
w, y, g = df.weight.values, df.event_detected.values, df.event_gt.values
hatR = (w * y * g).sum() / (w * g).sum()
# bootstrap
nboot = 1000
boot = []
for _ in range(nboot):
    idx = np.random.choice(len(w), len(w), replace=True)
    num = (w[idx] * y[idx] * g[idx]).sum(); den = (w[idx] * g[idx]).sum()
    boot.append(num/den if den>0 else 0.0)
ci = np.percentile(boot, [2.5,97.5])
print(disengagement_rate, infraction_rate, comfort_p90, hatR, ci)