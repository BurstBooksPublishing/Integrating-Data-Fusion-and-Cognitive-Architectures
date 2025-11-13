import pandas as pd, math
# Load events: timestamp,episode_id,event_type,duration,violation_flag,reset_flag
df = pd.read_csv("events.csv", parse_dates=["timestamp"])

# Build per-episode summary
grp = df.sort_values("timestamp").groupby("episode_id")
episodes = grp.agg(
    start=("timestamp","first"),
    end=("timestamp","last"),
    success=("event_type", lambda s: int("success" in s.values)),
    reset_count=("reset_flag", "sum"),
    violation_count=("violation_flag","sum"),
    total_time=("duration","sum"),
).reset_index()

# Basic metrics
n = len(episodes)
k = episodes["success"].sum()
p = k / n  # success rate

# Efficiency: budget normalized (example budget 60s)
episodes["efficiency"] = (60.0 / episodes["total_time"]).clip(0,1)
mean_eff = episodes["efficiency"].mean()

# Reset and violation rates per 100 episodes
resets_per_100 = episodes["reset_count"].sum() / n * 100
violations_per_100 = episodes["violation_count"].sum() / n * 100

# Wilson CI (95%)
z = 1.96
p_hat = (k + z*z/2) / (n + z*z)
ci_half = z * math.sqrt(p_hat*(1-p_hat)/(n+z*z))

# Composite KPI example weights
w_s,w_e,w_r,w_v = 0.5,0.25,0.125,0.125
kpi = w_s*p + w_e*mean_eff - w_r*(resets_per_100/100) - w_v*(violations_per_100/100)

print(f"n={n}, success={p:.3f} ± {ci_half:.3f}, mean_eff={mean_eff:.3f}")
print(f"resets/100={resets_per_100:.1f}, violations/100={violations_per_100:.1f}, KPI={kpi:.3f}")