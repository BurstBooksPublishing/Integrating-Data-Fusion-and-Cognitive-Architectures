import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# sample dataframe with session rows (real system would stream and persist these)
df = pd.DataFrame({
    'session': np.arange(1,11),
    # six TLX subscale scores 0-100
    'mental': np.random.uniform(20,80,10),
    'physical': np.random.uniform(0,40,10),
    'temporal': np.random.uniform(10,70,10),
    'performance': np.random.uniform(30,90,10),
    'effort': np.random.uniform(20,80,10),
    'frustration': np.random.uniform(10,60,10),
    # pairwise weights (sum nonzero)
    'w_mental': 4, 'w_physical':1, 'w_temporal':3,
    'w_performance':4, 'w_effort':2, 'w_frustration':1,
    # behavioural: alerts accepted / total
    'alerts_accepted': np.random.randint(5,15,10),
    'alerts_total': np.random.randint(10,20,10),
    # performance metric to fit (higher is better)
    'success_rate': np.linspace(0.5,0.9,10) + np.random.normal(0,0.02,10)
})

# compute weighted TLX per Eq. (1)
weights = np.array([df.w_mental[0], df.w_physical[0], df.w_temporal[0],
                    df.w_performance[0], df.w_effort[0], df.w_frustration[0]])
scores = df[['mental','physical','temporal','performance','effort','frustration']].values
df['wtlx'] = (scores * weights).sum(axis=1) / weights.sum()

# compute reliance rate and trust gap (R* provided externally)
df['reliance'] = df.alerts_accepted / df.alerts_total
R_star = 0.7  # normative reliance from domain model
df['trust_gap'] = df.reliance - R_star

# fit learning curve per Eq. (3)
def exp_learn(t, P_inf, A, b): return P_inf - A * np.exp(-b * t)
popt, _ = curve_fit(exp_learn, df.session.values, df.success_rate.values,
                    bounds=([0.5,0.0,0.0],[1.0,1.0,2.0]))
P_inf, A, b = popt

# outputs for telemetry/alerting (would be pushed to metrics DB)
print("WTLX mean", df.wtlx.mean())
print("Learning params P_inf, A, b", P_inf, A, b)