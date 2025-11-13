import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, precision_score
from scipy.stats import ks_2samp
# load logged stream with labels for silent trial
df = pd.read_csv("stream_log.csv")  # columns: patient_id, score, label, subgroup
# compute global AUROC and PPV at operating threshold
auc = roc_auc_score(df['label'], df['score'])
threshold = 0.7
pred = (df['score'] >= threshold).astype(int)
ppv = precision_score(df['label'], pred)
# subgroup breakdown
group_metrics = df.groupby('subgroup').apply(
    lambda g: pd.Series({'auc': roc_auc_score(g['label'], g['score']),
                         'ppv': precision_score(g['label'], (g['score']>=threshold).astype(int))}))
print("global", auc, ppv); print(group_metrics)
# canary logic: simulate rollout fraction p and simple rollback triggers
def canary_step(log_df, p, alert_budget):
    sample = log_df.sample(frac=p, random_state=0)
    expected_alerts = sample['score'].ge(threshold).sum()
    # trigger if alert budget exceeded or PPV drop
    if expected_alerts > alert_budget:
        return False, "alert_budget_exceeded"
    if precision_score(sample['label'], (sample['score']>=threshold).astype(int)) < 0.9*ppv:
        return False, "ppv_drop"
    return True, "ok"
# drift sentinel: KS on score distribution vs baseline
baseline_scores = pd.read_pickle("baseline_scores.pkl")
stat, pval = ks_2samp(baseline_scores, df['score'])
if pval < 0.01:
    print("drift_detected", stat, pval)
# write telemetry
df.to_parquet("silent_trial_telemetry.parquet")