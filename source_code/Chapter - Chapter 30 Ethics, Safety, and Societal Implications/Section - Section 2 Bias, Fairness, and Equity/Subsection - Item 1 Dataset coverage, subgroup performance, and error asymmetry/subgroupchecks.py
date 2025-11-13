import pandas as pd
from sklearn.metrics import confusion_matrix

# load fused outputs: columns: group,label,pred
df = pd.read_csv("fused_outputs.csv")  # group identifiers and true/pred labels

threshold = 0.05  # alert threshold for Delta_err
groups = df['group'].unique()
N = len(df)

metrics = []
for g in groups:
    sub = df[df['group']==g]
    n = len(sub)
    c = n / N                     # coverage fraction
    tn, fp, fn, tp = confusion_matrix(sub['label'], sub['pred'], labels=[0,1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp)>0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn)>0 else 0.0
    metrics.append((g, n, c, fnr, fpr))

metrics_df = pd.DataFrame(metrics, columns=['group','n','coverage','FNR','FPR'])
global_err = df['label'].ne(df['pred']).mean()
# compute Delta_err using FNR as the targeted error
metrics_df['err_diff'] = (metrics_df['FNR'] - metrics_df['FNR'].mean()).abs()
delta_err = metrics_df['err_diff'].max()

# action: alert and list low-coverage groups
low_coverage = metrics_df[metrics_df['coverage'] < 0.01]  # example min coverage
if delta_err > threshold or not low_coverage.empty:
    print("ALERT: asymmetric errors or low coverage detected")
    print(metrics_df.sort_values('err_diff', ascending=False).head())
    print("Low coverage groups:\n", low_coverage)
# outputs can feed L4 policy or human-in-the-loop queue