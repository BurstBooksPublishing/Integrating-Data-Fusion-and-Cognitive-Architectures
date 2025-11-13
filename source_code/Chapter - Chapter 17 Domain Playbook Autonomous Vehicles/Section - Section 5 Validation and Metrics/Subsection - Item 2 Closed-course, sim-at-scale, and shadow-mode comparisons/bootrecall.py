import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

def bootstrap_metric_diff(df, metric_fn, n_boot=1000, seed=0):
    np.random.seed(seed)
    # df must contain columns: 'mode' ('closed'|'sim'|'shadow'), 'seed', 'label', 'pred'
    paired = df.pivot_table(index='seed', columns='mode', values=['label','pred'])
    # keep seeds present in both modes
    paired = paired.dropna()
    labels_closed = paired[('label','closed')].values
    preds_closed  = paired[('pred','closed')].values
    labels_sim    = paired[('label','sim')].values
    preds_sim     = paired[('pred','sim')].values

    diffs = []
    n = len(labels_closed)
    idx = np.arange(n)
    for _ in range(n_boot):
        samp = np.random.choice(idx, size=n, replace=True)
        r_closed = metric_fn(labels_closed[samp], preds_closed[samp])
        r_sim    = metric_fn(labels_sim[samp], preds_sim[samp])
        diffs.append(r_closed - r_sim)
    diffs = np.array(diffs)
    return np.percentile(diffs, [2.5, 50, 97.5])  # lower, median, upper CI

# Example metric function (binary recall)
def recall(labels, preds):
    return recall_score(labels, preds)

# Usage: df = pd.read_csv('paired_runs.csv'); ci = bootstrap_metric_diff(df, recall)