import numpy as np
from scipy.stats import ks_2samp

# compute Population Stability Index for categorical feature
def psi(reference, current, eps=1e-6):
    # reference/current are arrays of category labels
    ref_counts = np.bincount(reference)
    cur_counts = np.bincount(current, minlength=ref_counts.size)
    ref_pct = ref_counts / (ref_counts.sum()+eps)
    cur_pct = cur_counts / (cur_counts.sum()+eps)
    # avoid log(0)
    idx = (ref_pct>0)
    return np.sum((ref_pct[idx]-cur_pct[idx]) * np.log((ref_pct[idx]+eps)/(cur_pct[idx]+eps)))

# streaming loop (simplified)
reference_window = np.load("golden_feature_cat.npy")  # vetted golden-trace feature
while True:
    batch = read_next_ingest_batch()  # blocking call from feature store
    feature_vals = batch["feature_cat"]  # categorical feature used in decisions
    score = psi(reference_window, feature_vals)
    # additional check: KS on numeric surrogate
    if score > 0.2 or ks_2samp(reference_window, feature_vals).pvalue < 0.01:
        quarantine(batch)  # route to quarantine for human review
    else:
        publish_to_feature_store(batch)  # safe path