import numpy as np
# preds, labels, groups are numpy arrays; groups contain subgroup ids
def subgroup_rates(preds, labels, groups):
    rates = {}
    for g in np.unique(groups):
        mask = (groups == g)
        tp = np.sum((preds[mask]==1) & (labels[mask]==1))
        tn = np.sum((preds[mask]==0) & (labels[mask]==0))
        fp = np.sum((preds[mask]==1) & (labels[mask]==0))
        fn = np.sum((preds[mask]==0) & (labels[mask]==1))
        n_pos = tp + fn
        n_neg = tn + fp
        rates[g] = {'FPR': fp / max(1, n_neg), 'FNR': fn / max(1, n_pos)}
    return rates

# flagging logic
def flag_disparity(rates, thresh_abs=0.05, thresh_ratio=1.25):
    # compare each group to reference (e.g., population mean)
    mean_fpr = np.mean([r['FPR'] for r in rates.values()])
    flags = {}
    for g, r in rates.items():
        if abs(r['FPR'] - mean_fpr) > thresh_abs or r['FPR'] / max(1e-6, mean_fpr) > thresh_ratio:
            flags[g] = True  # needs review
        else:
            flags[g] = False
    return flags

# Example usage; downstream: alerting, soft-stop, and audit logging