import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Example inputs: model scores and true binary labels (1=deterioration)
# Replace these with held-out test arrays.
scores = np.load("scores.npy")  # model probabilistic scores
labels = np.load("labels.npy")  # binary 0/1

# Operational parameters
patients_per_clinician = 20       # W
alerts_per_clinician_per_shift = 2  # B
shift_length_hours = 8

# Determine target fraction f of patients to alert per shift
f_target = alerts_per_clinician_per_shift / patients_per_clinician

# Find threshold t that yields top-f_target fraction of scores
t = np.percentile(scores, 100*(1-f_target))  # threshold on score

# Compute AUROC
auroc = roc_auc_score(labels, scores)

# Compute precision/recall at all thresholds and extract PPV at t
precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
# Find nearest threshold index for t
# Note: pr_thresholds is sorted ascending; handle edge cases
idx = np.argmin(np.abs(pr_thresholds - t)) if pr_thresholds.size else 0
ppv_at_t = precision[idx] if pr_thresholds.size else precision[0]

# Expected alerts per shift after thresholding (empirical)
alerts_empirical = int(np.sum(scores >= t))
emp_alerts_per_clinician = alerts_empirical / (len(scores) / patients_per_clinician)

print(f"AUROC: {auroc:.3f}")
print(f"Threshold t: {t:.3f}")
print(f"PPV at t (precision): {ppv_at_t:.3f}")
print(f"Empirical alerts per clinician per shift: {emp_alerts_per_clinician:.2f}")