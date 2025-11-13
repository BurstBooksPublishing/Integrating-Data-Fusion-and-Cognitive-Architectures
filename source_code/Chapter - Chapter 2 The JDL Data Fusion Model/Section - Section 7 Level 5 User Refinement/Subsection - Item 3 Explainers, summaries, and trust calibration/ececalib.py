import numpy as np
from sklearn.isotonic import IsotonicRegression

def compute_ece(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        idx = (probs > bins[i]) & (probs <= bins[i+1])
        if not np.any(idx):
            continue
        acc = labels[idx].mean()
        conf = probs[idx].mean()
        ece += (idx.sum()/n) * abs(acc - conf)
    return ece

def calibrate_probs(probs, labels):
    # isotonic regression fits a monotonic mapping for calibration
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probs, labels)
    return ir.predict(probs), ir

def generate_summary(hypotheses, confidences, provenance_counts):
    # concise, provenance-aware explanation template
    top = np.argmax(confidences)
    return (
        f"Top hypothesis: {hypotheses[top]} (confidence {confidences[top]:.2f}). "
        f"Supporting sensors: {provenance_counts[hypotheses[top]]} evidence items. "
        "Alternatives shown with calibrated confidences; operator feedback will update trust."
    )

if __name__ == "__main__":
    # synthetic example data
    hyps = ["vehicle", "pedestrian", "unknown"]
    probs = np.array([0.78, 0.15, 0.07])
    labels = np.array([1, 0, 0])  # ground truth for calibration test
    prov = {"vehicle": 4, "pedestrian": 2, "unknown": 1}

    ece_before = compute_ece(probs, labels)
    calibrated, ir_model = calibrate_probs(probs, labels)
    summary = generate_summary(hyps, calibrated, prov)

    print(f"ECE before: {ece_before:.3f}")
    print("Calibrated confidences:", np.round(calibrated, 3))
    print(summary)