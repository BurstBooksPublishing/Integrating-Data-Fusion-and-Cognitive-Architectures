import numpy as np

# Mock scorer: weighted sum of modality confidences -> situation probability
def situation_scorer(evidence, weights):
    return 1/(1+np.exp(-np.dot(evidence, weights)))  # logistic transform

def compute_ece(probs, truths, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        idx = (probs >= bins[i]) & (probs < bins[i+1])
        if not np.any(idx): continue
        conf = probs[idx].mean()
        acc = truths[idx].mean()
        ece += (idx.sum()/N) * abs(acc - conf)
    return ece

# Simulate dataset
rng = np.random.default_rng(0)
N = 1000
modalities = 3
weights = np.array([1.2, 0.8, 0.5])  # true contribution to situation
evidence = rng.normal(loc=0.5, scale=0.3, size=(N, modalities))
# truth generation with threshold on linear score (serves as ground truth)
truths = (np.dot(evidence, weights) + rng.normal(scale=0.4, size=N) > 1.0).astype(float)
probs = situation_scorer(evidence, weights)

# Baseline ECE
baseline_ece = compute_ece(probs, truths)
print("baseline ECE:", baseline_ece)

# Counterfactual probe: ablate each modality and measure sensitivity
sensitivity = []
for m in range(modalities):
    evidence_cf = evidence.copy()
    evidence_cf[:, m] = 0.0  # ablation counterfactual
    probs_cf = situation_scorer(evidence_cf, weights)
    delta = np.abs(probs - probs_cf).mean()
    sensitivity.append(delta)
    print(f"modality {m} mean prob delta:", delta)

# Stress scenario: random dropout of modality 0 with rate p
p = 0.3
drop_mask = rng.random(size=N) < p
evidence_stress = evidence.copy()
evidence_stress[drop_mask, 0] = 0.0
probs_stress = situation_scorer(evidence_stress, weights)
ece_stress = compute_ece(probs_stress, truths)
print("stress ECE:", ece_stress)
# Flag if ECE increases beyond acceptable delta
if ece_stress - baseline_ece > 0.02:
    print("ALERT: calibration degraded under stress")