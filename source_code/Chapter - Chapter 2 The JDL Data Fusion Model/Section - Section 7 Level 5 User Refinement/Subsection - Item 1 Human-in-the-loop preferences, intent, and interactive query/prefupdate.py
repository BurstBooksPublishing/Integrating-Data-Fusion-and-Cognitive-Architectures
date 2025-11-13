#!/usr/bin/env python3
import numpy as np

# Define categories and Dirichlet prior (alpha)
C = ["vessel", "small_boat", "swimmer"]
alpha = np.array([1.0, 1.0, 1.0])  # weak prior
counts = np.zeros_like(alpha)      # feedback counts

# Example hypotheses: (id, category, base_score)
hypotheses = [
    ("h1", "vessel", 0.75),
    ("h2", "small_boat", 0.60),
    ("h3", "swimmer", 0.40),
]

lambda_bias = 1.0  # trade-off between base score and preference

def dirichlet_mean(alpha, counts):
    # posterior mean per Eq. (1)
    a = alpha + counts
    return a / a.sum()

def rescore(hypotheses, pref_mean, lam=lambda_bias):
    # preference-weighted rescore
    cat_idx = {c:i for i,c in enumerate(C)}
    rescored = []
    for hid, cat, s in hypotheses:
        w = pref_mean[cat_idx[cat]]
        new_score = s * (1.0 + lam * w)  # simple multiplicative bias
        rescored.append((hid, cat, new_score))
    return sorted(rescored, key=lambda x: x[2], reverse=True)

# Interactive loop (simulate operator input)
if __name__ == "__main__":
    print("Initial top hypotheses:")
    pref = dirichlet_mean(alpha, counts)
    for h in rescore(hypotheses, pref):
        print(h)
    # Simulate operator marking 'swimmer' as high priority twice
    counts[2] += 2  # operator feedback (could be from clicks)
    pref = dirichlet_mean(alpha, counts)
    print("\nAfter feedback, updated preference mean:", pref)
    for h in rescore(hypotheses, pref):
        print(h)
# End of script