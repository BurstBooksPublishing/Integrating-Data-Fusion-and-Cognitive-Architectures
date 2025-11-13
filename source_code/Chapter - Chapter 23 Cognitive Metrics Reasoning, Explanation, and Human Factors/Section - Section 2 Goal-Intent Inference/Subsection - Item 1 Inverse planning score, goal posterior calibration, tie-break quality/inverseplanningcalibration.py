import numpy as np

# synthetic per-goal log-likelihoods for N examples and K goals
np.random.seed(0)
N, K = 200, 3
loglikes = np.random.randn(N, K)  # replace with planner log-likelihoods
priors = np.array([0.5, 0.3, 0.2])  # domain prior

def softmax(lv, temp=1.0):
    a = lv / temp
    a = a - a.max(axis=1, keepdims=True)
    ex = np.exp(a)
    return ex / ex.sum(axis=1, keepdims=True)

# posterior with default temperature
post = softmax(loglikes + np.log(priors), temp=1.0)

# simulate ground truth (for calibration evaluation)
truth = np.random.choice(K, size=N, p=priors)

# temperature scaling: grid search minimize NLL on validation set
temps = np.linspace(0.5, 3.0, 51)
best_temp, best_nll = 1.0, np.inf
for t in temps:
    p = softmax(loglikes + np.log(priors), temp=t)
    nll = -np.mean(np.log(np.clip(p[np.arange(N), truth], 1e-12, 1.0)))
    if nll < best_nll:
        best_nll = nll; best_temp = t
# apply best temperature
post_cal = softmax(loglikes + np.log(priors), temp=best_temp)

# ECE computation (K-way treated by top-probability bins)
def ece(pred_probs, labels, M=10):
    top_probs = pred_probs.max(axis=1)
    preds = pred_probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, M+1)
    ece_val = 0.0
    N = len(labels)
    for m in range(M):
        idx = (top_probs > bin_edges[m]) & (top_probs <= bin_edges[m+1])
        if not np.any(idx): continue
        acc = np.mean(preds[idx] == labels[idx])
        avg_conf = np.mean(top_probs[idx])
        ece_val += (idx.sum()/N) * abs(avg_conf - acc)
    return ece_val

ece_before = ece(post, truth)
ece_after = ece(post_cal, truth)

# tie-break quality for each example
def tie_break_score(posterior, rationale_score=0.5, alpha=1.0, beta=1.0, gamma=1.0):
    sorted_p = np.sort(posterior)[::-1]
    margin = sorted_p[0] - sorted_p[1]
    H = -np.sum(posterior*np.log(np.clip(posterior,1e-12,1.0)))
    H_base = np.log(len(posterior))
    R = rationale_score  # normalized [0,1]
    return alpha*margin + beta*(H_base-H) + gamma*R

# compute metrics for the first 5 examples
for i in range(5):
    tb = tie_break_score(post_cal[i], rationale_score=np.random.rand())
    print(f"example {i}: topP={post_cal[i].max():.3f}, margin={(np.sort(post_cal[i])[::-1][0]-np.sort(post_cal[i])[::-1][1]):.3f}, TQ={tb:.3f}")

print("ECE before/after:", ece_before, ece_after)