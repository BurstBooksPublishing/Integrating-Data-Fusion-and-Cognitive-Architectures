import numpy as np

# Hypotheses and priors
hyps = ["benign","hostile","spoof"]
priors = np.array([0.6, 0.2, 0.2])  # prior P(h)

# Evidence features (example values)
kin_residual = 0.8  # lower means consistent with planned rendezvous
ais_auth_score = 0.3  # 0..1, low => likely spoof
sar_gap_hours = 6.0

# Likelihood models (independent factors multiplied)
def kin_likelihood(residual):
    # Gaussian-like; tuned by real validation
    sigma = 0.5
    return np.exp(-0.5*(residual/sigma)**2)

def ais_likelihood(score, hyp):
    # Template: benign expects high auth, spoof expects low auth
    if hyp=="benign":
        return score
    if hyp=="spoof":
        return 1.0-score
    return 0.5  # hostiles may or may not spoof

def sar_likelihood(gap, hyp):
    # long revisit gaps increase hostile/spoof likelihood
    if hyp=="benign":
        return np.exp(-0.2*gap)
    return 1 - np.exp(-0.1*gap)

# Compute likelihoods P(E|h_i)
lik = np.array([kin_likelihood(kin_residual) * ais_likelihood(ais_auth_score,h)
                * sar_likelihood(sar_gap_hours,h) for h in hyps])

# Posterior via Bayes
unnorm = lik * priors
posterior = unnorm / unnorm.sum()

# Define utilities U(a,h)
actions = ["monitor","shadow","interdict"]
U = {
    ("monitor","benign"):  1.0, ("monitor","hostile"): -2.0, ("monitor","spoof"): -0.5,
    ("shadow","benign"):   0.0, ("shadow","hostile"):  1.5, ("shadow","spoof"): 0.5,
    ("interdict","benign"): -5.0,("interdict","hostile"): 5.0, ("interdict","spoof"): 2.0
}

# Expected utility per action
eu = {a: sum(posterior[i]*U[(a,hyps[i])] for i in range(len(hyps))) for a in actions}

# Select best COA with simple safety threshold
best = max(eu, key=eu.get)
print("posterior", dict(zip(hyps, posterior.round(3))))
print("EU", eu, "best COA:", best)