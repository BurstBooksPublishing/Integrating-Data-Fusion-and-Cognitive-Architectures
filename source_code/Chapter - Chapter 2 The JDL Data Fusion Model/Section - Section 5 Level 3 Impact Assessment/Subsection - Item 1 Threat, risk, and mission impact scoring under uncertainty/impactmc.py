import numpy as np

# Hypotheses priors and consequence means (toy values)
p_mean = np.array([0.6, 0.3, 0.1])
C_mean = np.array([100.0, 400.0, 900.0])

# Epistemic uncertainty: Dirichlet for posterior probabilities, normal for consequences
def sample_expected_impact(n_samples=5000):
    impacts = np.empty(n_samples)
    for i in range(n_samples):
        p = np.random.dirichlet(10.0 * p_mean)          # small uncertainty around p_mean
        C = np.random.normal(C_mean, 50.0)              # consequence uncertainty
        impacts[i] = np.dot(p, C)
    return impacts

imp_samples = sample_expected_impact()
print("E[I]:", imp_samples.mean())                     # posterior mean
print("95% CI:", np.percentile(imp_samples, [2.5,97.5]))# credible interval

# Simple VoI: simulate an information probe that shifts p_mean towards hypothesis 1
def estimate_voi(n_samples=2000):
    before = sample_expected_impact(n_samples)
    # hypothetical evidence D that increases weight on hypothesis 2
    p_mean_after = np.array([0.4, 0.5, 0.1])
    def sample_after():
        return np.dot(np.random.dirichlet(10.0 * p_mean_after),
                      np.random.normal(C_mean, 50.0))
    after = np.fromiter((sample_after() for _ in range(n_samples)), dtype=float)
    return before.mean() - after.mean()  # expected reduction in impact

print("Estimated VoI (toy):", estimate_voi())