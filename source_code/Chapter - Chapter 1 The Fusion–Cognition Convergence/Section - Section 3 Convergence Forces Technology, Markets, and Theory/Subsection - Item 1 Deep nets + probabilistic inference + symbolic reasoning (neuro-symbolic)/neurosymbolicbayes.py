import numpy as np

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def neuro_symbolic_posterior(logits, prior, symbol_pot):
    # logits: raw net outputs for hypotheses (shape (H,))
    # prior: prior probabilities (shape (H,))
    # symbol_pot: symbolic potential in [0,1] (shape (H,))
    lik = softmax(logits)               # P(X|h) approx from network
    unnorm = lik * prior * symbol_pot  # multiply evidence + symbolic weight
    posterior = unnorm / (unnorm.sum() + 1e-12)  # normalize, avoid div0
    return posterior

# Example usage
logits = np.array([2.0, 0.5, -1.0])             # detector outputs for 3 intents
prior = np.array([0.6, 0.3, 0.1])               # mission context prior
symbol_pot = np.array([1.0, 0.01, 0.5])         # rule downgrades second hypothesis
post = neuro_symbolic_posterior(logits, prior, symbol_pot)
print(post)  # posterior over hypotheses (traceable factors)