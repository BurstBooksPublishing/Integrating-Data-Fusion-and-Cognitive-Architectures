import numpy as np

np.random.seed(0)
# chunk base strengths and cue weights
B = np.array([0.5, 0.2, 0.1])         # base activations
W = np.array([1.0, 0.5])              # cue weights
S = np.array([[0.6, 0.1, 0.0],        # cue->chunk similarities
              [0.2, 0.8, 0.3]])
# compute activations
A = B + W @ S                         # spreading activation
theta = 0.0                           # retrieval threshold
s = 0.2                               # noise scale (logistic)
t0, F = 0.02, 0.1                      # base and latency factor
# sample retrieval noise and compute latencies and success prob
eps = np.random.normal(0, s, size=A.shape)   # Gaussian noise sample
latencies = t0 + F * np.exp(-(A + eps))      # eqn (1) computation
p_retrieve = 1.0 / (1.0 + np.exp((theta - A) / s))  # logistic prob
# utilities for productions, softmax choice
U = np.array([1.2, 0.8, 0.5])
tau = 0.3
probs = np.exp(U / tau) / np.sum(np.exp(U / tau))  # softmax
print("A:", A)                    # activation vector
print("latencies (s):", latencies)
print("P(retrieve):", p_retrieve)
print("production probs:", probs)  # sampling choice uses these