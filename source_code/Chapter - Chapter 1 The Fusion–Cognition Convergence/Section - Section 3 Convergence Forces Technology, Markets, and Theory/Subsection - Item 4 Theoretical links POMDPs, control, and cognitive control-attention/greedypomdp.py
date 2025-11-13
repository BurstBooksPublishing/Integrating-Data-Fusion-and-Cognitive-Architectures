import numpy as np

# simple discrete problem: 3 states, 2 sensors (actions)
states = np.array([0,1,2])
nS = len(states)
# transition: identity (stationary) for simplicity
T = np.eye(nS)
# observation models: sensors give different confusion matrices
O = {
    0: np.array([[0.9,0.08,0.02],
                 [0.1,0.8,0.1],
                 [0.05,0.15,0.8]]),  # sensor A
    1: np.array([[0.7,0.2,0.1],
                 [0.2,0.7,0.1],
                 [0.1,0.1,0.8]])    # sensor B (cheaper)
}
cost = {0:1.0, 1:0.3}           # sensing costs
alpha_info = 1.0                # weight for info gain in utility

def entropy(p):
    p = np.clip(p,1e-12,1.0)
    return -np.sum(p*np.log(p))

def belief_update(b, action, obs):
    # Bayes rule for discrete models
    lik = O[action][:,obs]      # P(o|s)
    b_prime = lik * (T.T @ b)   # here T identity simplifies
    return b_prime / (np.sum(b_prime)+1e-12)

# initial belief
b = np.array([0.5,0.3,0.2])

# simulated observation probabilities per action
def expected_info_gain(b, action):
    # enumerate possible observations (obs index corresponds to columns)
    P_o = O[action].T @ b        # P(o|b,a)
    H_before = entropy(b)
    exp_H_after = 0.0
    for o, p_o in enumerate(P_o):
        b2 = belief_update(b, action, o)
        exp_H_after += p_o * entropy(b2)
    return H_before - exp_H_after

# one-step greedy selection
utilities = {}
for a in O:
    eig = expected_info_gain(b,a)
    utilities[a] = alpha_info*eig - cost[a]  # net utility
best = max(utilities, key=utilities.get)
print("Select sensor", best, "utilities", utilities)