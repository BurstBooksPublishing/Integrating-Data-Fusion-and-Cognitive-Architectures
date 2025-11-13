import random
import math

class BudgetedEpsilonGreedy:
    def __init__(self, actions, eps0=0.2, budget=10.0):
        self.actions = actions                # list of actions (sensor tasks)
        self.eps0 = eps0                      # initial exploration rate
        self.budget = budget                  # allowed cumulative safety cost
        self.spent = 0.0                      # accumulated safety cost
        self.counts = {a:1 for a in actions}  # simple counts for estimates
        self.values = {a:0.0 for a in actions}# estimated reward per action
        self.log = []                         # store (s,a,r,c,behaviour_prob)
    def select(self, state):
        # exploration rate decays as budget is consumed
        eps = self.eps0 * max(0.0, 1.0 - self.spent / max(self.budget, 1e-6))
        if random.random() < eps:
            a = random.choice(self.actions)
            pi = eps / len(self.actions) + (1-eps)*0.0  # behaviour prob approximated
        else:
            a = max(self.actions, key=lambda x: self.values[x])  # greedy
            pi = 1 - eps + eps / len(self.actions)
        return a, pi
    def update(self, state, action, reward, cost, beh_prob):
        # incremental update to value estimate
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n
        self.spent += cost
        self.log.append((state, action, reward, cost, beh_prob))
    def off_policy_estimate(self, target_policy_fn):
        # simple importance-weighted policy value estimator
        num, den = 0.0, 0.0
        for s,a,r,c,pi in self.log:
            p_star = target_policy_fn(s,a)
            w = p_star / max(pi, 1e-12)
            num += w * r
            den += w
        return num / max(den, 1e-12)  # normalized estimate
# Note: integrate safety filter and human veto externally.