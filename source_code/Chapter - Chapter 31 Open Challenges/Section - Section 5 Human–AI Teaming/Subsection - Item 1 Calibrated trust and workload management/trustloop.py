import math, time
# EWMA workload estimator and simple temp-scaling calibrator.

class TrustManager:
    def __init__(self, alpha=0.8, beta=2.0, gamma=1.0, ewma_lambda=0.9, temp=1.0):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.lam = ewma_lambda
        self.temp = temp
        self.W = 0.0  # workload EWMA in [0,1]

    def temp_scale(self, p):  # placeholder calibrator (softmax temp)
        # avoid raw logits; operate on probability p
        p = max(min(p, 1-1e-6), 1e-6)
        logit = math.log(p/(1-p))
        scaled = 1/(1+math.exp(-logit/self.temp))
        return scaled

    def update_workload(self, obs):  # obs in [0,1]
        self.W = self.lam * self.W + (1 - self.lam) * obs
        return self.W

    def compute_trust(self, raw_conf, workload_obs, criticality):
        C = self.temp_scale(raw_conf)
        W = self.update_workload(workload_obs)
        # logistic synthesis per (1)
        score = 1/(1+math.exp(-(self.alpha*math.log(C/(1-C)) - self.beta*W + self.gamma*criticality)))
        return score

    def decide_action(self, trust_score, thresholds=(0.3,0.7)):
        low, high = thresholds
        if trust_score < low:
            return "human_only"   # hand back control
        if trust_score > high:
            return "automate"     # increase automation authority
        return "assist"          # shared control with high-explainability

# Example loop
tm = TrustManager()
for t in range(10):
    raw_conf = 0.6  # fused system confidence
    workload_obs = 0.2 + 0.05*t  # synthetic rising load
    criticality = 0.8 if t>5 else 0.3
    trust = tm.compute_trust(raw_conf, workload_obs, criticality)
    action = tm.decide_action(trust)
    print(f"t={t}, W={tm.W:.3f}, trust={trust:.3f}, action={action}")
    time.sleep(0.1)