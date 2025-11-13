import time, math
import numpy as np

def cos_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)

class Hypothesis:
    def __init__(self, hid, theta, prior, evidence_ids):
        self.hid = hid                        # unique id
        self.theta = np.array(theta)          # parameter vector
        self.prior = float(prior)             # P(H)
        self.evidence = set(evidence_ids)     # provenance links
        self.birth = time.time()
        self.posterior = prior                # init posterior

class HypothesisManager:
    def __init__(self, retire_thresh=0.01, merge_sim=0.95, tau=0.5):
        self.active = {}                      # hid -> Hypothesis
        self.retired = {}
        self.retire_thresh = retire_thresh
        self.merge_sim = merge_sim
        self.tau = tau

    def spawn(self, hid, theta, prior, evidence_ids):
        self.active[hid] = Hypothesis(hid, theta, prior, evidence_ids)

    def score_likelihood(self, H, data_vec):
        # simple Gaussian likelihood on theta distance
        dist = np.linalg.norm(H.theta - data_vec)
        return math.exp(-0.5*(dist**2))

    def compete(self, data_vec):
        # Bayes update then softmax competition
        logs = {}
        for hid,H in list(self.active.items()):
            like = self.score_likelihood(H, data_vec)
            logs[hid] = math.log(like+1e-12) + math.log(H.prior+1e-12)
        # softmax with temperature tau
        maxlog = max(logs.values())
        exps = {hid: math.exp((v-maxlog)/self.tau) for hid,v in logs.items()}
        Z = sum(exps.values())
        for hid,H in self.active.items():
            H.posterior = exps[hid]/Z

    def merge(self):
        # greedy pairwise merge by cosine similarity
        ids = list(self.active.keys())
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                a,b = ids[i],ids[j]
                if a not in self.active or b not in self.active: continue
                A,B = self.active[a], self.active[b]
                if cos_sim(A.theta, B.theta) >= self.merge_sim:
                    # weighted merge
                    wa, wb = A.posterior, B.posterior
                    theta = (wa*A.theta + wb*B.theta)/(wa+wb+1e-12)
                    evid = A.evidence.union(B.evidence)
                    # create new id and replace
                    nid = a + "+" + b
                    self.active.pop(a); self.active.pop(b)
                    self.spawn(nid, theta, wa+wb, evid)

    def retire(self, age_limit=300):
        now = time.time()
        for hid,H in list(self.active.items()):
            if H.posterior < self.retire_thresh or (now - H.birth) > age_limit:
                self.retired[hid] = H
                del self.active[hid]

# Example usage (simple): spawn two similar hypotheses, update, merge, retire
if __name__ == "__main__":
    mgr = HypothesisManager()
    mgr.spawn("h1",[1.0,0.0],0.4,["e1"])
    mgr.spawn("h2",[0.99,0.01],0.3,["e2"])
    mgr.compete(np.array([1.0,0.0]))   # evidence vector
    mgr.merge()
    mgr.retire()
    print("Active:", list(mgr.active.keys()), "Retired:", list(mgr.retired.keys()))