import math, logging, time
import numpy as np

logging.basicConfig(level=logging.INFO)

class Workspace:
    def __init__(self, beta=5.0, tau=0.05):
        self.beta = beta    # selectivity
        self.tau = tau      # min selection probability
        self.subscribers = []  # callbacks

    def subscribe(self, cb):
        self.subscribers.append(cb)

    def salience(self, e, n, a, age, w=(1.0,0.8,0.5), gamma=0.1):
        # Eq. (2): weighted linear salience with age penalty
        return w[0]*e + w[1]*n + w[2]*a - gamma*age

    def softmax(self, S):
        # Eq. (1)
        ex = np.exp(self.beta * (S - np.max(S)))
        return ex / np.sum(ex)

    def select_and_broadcast(self, candidates, topk=3):
        # candidates: list of dicts with keys e,n,a,age,prov,id
        S = np.array([self.salience(**c) for c in candidates])
        P = self.softmax(S)
        # choose indices with P >= tau, then topk
        idx = np.where(P >= self.tau)[0]
        if len(idx) == 0:
            idx = np.argsort(-P)[:1]  # ensure at least one winner
        winners = sorted(idx, key=lambda i: -P[i])[:topk]
        for i in winners:
            msg = {'id': candidates[i]['id'], 'salience': float(S[i]),
                   'prob': float(P[i]), 'prov': candidates[i]['prov'],
                   'timestamp': time.time()}
            logging.info("Broadcasting %s", msg)   # rationale trace
            for cb in self.subscribers:
                cb(msg)

# Example consumer
def planner_cb(msg):
    # simple veto on low evidence in provenance
    if msg['salience'] < 0.5:
        logging.info("Planner defers %s", msg['id'])
    else:
        logging.info("Planner accepts %s", msg['id'])

# Demo
ws = Workspace(beta=6.0, tau=0.02)
ws.subscribe(planner_cb)
cands = [
    {'id':'trackA','e':0.9,'n':0.1,'a':0.0,'age':1.0,'prov':'radar'},
    {'id':'trackB','e':0.6,'n':0.5,'a':0.2,'age':0.2,'prov':'multi'},
    {'id':'eventC','e':0.4,'n':0.9,'a':0.6,'age':0.1,'prov':'visual'}
]
ws.select_and_broadcast(cands, topk=2)