import math, time

# Plan templates: ordered steps with minimal temporal windows (s).
PLANS = {
    'rendezvous': [('move_to_A',0,60), ('hold',0,30), ('transfer',0,20)],
    'resupply':  [('approach_store',0,40), ('load',0,20), ('depart',0,60)]
}

# Simple priors
PRIORS = {'rendezvous': 0.3, 'resupply': 0.1}

class Hypothesis:
    def __init__(self, name, evidence=[]):
        self.name = name
        self.evidence = evidence[:]  # list of (step, t)
        self.weight = PRIORS.get(name, 1e-3)
        self.spawn_time = time.time()
    def update(self, event):
        self.evidence.append(event); self.weight *= likelihood(self, event)

def likelihood(hyp, event):
    # event: (step_name, timestamp)
    step_name, t = event
    expected = [s for (s,_,_) in PLANS[hyp.name]]
    # match score
    match = 0.9 if step_name in expected else 0.01
    # timing penalty: simple recency factor
    age = time.time() - t
    timing = math.exp(-age/60.0)
    return match * timing

# Evidence buffer simulated as list of (step, t)
evidence_stream = [('move_to_A', time.time()-5), ('hold', time.time()-3)]

# Spawn and update
hyps = []
for evt in evidence_stream:
    # spawn any plan whose first two steps match prefix
    for plan, steps in PLANS.items():
        step_names = [s for s,_,_ in steps]
        if evt[0] in step_names:
            # weak spawn
            h = Hypothesis(plan, evidence=[evt])
            hyps.append(h)
    # update existing hyps
    for h in hyps:
        h.update(evt)

# Normalize and prune
total = sum(h.weight for h in hyps) + 1e-12
hyps = [h for h in hyps if (h.weight/total) > 0.05]  # prune below 5%
for h in hyps:
    print(h.name, h.weight/total, len(h.evidence))