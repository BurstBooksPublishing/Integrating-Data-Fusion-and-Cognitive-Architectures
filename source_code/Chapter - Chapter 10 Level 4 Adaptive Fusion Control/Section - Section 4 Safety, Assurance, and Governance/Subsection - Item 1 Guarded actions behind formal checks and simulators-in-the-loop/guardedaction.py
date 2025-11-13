import math, random

R_MAX = 0.05  # allowable risk threshold

def formal_checks(state, action):
    # example hard constraints: no-go zones, resource budgets
    return not state['no_go'][action['sector']] and action['cost'] <= state['budget']

class FastSimulator:
    def __init__(self, model): self.model = model
    def simulate(self, state, action, n=50):
        # Monte Carlo over sensor/model noise (simple stub)
        outcomes = []
        for _ in range(n):
            perturbed = {k: v + random.gauss(0, state['noise_std']) for k,v in state['obs'].items()}
            outcomes.append(self.model(perturbed, action))
        return outcomes

def risk_from_outcomes(outcomes):
    # risk = P(failure) estimated by fraction of bad outcomes
    failures = sum(1 for o in outcomes if o['mission_score'] < 0.5)
    return failures / len(outcomes)

def guarded_execute(state, action, sim, fallback):
    # 1. provenance & formal checks
    if not formal_checks(state, action):
        return {'status':'blocked','reason':'formal_checks_failed'}
    # 2. SIL
    outcomes = sim.simulate(state, action)
    r = risk_from_outcomes(outcomes)
    # 3. thresholding
    if r > R_MAX:
        return {'status':'rejected','risk':r,'action':fallback}
    # 4. commit (idempotent enactment stub)
    return {'status':'committed','risk':r,'action':action}

# Example model and invocation
def simple_model(obs, action):
    # mission_score degrades with occlusion probability (stub)
    occl = obs.get('occlusion',0)
    score = max(0.0, 1.0 - 0.8*occl - 0.1*action['cost'])
    return {'mission_score':score}

state = {'no_go':{'A':False,'B':True}, 'budget':10, 'noise_std':0.05, 'obs':{'occlusion':0.2}}
action = {'sector':'A','cost':3}
sim = FastSimulator(simple_model)
fallback = {'sector':'A','cost':1}  # conservative fallback
print(guarded_execute(state, action, sim, fallback))  # execute