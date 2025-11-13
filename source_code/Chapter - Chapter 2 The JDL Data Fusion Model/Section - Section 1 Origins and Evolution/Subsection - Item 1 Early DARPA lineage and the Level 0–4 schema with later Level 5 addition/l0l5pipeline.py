from dataclasses import dataclass, asdict
import math, uuid

# L0: raw detection (signal features + cov)
@dataclass
class L0_Detection:
    id: str
    features: dict  # e.g., {'rf_power':..., 'bearing':...}
    cov: list       # covariance matrix (serialized)

# L1: track hypothesis
@dataclass
class L1_Track:
    id: str
    state: list     # state vector
    cov: list
    history: list   # provenance list of L0 ids

# L2/L3: situation/impact hypothesis with score and provenance
@dataclass
class Hypothesis:
    id: str
    description: str
    score: float    # posterior or unnormalized likelihood
    provenance: list

def bayes_score(likelihood, prior):
    # simple Bayesian combine (unnormalized)
    return likelihood * prior

# Example promotion rule: if entropy high, send to Level 5 (human)
def entropy_from_scores(scores):
    # scores assumed normalized probabilities
    return -sum(p*math.log(p+1e-12) for p in scores)

# pipeline steps (illustrative)
d = L0_Detection(id='d1', features={'rf': 12.3}, cov=[[1.0]])
t = L1_Track(id='t1', state=[0,0], cov=[[1.0]], history=[d.id])
hyp = Hypothesis(id=str(uuid.uuid4()), description='contact moving NE',
                 score=bayes_score(likelihood=0.3, prior=0.5),
                 provenance=[t.id])
# thresholding for human review
scores = [0.6, 0.4]  # competing hypotheses
if entropy_from_scores(scores) > 0.65:
    # flag to Level 5: present compact rationale with provenance
    review_packet = {'hypothesis': asdict(hyp), 'scores': scores}
    # send to UI/analyst channel (pseudo)
    print('SEND_TO_LEVEL5', review_packet)