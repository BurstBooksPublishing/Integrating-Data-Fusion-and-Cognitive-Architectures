from dataclasses import dataclass, field
import math, json, time

# Data models
@dataclass
class Track:
    track_id: int
    state: tuple  # (x,y,vx,vy)
    cov: list     # covariance (flattened) ; simplified
    source: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class Hypothesis:
    label: str
    score: float
    evidence: dict

# Simple likelihood model (Gaussian distance proxy)
def likelihood(track, template_state, sigma=10.0):
    dx = track.state[0] - template_state[0]
    dy = track.state[1] - template_state[1]
    r2 = dx*dx + dy*dy
    return math.exp(-0.5*r2/(sigma*sigma))

# Hypothesis scoring using prior and likelihood
def score_hypotheses(track, templates, priors):
    hyps = []
    for label, tmpl in templates.items():
        lik = likelihood(track, tmpl)
        post = lik * priors.get(label, 1e-3)
        hyps.append(Hypothesis(label=label, score=post, evidence={'lik':lik,'prior':priors.get(label)}))
    # normalize
    s = sum(h.score for h in hyps) or 1.0
    for h in hyps: h.score /= s
    return hyps

# Operator selection (production-like)
def choose_operator(hyps, operator_catalog):
    best_op, best_val = None, -math.inf
    for op, params in operator_catalog.items():
        val = sum(h.score * params['utility'].get(h.label, 0.0) for h in hyps) - params.get('cost',0.0)
        if val > best_val:
            best_val, best_op = val, op
    return best_op, best_val

# Example run
templates = {'vehicle':(100,50,0,0), 'pedestrian':(102,52,0,0)}
priors = {'vehicle':0.6, 'pedestrian':0.4}
operators = {'increase_sensor_rate':{'utility':{'vehicle':0.1,'pedestrian':0.5}, 'cost':0.2},
             'task_camera':{'utility':{'vehicle':0.7,'pedestrian':0.2}, 'cost':0.5}}

track = Track(1,(101.2,51.1,0,0),[],source='radar')
hyps = score_hypotheses(track, templates, priors)
op, val = choose_operator(hyps, operators)

# Provenance log (audit-friendly)
log = {'track_id':track.track_id, 'hyps':[h.__dict__ for h in hyps], 'chosen_op':op, 'value':val}
print(json.dumps(log, indent=2))