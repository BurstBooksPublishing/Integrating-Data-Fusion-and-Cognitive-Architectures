from typing import List, Dict
import math, time

class Candidate:
    def __init__(self, cid, confidence, expected_utility, resource_cost, provenance):
        self.cid = cid
        self.confidence = confidence
        self.expected_utility = expected_utility
        self.resource_cost = resource_cost
        self.provenance = provenance

def safety_gate(c: Candidate, policy_params: Dict) -> bool:
    # hard rule: block kinetic if confidence < threshold
    if policy_params['kinetic'] and c.expected_utility > policy_params['kinetic_min_utility']:
        return c.confidence >= policy_params['kinetic_confidence']
    # other invariant checks here
    return True

def score(c: Candidate, weights: Dict) -> float:
    # composite score: confidence-weighted utility minus resource penalty
    return c.confidence * c.expected_utility - weights['lambda'] * c.resource_cost

def arbitrate(candidates: List[Candidate], params: Dict) -> Candidate:
    allowed = [c for c in candidates if safety_gate(c, params['policy'])]
    if not allowed:
        return None  # fallback action: safe-hold
    # compute scores and apply hysteresis if same candidate recently chosen
    scored = sorted(allowed, key=lambda c: score(c, params['weights']), reverse=True)
    winner = scored[0]
    # log decision artifact
    decision = {
        'time': time.time(), 'winner': winner.cid,
        'score': score(winner, params['weights']),
        'provenance': winner.provenance
    }
    print("DECISION:", decision)  # replace with structured audit log
    return winner

# Example use (real system would stream candidates)