import numpy as np

def conformity_score(violations, aligns, w_c=None, w_p=None, alpha=2.0, beta=1.0):
    # violations: array of v_i in [0,1]; aligns: array of a_j in [0,1]
    w_c = np.ones_like(violations) if w_c is None else np.array(w_c)
    w_p = np.ones_like(aligns) if w_p is None else np.array(w_p)
    num = alpha * np.sum(w_c * violations) + beta * np.sum(w_p * (1 - aligns))
    den = alpha * np.sum(w_c) + beta * np.sum(w_p)
    return float(max(0.0, 1.0 - num / den))

def shield(policy, state, constraints, fallback_policy):
    # constraints: list of callables(state) -> (v,desc) where v in [0,1]
    # policy: callable(state) -> action; fallback_policy: safe action
    violations = []
    descs = []
    for c in constraints:
        v, d = c(state)
        violations.append(v); descs.append(d)
    aligns = policy.preference_alignments(state)  # returns array in [0,1]
    score = conformity_score(np.array(violations), np.array(aligns))
    if score < 0.6:  # operational gate threshold
        return fallback_policy(state), {'score': score, 'violations': descs}
    return policy(state), {'score': score, 'violations': descs}

# Example constraint callable (distance to forbidden zone)
def no_fly_check(state):
    dist_norm = max(0.0, (state['min_dist'] - state['threshold'])/state['threshold'])
    v = max(0.0, 1.0 - dist_norm)  # violation magnitude
    return v, f"no-fly prox {state['min_dist']:.1f}m"