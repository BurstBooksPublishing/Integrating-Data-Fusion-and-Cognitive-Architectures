import numpy as np

def simulate_counterfactual(state, action):
    # simple structural model: state dict -> next outcome dict
    # deterministic core with stochastic noise added
    next_pos = state['pos'] + action['move'] + np.random.normal(0, state['process_sigma'])
    damage = state['threat_level'] * (1.0 if action['engage'] else 0.1)
    return {'pos': next_pos, 'damage': damage}

def utility(outcome, action):
    # mission value minus resource cost and collateral penalty
    mission_value = -np.linalg.norm(outcome['pos'] - action['goal'])  # closer is better
    cost = action['cost']
    collateral = -10.0 * outcome['damage']
    return mission_value - cost + collateral

def expected_utility(posterior_samples, actions):
    # posterior_samples: list of state dicts with weight field 'w'
    results = {}
    for a in actions:
        sims = []
        for s in posterior_samples:
            o = simulate_counterfactual(s, a)                # simulate outcome
            sims.append(utility(o, a) * s['w'])              # weight by posterior
        results[a['name']] = float(np.sum(sims))
    return results

# Example usage
posterior = [{'pos': np.array([0.0,0.0]), 'threat_level': 0.8, 'process_sigma': 0.1, 'w': 0.5},
             {'pos': np.array([1.0,0.2]), 'threat_level': 0.2, 'process_sigma': 0.2, 'w': 0.5}]
actions = [{'name':'engage','move':np.array([0.5,0.0]), 'engage':True, 'cost':5.0,'goal':np.array([2.0,0.0])},
           {'name':'observe','move':np.array([0.1,0.0]), 'engage':False,'cost':1.0,'goal':np.array([2.0,0.0])}]
print(expected_utility(posterior, actions))  # prints expected utility per action