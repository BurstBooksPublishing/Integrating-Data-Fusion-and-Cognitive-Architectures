import time, math, json
import numpy as np
from scipy.stats import entropy  # KL via entropy(p, q)

# p_fusion, p_cog: numpy arrays over hypotheses (sum=1)
# util: numpy array of utilities per hypothesis
# params: dictionary with thresholds and dwell (seconds)

def should_override(p_fusion, p_cog, util, params, last_override_ts):
    # expected utility delta
    eu_f = np.dot(p_fusion, util)
    eu_c = np.dot(p_cog, util)
    delta = eu_c - eu_f

    # KL divergence
    kl = entropy(p_cog, p_fusion)

    # epistemic variance heuristic
    var_f, var_c = p_fusion.var(), p_cog.var()

    now = time.time()
    dwell_ok = (now - last_override_ts) > params['dwell']

    decision = (delta > params['C_override'] and kl > params['tau']
                and var_c < var_f and dwell_ok)

    # audit record
    audit = dict(timestamp=now, delta=float(delta), kl=float(kl),
                 var_f=float(var_f), var_c=float(var_c),
                 decision=bool(decision))
    print(json.dumps(audit))  # inline audit; in prod write to tamper-evident store

    return decision

# Example usage: last_override_ts = 0; params = {'C_override':0.5, 'tau':0.3, 'dwell':2.0}