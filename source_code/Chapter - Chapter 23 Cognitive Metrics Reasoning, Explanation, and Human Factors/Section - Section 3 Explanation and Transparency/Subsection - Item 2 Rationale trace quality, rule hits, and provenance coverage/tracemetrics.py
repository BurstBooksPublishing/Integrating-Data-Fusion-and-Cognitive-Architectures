import math
from collections import Counter

def compute_metrics(trace, expected_evidence, applicable_rules, source_weights):
    # trace: dict with keys 'evidence' (set), 'rules' (list), 'sources' (set), 'attested' (set)
    C = len(trace['evidence'] & expected_evidence) / max(1, len(expected_evidence))  # completeness
    R = len(trace['rules']) / max(1, len(applicable_rules))  # rule-hit rate
    # provenance coverage weighted by source importance
    covered_weight = sum(source_weights.get(s,0.0) for s in trace['attested'])
    total_weight = sum(source_weights.get(s,0.0) for s in source_weights)
    P = covered_weight / max(1e-9, total_weight)
    # entropy-like hit diversity (diagnostic)
    rule_counts = Counter(trace['rules'])
    probs = [c/len(trace['rules']) for c in rule_counts.values()] if trace['rules'] else [1.0]
    entropy = -sum(p*math.log(p) for p in probs)
    return {'C':C, 'R':R, 'P':P, 'hit_entropy':entropy}

# simple run (executable)
trace = {'evidence':{'trackA','radar1'}, 'rules':['r_close','r_speed'], 'sources':{'radar1'}, 'attested':{'radar1'}}
expected = {'trackA','radar1','camera3'}
applicable = {'r_close','r_speed','r_bearing'}
weights = {'radar1':0.7, 'camera3':0.3}
print(compute_metrics(trace, expected, applicable, weights))  # prints metrics