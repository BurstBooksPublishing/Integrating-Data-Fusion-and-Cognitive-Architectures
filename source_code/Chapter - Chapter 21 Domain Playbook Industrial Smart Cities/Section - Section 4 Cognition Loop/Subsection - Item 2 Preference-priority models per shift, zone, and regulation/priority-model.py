import numpy as np, datetime, json

# Simple in-memory policy store (real systems use signed policy DB)
POLICIES = {
  ('day','floor','std_reg'): {'w': np.array([0.25,0.45,0.20,0.10]),
                              'constraints': {'emissions': 100.0, 'max_human_exposure': 1}},
  ('night','hot_zone','safety_reg'): {'w': np.array([0.60,0.20,0.10,0.10]),
                                      'constraints': {'emissions': 50.0, 'max_human_exposure': 0}}
}

def select_action(ctx, actions, diagnostics):
    # ctx: (shift, zone, regulation); actions: list of dicts with 'id','u' vector,'metrics'
    policy = POLICIES.get(ctx)
    if policy is None:
        raise KeyError('Policy missing for context')  # governance check
    w = policy['w'] / np.sum(policy['w'])            # normalize
    # Prefilter by hard constraints
    cand = []
    for a in actions:
        ok = True
        for k,v in policy['constraints'].items():
            if a['metrics'].get(k, float('inf')) > v:
                ok = False; break
        if ok: cand.append(a)
    # If none pass, fallback to safest available (diagnostic signal)
    if not cand:
        diagnostics['fallback'] = 'no_action_meets_constraints'
        return min(actions, key=lambda a: a['metrics'].get('safety_risk', 1.0))
    # Score remaining candidates
    scores = [w.dot(np.array(a['u'])) for a in cand]
    choice = cand[int(np.argmax(scores))]
    # Provenance record
    provenance = {'timestamp': datetime.datetime.utcnow().isoformat(),
                  'context': ctx, 'policy_w': w.tolist(), 'choice': choice['id']}
    diagnostics['provenance'] = provenance
    print(json.dumps(provenance))  # replace with signed audit log
    return choice

# Example call (would be called by L4 controller)
if __name__ == '__main__':
    ctx=('night','hot_zone','safety_reg')
    actions=[{'id':'A1','u':[0.6,0.3,0.05,0.05],'metrics':{'emissions':40,'max_human_exposure':0,'safety_risk':0.2}},
             {'id':'A2','u':[0.3,0.6,0.05,0.05],'metrics':{'emissions':60,'max_human_exposure':0,'safety_risk':0.5}}]
    diag={}
    chosen=select_action(ctx, actions, diag)
    # chosen returned to planner; diag logged for audit