import numpy as np, json, time
# inputs (example) -- in practice from message bus
fusion = {'id':1, 'state':np.array([10.0,0.0]), 'cov':np.array([[4.0,0.0],[0.0,1.0]]),
          'provenance':{'sensors':['radar','cam'],'ts':time.time()}}
# simple utility models (domain-specific)
def expected_utility(action, state):
    # reward higher for aggressive action near target; negative for collisions (toy)
    if action=='proceed': return -np.linalg.norm(state) + 5.0
    if action=='stop':    return -1.0
    return 0.0
def uncertainty_penalty(cov): return np.trace(cov)  # simple surrogate

alpha = 0.5          # risk aversion
a_f, a_s = 'proceed','stop'
eu_f = expected_utility(a_f, fusion['state'])
eu_s = expected_utility(a_s, fusion['state'])
pen = uncertainty_penalty(fusion['cov'])
# veto rule (Eq. 1)
if eu_f + alpha*pen < eu_s:
    decision = {'action':'veto','fallback':a_s}
else:
    decision = {'action':'commit','selected':a_f}
# audit record
audit = {'fusion_id':fusion['id'],'decision':decision,'eu_f':eu_f,'eu_s':eu_s,'pen':pen,'prov':fusion['provenance']}
print(json.dumps(audit))  # in practice publish to signed ledger