import math, time

def sigmoid(x): return 1/(1+math.exp(-x))

def decide_interrupt(fused_evidence, user_state, policy):
    """
    fused_evidence: dict with keys 'p_critical', 'severity', 'provenance'
    user_state: dict with 'engagement' (0..1), 'last_interaction_ts'
    policy: dict with 'Vb_map','tau','gamma','min_dwell_s'
    """
    now = time.time()
    # compute workload proxy w (higher = more busy)
    idle_time = now - user_state['last_interaction_ts']
    w = max(0.0, 1.0 - min(idle_time/600.0, 1.0))  # busy if recent interactions
    # benefit scaled by severity
    Vb = policy['Vb_map'].get(fused_evidence['severity'], 1.0)
    p_c = fused_evidence['p_critical']
    # disruption cost increases with engagement and decreases with acceptance probability
    C_i =  (0.5 + 0.5 * user_state['engagement']) * (0.5 + 0.5*w)
    EU = p_c * Vb - C_i - policy['gamma'] * fused_evidence.get('privacy_risk',0.0)
    # hysteresis/dwell: avoid re-interrupting too quickly
    if now - user_state.get('last_interrupt_ts',0) < policy['min_dwell_s']:
        action = 'defer'
    elif EU > policy['tau']:
        # choose modality by social cost ranking
        action = 'voice' if user_state['engagement']<0.7 else 'ambient_light_then_voice'
    else:
        action = 'suppress'
    rationale = {
        'EU': EU, 'p_c': p_c, 'Vb': Vb, 'C_i': C_i,
        'policy_tau': policy['tau'], 'provenance': fused_evidence.get('provenance')
    }
    return action, rationale

# Example policy and call (operational values)
policy = {'Vb_map':{'high':5.0,'med':2.0,'low':0.5}, 'tau':0.8, 'gamma':1.0, 'min_dwell_s':30}
fused = {'p_critical':0.92,'severity':'high','provenance':['vision.track.42','mic.energy']}
user = {'engagement':0.2,'last_interaction_ts':time.time()-120,'last_interrupt_ts':0}
print(decide_interrupt(fused,user,policy))  # prints action and rationale