import time, jwt, math

# Simple in-memory stores (replace with DB/services in production)
CONSENT_DB = {'data_id_42': {'state':'granted','scope':'ops','ts':1630000000}}
APPROVAL_CACHE = {'approver_A': {'level':3,'expires':time.time()+300}}  # cached token

# Compute risk: capability severity C, exposure E, consent score S in [0,1]
def compute_risk(C:float, E:float, S:float, alpha=1.0, beta=1.0, gamma=2.0)->float:
    return alpha*C + beta*E + gamma*(1.0 - S)

def check_consent(data_id):
    c = CONSENT_DB.get(data_id)
    return 1.0 if c and c['state']=='granted' else 0.0

def has_approval(token_b64, required_level):
    try:
        claims = jwt.decode(token_b64, options={"verify_signature": False})  # demo only
    except Exception:
        return False
    role = claims.get('role')
    cache = APPROVAL_CACHE.get(role)
    if cache and cache['level'] >= required_level and cache['expires'] > time.time():
        return True
    return False

def policy_gate(action, data_id, capability_tier, exposure, approver_token=None):
    S = check_consent(data_id)
    R = compute_risk(capability_tier, exposure, S)
    THRESHOLD = 5.0  # derived from governance & approvals
    if R <= THRESHOLD:
        # log allow with provenance
        print(f"ALLOW action={action} data={data_id} risk={R:.2f}")
        return True
    # require higher approval if available
    if approver_token and has_approval(approver_token, required_level=capability_tier):
        print(f"ALLOW w/ approval action={action} data={data_id} risk={R:.2f}")
        return True
    # otherwise block and emit audited rejection
    print(f"BLOCK action={action} data={data_id} risk={R:.2f}")
    return False

# Example usage (replace token with real signed JWT in production)
if __name__ == "__main__":
    fake_token = jwt.encode({'role':'approver_A'}, key='', algorithm='none')
    policy_gate('escalate_to_kinetic','data_id_42', capability_tier=4, exposure=2.0,
                approver_token=fake_token)