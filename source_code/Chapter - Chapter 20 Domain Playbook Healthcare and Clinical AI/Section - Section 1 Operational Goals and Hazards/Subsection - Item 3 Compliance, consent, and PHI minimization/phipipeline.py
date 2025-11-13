import json, hmac, hashlib, time
from functools import wraps
import numpy as np

# Simple consent store (in prod, use AC/ABAC service)
CONSENT = {'patient_token_abc': {'purpose':'triage', 'expires': 1700000000}}

def check_consent(token, purpose):  # return bool
    c = CONSENT.get(token)
    return bool(c and c['purpose']==purpose and c['expires']>time.time())

def require_consent(purpose):
    def deco(f):
        @wraps(f)
        def wrapped(record, *a, **kw):
            if not check_consent(record.get('patient_token'), purpose):
                raise PermissionError('consent_missing')  # logged by caller
            return f(record, *a, **kw)
        return wrapped
    return deco

def pseudonymize(patient_id, salt):
    # stable pseudonym for authorized joins (HMAC with rotated salt)
    return hmac.new(salt.encode(), patient_id.encode(), hashlib.sha256).hexdigest()

def laplace_mechanism(value, sensitivity, epsilon):
    scale = sensitivity/epsilon
    return float(value + np.random.laplace(0, scale))

@require_consent('triage')
def ingest_record(record, salt, epsilon=1.0):
    # remove direct PHI; keep pseudonym and time-limited provenance
    out = {}
    out['patient_tok'] = pseudonymize(record['patient_id'], salt)  # join key
    out['vitals_noisy'] = {k: laplace_mechanism(v, 1.0, epsilon) 
                           for k,v in record['vitals'].items()}  # DP on numeric fields
    out['provenance'] = {'source': record.get('source'), 'ts': record.get('ts')}
    # write to secure store (omitted)
    return out

# Example usage
rec = {'patient_id':'12345', 'patient_token':'patient_token_abc',
       'vitals':{'hr':82,'rr':18}, 'source':'bedside', 'ts': 1699990000}
print(ingest_record(rec, salt='rotate_me_2025', epsilon=0.5))