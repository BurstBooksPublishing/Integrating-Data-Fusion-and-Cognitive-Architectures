import json, hashlib, time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

# generate or load key (demo generates ephemeral key)
priv = Ed25519PrivateKey.generate()
pub = priv.public_key()

def minimize(record):
    # redact direct identifiers; keep hashed token reference
    rec = record.copy()
    if 'person_name' in rec:
        rec['person_token'] = hashlib.sha256(rec['person_name'].encode()).hexdigest()
        del rec['person_name']  # remove PII
    return rec

def serialize(entry):
    return json.dumps(entry, sort_keys=True, ensure_ascii=False).encode()

def append_entry(log_path, entry):
    # read last hash
    try:
        with open(log_path, 'rb') as f:
            last = f.read().splitlines()[-1]
            last_hash = last.split(b' ')[0]
    except Exception:
        last_hash = b'0'*64
    # chain hash
    h = hashlib.sha256(last_hash + serialize(entry)).hexdigest()
    sig = priv.sign(h.encode()).hex()
    line = f"{h} {sig} {json.dumps(entry, separators=(',',':'))}\n"
    with open(log_path, 'ab') as f:
        f.write(line.encode())  # append-only write
    return h

# pipeline demo
raw = {'source_id':'cam-12','capture_ts':time.time(), 'person_name':'Alice', 'event':'enter'}
mined = minimize(raw)  # minimization step
prov = {'source_id':mined['source_id'],'capture_ts':mined['capture_ts'],
        'transform_chain':['minimize_v1'],'data_class':'pseudonymized',
        'schema_v':'1.0','entry':mined}
h = append_entry('prov.log', prov)  # append with chain+signature
print("appended hash:", h)