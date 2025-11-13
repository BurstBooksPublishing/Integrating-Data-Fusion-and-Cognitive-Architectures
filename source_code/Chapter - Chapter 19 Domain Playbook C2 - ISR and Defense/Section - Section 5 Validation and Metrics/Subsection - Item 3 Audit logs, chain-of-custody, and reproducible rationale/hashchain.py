import json, hashlib, hmac, base64, time, os
KEY = b'secret-key'                        # replace with secure key or private key handling
def canonical(s):                          # deterministic serialization
    return json.dumps(s, sort_keys=True, separators=(',',':'), ensure_ascii=False).encode()
def entry_hash(entry, prev_hash):
    data = canonical(entry) + (prev_hash or b'')
    return hashlib.sha256(data).digest()
def hmac_sign(data):
    return base64.b64encode(hmac.new(KEY, data, hashlib.sha256).digest()).decode()
# Example usage
prev = None
for i, ev in enumerate([{'id':1,'src':'radar','val':0.87},{'id':2,'src':'cam','val':0.93}]):
    ev['ts'] = time.time()
    ev['process'] = 'assoc_v1'            # model/artifact id here
    h = entry_hash(ev, prev)
    att = hmac_sign(h)
    record = {'entry':ev,'hash':base64.b64encode(h).decode(),'attestation':att}
    open(f'log_{i}.json','wb').write(canonical(record))  # append-only store in practice
    prev = h