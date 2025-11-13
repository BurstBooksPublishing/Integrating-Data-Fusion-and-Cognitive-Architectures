import hashlib, hmac, json, uuid, time
# secret key for HMAC (rotate/manage in KMS in production)
K = b'supersecretkey'

def now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()  # content fingerprint

def sign_provenance(record: dict) -> str:
    b = json.dumps(record, sort_keys=True).encode('utf-8')
    return hmac.new(K, b, hashlib.sha256).hexdigest()  # MAC signature

def write_provenance_capture(raw_bytes: bytes, sensor_id: str, seq: int):
    cid = str(uuid.uuid4())
    h = content_hash(raw_bytes)
    rec = {
        'capture_id': cid,
        'sensor_id': sensor_id,
        'content_hash': h,
        'ts': now_iso(),
        'seq': seq
    }
    rec['signature'] = sign_provenance(rec)
    with open('provenance.log','ab') as f:  # append-only audit log
        f.write((json.dumps(rec)+'\n').encode('utf-8'))
    # store content addressed (example: filename = content_hash)
    with open(h, 'wb') as c: c.write(raw_bytes)
    return rec

def emit_decision(decision: dict, input_provenance: list):
    # record links to inputs' capture_ids and content_hashes
    dec = {
        'decision_id': str(uuid.uuid4()),
        'ts': now_iso(),
        'decision': decision,
        'inputs': [{'capture_id': p['capture_id'],'content_hash': p['content_hash']} for p in input_provenance]
    }
    dec['signature'] = sign_provenance(dec)
    with open('decisions.log','ab') as f:
        f.write((json.dumps(dec)+'\n').encode('utf-8'))
    return dec

def verify_record(rec: dict) -> bool:
    sig = rec.pop('signature', None)
    ok = sig == sign_provenance(rec)
    rec['signature'] = sig
    return ok

# usage: capture -> process -> decision
# raw = b'camera frame bytes...'
# prov = write_provenance_capture(raw, 'cam_front', seq=123)
# decision = emit_decision({'action':'brake','reason':'pedestrian'}, [prov])