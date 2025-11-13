import json, hmac, hashlib, requests, time
# secret_key should be device-protected (HSM or secure element)
secret_key = b'supersecret_hmac_key'  

def redact(obj, rules):
    # rules: list of (json_path_predicate, replacer) functions
    if isinstance(obj, dict):
        return {k: redact(v, rules) for k, v in obj.items() if not is_sensitive_key(k, rules)}
    if isinstance(obj, list):
        return [redact(x, rules) for x in obj]
    return obj

def is_sensitive_key(k, rules):
    # simple predicate: redact keys matching patterns (PII, raw_image)
    sensitive = ['ssn', 'person_name', 'raw_image', 'audio_blob']
    return k in sensitive

def canonicalize(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def sign(payload_bytes):
    return hmac.new(secret_key, payload_bytes, hashlib.sha256).hexdigest()

def create_and_send_snapshot(context, rules, endpoint):
    # 1. sample selection already applied to context
    redacted = redact(context, rules)                      # redact PII and raw blobs
    payload = { 'ts': int(time.time()), 'seq': context['seq'], 'body': redacted }
    payload_b = canonicalize(payload)
    sig = sign(payload_b)
    envelope = { 'payload': payload, 'hmac': sig }
    # 2. upload over TLS with client auth (omitted for brevity)
    r = requests.post(endpoint, json=envelope, timeout=10) # server verifies HMAC
    r.raise_for_status()
    return r.status_code