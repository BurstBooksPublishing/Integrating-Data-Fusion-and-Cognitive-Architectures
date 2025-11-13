#!/usr/bin/env python3
import time, json, hmac, hashlib, sqlite3
# Simple monitor: compute score and trigger fallback; sign audits with HMAC.

SECRET=b'supersecretkey'               # secure key management required in prod
DB='audit.db'

# init DB (idempotent)
conn=sqlite3.connect(DB); c=conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS audit
(id INTEGER PRIMARY KEY AUTOINCREMENT, seq INTEGER, t REAL, payload TEXT, mac TEXT)''')
conn.commit()

def sign(payload:bytes)->str:
    return hmac.new(SECRET,payload,hashlib.sha256).hexdigest()

def compute_score(metrics:dict)->float:
    # weights tuned in validation; normalize externally
    w={'sensor_ok':0.5,'model_conf':0.3,'assoc_rate':0.2}
    return sum(w[k]*metrics.get(k,0.0) for k in w)

def record_audit(seq:int, payload:dict):
    raw=json.dumps(payload,separators=(',',':')).encode()
    mac=sign(raw)
    c.execute('INSERT INTO audit(seq,t,payload,mac) VALUES(?,?,?,?)',
              (seq,time.time(),raw.decode(),mac))
    conn.commit()

def apply_fallback(state:str):
    # stub: invoke control API; idempotent and logged
    print("Applying fallback:",state)

# runtime loop (simulated)
seq=0
while True:
    seq+=1
    # example metrics from fusion & cognition layers
    metrics={'sensor_ok':0.6,'model_conf':0.4,'assoc_rate':0.8}
    S=compute_score(metrics)
    payload={'seq':seq,'score':S,'metrics':metrics}
    if S < 0.5:
        payload['action']='fallback'
        payload['safe_state']='hold_and_notify'
        apply_fallback(payload['safe_state'])
    else:
        payload['action']='nominal'
    record_audit(seq,payload)
    time.sleep(1)  # loop cadence matched to system SLOs