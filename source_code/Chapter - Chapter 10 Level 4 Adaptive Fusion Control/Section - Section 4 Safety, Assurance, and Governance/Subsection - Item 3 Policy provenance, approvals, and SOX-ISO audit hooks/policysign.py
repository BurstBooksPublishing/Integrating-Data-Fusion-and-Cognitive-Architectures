import json, hashlib, sqlite3, time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# generate keypair (production: use persistent keys/ HSM)
sk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
pk = sk.public_key()

def sha256(b): return hashlib.sha256(b).hexdigest()

# create policy artifact
policy = {"policy_id":"P-2025-001","version":1,"rules":["sensor_rate=30Hz"]}
payload = json.dumps(policy, sort_keys=True).encode()

# sign payload
signature = sk.sign(payload, padding.PKCS1v15(), hashes.SHA256())

# compute chained hash (previous read from ledger)
conn = sqlite3.connect("policy_ledger.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS ledger(idx INTEGER PRIMARY KEY, h TEXT, payload TEXT, sig BLOB, ts REAL)")
c.execute("SELECT h FROM ledger ORDER BY idx DESC LIMIT 1")
row = c.fetchone()
prev_h = row[0] if row else "0"*64
entry_h = sha256((prev_h + payload.decode()).encode())

# append to ledger
c.execute("INSERT INTO ledger(h,payload,sig,ts) VALUES(?,?,?,?)",
          (entry_h, payload.decode(), signature, time.time()))
conn.commit()
conn.close()
# runtime: include policy["policy_id"], policy["version"], signature with decisions