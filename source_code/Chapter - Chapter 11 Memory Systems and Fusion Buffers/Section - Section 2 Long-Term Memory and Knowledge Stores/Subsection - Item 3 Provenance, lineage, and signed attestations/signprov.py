import json, hashlib, time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives.asymmetric.utils import (
    encode_dss_signature, decode_dss_signature)

# generate ephemeral key (replace with KMIP/HSM in production)
priv = ec.generate_private_key(ec.SECP256R1())
pub = priv.public_key()
pub_pem = pub.public_bytes(
    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)

def hash_chain(prev_hash, data, meta):
    payload = (prev_hash or "").encode() + json.dumps(data, sort_keys=True).encode() \
              + json.dumps(meta, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()

# example artifact: derived track update
prev = None  # previously stored chain root
data = {"track_id":"T42","pos":[12.3,45.6],"cov":[[0.1,0.],[0.,0.1]]}
meta = {"sensor":"radarA","proc":"kf_v1","ts":time.time()}

H = hash_chain(prev, data, meta)
sig = priv.sign(H.encode(), ec.ECDSA(hashes.SHA256()))
r, s = decode_dss_signature(sig)

record = {
  "hash": H, "data": data, "meta": meta,
  "signature": {"r": r, "s": s},
  "pubkey_pem": pub_pem.decode()
}
print(json.dumps(record, indent=2))  # store into DB/graph
# verification: recompute hash and verify signature
from cryptography.hazmat.primitives.asymmetric import ec as _ec
pub2 = serialization.load_pem_public_key(record["pubkey_pem"].encode())
sig2 = encode_dss_signature(record["signature"]["r"], record["signature"]["s"])
pub2.verify(sig2, record["hash"].encode(), _ec.ECDSA(hashes.SHA256()))
# raises exception if invalid