from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

# Simulate CMK stored in HSM/KMS (RSA key pair)
cmk_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
cmk_pub = cmk_priv.public_key()

def encrypt_payload(plaintext: bytes) -> dict:
    dek = AESGCM.generate_key(bit_length=256)               # per-object DEK
    aesg = AESGCM(dek)
    nonce = os.urandom(12)
    ciphertext = aesg.encrypt(nonce, plaintext, None)      # AEAD
    wrapped_dek = cmk_pub.encrypt(                          # wrap DEK with CMK
        dek,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None)
    )
    return {"ciphertext": ciphertext, "nonce": nonce, "wrapped_dek": wrapped_dek}

def decrypt_payload(record: dict) -> bytes:
    wrapped_dek = record["wrapped_dek"]
    dek = cmk_priv.decrypt(                                 # unwrap DEK (KMS/HSM op)
        wrapped_dek,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None)
    )
    aesg = AESGCM(dek)
    return aesg.decrypt(record["nonce"], record["ciphertext"], None)

# Usage: encrypt before storing into object store; unwrap when reading.