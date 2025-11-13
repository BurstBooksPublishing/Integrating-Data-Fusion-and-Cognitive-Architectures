# Requires numpy
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class UncertaintyToken:
    mean: np.ndarray               # state vector
    cov: np.ndarray                # aleatoric covariance
    epistemic_var: float           # scalar epistemic variance (isotropic)
    symbolic_conf: float           # [0,1] confidence for symbolic assertion
    valid_until: datetime
    provenance: str

def to_information(tok: UncertaintyToken):
    # add isotropic epistemic variance
    cov_aug = tok.cov + tok.epistemic_var * np.eye(tok.cov.shape[0])
    info = np.linalg.inv(cov_aug)
    info_vec = info @ tok.mean
    return info, info_vec

def fuse_tokens(tokens):
    infos = []
    info_vecs = []
    for t in tokens:
        i, iv = to_information(t)
        # incorporate symbolic confidence as a weight on information
        w = max(1e-3, t.symbolic_conf)  # avoid zero info
        infos.append(w * i)
        info_vecs.append(w * iv)
    I_fused = sum(infos)
    iv_fused = sum(info_vecs)
    cov_fused = np.linalg.inv(I_fused)
    mu_fused = cov_fused @ iv_fused
    # aggregate provenance and validity conservatively
    valid_until = min(t.valid_until for t in tokens)
    provenance = ";".join(t.provenance for t in tokens)
    # approximate fused epistemic as max of inputs (conservative)
    epi = max(t.epistemic_var for t in tokens)
    return UncertaintyToken(mu_fused, cov_fused, epi, 1.0, valid_until, provenance)

def control_safe(tok: UncertaintyToken, d_min: float, kappa: float=3.0):
    # use largest principal axis for conservative margin
    eigvals = np.linalg.eigvalsh(tok.cov + tok.epistemic_var*np.eye(tok.cov.shape[0]))
    margin = kappa * np.sqrt(np.max(eigvals))
    return margin < d_min

# Example usage
if __name__ == "__main__":
    now = datetime.utcnow()
    t1 = UncertaintyToken(np.array([10.,0.]), np.diag([4.,1.]), 1.0, 0.9, now+timedelta(seconds=2), "radar:v1")
    t2 = UncertaintyToken(np.array([9.5,0.2]), np.diag([1.,2.]), 0.5, 0.8, now+timedelta(seconds=1), "lidar:v2")
    fused = fuse_tokens([t1,t2])
    safe = control_safe(fused, d_min=2.0)
    print("Fused mean:", fused.mean, "Safe:", safe)  # simple observable