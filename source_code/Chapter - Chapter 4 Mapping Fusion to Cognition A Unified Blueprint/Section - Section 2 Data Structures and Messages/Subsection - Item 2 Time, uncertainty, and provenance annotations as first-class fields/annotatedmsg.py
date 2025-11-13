from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import numpy as np

@dataclass
class ProvenanceRecord:
    source: str
    op: str
    timestamp: datetime
    version: str

@dataclass
class AnnotatedMessage:
    mean: np.ndarray                 # state vector
    cov: np.ndarray                  # covariance matrix (aleatoric)
    phys_time: datetime              # UTC physical time
    seq: int                         # monotonic sequence number
    epistemic_conf: float = 1.0      # model confidence [0,1]
    prov: list = field(default_factory=list)  # list[ProvenanceRecord]

    def append_prov(self, source, op, version="v1"):
        self.prov.append(ProvenanceRecord(source, op, datetime.now(timezone.utc), version))

    def propagate(self, A, Q, dt):
        # simple linear propagation to align time by dt seconds
        self.mean = A @ self.mean
        self.cov = A @ self.cov @ A.T + Q
        self.phys_time = self.phys_time + timedelta(seconds=dt)
        self.append_prov("propagator", f"propagate dt={dt}", version="v1")

    def fuse_information(self, other):
        # fuse two Gaussian estimates via information form (assumes independence)
        J1 = np.linalg.inv(self.cov)
        J2 = np.linalg.inv(other.cov)
        J = J1 + J2
        cov = np.linalg.inv(J)
        mean = cov @ (J1 @ self.mean + J2 @ other.mean)
        new = AnnotatedMessage(mean, cov, max(self.phys_time, other.phys_time),
                               max(self.seq, other.seq), 
                               epistemic_conf=min(self.epistemic_conf, other.epistemic_conf),
                               prov=self.prov + other.prov)
        new.append_prov("fuser", "information_fusion", version="v1")
        return new

# Example usage (requires numpy)
if __name__ == "__main__":
    x1 = AnnotatedMessage(np.array([0.,0.]), np.eye(2)*2.0, datetime.now(timezone.utc), seq=1)
    x2 = AnnotatedMessage(np.array([1.,0.5]), np.eye(2)*1.0, datetime.now(timezone.utc), seq=2)
    fused = x1.fuse_information(x2)  # fuse with provenance preserved
    print(fused.mean, fused.cov, len(fused.prov))  # quick diagnostic