import numpy as np, time

class Anchor:
    def __init__(self, mean, cov, weight=1.0, meta=None):
        self.mean = np.array(mean)      # 2D or 3D position
        self.cov = np.array(cov)        # covariance matrix
        self.weight = weight
        self.meta = meta or {}
        self.ts = time.time()

class MemoryStore:
    def __init__(self, lambda_decay=0.001):
        self.anchors = {}               # id -> Anchor
        self.constraints = {}           # user_id -> dict of constraints
        self.lambda_decay = lambda_decay

    def add_observation(self, aid, z_mean, z_cov, source):
        # create or update anchor using Gaussian fusion (Kalman update)
        z_mean = np.array(z_mean); z_cov = np.array(z_cov)
        if aid not in self.anchors:
            self.anchors[aid] = Anchor(z_mean, z_cov, weight=1.0, meta={'sources':[source]})
            return
        A = self.anchors[aid]
        # Kalman-gain style update
        S = A.cov + z_cov
        K = A.cov @ np.linalg.inv(S)
        A.mean = A.mean + K @ (z_mean - A.mean)
        A.cov = (np.eye(len(A.mean)) - K) @ A.cov
        A.weight = A.weight + 1.0
        A.meta.setdefault('sources',[]).append(source)
        A.ts = time.time()

    def apply_user_constraints(self, user_id, plan):
        # simple veto: if user marks anchor immutable, block moves
        blocked = []
        for aid, anchor in self.anchors.items():
            c = self.constraints.get(user_id, {}).get(aid, {})
            if c.get('immutable') and plan.targets_move(aid):
                blocked.append(aid)
        return blocked

    def evict_stale(self, now=None):
        now = now or time.time()
        to_delete = []
        for aid, anchor in self.anchors.items():
            age = now - anchor.ts
            weight = anchor.weight * np.exp(-self.lambda_decay * age)  # decay
            if weight < 0.1: to_delete.append(aid)
        for aid in to_delete: del self.anchors[aid]  # quarantine in real systems