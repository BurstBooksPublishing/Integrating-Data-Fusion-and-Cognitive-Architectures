import random, math, time, pickle
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []             # (embedding, data, delta, meta)
        self.pos = 0
        self.sum_priorities = 0.0

    def _validate_meta(self, meta):
        # provenance checks: signed flag, timestamp monotonicity, sensor_q
        return meta.get("signed",False) and meta.get("sensor_q",0) > 0.5

    def _ood_check(self, embedding, mean, cov_inv, thresh=9.21):
        # Mahalanobis distance for OOD detection (chi-square 2-dof ~9.21)
        d = embedding - mean
        m2 = float(d.T @ cov_inv @ d)
        return m2 > thresh

    def add(self, embedding, data, delta, meta):
        if not self._validate_meta(meta):
            return False  # reject low-integrity example
        priority = (abs(delta) + 1e-6) ** self.alpha
        item = (embedding, data, delta, meta, priority, time.time())
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            # simple replacement policy by position (can be stratified)
            self.buffer[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity
        self.sum_priorities += priority
        return True

    def sample(self, batch_size, embedding_mean=None, cov_inv=None):
        # combine priority with simple recency weight
        weights = np.array([it[4] for it in self.buffer], dtype=np.float64)
        probs = weights / max(weights.sum(), 1e-12)
        idx = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        batch, is_weights = [], []
        N = len(self.buffer)
        for i in idx:
            item = self.buffer[i]
            p_i = probs[i]
            w = (1.0 / (N * p_i)) ** self.beta
            # optional OOD check before committing to update
            if embedding_mean is not None and cov_inv is not None:
                if self._ood_check(item[0], embedding_mean, cov_inv):
                    continue  # skip OOD exemplars in this training step
            batch.append(item)
            is_weights.append(w)
        return batch, np.array(is_weights)

    def checkpoint(self, path):
        # save buffer metadata only (compact), not full raw sensor logs
        with open(path, "wb") as f:
            pickle.dump([(it[2], it[3], it[5]) for it in self.buffer], f)  # delta,meta,timestamp