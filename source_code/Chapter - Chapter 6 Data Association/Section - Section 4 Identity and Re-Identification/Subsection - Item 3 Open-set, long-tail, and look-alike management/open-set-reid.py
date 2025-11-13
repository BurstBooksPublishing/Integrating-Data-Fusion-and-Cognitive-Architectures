import numpy as np
from collections import defaultdict

# simple cosine similarity
def cos_sim(a, b): return a.dot(b) / (np.linalg.norm(a)*np.linalg.norm(b))

class ReIDStore:
    def __init__(self, open_thresh=0.15, lookalike_delta=0.05, alpha=0.1):
        self.protos = {}               # id -> prototype vector
        self.counts = defaultdict(int) # id -> observations count
        self.open_thresh = open_thresh
        self.lookalike_delta = lookalike_delta
        self.alpha = alpha             # ema update weight

    def score(self, e):                # e is unit-normalized embedding
        sims = {k: cos_sim(e, v) for k, v in self.protos.items()}
        if not sims:
            return None, 0.0, sims
        topk = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        return topk[0][0], topk[0][1], dict(topk[:5]) # top id, score, top-5

    def ingest(self, e, cross_modal=None):
        top_id, top_score, top_sims = self.score(e)
        open_score = 1.0 - top_score if top_id is not None else 1.0
        if open_score > self.open_thresh:
            # create new identity (open-set)
            new_id = f"id_{len(self.protos)+1}"
            self.protos[new_id] = e.copy()
            self.counts[new_id] = 1
            return ("new", new_id, open_score)
        # look-alike detection: check second-best close to top
        sims_sorted = sorted(top_sims.items(), key=lambda kv: kv[1], reverse=True)
        if len(sims_sorted) > 1 and (sims_sorted[0][1]-sims_sorted[1][1]) < self.lookalike_delta:
            # require cross-modal corroboration or defer to cognition
            if cross_modal and cross_modal == top_id:
                self._update_proto(top_id, e)
                return ("accept", top_id, top_score)
            else:
                return ("ambiguous", top_id, top_score)
        # confident match; update prototype with EMA
        self._update_proto(top_id, e)
        return ("accept", top_id, top_score)

    def _update_proto(self, id_, e):
        if self.counts[id_] == 0:
            self.protos[id_] = e.copy()
            self.counts[id_] = 1
            return
        # EMA update to avoid domination by frequent classes
        self.protos[id_] = (1-self.alpha)*self.protos[id_] + self.alpha*e
        self.counts[id_] += 1