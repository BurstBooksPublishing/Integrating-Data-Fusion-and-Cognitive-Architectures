from dataclasses import dataclass, field
from typing import List
import time, math, heapq

@dataclass
class Episode:
    id: str
    last_ts: float
    salience: float
    info_gain: float
    size_bytes: int
    immutable: bool = False  # e.g., legal/audit hold
    data: dict = field(default_factory=dict)

def retention_score(ep: Episode, alpha=1.0, beta=1.0, gamma=1.0, lamb=1e-5):
    age = time.time() - ep.last_ts
    return alpha * math.exp(-lamb * age) + beta*ep.salience + gamma*ep.info_gain

def compact_episode(ep: Episode):
    # compacting strategy: downsample state history and compress embeddings
    # here we simulate by reducing size and annotating provenance
    ep.data['compacted'] = True
    ep.data['compaction_ts'] = time.time()
    ep.size_bytes = max(int(ep.size_bytes * 0.2), 256)  # reduce size, keep minimum
    return ep

def garbage_collect(episodes: List[Episode], byte_budget: int):
    total = sum(e.size_bytes for e in episodes)
    if total <= byte_budget:
        return episodes
    # build min-heap by retention score
    heap = [(retention_score(e), e) for e in episodes if not e.immutable]
    heapq.heapify(heap)
    # compact lowest-score episodes until under budget
    while total > byte_budget and heap:
        score, ep = heapq.heappop(heap)
        ep = compact_episode(ep)            # inplace compaction
        total = sum(e.size_bytes for e in episodes)
        # re-evaluate and re-insert if further compaction possible
        if ep.size_bytes > 512:
            heapq.heappush(heap, (retention_score(ep), ep))
    return episodes

# Example usage: run worker periodically (scheduler omitted).