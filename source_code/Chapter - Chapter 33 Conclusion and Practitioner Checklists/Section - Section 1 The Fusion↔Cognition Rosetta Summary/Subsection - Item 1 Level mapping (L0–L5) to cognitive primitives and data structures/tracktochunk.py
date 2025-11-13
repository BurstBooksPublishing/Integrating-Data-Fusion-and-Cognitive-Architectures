import numpy as np
import networkx as nx

# simple track representation (id, state(3), cov(3x3), appearance vec)
tracks = [
    (1, np.array([10.0, 5.0, 0.0]), np.eye(3)*0.5, np.random.rand(16)),
    (2, np.array([12.0, 5.5, 0.0]), np.eye(3)*0.8, np.random.rand(16)),
]

# package tracks into cognitive chunks with provenance
chunks = []
for tid, state, cov, emb in tracks:
    chunk = {
        "chunk_id": f"track-{tid}",            # use string ids for registries
        "state": state,
        "covariance": cov,
        "embedding": emb.tolist(),             # serializable embedding
        "provenance": {"source":"L1_tracker","timestamp":0.0}
    }
    chunks.append(chunk)

# build L2 situation graph by geometric proximity relation
G = nx.DiGraph()
for c in chunks:
    G.add_node(c["chunk_id"], **c)

# relate nodes within distance threshold
threshold = 3.0
for i in range(len(chunks)):
    for j in range(i+1, len(chunks)):
        a = chunks[i]["state"][:2]; b = chunks[j]["state"][:2]
        d = np.linalg.norm(a-b)
        if d < threshold:
            G.add_edge(chunks[i]["chunk_id"], chunks[j]["chunk_id"],
                       relation="near", distance=float(d))
# G is a queryable situation graph for downstream reasoning