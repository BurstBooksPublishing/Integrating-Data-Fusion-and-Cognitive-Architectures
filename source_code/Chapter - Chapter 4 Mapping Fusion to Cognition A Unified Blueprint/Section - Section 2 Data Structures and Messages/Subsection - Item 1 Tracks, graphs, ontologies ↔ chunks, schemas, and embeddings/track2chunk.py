import numpy as np
import networkx as nx

# sample track: list of (time, feature_vec, conf)
track = [(0, np.array([0.1, 0.9]), 0.8),
         (1, np.array([0.2, 0.8]), 0.9),
         (2, np.array([0.4, 0.6]), 0.7)]

# simple graph: nodes are track id and nearby object id
G = nx.DiGraph()
G.add_node("track_1", kind="track")
G.add_node("obj_A", kind="object", class="vehicle")
G.add_edge("track_1", "obj_A", rel="near")

# weighted pooling for embedding (weights = confidence here)
feats = np.stack([f for (_, f, _) in track])
confs = np.array([c for (_, _, c) in track])
weights = confs / confs.sum()
embedding = (weights[:,None] * feats).sum(axis=0)  # dense vector

# simple symbolization against a toy ontology
ontology = {"vehicle": {"speed_slot": "float", "role": "string"}}
chunk = {
    "id": "chunk_123",
    "type": "observation",
    "class": "vehicle",                    # grounded by graph/ontology
    "embedding": embedding.tolist(),       # stored for retrieval
    "schema_slots": {"speed": float(embedding[0]*30)}, # heuristic mapping
    "timestamp": track[-1][0],
    "uncertainty": float(1.0 - weights.max()),
    "provenance": {"source": "radar_track", "track_id": "track_1"}
}

print("Graph nodes:", G.nodes(data=True))
print("Chunk:", chunk)