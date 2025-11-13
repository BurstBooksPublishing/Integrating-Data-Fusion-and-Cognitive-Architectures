import networkx as nx
import math

# Build directed evidence graph
G = nx.DiGraph()
# Add nodes: (id, dict of attrs)
G.add_node("E1", type="event", desc="ego_decel", conf=0.9, prov="brake01")
G.add_node("E2", type="event", desc="ped_emerge", conf=0.7, prov="fusion_cam_lidar")
G.add_node("C1", type="condition", desc="occlusion", conf=0.8, prov="map+vision")
G.add_node("H",  type="hypothesis", desc="imminent_collision", conf=0.5, prov="planner")

# Add support/causal edges with weights
G.add_edge("E1", "H", rel="supports", weight=0.6, rule="r1")
G.add_edge("E2", "H", rel="enables",  weight=0.85, rule="r2")
G.add_edge("C1", "E2", rel="enables", weight=0.7, rule="r3")

# Function: score a chain given ordered node list
def score_chain(G, chain):
    logp = 0.0
    for n in chain:
        node_conf = G.nodes[n].get("conf", 1e-6)
        logp += math.log(max(node_conf, 1e-6))
    for u, v in zip(chain, chain[1:]):
        w = G.edges[u, v].get("weight", 1e-6)
        logp += math.log(max(w, 1e-6))
    return math.exp(logp)

# Define chain: C1 -> E2 -> H (including E1 via separate support path)
chain1 = ["C1", "E2", "H"]
print("Chain score C1->E2->H:", score_chain(G, chain1))
# Rationale extraction: minimal supporting nodes for H
supporting = [u for u in G.predecessors("H")]
trace = supporting + ["H"]
print("Rationale trace for H:", trace)