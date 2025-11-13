import math
import networkx as nx
from networkx.algorithms import isomorphism as iso

# Build event graph from perception outputs (type, t, x,y,score).
G = nx.DiGraph()
detections = [
    {'id':'a1','type':'approach','t':10.0,'pos':(0.0,0.0),'p':0.9},
    {'id':'b1','type':'approach','t':12.0,'pos':(1.0,1.0),'p':0.85},
    {'id':'a2','type':'loiter','t':18.0,'pos':(0.2,0.1),'p':0.7},
]
for d in detections:
    G.add_node(d['id'], **d)

# Add simple temporal edges (precedence) and proximity edges (distance)
def dist(u,v):
    x1,y1 = G.nodes[u]['pos']; x2,y2 = G.nodes[v]['pos']
    return math.hypot(x1-x2,y1-y2)

for u in G.nodes:
    for v in G.nodes:
        if u==v: continue
        if G.nodes[u]['t'] < G.nodes[v]['t']:
            G.add_edge(u,v,rel='before')
        if dist(u,v) < 2.0:
            G.add_edge(u,v,rel='near')

# Define symbolic template graph for "rendezvous": two 'approach' then 'loiter'
T = nx.DiGraph()
T.add_node('X', type='approach')
T.add_node('Y', type='approach')
T.add_node('Z', type='loiter')
T.add_edge('X','Y', rel='before')
T.add_edge('Y','Z', rel='before')
T.add_edge('X','Y', rel='near')  # require proximity

# Node attribute matcher: types must match
nm = iso.categorical_node_match('type', None)
# Edge attribute matcher: relation must be subset
em = iso.categorical_edge_match('rel', None)

GM = iso.DiGraphMatcher(G, T, node_match=nm, edge_match=em)

# Scoring weights (calibrated offline)
alpha,beta,gamma = 1.0, 0.8, 0.5
prior_pi = 0.01

def temporal_consistency(nodes):
    times = [G.nodes[n]['t'] for n in nodes]
    # penalty if gaps exceed window (simple heuristic)
    max_gap = max(times) - min(times)
    return math.exp(-0.1 * max_gap)

hypotheses = []
for mapping in GM.subgraph_isomorphisms_iter():
    node_ids = [mapping[k] for k in ['X','Y','Z']]
    p_prod = 1.0
    for n in node_ids:
        p_prod *= G.nodes[n]['p']  # multiplicative evidence
    r = 1.0  # template match binary here; could be fuzzy
    c = temporal_consistency(node_ids)
    logS = alpha * sum(math.log(G.nodes[n]['p']) for n in node_ids) \
           + beta * math.log(r) + gamma * math.log(c) + math.log(prior_pi)
    S = math.exp(logS)
    hypotheses.append({'nodes':node_ids,'score':S,'mapping':mapping})

# Output top hypothesis
hypotheses.sort(key=lambda h: h['score'], reverse=True)
print(hypotheses)
# (In production, attach provenance, rule hits, and confidence bands.)