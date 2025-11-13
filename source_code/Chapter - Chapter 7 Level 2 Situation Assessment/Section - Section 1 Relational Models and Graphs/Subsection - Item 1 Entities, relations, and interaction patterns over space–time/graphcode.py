import numpy as np
import networkx as nx

# sample tracks: id -> {pos, vel, class_prob, cov}
tracks = {
  1: {'pos': np.array([0.,0.]), 'vel': np.array([1.,0.]), 'cls': 0.9, 'cov': np.eye(2)*0.5},
  2: {'pos': np.array([2.,0.2]), 'vel': np.array([-0.5,0.0]), 'cls': 0.8,'cov': np.eye(2)*0.7},
  3: {'pos': np.array([5.,1.]), 'vel': np.array([0.,-1.]), 'cls': 0.6,'cov': np.eye(2)*1.0},
}

sigma_d = 2.0; kappa = 1.0; edge_thresh = 0.05

G = nx.Graph()
for tid, tr in tracks.items():
    G.add_node(tid, pos=tr['pos'], vel=tr['vel'], cls=tr['cls'])

# pairwise affinity as in eq. (1)
def affinity(a,b):
    d = np.linalg.norm(a['pos']-b['pos'])
    rel_head = np.dot(a['vel'], (b['pos']-a['pos'])) # proxy alignment
    score = np.exp(-d**2/(2*sigma_d**2))*np.exp(kappa * (rel_head/(1e-6+np.linalg.norm(a['vel'])*d)))
    score *= min(a['cls'], b['cls']) # semantic compatibility
    return float(score)

# build edges
for i in tracks:
    for j in tracks:
        if i>=j: continue
        w = affinity(tracks[i], tracks[j])
        if w>edge_thresh:
            G.add_edge(i,j,weight=w)

# simple message passing: weighted neighbor average of velocities
new_feats = {}
for n in G.nodes:
    nbrs = list(G[n])
    if not nbrs:
        new_feats[n] = G.nodes[n]['vel']
        continue
    weights = np.array([G[n][m]['weight'] for m in nbrs])
    weights /= weights.sum()
    agg = sum(weights[k]*G.nodes[nbrs[k]]['vel'] for k in range(len(nbrs)))
    new_feats[n] = 0.5*G.nodes[n]['vel'] + 0.5*agg  # residual update

# commit updates
for n, v in new_feats.items():
    G.nodes[n]['vel'] = v
# G now has pruned edges and updated node features