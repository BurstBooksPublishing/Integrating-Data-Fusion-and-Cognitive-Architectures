import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian
import networkx as nx

# Simulated tracks and detections (2D states with covariances)
rng = np.random.default_rng(0)
m, n = 5, 7
tracks = rng.normal(size=(m,2))
tracks_cov = np.array([np.eye(2)*0.5 for _ in range(m)])
dets = rng.normal(loc=0.2, size=(n,2))
# Mahalanobis cost matrix
C = np.full((m,n), 1e6)
gating_r2 = 9.0  # squared gating radius
for i in range(m):
    P = tracks_cov[i]
    P_inv = np.linalg.inv(P)
    for j in range(n):
        v = dets[j]-tracks[i]
        d2 = float(v.T @ P_inv @ v)
        if d2 <= gating_r2:
            C[i,j] = d2  # cost = Mahalanobis distance

# Hungarian (handles rectangular by padding)
row_ind, col_ind = linear_sum_assignment(C)
matches_hungarian = [(i,j) for i,j in zip(row_ind, col_ind) if C[i,j] < 1e5]

# Simple auction algorithm (epsilon-scaling, synchronous version)
def auction(cost, eps=1e-2):
    m,n = cost.shape
    price = np.zeros(n)
    owner = -np.ones(n, dtype=int)
    unassigned = list(range(m))
    while unassigned:
        i = unassigned.pop()
        # bidder i finds best and second-best
        vals = cost[i,:] - price
        j = int(np.argmin(vals))
        best = vals[j]
        vals[j] = np.inf
        second = np.min(vals)
        bid = best - second + eps
        # award
        if owner[j] != -1:
            unassigned.append(owner[j])  # previous owner becomes unassigned
        owner[j] = i
        price[j] += bid
    # collect matches
    return [(owner[j], j) for j in range(n) if owner[j] != -1 and cost[owner[j], j] < 1e5]

matches_auction = auction(C)

# Min-cost flow formulation with networkx (supply = -demand convention)
G = nx.DiGraph()
# add nodes with demands: tracks supply=1 -> demand=-1, detections demand=1
for i in range(m):
    G.add_node(f"t{i}", demand=-1)
for j in range(n):
    G.add_node(f"d{j}", demand=1)
# edges t->d with capacity 1 and weight=cost
for i in range(m):
    for j in range(n):
        if C[i,j] < 1e5:
            G.add_edge(f"t{i}", f"d{j}", weight=float(C[i,j]), capacity=1)
# network_simplex returns (cost, flow_dict)
cost_flow, flow_dict = nx.network_simplex(G)
matches_flow = []
for u, nbrs in flow_dict.items():
    for v, f in nbrs.items():
        if f > 0 and u.startswith('t') and v.startswith('d'):
            matches_flow.append((int(u[1:]), int(v[1:])))
# Print succinct diagnostics
print("Hungarian:", matches_hungarian)
print("Auction:  ", matches_auction)
print("Flow:     ", matches_flow)