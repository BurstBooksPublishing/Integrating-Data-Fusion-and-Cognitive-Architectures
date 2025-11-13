import numpy as np
import networkx as nx

# Build graph (5 nodes) and capacities
G = nx.path_graph(5)
W = nx.to_numpy_array(G)         # adjacency for redistribution
capacity = np.array([1.0,0.8,1.2,1.0,0.9])   # per-node capacity (pu)
# current loads (pu) from fused telemetry + forecast
load = np.array([0.9,0.95,1.1,0.85,0.7])

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def simulateStep(load, capacity, W, k=20.0, alpha=0.6):
    # compute overload ratio and failure probability
    ratio = load/capacity
    pFail = sigmoid(k*(ratio-1.0))
    # sample failures (deterministic threshold could be used instead)
    fail = (pFail > 0.5).astype(float)   # simple decision rule
    # compute redistributed load from failed nodes
    redistributed = (fail * (alpha*capacity)) @ W   # sends amount to neighbors
    # update loads (neighbors receive redistributed load / degree)
    deg = W.sum(axis=1); deg[deg==0]=1.0
    load_next = load + redistributed/deg
    # mitigation: if nodes near overload, propose controlled shed
    mitigation = np.minimum(0.1, np.maximum(0.0, ratio-0.95))  # fraction to shed
    return load_next, fail, mitigation

load_next, fail, mitigation = simulateStep(load, capacity, W)
print("next loads", load_next)         # telemetry update to working memory
print("failures", fail)                # L2 hypotheses for operator
print("mitigation fractions", mitigation)  # L4 action suggestions