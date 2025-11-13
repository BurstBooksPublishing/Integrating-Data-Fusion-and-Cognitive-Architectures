import numpy as np
import networkx as nx

np.random.seed(0)

# synthetic tracks: (id, time, state(2), cov(2x2))
tracks = [
    ('A', 0.0, np.array([0.0,0.0]), np.eye(2)*0.1),
    ('B', 0.1, np.array([0.2,0.0]), np.eye(2)*0.2),
    ('C', 0.2, np.array([5.0,5.0]), np.eye(2)*0.5),
]

# simple linear encoder (random projection) and Jacobian constant
W = np.random.randn(3,2)  # maps 2->3 dims
def encode(mu):
    return W.dot(mu)         # embedding

def emb_cov(sigma):
    J = W                   # Jacobian of linear map
    return J.dot(sigma).dot(J.T)

# build graph with node features
G = nx.Graph()
for tid,t,mu,sig in tracks:
    e = encode(mu)
    Se = emb_cov(sig)
    G.add_node(tid, time=t, mu=mu, Sigma=sig, emb=e, Se=Se)

# edge gating using Mahalanobis threshold tau
tau = 9.21  # chi-square(2,0.99)
for i in G.nodes:
    for j in G.nodes:
        if i>=j: continue
        mu_i,mu_j = G.nodes[i]['mu'], G.nodes[j]['mu']
        Si, Sj = G.nodes[i]['Sigma'], G.nodes[j]['Sigma']
        d2 = (mu_i-mu_j).T.dot(np.linalg.inv(Si+Sj)).dot(mu_i-mu_j)
        if d2 <= tau:
            G.add_edge(i,j,weight=np.exp(-0.5*d2))  # soft weight

# chunk: window 0.0..0.2, inverse-covariance weighted avg of embeddings
nodes_in_window = [n for n,d in G.nodes(data=True) if 0.0 <= d['time'] <= 0.2]
num = np.zeros(3); den = np.zeros((3,3))
for n in nodes_in_window:
    e = G.nodes[n]['emb']
    Se = G.nodes[n]['Se']
    Wi = np.linalg.pinv(Se)         # weight matrix
    num += Wi.dot(e)
    den += Wi
e_chunk = np.linalg.solve(den,num)  # chunk embedding
# store chunk as node
G.add_node('chunk_0', emb=e_chunk, members=nodes_in_window)
print("Chunk embedding:", G.nodes['chunk_0']['emb'])