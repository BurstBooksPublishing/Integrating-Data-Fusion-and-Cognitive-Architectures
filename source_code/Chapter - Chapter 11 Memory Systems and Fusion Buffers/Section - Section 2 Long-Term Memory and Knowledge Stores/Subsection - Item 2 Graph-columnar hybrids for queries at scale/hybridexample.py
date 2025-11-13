import networkx as nx
import pandas as pd
import duckdb  # pip install duckdb
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Build graph (small example)
G = nx.Graph()
G.add_edges_from([('A','B'),('B','C'),('A','D')])         # # relation edges
G.nodes['A']['type']='track'

# Columnar table: embeddings and metadata
df = pd.DataFrame([
  {'node_id':'A','ts':1000,'embedding':np.random.rand(128).tolist()},
  {'node_id':'B','ts':1005,'embedding':np.random.rand(128).tolist()},
  {'node_id':'C','ts': 990,'embedding':np.random.rand(128).tolist()},
  {'node_id':'D','ts':1010,'embedding':np.random.rand(128).tolist()},
])
con = duckdb.connect()           # columnar SQL runtime
con.register('episodes_df', df)
# Candidate set via graph traversal
candidates = list(nx.single_source_shortest_path_length(G,'A',cutoff=1).keys())

# Fetch candidate embeddings from columnar store
q = f"SELECT node_id, ts, embedding FROM episodes_df WHERE node_id IN {tuple(candidates)}"
res = con.execute(q).df()

# Prepare ANN index restricted to candidates
X = np.vstack(res['embedding'].apply(np.array).values)
nn = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='cosine').fit(X)
probe = np.random.rand(128).reshape(1,-1)  # plan prototype embedding
dist, idx = nn.kneighbors(probe)
matches = res.iloc[idx[0]]['node_id'].tolist()
print('Matched candidates:', matches)  # use for downstream reasoning