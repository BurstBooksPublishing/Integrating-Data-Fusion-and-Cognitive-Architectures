import numpy as np
# prototype embeddings for known classes (rows) and their symbolic labels
prototypes = np.array([[0.5,0.5,0.7],[0.1,0.9,0.4]])  # small example
labels = ["vehicle","pedestrian"]
def normalize(x): return x/np.linalg.norm(x)
# observation features → embedding (identity here; replace with model.encode)
obs_feat = np.array([0.45,0.48,0.65])
e = normalize(obs_feat)                                 # subsymbolic representation
protos = np.vstack([normalize(p) for p in prototypes])
s = protos @ e                                          # cosine scores
alpha = 10.0
p = np.exp(alpha*s) / np.sum(np.exp(alpha*s))           # Eq. (1)
best = int(np.argmax(p))
if p[best] > 0.7:                                       # grounding threshold
    triple = ("obs_123","hasLabel", labels[best])       # symbolic assertion
    meta = {"belief": float(p[best]), "method":"cosine_softmax"}  # provenance
else:
    triple = ("obs_123","hasLabel","unknown")
    meta = {"belief": float(p[best]), "method":"cosine_softmax", "action":"defer"}
print(triple, meta)  # emit to symbolic store / working memory