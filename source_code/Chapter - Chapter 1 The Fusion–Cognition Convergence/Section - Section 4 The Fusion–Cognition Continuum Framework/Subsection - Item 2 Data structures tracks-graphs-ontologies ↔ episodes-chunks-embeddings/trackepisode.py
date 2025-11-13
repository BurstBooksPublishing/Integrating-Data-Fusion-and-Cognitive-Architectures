import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# observations: small textual descriptors per timestamp (toy)
obs_texts = [
    "slow loitering near buoy, bearing 120°, speed 2 kn",
    "course change toward port, reduced speed",
    "radar cross-section increased, closer to commercial lane"
]
# synthetic covariances (trace proxies) per obs
cov_traces = np.array([0.5, 1.2, 0.8])  # lower is more certain

# vectorize observations (serves as embeddings here)
vec = TfidfVectorizer().fit(obs_texts)
e_obs = vec.transform(obs_texts).toarray()  # shape (N, D)

# compute weights per eq. (1)
eps = 1e-6
weights = 1.0 / (cov_traces + eps)
weights = weights / weights.sum()  # normalized

# weighted aggregation to track embedding
e_track = (weights[:, None] * e_obs).sum(axis=0)

# episodic store: simple list of precomputed embeddings
episodes = {
  "fishing_pattern": vec.transform(["repetitive loitering near buoy"]).toarray()[0],
  "transit": vec.transform(["straight transit through lane"]).toarray()[0]
}

# retrieval by cosine similarity
scores = {k: float(cosine_similarity(e_track.reshape(1,-1), v.reshape(1,-1))) 
          for k,v in episodes.items()}
print("Retrieval scores:", scores)  # higher = more similar