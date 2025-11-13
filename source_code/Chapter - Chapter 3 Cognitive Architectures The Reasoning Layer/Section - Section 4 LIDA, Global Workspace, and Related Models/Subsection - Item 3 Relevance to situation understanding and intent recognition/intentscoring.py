import numpy as np
import networkx as nx

# simple fused evidence and episodic priors (dicts keyed by hypothesis)
Lf = {'rendezvous': 0.6, 'intercept': 0.4}   # fusion likelihoods
Pc = {'rendezvous': 0.7, 'intercept': 0.3}   # episodic priors
s  = {'rendezvous': 1.0, 'intercept': 1.5}   # salience scores
alpha = 1.2                                   # attention gain

# compute unnormalized scores
scores = {h: (s[h]**alpha) * Lf[h] * Pc[h] for h in Lf}
# normalize to posterior
tot = sum(scores.values())
post = {h: scores[h]/tot for h in scores}

# build a minimal workspace graph and broadcast top hypothesis
G = nx.DiGraph()
G.add_node('workspace')                       # global workspace node
for h,p in post.items():
    G.add_node(h, posterior=p)
    G.add_edge(h, 'workspace', weight=p)      # broadcast link (weight=posterior)

top = max(post, key=post.get)
print(f"Top intent: {top}, posterior={post[top]:.3f}")   # actionable output
# If posterior below threshold, trigger sensor tasking or human query.