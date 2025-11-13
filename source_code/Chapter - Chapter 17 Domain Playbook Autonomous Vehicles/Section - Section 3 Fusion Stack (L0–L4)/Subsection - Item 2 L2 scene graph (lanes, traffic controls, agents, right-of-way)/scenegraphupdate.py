import networkx as nx
import math
# simple logistic helpers
def logit(p): return math.log(p/(1-p))
def sigmoid(x): return 1/(1+math.exp(-x))

def score_evidence(agent, lane, map_prior=0.5):
    # geometric overlap and heading alignment produce a pseudo-likelihood
    overlap = min(1.0, agent['lane_overlap'])           # fraction
    heading_score = max(0.0, 1 - abs(agent['heading_err'])/math.pi)
    # combine into a likelihood ratio via heuristic
    lik_r = 0.5 + 0.5*(0.6*overlap + 0.4*heading_score)
    lik_not = 1 - lik_r
    return lik_r, lik_not

G = nx.DiGraph()
# create nodes (example)
G.add_node('lane_42', type='lane', prior=0.7)
G.add_node('agent_7', type='agent', lane_overlap=0.8, heading_err=0.1)
# initialize edge belief as log-odds
G.add_edge('agent_7','lane_42', rel='on_lane', logodds=logit(0.5))

# evidence loop (single step)
lik_r, lik_not = score_evidence(G.nodes['agent_7'], G.nodes['lane_42'])
edge = G['agent_7']['lane_42']
edge['logodds'] += math.log(lik_r/lik_not)   # eq. (1) incremental update
edge['belief'] = sigmoid(edge['logodds'])   # store posterior
# attach provenance minimal record
edge.setdefault('provenance',[]).append({'source':'vision+map','lik_r':lik_r})
print("Updated belief on right-of-way/on_lane:", edge['belief'])