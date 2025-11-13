import math, networkx as nx

# Mock priors and likelihoods (from models/sensors)
room_prior = {"kitchen": 0.3, "living": 0.4, "bedroom": 0.3}
affordance_prior = {"sit":0.2,"cook":0.1,"read":0.05,"charge":0.05}
# Evidence likelihoods p(E|affordance) from classifiers
evidence = {"sit":0.8,"cook":0.6,"read":0.4,"charge":0.1}
# Activity templates mapping activities -> required affordances
activities = {"dine":["sit","cook"], "relax":["sit","read"], "charge_device":["charge"]}

def log_prob(x):
    return math.log(max(x,1e-9))

# Build scene graph
G = nx.DiGraph()
G.add_node("room_1", type="room", priors=room_prior)

# Fuse affordances into log-posteriors using Bayes-product approximation
aff_logpost = {}
for a in affordance_prior:
    # log P(A)+sum log P(E|A) with single evidence item here
    aff_logpost[a] = log_prob(affordance_prior[a]) + log_prob(evidence.get(a,1e-9))

# Rank activities
activity_scores = {}
for act, req in activities.items():
    # Combine using simple sum of affordance log-posteriors
    score = sum(aff_logpost.get(r, log_prob(1e-9)) for r in req)
    activity_scores[act] = score + log_prob(0.05)  # prior on activity base
# Output sorted activities
for act, score in sorted(activity_scores.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{act}: score={score:.3f}")