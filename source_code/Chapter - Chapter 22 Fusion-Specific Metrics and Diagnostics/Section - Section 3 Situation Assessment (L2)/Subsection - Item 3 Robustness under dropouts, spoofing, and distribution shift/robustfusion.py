import math, numpy as np

# inputs: priors, per-sensor log-likelihoods logP(E_i|H), and sentinel rates
priors = {'H1': -1.2, 'H0': -0.7}   # log priors
loglik = { 'cam': {'H1': -0.3,'H0': -2.0},
           'radar': {'H1': -0.5,'H0': -1.5},
           'gps': {'H1': -5.0,'H0': -0.1} }
p_drop = {'cam':0.05,'radar':0.02,'gps':0.6}   # high gps dropout
p_spoof = {'cam':0.01,'radar':0.0,'gps':0.05}  # gps some spoof risk
S_loglik = -3.0   # log-likelihood under generic spoof model (constant)

def effective_loglik(sensor, h):
    # compute log L_i from Eq. (1) using log-sum-exp for stability
    a = math.log(max(0.0,1 - p_drop[sensor] - p_spoof[sensor])) + loglik[sensor][h]
    b = math.log(p_spoof[sensor]) + S_loglik
    c = math.log(p_drop[sensor]) + math.log(1e-6)  # neutral evidence small const
    return np.logaddexp(np.logaddexp(a,b), c)

# quarantine if sentinel indicates poor calibration (simple rule)
quarantine = {s: p_drop[s]>0.5 or p_spoof[s]>0.3 for s in p_drop}

post = {}
for h in priors:
    acc = priors[h]
    for s in loglik:
        if quarantine[s]:
            continue   # skip quarantined sensors
        acc += effective_loglik(s,h)
    post[h] = acc
# normalize
Z = np.logaddexp.reduce(list(post.values()))
for h in post: post[h] = math.exp(post[h]-Z)

print(post)   # posterior probabilities for hypotheses