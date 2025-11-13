import numpy as np
from math import log, exp
# Simple Gaussian log-likelihood
def log_gauss(x, mu, sigma): 
    return -0.5*((x-mu)**2)/(sigma**2) - 0.5*log(2*np.pi*sigma**2)

def temporal_weight(ts, now, half_life):
    # Exponential decay weight for evidence age
    return exp(-(now-ts)/half_life)

def fuse_hypotheses(evidence, priors, now, half_life=3600.0):
    # evidence: dict modality -> list of (timestamp, value) tuples
    # priors: dict hypothesis -> log prior
    log_post = {h: priors[h] for h in priors}
    for mod, items in evidence.items():
        for ts, val in items:
            w = temporal_weight(ts, now, half_life)
            # Example per-modality likelihoods per hypothesis (domain-specific)
            for h in log_post:
                if mod == "vitals":
                    # expect vector val = (HR,RR,MAP)
                    mu = np.array([90,20,70]) if h=="sepsis" else np.array([75,16,80])
                    sigma = np.array([15,4,10])
                    ll = sum(log_gauss(v, m, s) for v,m,s in zip(val,mu,sigma))
                elif mod == "lactate":
                    mu, sigma = (2.5,0.6) if h=="sepsis" else (1.0,0.3)
                    ll = log_gauss(val, mu, sigma)
                elif mod == "notes":
                    # val is classifier probability p(text|h); convert to log-likelihood
                    p = max(1e-6, min(1-1e-6, val))
                    ll = log(p)
                else:
                    ll = 0.0
                log_post[h] += w * ll
    # normalize to probabilities
    maxlp = max(log_post.values())
    norm = sum(exp(log_post[h]-maxlp) for h in log_post)
    return {h: exp(log_post[h]-maxlp)/norm for h in log_post}

# Synthetic run
now = 1_600_000_000
evidence = {
  "vitals":[(now-30, (110,28,60))],         # recent tachycardia, hypotension
  "lactate":[(now-900, 3.2)],               # 15 minutes old
  "notes":[(now-60, 0.8)]                   # classifier strong for infection
}
priors = {"sepsis": log(0.01), "SIRS": log(0.02), "dehydration": log(0.05), "no-acute": log(0.92)}
post = fuse_hypotheses(evidence, priors, now)
print(post)  # posterior distribution over hypotheses