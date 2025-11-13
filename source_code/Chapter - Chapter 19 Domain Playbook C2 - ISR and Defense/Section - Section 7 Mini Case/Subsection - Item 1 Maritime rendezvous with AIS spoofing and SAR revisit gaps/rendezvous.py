import math,heapq,random
import numpy as np
from datetime import datetime, timedelta

# simple likelihoods (tunable)
def ais_likelihood(match_score, spoof_prob_prior):
    # higher match -> less likely spoofed AIS
    return max(1e-6, (1-spoof_prob_prior)*(match_score) + 1e-6)

def sar_likelihood(detected, pd=0.85, pfa=0.01):
    return pd if detected else (1-pd)  # simplified

# prior beliefs
pV = 0.01                 # prior rendezvous probability
p_spoof = 0.05            # prior AIS spoof chance

# synthetic evidence
ais_match = 0.9           # AIS consistency score [0,1]
sar_detected = False      # SAR revisit currently missed

# compute posterior P(V|A,S) via Bayes (unnormalized)
lik_A = ais_likelihood(ais_match, p_spoof)
lik_S = sar_likelihood(sar_detected)
unnorm_true = lik_A * lik_S * pV
unnorm_false = (1-lik_A) * (1-lik_S) * (1-pV)
pV_post = unnorm_true / (unnorm_true + unnorm_false + 1e-12)

# decision: schedule SAR if EVoI exceeds cost threshold
def expected_value_of_info(p_post, cost_task, utility_correct=1.0):
    # assume revisit will reveal truth with prob r (revisit reliability)
    r = 0.9
    # expected reduction in decision loss (simplified)
    return r * utility_correct * abs(p_post - 0.5) - cost_task

cost = 0.02
evoi = expected_value_of_info(pV_post, cost)
schedule = evoi > 0

print("P(V|A,S)=",round(pV_post,3),"EVoI=",round(evoi,4),"schedule_sar=",schedule)