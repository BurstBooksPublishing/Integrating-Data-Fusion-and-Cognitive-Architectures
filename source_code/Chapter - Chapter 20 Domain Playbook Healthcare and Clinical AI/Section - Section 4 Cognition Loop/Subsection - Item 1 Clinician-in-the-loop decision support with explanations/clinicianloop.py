import math, json, time

# Example model weights (learned offline), utilities, and governance tau.
WEIGHTS = {'hr':0.02,'sysbp':-0.03,'lactate':0.8,'wbc':0.1}
BIAS = -3.0
UTIL = {'treat_if_sepsis': 10.0, 'treat_if_not': -2.0,
        'observe_if_sepsis': -8.0, 'observe_if_not': 0.0}
TAU = 1.5  # alert threshold

def logistic(x): return 1.0/(1.0+math.exp(-x))

def sepsis_probability(features):
    # features: dict of {name: mean}; variances available for provenance.
    s = BIAS + sum(WEIGHTS.get(k,0.0)*v for k,v in features.items())
    return logistic(s)

def expected_utility(p_sepsis):
    EU_treat = p_sepsis*UTIL['treat_if_sepsis'] + (1-p_sepsis)*UTIL['treat_if_not']
    EU_observe = p_sepsis*UTIL['observe_if_sepsis'] + (1-p_sepsis)*UTIL['observe_if_not']
    return EU_treat, EU_observe

def top_contributors(features, top_n=3):
    # simple linear attribution (faithful for linear logit)
    contribs = [(k, WEIGHTS.get(k,0.0)*v) for k,v in features.items()]
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs[:top_n]

def evaluate(fused_message):
    features = fused_message['features']  # means only for brevity
    p = sepsis_probability(features)
    EU_t, EU_o = expected_utility(p)
    decision = 'alert' if (EU_t - EU_o) > TAU else 'no_alert'
    explanation = {
        'sepsis_prob': round(p,3),
        'decision': decision,
        'EU_treat': round(EU_t,2),
        'EU_observe': round(EU_o,2),
        'top_contributors': top_contributors(features),
        'provenance': fused_message.get('provenance', {}),
        'model_version': fused_message.get('model_version','v1.0'),
        'timestamp': time.time()
    }
    # log for audit (append-only)
    print(json.dumps(explanation))
    return explanation

# Example fused message (would come from L0-L3 fusion pipeline)
msg = {'features': {'hr':110,'sysbp':95,'lactate':2.6,'wbc':13.0},
       'provenance': {'hr_src':'monitor12','lab_src':'lab42'},
       'model_version':'sepsis_v2'}
evaluate(msg)  # run evaluation; UI consumes returned explanation