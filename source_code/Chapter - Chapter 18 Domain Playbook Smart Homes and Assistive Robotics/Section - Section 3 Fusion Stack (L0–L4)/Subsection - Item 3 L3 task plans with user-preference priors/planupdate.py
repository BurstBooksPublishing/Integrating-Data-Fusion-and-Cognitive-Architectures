import numpy as np

# Define plan library (list of plan ids and required affordances)
plans = [
    {"id":"bedside_fetch","req":["reachable","container"]},
    {"id":"kitchen_fetch","req":["kitchen_access","container"]},
    {"id":"suggest_self","req":["user_capable"]}
]

# Preference priors (learned frequencies, sum to 1)
pref_prior = np.array([0.6, 0.3, 0.1])  # bedtime preference for bedside

def affordance_likelihood(plan, observ):
    # Simple product of per-requirement matches with track confidence
    probs = []
    for r in plan["req"]:
        probs.append(observ.get("affordances",{}).get(r,0.01))  # small default
    return float(np.product(probs))

def update_posteriors(observ):
    likes = np.array([affordance_likelihood(p,observ) for p in plans])
    post_unnorm = likes * pref_prior
    post = post_unnorm / (post_unnorm.sum()+1e-12)
    return post

# Example observation: bottle on nightstand reachable with high confidence
observation = {"affordances":{"reachable":0.9,"container":0.8,"kitchen_access":0.1,"user_capable":0.05}}
posterior = update_posteriors(observation)
choice_idx = int(np.argmax(posterior))
print("Posterior:", posterior)          # plan probabilities
print("Selected plan:", plans[choice_idx]["id"])
# Next: expand selected plan into actions and check safety guards (omitted)