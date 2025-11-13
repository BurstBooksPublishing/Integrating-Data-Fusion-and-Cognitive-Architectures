import math, time, json
# Simple in-memory priors and likelihood tables (example values)
PRIORS = {"make_coffee": 0.4, "find_glasses": 0.1, "idle": 0.5}
# Routine likelihoods estimated from short templates
ROUTINE_LIK = {"make_coffee": 0.6, "find_glasses": 0.05, "idle": 0.3}
# Dialogue-act likelihoods (calibrated)
DIALOG_LIK = {"request_find": {"find_glasses":0.9, "make_coffee":0.01},
              "confirm": {"find_glasses":0.2, "make_coffee":0.6}}
DECISION_THRESHOLD = 0.3

def log_posteriors(routine_obs, dialog_act):
    # routine_obs currently collapsed to precomputed likelihoods; extend as needed
    post = {}
    # compute unnormalized log posterior
    for g in PRIORS:
        logp = math.log(ROUTINE_LIK.get(g,1e-6))
        logp += math.log(DIALOG_LIK.get(dialog_act, {}).get(g, 1e-6))
        logp += math.log(PRIORS[g])
        post[g] = logp
    # normalize
    maxlog = max(post.values())
    exps = {g: math.exp(l - maxlog) for g,l in post.items()}
    Z = sum(exps.values())
    return {g: v/Z for g,v in exps.items()}

# Example runtime
if __name__ == "__main__":
    # incoming fused evidence (from perception & speech subsystems)
    dialog = "request_find"                      # parsed dialogue act
    routine = {"kettle": True}                   # placeholder
    posts = log_posteriors(routine, dialog)
    print(json.dumps(posts, indent=2))           # auditable posterior
    # decision rule
    best, pbest = max(posts.items(), key=lambda kv: kv[1])
    if pbest > DECISION_THRESHOLD:
        print("Action:", best)                   # trigger policy executor
    else:
        print("Defer: seek clarification or gather more evidence")