import json, math, random

def isotonic_calibrate(probs, targets):  # simple placeholder calibrator
    # fit monotonic mapping; here use identity for brevity
    return probs

def expected_utility(cal_probs, utilities):  # cal_probs:list, utilities:matrix
    # returns E[U] per action and simple std via sampling
    n_actions = len(utilities[0])
    samples = 200
    est = []
    for a in range(n_actions):
        draws = []
        for _ in range(samples):
            # sample hypothesis according to cal_probs
            i = random.choices(range(len(cal_probs)), weights=cal_probs)[0]
            draws.append(utilities[i][a])
        est.append((sum(draws)/samples, math.sqrt(sum((d-est[-1][0])**2 for d in draws)/samples) if est else 0))
    return est

# Example inputs (from fusion + cognition)
hypotheses = [
    {"id":"H1","prob":0.6,"provenance":["radar_track:42","camera:img123"],"note":"approaching"},
    {"id":"H2","prob":0.4,"provenance":["radar_track:43"],"note":"loitering"}
]
probs = [h["prob"] for h in hypotheses]
cal_probs = isotonic_calibrate(probs, None)  # calibrate

# utilities per hypothesis for two actions: hold, engage
utilities = [[-10, 5], [-1, 1]]  # U(a|H1), U(a|H2)
eu = expected_utility(cal_probs, utilities)  # compute E[U]

# prepare payload for UI
payload = {
  "hypotheses":[{"id":h["id"],"prob":p,"provenance":h["provenance"]} for h,p,h in zip(hypotheses,cal_probs,hypotheses)],
  "actions":[{"name":"hold","E_U":eu[0][0],"uncertainty":eu[0][1]},
             {"name":"engage","E_U":eu[1][0],"uncertainty":eu[1][1]}],
  "rationale":{"why":"radar+camera indicate approach","why_not":"loitering hypothesis reduces urgency"},
  "audit":{"trace_id":"tx-0001","timestamp":"2025-11-07T12:00:00Z"}
}
print(json.dumps(payload,indent=2))  # serialized UI payload