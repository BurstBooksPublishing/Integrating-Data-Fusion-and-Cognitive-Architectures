import numpy as np

# sample actions, estimated benefit and harm vectors (from fusion+cognition)
actions = ["A", "B", "C"]
benefit = np.array([0.6, 0.8, 0.7])   # expected benefit per action
harm = np.array([0.1, 0.4, 0.15])     # expected harm per action
group_impacts = {"g1": np.array([0.5,0.9,0.6]), "g2": np.array([0.7,0.7,0.8])}

EPS = 0.2   # total harm budget threshold
DELTA = 0.25  # subgroup benefit disparity allowed

# filter actions that exceed harm budget
safe_idx = np.where(harm <= EPS)[0]
candidates = list(safe_idx)

# check justice constraint per candidate
def passes_justice(idx):
    b = benefit[idx]
    for g, vals in group_impacts.items():
        if abs(vals[idx] - benefit.mean()) > DELTA:
            return False
    return True

valid = [i for i in candidates if passes_justice(i)]

# choose highest-benefit valid action; fallback to conservative safe action
if valid:
    choice = actions[int(max(valid, key=lambda i: benefit[i]))]
else:
    choice = actions[int(np.argmin(harm))]  # conservative fallback

print("selected action:", choice)  # runtime decision recorded for audit