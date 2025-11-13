import numpy as np

rooms = ["living", "kitchen", "bedroom"]
# prior from long-term memory (frequency of misplacements)
prior = np.array([0.5, 0.3, 0.2])  # sum=1

def sensor_likelihood(obs, rooms):
    # simple model: obs is room name or None; false positive rate fp
    fp = 0.1
    L = np.full(len(rooms), fp)
    if obs in rooms:
        L[rooms.index(obs)] = 1 - fp
    return L

def bayes_update(prior, likelihood):
    unnorm = likelihood * prior
    return unnorm / unnorm.sum()

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p))

# simulated interaction loop
belief = prior.copy()
threshold = 0.7  # decide when confident
for step in range(6):
    # choose action: sense noisy visual in highest-prob room
    sense_room = rooms[np.argmax(belief)]
    # simulate an observation (ground truth bedroom)
    ground_truth = "bedroom"
    obs = sense_room if np.random.rand() < 0.6 and sense_room==ground_truth else None
    L = sensor_likelihood(obs, rooms)
    belief = bayes_update(belief, L)
    print(f"step {step} obs={obs} belief={belief}")

    if belief.max() >= threshold:
        print("Decision: fetch from", rooms[np.argmax(belief)])
        break

    # generate simple question if uncertain
    q = "Did you leave them in the kitchen?"
    # simulated user answer: yes/no/unsure
    user_ans = "no"  # simulated; could be sampled or from UI
    if user_ans == "yes":
        # strong likelihood for kitchen
        Lq = np.array([0.05, 0.9, 0.05])
    elif user_ans == "no":
        Lq = np.array([0.8, 0.05, 0.15])
    else:
        Lq = np.array([0.33, 0.33, 0.34])
    belief = bayes_update(belief, Lq)
# end