import numpy as np, math, random
# simple 1D state space with Gaussian likelihoods
def transition(b, P): return P @ b                 # linear transition
def observation_likelihood(x, z, sigma):            # P(z|x)
    return np.exp(-0.5*((z-x)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
def belief_update(b, P, z, sigma):
    b_pred = transition(b, P)                      # prior prediction
    lik = np.array([observation_likelihood(x,z,sigma) for x in xs])
    b_post = b_pred * lik
    return b_post / (b_post.sum()+1e-12)           # normalize
def planner_expected_reward(b, actions, reward_fn):
    # pick action with max expected immediate reward
    ex = [np.dot(b, reward_fn(a)) for a in actions]
    return actions[int(np.argmax(ex))]
# state/grid
xs = np.linspace(-5,5,101)
# transition kernel (small diffusion)
P = np.eye(len(xs))*0.9 + np.roll(np.eye(len(xs)),1,axis=1)*0.05
P /= P.sum(axis=0)
# actions and reward: move toward goal at +3
actions = [-1,0,1]
def reward_fn(a): return -np.abs(xs - (3 - a))     # nearer to goal better
# init belief (broad)
b = np.exp(-0.5*(xs/2)**2); b /= b.sum()
# loop: sense, update, plan, act, safety check
for t in range(20):
    z = 3 + np.random.randn()*0.7                   # noisy sensor reading
    b = belief_update(b, P, z, sigma=0.7)           # update belief
    a = planner_expected_reward(b, actions, reward_fn)
    # safety: forbid large moves if variance high
    var = np.dot(b, (xs - np.dot(b,xs))**2)
    if var > 2.0: a = 0                              # conservative fallback
    print(f"t={t}, action={a}, var={var:.2f}")
    # apply action as shift in belief (approx)
    b = np.roll(b, a) * 0.98; b /= b.sum()