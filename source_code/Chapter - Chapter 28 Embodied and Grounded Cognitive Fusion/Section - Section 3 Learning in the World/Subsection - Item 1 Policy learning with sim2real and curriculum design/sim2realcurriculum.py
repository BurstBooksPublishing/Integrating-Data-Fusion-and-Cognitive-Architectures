import gym, math, random, torch, torch.nn as nn, torch.optim as optim
# Simple policy network
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(),
                                 nn.Linear(64,act_dim), nn.Softmax(dim=-1))
    def forward(self,x): return self.net(x)
# Env wrapper that randomizes dynamics and observation noise
class RandEnv(gym.Wrapper):
    def __init__(self, env, theta):
        super().__init__(env); self.theta=theta
    def step(self,a):
        # scale action effect to emulate dynamics mismatch
        scaled = a * self.theta['act_scale']
        obs, r, d, info = self.env.step(int(scaled))
        # add observation noise
        obs = obs + self.theta['obs_noise']*np.random.randn(*obs.shape)
        return obs, r, d, info
# Training loop: curriculum of increasing noise
import numpy as np
env0 = gym.make('CartPole-v1'); obs_dim=env0.observation_space.shape[0]
act_dim=env0.action_space.n
policy=Policy(obs_dim,act_dim); opt=optim.Adam(policy.parameters(),1e-3)
def rollout(theta, episodes=5):
    rew_total=0
    for _ in range(episodes):
        e=RandEnv(gym.make('CartPole-v1'), theta)
        obs=e.reset(); done=False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_t); m=torch.distributions.Categorical(probs)
            a = m.sample().item()
            obs, r, done, _ = e.step(a)
            rew_total += r
    return rew_total/episodes
# Curriculum schedule: increase noise and action scale mismatch
stages = [{'obs_noise':0.0,'act_scale':1.0},
          {'obs_noise':0.05,'act_scale':0.95},
          {'obs_noise':0.1,'act_scale':0.9}]
for stage in stages:
    for epoch in range(100): # training epochs per stage
        # collect a single episode gradient (REINFORCE minimal)
        e=RandEnv(gym.make('CartPole-v1'), stage)
        obs=e.reset(); logps=[]; rewards=[]
        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_t); m=torch.distributions.Categorical(probs)
            a = m.sample()
            logps.append(m.log_prob(a)); obs, r, done, _ = e.step(a.item()); rewards.append(r)
            if done:
                G = sum(rewards)
                loss = -torch.stack(logps).sum() * G  # REINFORCE surrogate
                opt.zero_grad(); loss.backward(); opt.step()
                break
    val = rollout({'obs_noise':0.15,'act_scale':0.88}, episodes=20) # holdout eval
    print('Stage',stage,'holdout_mean_rew',val)