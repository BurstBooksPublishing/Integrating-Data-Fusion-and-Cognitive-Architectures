import torch, torch.nn as nn, torch.optim as optim

# Simple policy network
class Policy(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim,64), nn.ReLU(), nn.Linear(64,a_dim))
    def forward(self,x): return self.net(x)

def task_loss(policy, data): # data: (states, actions, rewards, latency, fa_count)
    # reward-shaped loss: negative expected reward + penalties
    states, actions, rewards, latency, fa = data
    pred = policy(states)
    # surrogate: MSE to targets (placeholder); real impl uses policy gradient
    return -rewards.mean() + 0.1*latency.mean() + 1.0*(fa.mean())

# meta-training loop
meta_policy = Policy(s_dim=10, a_dim=4)
meta_opt = optim.Adam(meta_policy.parameters(), lr=1e-3)
alpha = 1e-2  # inner lr

for meta_epoch in range(1000):
    meta_opt.zero_grad()
    meta_loss = 0.0
    for task in sample_tasks(batch=4):  # sample tasks from p(T)
        policy_clone = Policy(10,4)
        policy_clone.load_state_dict(meta_policy.state_dict())
        # inner update (one gradient step)
        data_train = task.sample_init_rollout()  # small init buffer
        loss_in = task_loss(policy_clone, data_train)
        grads = torch.autograd.grad(loss_in, policy_clone.parameters(), create_graph=True)
        for p, g in zip(policy_clone.parameters(), grads):
            p.data = p.data - alpha * g  # fast-weights update
        # validation loss after adaptation
        data_val = task.sample_validation_rollout()
        loss_val = task_loss(policy_clone, data_val)
        meta_loss = meta_loss + loss_val
    meta_loss = meta_loss / 4.0
    meta_loss.backward()
    meta_opt.step()

# fast adapt at deployment with safety projection
def fast_adapt_and_safe(policy, init_data):
    # one or few gradient steps
    opt = optim.SGD(policy.parameters(), lr=alpha)
    loss = task_loss(policy, init_data)
    opt.zero_grad(); loss.backward(); opt.step()
    # safety projection: clip action logits to safe range
    with torch.no_grad():
        for p in policy.parameters(): p.clamp_(-5.0,5.0)
    return policy