import random
import torch, torch.nn as nn, torch.optim as optim

class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0
    def add(self, x, y):
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((x.clone(), y.clone()))
        else:
            j = random.randrange(self.n_seen)
            if j < self.capacity:
                self.buffer[j] = (x.clone(), y.clone())
    def sample(self, k):
        return random.sample(self.buffer, min(k, len(self.buffer)))

# simple model
model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
opt = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
buffer = ReservoirBuffer(capacity=500)  # rehearsal budget

def online_update(batch, replay_k=32, lam=0.7):
    x, y = batch
    # add exemplars to buffer (one-by-one)
    for xi, yi in zip(x, y):
        buffer.add(xi, yi)
    # current loss
    logits = model(x)
    loss_curr = criterion(logits, y)
    # replay loss
    replay = buffer.sample(replay_k)
    if replay:
        xr = torch.stack([r[0] for r in replay])
        yr = torch.tensor([r[1] for r in replay])
        loss_replay = criterion(model(xr), yr)
    else:
        loss_replay = 0.0
    loss = loss_curr + lam * loss_replay
    opt.zero_grad()
    loss.backward()
    opt.step()
# streamer would call online_update for each incoming batch