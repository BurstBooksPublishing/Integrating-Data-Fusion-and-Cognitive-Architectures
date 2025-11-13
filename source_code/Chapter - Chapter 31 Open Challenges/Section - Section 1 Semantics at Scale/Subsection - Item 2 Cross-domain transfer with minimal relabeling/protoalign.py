import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# encoder: pretrained embedding (simple MLP here)
class Encoder(nn.Module):
    def __init__(self, d_in, d_z=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in,d_z), nn.ReLU(), nn.Linear(d_z,d_z))
    def forward(self,x): return self.net(x)

# contrastive loss toward prototypes
def prototype_loss(z, targets, prototypes):
    # targets: -1 for unlabeled, else class idx
    mask = (targets >= 0)
    if mask.sum()==0: return torch.tensor(0., device=z.device)
    z_m = z[mask]; t_m = targets[mask]
    p = prototypes[t_m]                        # batch prototypes
    return ((z_m - p).pow(2).sum(dim=1)).mean()

# acquisition: highest entropy over soft-similarities
def acquire_indices(z, prototypes, budget):
    sims = torch.matmul(z, prototypes.t())    # cosine or dot similarities
    probs = sims.softmax(dim=1)
    entropy = -(probs*probs.clamp(min=1e-12).log()).sum(dim=1)
    return torch.topk(entropy, k=budget).indices

# synthetic example: features, some labels
X = torch.randn(1000,128)
labels = -1*torch.ones(1000,dtype=torch.long)  # -1 unlabeled
labels[:20] = torch.randint(0,2,(20,))         # small seed labels

enc = Encoder(128,64)
prototypes = torch.randn(2,64, requires_grad=True)  # two source classes

opt = torch.optim.Adam(list(enc.parameters())+[prototypes], lr=1e-3)
ds = TensorDataset(X, labels)
loader = DataLoader(ds, batch_size=128, shuffle=True)

for epoch in range(30):
    for xb, yb in loader:
        z = enc(xb)
        loss = prototype_loss(z, yb, prototypes)
        # ontology/fidelity regularizers would add here
        opt.zero_grad(); loss.backward(); opt.step()
    # active acquire 5 labels per epoch (simulate oracle)
    with torch.no_grad():
        z_all = enc(X)
        idx = acquire_indices(z_all, prototypes, budget=5)
        labels[idx] = torch.randint(0,2,(len(idx),))  # oracle provides labels