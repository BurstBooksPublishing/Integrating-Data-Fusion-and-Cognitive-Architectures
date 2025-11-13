import torch, torch.nn as nn

# Perception encoder: maps sensor bundle to obs embedding
class Encoder(nn.Module):
    def __init__(self, dim): super().__init__(); self.lin=nn.Linear(256,dim)
    def forward(self, obs): return self.lin(obs)               # obs tensor

# Latent dynamics: simple RNN for z_t
class Dyn(nn.Module):
    def __init__(self, dim): super().__init__(); self.rnn=nn.GRUCell(dim,dim)
    def forward(self, z, e): return self.rnn(e,z)              # update latent

# Symbolic checker: returns True if constraints hold
def symbolic_check(symbolic_state):
    # lightweight checks, call external reasoner in real systems
    return (symbolic_state['speed'] <= symbolic_state['limit'])

enc = Encoder(dim=64)
dyn = Dyn(dim=64)
opt = torch.optim.Adam(list(enc.parameters())+list(dyn.parameters()), lr=1e-4)

for batch in dataloader:
    obs = batch['sensor_bundle']            # fused sensor vector
    e = enc(obs)                            # encode
    z = torch.zeros(e.size(0),64)           # init latent
    z = dyn(z,e)                            # one-step update
    pred = decoder(z)                       # predict next obs (omitted)
    loss_pred = criterion(pred, batch['next_obs'])
    # symbolic penalty (illustrative)
    sym = batch['symbolic_state']
    sym_violation = (~torch.tensor([symbolic_check(s) for s in sym])).float().mean()
    loss = loss_pred + 0.1*sym_violation
    opt.zero_grad(); loss.backward(); opt.step()
    # runtime guard: if sym_violation large, mark for human review