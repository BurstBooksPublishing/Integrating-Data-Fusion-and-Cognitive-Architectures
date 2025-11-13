import torch, math, random
torch.manual_seed(0)

# Synthetic data: n predicates, binary scenario label
n, m, N = 6, 8, 512
# Random predicate generator (simulate event-graph lift)
def gen_sample():
    p = torch.rand(n)  # predicate truths in [0,1]
    # label is 1 if predicates 0 and 2 co-occur strongly, else 0
    y = 1.0 if (p[0]>0.6 and p[2]>0.5) else 0.0
    return p, torch.tensor([y])

data = [gen_sample() for _ in range(N)]

# Model params: selector logits alpha (m x n), clause weights w and bias b
alpha = torch.randn(m, n, requires_grad=True)
w = torch.randn(m, requires_grad=True)
b = torch.randn(1, requires_grad=True)
opt = torch.optim.Adam([alpha, w, b], lr=1e-2)

eps = 1e-6
for epoch in range(200):
    loss_tot = 0.0
    for p, y in data:
        s = torch.sigmoid(alpha)              # soft selectors s_ji
        # clause truth phi_j = exp(sum s_ji * log(p_i+eps))
        phi = torch.exp((s * torch.log(p+eps)).sum(dim=1))
        logits = (w * phi).sum() + b
        pred = torch.sigmoid(logits)
        loss = torch.nn.functional.binary_cross_entropy(pred, y)
        # sparsity regularizer on selectors to encourage compact rules
        loss = loss + 1e-2 * s.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        loss_tot += loss.item()
    if epoch % 50 == 0:
        print(f"epoch {epoch}: loss {loss_tot/len(data):.4f}")

# Inspect learned selectors: print top predicates per clause
s_final = torch.sigmoid(alpha).detach()
for j in range(m):
    top_idx = torch.topk(s_final[j], 2).indices.tolist()
    print(f"clause {j}: top predicates {top_idx}, weight {w[j].item():.2f}")