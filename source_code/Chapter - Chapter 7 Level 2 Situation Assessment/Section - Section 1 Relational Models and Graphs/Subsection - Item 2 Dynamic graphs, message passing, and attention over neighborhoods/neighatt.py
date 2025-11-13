import torch
import torch.nn as nn
import math

class StreamingNeighborhoodAttention(nn.Module):
    def __init__(self, dim, topk=16, decay_lambda=1.0):
        super().__init__()
        self.dim = dim
        self.topk = topk
        self.decay_lambda = decay_lambda
        self.W_self = nn.Linear(dim, dim, bias=False)
        self.W_msg = nn.Linear(dim, dim, bias=False)
        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, node_feats, edge_index, edge_dt):
        """
        node_feats: (N, D) tensor of current node states
        edge_index: (2, M) long tensor of directed edges (src, dst)
        edge_dt: (M,) float tensor of message age (seconds)
        """
        N, D = node_feats.shape
        src, dst = edge_index
        # compute queries/keys
        Qv = self.Q(node_feats)      # (N, D)
        Ku = self.K(node_feats)      # (N, D)
        # per-edge scores
        score = (Qv[dst] * Ku[src]).sum(dim=1) / math.sqrt(D)  # (M,)
        # apply temporal decay to scores (older messages downweighted)
        decay = torch.exp(-self.decay_lambda * edge_dt)       # (M,)
        score = score * decay
        # aggregate per-destination with top-k selection
        # naive per-destination grouping (works for moderate sizes)
        agg = torch.zeros(N, D, device=node_feats.device)
        for v in range(N):
            mask = (dst == v).nonzero(as_tuple=False).squeeze(1)
            if mask.numel() == 0:
                continue
            scores_v = score[mask]
            src_v = src[mask]
            # top-k neighbors by score
            k = min(self.topk, scores_v.numel())
            topk_idx = torch.topk(scores_v, k=k, largest=True).indices
            selected_src = src_v[topk_idx]
            selected_scores = scores_v[topk_idx]
            alpha = torch.softmax(selected_scores, dim=0).unsqueeze(1)  # (k,1)
            msgs = self.W_msg(node_feats[selected_src])                 # (k,D)
            agg[v] = (alpha * msgs).sum(dim=0)
        # self + messages update
        updated = self.act(self.W_self(node_feats) + agg)
        return updated

# Quick usage example (N=100 nodes, M random edges)
if __name__ == "__main__":
    N, D, M = 100, 64, 800
    feats = torch.randn(N, D)
    src = torch.randint(0, N, (M,))
    dst = torch.randint(0, N, (M,))
    dt = torch.rand(M) * 2.0  # ages in seconds
    layer = StreamingNeighborhoodAttention(D, topk=8, decay_lambda=0.5)
    out = layer(feats, torch.stack([src, dst]), dt)  # (N,D)
    print("Updated shape:", out.shape)  # diagnostic