import torch
import torch.nn.functional as F
from torch import nn

class StreamingAttnLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, window=128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.nh = n_heads
        self.w = window
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward_stream(self, x_chunk, K_cache, V_cache):
        # x_chunk: [L, B, D]; K_cache, V_cache: [T, B, D] or None
        L, B, D = x_chunk.shape
        qkv = self.qkv(x_chunk).view(L, B, 3, self.nh, self.d_k)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [L,B,nh,d_k]
        # concat cache if present
        if K_cache is not None:
            K = torch.cat([K_cache, k], dim=0)  # [T+L,B,nh,d_k]
            V = torch.cat([V_cache, v], dim=0)
        else:
            K, V = k, v
        T = K.shape[0]
        # build causal mask limited to window self.w
        idx = torch.arange(T, device=x_chunk.device)
        mask = (idx[None, :] > idx[:, None]).float() * -1e9  # full causal
        if T > self.w:
            # allow only last window entries to be attended
            valid = torch.zeros(T, device=x_chunk.device)
            valid[-self.w:] = 1.0
            mask = mask + ((1.0 - valid[None, :]) * -1e9)  # block old keys
        # compute attention for queries only (last L positions)
        Q = q.permute(1,2,0,3).reshape(B*self.nh, L, self.d_k)  # [B*nh,L,d_k]
        Kc = K.permute(1,2,0,3).reshape(B*self.nh, T, self.d_k)
        Vc = V.permute(1,2,0,3).reshape(B*self.nh, T, self.d_k)
        logits = torch.matmul(Q, Kc.transpose(-2,-1)) / (self.d_k**0.5)
        logits = logits + mask[-L:, :].unsqueeze(0)  # apply mask aligned to queries
        weights = F.softmax(logits, dim=-1)
        out = torch.matmul(weights, Vc)  # [B*nh,L,d_k]
        out = out.view(B, self.nh, L, self.d_k).permute(2,0,1,3).reshape(L,B,D)
        return self.out(out), K.detach(), V.detach()  # return outputs and updated caches

# Example usage omitted for brevity.