import torch, torch.nn as nn

# TCN block (causal conv with dilation)
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=(k-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
    def forward(self, x):            # x: (B, C, T)
        y = self.conv(x)
        return self.relu(y[:, :, :x.size(2)])  # crop to causal length

# Simple streaming wrapper for TCN using a fixed buffer
class StreamingTCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, k, dilations):
        super().__init__()
        self.blocks = nn.ModuleList([TCNBlock(in_ch if i==0 else hidden_ch,
                                              hidden_ch, k, d)
                                     for i,d in enumerate(dilations)])
        self.buffer = None
    def forward_step(self, x_step):  # x_step: (B, C) single timestep
        x_step = x_step.unsqueeze(-1)               # (B, C, 1)
        if self.buffer is None:
            self.buffer = x_step
        else:
            self.buffer = torch.cat([self.buffer, x_step], dim=2)
            max_len = 1 + (k-1)*sum(dilations)
            if self.buffer.size(2) > max_len:      # evict old
                self.buffer = self.buffer[:, :, -max_len:]
        out = self.buffer
        for b in self.blocks: out = b(out)
        return out[:, :, -1]  # last output (B, hidden_ch)

# LSTM single-step usage (stateful built-in)
lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
h0 = (torch.zeros(1,4,32), torch.zeros(1,4,32))  # initial state
# single-step forward: feed 1-length sequence
x1 = torch.randn(4,1,16)
out, h1 = lstm(x1, h0)  # out: (B,1,32); h1 contains updated state