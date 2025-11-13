import torch, numpy as np
from ortools.sat.python import cp_model

# Dummy learned scorer (replace with trained PyTorch model)
class Scorer(torch.nn.Module):
    def forward(self, det_feats, track_feats):
        # simple dot-product scores -> logits
        return det_feats @ track_feats.T

# sample features
det_feats = torch.randn(5,16)   # 5 detections
track_feats = torch.randn(7,16) # 7 tracks
scorer = Scorer()
logits = scorer(det_feats, track_feats).detach().numpy()

model = cp_model.CpModel()
assign = {}
for i in range(logits.shape[0]):
    for j in range(logits.shape[1]):
        assign[(i,j)] = model.NewBoolVar(f"a_{i}_{j}")
# constraints: each detection assigned at most once
for i in range(logits.shape[0]):
    model.Add(sum(assign[(i,j)] for j in range(logits.shape[1)]) <= 1)
# constraints: each track capacity <= 2 (example)
for j in range(logits.shape[1]):
    model.Add(sum(assign[(i,j)] for i in range(logits.shape[0)]) <= 2)

# objective: maximize summed logits (solver minimizes negative)
model.Maximize(sum(int(1000*logits[i,j])*assign[(i,j)]
                   for i in range(logits.shape[0]) for j in range(logits.shape[1)]))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 0.5  # latency budget
res = solver.Solve(model)
# extract assignment
solution = [(i,j) for (i,j) in assign if solver.Value(assign[(i,j)])==1]
print("Assigned pairs:", solution)  # simple output