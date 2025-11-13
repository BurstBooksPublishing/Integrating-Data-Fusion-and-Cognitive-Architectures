# Minimal executable example (requires torch, numpy)
import os, json, hashlib, random, numpy as np, torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

TRACE_DIR='golden_traces'          # local archive
os.makedirs(TRACE_DIR, exist_ok=True)

def make_trace(i):
    seed=i; random.seed(seed); np.random.seed(seed)
    obs=(np.random.randn(10).tolist())          # simple time-series
    gt=float(np.mean(obs)+0.1*seed)             # synthetic ground truth
    meta={'seed':seed,'version':'v1'}
    payload={'obs':obs,'gt':gt,'meta':meta}
    data=json.dumps(payload).encode()
    sig=hashlib.sha256(data).hexdigest()
    path=os.path.join(TRACE_DIR,f'trace_{i}.json')
    with open(path,'wb') as fh: fh.write(data)
    # write signature file
    with open(path+'.sig','w') as s: s.write(sig)

# create small corpus if missing
if not any(fname.endswith('.json') for fname in os.listdir(TRACE_DIR)):
    for i in range(20): make_trace(i)

class TraceDataset(Dataset):
    def __init__(self, ddir): self.files=[os.path.join(ddir,f) for f in os.listdir(ddir) if f.endswith('.json')]
    def __len__(self): return len(self.files)
    def __getitem__(self,idx):
        with open(self.files[idx],'r') as fh: payload=json.load(fh)
        # convert to tensors
        obs=torch.tensor(payload['obs'],dtype=torch.float32).unsqueeze(0) # (1,10)
        return {'obs':obs,'gt':torch.tensor(payload['gt'],dtype=torch.float32),'meta':payload['meta'],'path':self.files[idx]}

def set_deterministic(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# simple model
model=nn.Sequential(nn.Flatten(), nn.Linear(10,1))
opt=optim.SGD(model.parameters(), lr=1e-2)

ds=TraceDataset(TRACE_DIR); dl=DataLoader(ds, batch_size=1, shuffle=False)

# train one epoch with replay determinism
model.train()
for item in dl:
    set_deterministic(item['meta'][0]['seed'])        # ensure replay same per-trace
    pred=model(item['obs'])
    loss=(pred.squeeze()-item['gt']).pow(2).mean()
    loss.backward(); opt.step(); opt.zero_grad()

# validate (simple MSE)
model.eval()
mse=0.0
with torch.no_grad():
    for item in dl:
        set_deterministic(item['meta'][0]['seed'])
        pred=model(item['obs']).squeeze()
        mse+=((pred-item['gt'])**2).item()
mse /= len(dl)

# persist artifact with provenance
torch.save(model.state_dict(),'model_artifact.pt')
meta={'mse':mse,'trace_corpus':'v1','commit':'', 'seed':42}
with open('model_artifact.meta.json','w') as fh: json.dump(meta,fh)