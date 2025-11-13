import time, json, random
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,10))  # classifier
opt = optim.Adam(model.parameters(), lr=1e-4)
replay = []  # simple replay buffer

def compute_self_loss(feat_prev, feat_curr):
    # temporal consistency: L2 between normalized embeddings
    a = feat_prev / (feat_prev.norm()+1e-8)
    b = feat_curr / (feat_curr.norm()+1e-8)
    return ((a-b)**2).mean()

def ingest(fused_obs):
    # fused_obs contains 'features' (128-d), 'prev_features', optional 'corrective' dict
    feat = torch.from_numpy(fused_obs['features']).float()
    prev = torch.from_numpy(fused_obs['prev_features']).float()
    pseudo_logits = model(feat)
    pseudo_label = pseudo_logits.detach().argmax().item()
    # assemble training sample
    sample = {'feat':feat, 'prev':prev, 'pseudo':pseudo_label,
              'ts': time.time(), 'prov': fused_obs['prov']}
    if 'corrective' in fused_obs:
        sample['corrective']=fused_obs['corrective']  # {'label':int, 'operator':str}
    replay.append(sample)
    return sample

def update_step(alpha=0.7, batch_size=8):
    if len(replay) < batch_size: return
    batch = random.sample(replay, batch_size)
    feats = torch.stack([s['feat'] for s in batch])
    prevs = torch.stack([s['prev'] for s in batch])
    logits = model(feats)
    # corrective term
    corrective_idx = [i for i,s in enumerate(batch) if 'corrective' in s]
    if corrective_idx:
        labels = torch.tensor([batch[i]['corrective']['label'] for i in corrective_idx])
        preds = logits[corrective_idx]
        Lc = nn.CrossEntropyLoss()(preds, labels)
    else:
        Lc = torch.tensor(0.0)
    # self term: temporal consistency over batch
    Ls = torch.stack([compute_self_loss(prevs[i], feats[i]) for i in range(len(batch))]).mean()
    loss = alpha*Lc + (1-alpha)*Ls
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    # provenance logging minimal example
    prov = [s['prov'] for s in batch]
    print(json.dumps({'ts':time.time(),'loss':loss.item(),'prov_sample':prov[0]}))

# Example loop omitted: call ingest(...) with fused observations and periodically call update_step()