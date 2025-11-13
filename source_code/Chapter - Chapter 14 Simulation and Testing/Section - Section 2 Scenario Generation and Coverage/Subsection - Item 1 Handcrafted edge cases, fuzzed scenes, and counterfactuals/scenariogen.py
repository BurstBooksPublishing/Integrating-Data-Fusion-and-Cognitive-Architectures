#!/usr/bin/env python3
import json, os, numpy as np, hashlib, random
# deterministic seeding helper
def seed_from_name(name):
    return int(hashlib.sha256(name.encode()).hexdigest(),16) % (2**32)
# parameter sampler with seed control
def sample_params(template, seed):
    rng = np.random.default_rng(seed)
    params = {}
    for k,v in template.items():
        if v['type']=='uniform':
            params[k] = float(rng.uniform(v['min'], v['max']))
        elif v['type']=='normal':
            params[k] = float(rng.normal(v['mu'], v['sigma']))
        else:
            params[k] = v['value']
    return params

# template example
template = {
  "ped_speed": {"type":"uniform","min":0.5,"max":3.0},
  "occluder_width": {"type":"normal","mu":1.2,"sigma":0.3},
  "illum": {"type":"uniform","min":100.0,"max":10000.0} # lux
}

outdir = "scenarios"
os.makedirs(outdir, exist_ok=True)
N = 50
coverage_hits = 0
for i in range(N):
    name = f"fuzz_scene_{i:03d}"
    seed = seed_from_name(name)
    params = sample_params(template, seed)
    # handcrafted insertion rule: if illum low and occluder large, mark as glare-like edge
    edge_flag = (params['illum']<500.0 and params['occluder_width']>1.4)
    if edge_flag: coverage_hits += 1
    # counterfactual variant: delay pedestrian by 0.6s
    cf = dict(params); cf['ped_delay'] = 0.6
    scenario = {"id":name, "seed":seed, "params":params, "counterfactual":cf, "edge":edge_flag}
    with open(os.path.join(outdir, name+".json"), "w") as f:
        json.dump(scenario, f, indent=2)
# simple coverage estimate per Eq. (1) using empirical hit-rate
p_hat = coverage_hits / N
print(f"scenarios:{N} edge_hits:{coverage_hits} p_hat:{p_hat:.3f}")