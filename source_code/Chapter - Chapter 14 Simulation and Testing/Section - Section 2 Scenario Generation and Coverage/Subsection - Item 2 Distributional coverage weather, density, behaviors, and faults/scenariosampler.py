import json, numpy as np, pathlib
rng = np.random.default_rng(42)               # deterministic seeds
def sample_scenarios(N):
    # sample continuous axes; map to categories/buckets
    vis = rng.uniform(100.0, 5000.0, size=N)  # visibility meters
    lam = rng.uniform(0.1, 5.0, size=N)       # agent density (Poisson lambda)
    behavior = rng.integers(0, 3, size=N)     # 0=normal,1=aggr,2=adversarial
    fault_p = rng.beta(1.0, 9.0, size=N)      # sensor fault probability
    return [{"seed":int(rng.integers(0,2**31)),
             "visibility":float(v),"lambda":float(l),
             "behavior":int(b),"fault_p":float(fp)}
            for v,l,b,fp in zip(vis,lam,behavior,fault_p)]

def run_batch(scenarios):
    results = []
    for s in scenarios:
        # simulator call placeholder: replace with SIL/HIL API
        # sim_result = simulate_scenario(s, seed=s["seed"])
        sim_result = {"fail": int((s["visibility"]<300 and s["lambda"]>3) or (s["fault_p"]>0.5))}
        results.append({**s, **sim_result})
    return results

N=200
outdir = pathlib.Path("scenarios")
outdir.mkdir(exist_ok=True)
scens = sample_scenarios(N)
res = run_batch(scens)
json.dump(res, open(outdir/"batch.json","w"), indent=2)  # save traceable artifact
# Monte Carlo failure estimate
p_hat = sum(r["fail"] for r in res)/len(res)
print(f"Estimated failure rate: {p_hat:.3f}")