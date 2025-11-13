import itertools, json, random, multiprocessing as mp
# mock evaluation function (replace with real pipeline call)
def evaluate(scenario): 
    # scenario: dict with params; returns metrics dict
    random.seed(scenario["seed"])
    # simulate failure rates to illustrate ranking
    fail_prob = 0.1 + 0.4*scenario["spoof"] + 0.3*(1-scenario["visibility"])
    return {"scenario": scenario, "fail": random.random() < fail_prob}

# build param grid (lightweight example)
spoofer=[0.0, 0.5, 1.0]  # spoof intensity
visibility=[0.2, 0.6, 0.9]  # radar visibility
seeds=[42, 99]
scenarios=[{"spoof":s,"visibility":v,"seed":sd} for s,v,sd in itertools.product(spoofer,visibility,seeds)]

# run in parallel and collect results
with mp.Pool(4) as pool:
    results=pool.map(evaluate, scenarios)
# sort by observed failure frequency (simple diagnostic)
ranked=sorted(results, key=lambda r: r["fail"], reverse=True)
print(json.dumps(ranked, indent=2))  # example output for triage