import math, time, random
# candidates: models with cost estimate (ms) and prior weight
candidates = {'A': {'prior':0.5,'cost':20},
              'B': {'prior':0.3,'cost':50},
              'C': {'prior':0.2,'cost':10}}
dwell_seconds = 30  # minimum time to hold a model
last_switch = 0
current = 'A'

def likelihood(perf):  # map KPI (accuracy) to likelihood
    # assume beta-like scaling: higher accuracy => much higher likelihood
    return max(1e-6, math.exp(10*(perf-0.8)))  # tuned factor

def select_model(kpi_stream):  # kpi_stream: dict candidate->accuracy
    # update posterior-like weights
    weights = {}
    for m,acc in kpi_stream.items():
        w = candidates[m]['prior'] * likelihood(acc)
        weights[m] = w
    # compute utility = expected reward (accuracy) - lambda*cost
    lam = 0.01
    utilities = {m: kpi_stream[m] - lam*candidates[m]['cost'] for m in weights}
    # softmax over utilities for stochastic exploration, keep argmax for action
    z = sum(math.exp(u) for u in utilities.values())
    probs = {m: math.exp(u)/z for m,u in utilities.items()}
    chosen = max(utilities, key=utilities.get)
    return chosen, probs

def adaptation_loop():
    global current, last_switch
    while True:
        # collect per-model recent KPI (could be from shadow runs)
        kpis = {m: max(0.0, min(1.0, random.normalvariate(0.85 - 0.05*ord(m[0])%3, 0.02)))
                for m in candidates}
        chosen, probs = select_model(kpis)
        now = time.time()
        # safety/hysteresis: enforce dwell time before switching
        if chosen != current and (now - last_switch) < dwell_seconds:
            chosen = current  # defer switch
        # safety check: require posterior-like mass for switch
        if chosen != current:
            # simple confidence gate: chosen prob must exceed 0.6
            if probs[chosen] < 0.6:
                chosen = current
        if chosen != current:
            # commit switch and update prior (small learning rate)
            lr = 0.1
            candidates[chosen]['prior'] = (1-lr)*candidates[chosen]['prior'] + lr
            candidates[current]['prior'] *= (1-lr)
            last_switch = now
            # log rationale (would be sent to audit store)
            print(f"Switching {current} -> {chosen}, probs={probs}")
            current = chosen
        # periodic background shadow retraining/validation could update priors here
        time.sleep(1)  # pacing for demo

# Run adaptation loop (in production, integrate with monitors and safe-stop hooks)
# adaptation_loop()  # uncomment to run