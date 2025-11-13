import random, math, statistics, time
# Simulate streaming requests (context_id simulates track/entity)
NUM_EVENTS = 2000
CANARY_RATE = 0.01  # initial canary fraction
ASSIGN_PROBS = {'A':0.5, 'B':0.5}  # balanced after canary
sticky_assign = {}  # per-entity sticky assignment

# logs
logs = []  # each entry: (ctx, variant, propensity, reward)

def policy_reward(ctx, variant):
    # deterministic baseline plus small noise; B is slightly better on odd ctx
    base = 1.0 if variant=='A' else 1.05
    if ctx % 2 == 1: base += 0.02  # simulate subgroup effect
    return base + random.gauss(0, 0.05)

for i in range(NUM_EVENTS):
    ctx = random.randint(0,499)  # 500 concurrent entities
    # sticky assignment to avoid track-hopping unless in interleaving mode
    if ctx in sticky_assign:
        variant = sticky_assign[ctx]
        propensity = 1.0  # deterministic for sticky; use per-decision logging if randomized
    else:
        # canary first, then balanced assignment
        if random.random() < CANARY_RATE:
            variant = 'B'  # canary routes a tiny unbalanced sample
            propensity = CANARY_RATE
        else:
            variant = random.choices(['A','B'], weights=(ASSIGN_PROBS['A'],ASSIGN_PROBS['B']))[0]
            propensity = ASSIGN_PROBS[variant]
        sticky_assign[ctx] = variant
    r = policy_reward(ctx, variant)
    logs.append((ctx, variant, propensity, r))

# IPS estimation for candidate B vs A (treat A as logging baseline)
def ips_value(logs, eval_variant):
    weighted = [(r * (1.0 if v==eval_variant else 0.0) / p) for (_,v,p,r) in logs]
    return statistics.mean(weighted), statistics.pstdev(weighted)/math.sqrt(len(weighted))

vA, seA = ips_value(logs, 'A')
vB, seB = ips_value(logs, 'B')
z = (vB - vA) / math.sqrt(seA*seA + seB*seB)
print(f"IPS V(A)={vA:.4f} se={seA:.4f}, V(B)={vB:.4f} se={seB:.4f}, z={z:.2f}")
# simple decision rule
if z > 1.96:
    print("B wins (p<0.05).")
elif z < -1.96:
    print("A wins.")
else:
    print("Inconclusive; continue data collection.")