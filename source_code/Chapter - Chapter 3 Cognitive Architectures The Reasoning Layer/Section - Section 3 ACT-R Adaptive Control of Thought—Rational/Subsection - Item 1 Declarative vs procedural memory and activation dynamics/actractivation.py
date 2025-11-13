import time, math, numpy as np

np.random.seed(0)
# Example chunks with past rehearsal times (seconds ago)
now = 1000.0
chunks = {
    'trackA': [900.0, 950.0],      # two rehearsals
    'trackB': [980.0],             # recent single rehearsal
    'trackC': [700.0, 800.0, 900.0]# many older rehearsals
}

d = 0.5   # decay
F = 0.2   # latency scale
W = {'goal_protect': 1.0}               # one active context feature
S = {'trackA': 0.3, 'trackB': 0.8, 'trackC': 0.1}  # associative strengths

def base_level(times, t=now, d=d):
    # times: list of rehearsal timestamps
    return math.log(sum((t - tk) ** (-d) for tk in times))

activations = {}
for cid, times in chunks.items():
    B = base_level(times)
    spread = W['goal_protect'] * S[cid]  # single-context example
    noise = np.random.normal(0, 0.25)    # retrieval noise
    A = B + spread + noise
    T = F * math.exp(-A)
    activations[cid] = {'B':B, 'A':A, 'T':T}

# Select chunk with minimum retrieval latency (fastest available)
best = min(activations.items(), key=lambda kv: kv[1]['T'])
print(best)  # chosen chunk and its metrics