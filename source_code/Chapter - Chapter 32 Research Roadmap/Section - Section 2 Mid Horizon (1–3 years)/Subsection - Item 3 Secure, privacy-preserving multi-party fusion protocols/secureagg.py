import os, hmac, hashlib, math, random
import numpy as np

# parties' local estimates
x = [2.4, 3.1, 1.8]
n = len(x)

# cyclic masks generation (shared randomness simulated)
r = [random.uniform(-1,1) for _ in range(n)]
y = [(x[i] + r[i] - r[(i+1)%n]) for i in range(n)]  # each party sends y_i

# aggregator sums masked payloads
masked_sum = sum(y)

# aggregator requests mask reveals (or masks cancel automatically if protocol is cyclic)
recovered_sum = masked_sum  # equals sum(x) by construction
assert abs(recovered_sum - sum(x)) < 1e-9

# apply Gaussian DP per Equation (ref:eq:gauss_dp)
Delta = max(x) - min(x)  # simple sensitivity for sum query in this toy example
epsilon, delta = 0.5, 1e-5
sigma = (Delta/epsilon)*math.sqrt(2*math.log(1.25/delta))
dp_noise = np.random.normal(0, sigma)
dp_sum = recovered_sum + dp_noise

# attach simple HMAC attestation (shared key per partner in practice)
key = b'shared_attestation_key'
msg = f"{dp_sum:.6f}".encode()
tag = hmac.new(key, msg, hashlib.sha256).hexdigest()

print("dp_sum", dp_sum, "hmac", tag)  # consumer verifies tag before using