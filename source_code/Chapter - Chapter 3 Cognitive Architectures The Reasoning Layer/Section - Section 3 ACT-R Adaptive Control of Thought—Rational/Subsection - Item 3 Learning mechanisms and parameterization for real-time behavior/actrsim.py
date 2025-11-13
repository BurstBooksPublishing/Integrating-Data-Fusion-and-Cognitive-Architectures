import time, math, numpy as np

# params
d = 0.5           # decay
F = 0.3           # latency scale
t0 = 0.05         # base overhead seconds
alpha = 0.2       # utility learning rate
sigma_act = 0.1   # activation noise
sigma_u = 0.05    # utility noise
tau = 0.2         # retrieval threshold (latency cutoff)

def base_activation(times, now):
    # times: list of presentation timestamps
    if not times: return -np.inf
    return math.log(sum((now - t)**(-d) for t in times))

def retrieval_latency(A):
    return F * math.exp(-A) + t0

# synthetic loop: new fused confirmations arrive
chunk_times = []            # presentation history for one chunk
U = 0.5                     # initial production utility

for step in range(1,51):
    now = step * 1.0        # seconds
    # fused sensor confirms label with probability p
    p_confirm = 0.3 + 0.01*step
    if np.random.rand() < p_confirm:
        chunk_times.append(now)     # present chunk
    A = base_activation(chunk_times, now) + np.random.normal(0,sigma_act)
    T = retrieval_latency(A)
    # decide action: fast retrieval triggers auto-confirm, else escalate
    action = "auto_confirm" if T < tau else "escalate"
    # simulated reward: success if auto_confirm and true label confirmed (p=0.9)
    r = 1.0 if (action=="auto_confirm" and np.random.rand()<0.9) else 0.0
    U = U + alpha*(r - U) + np.random.normal(0,sigma_u)  # update utility
    print(f"{now:3.0f}s A={A:.2f} T={T:.3f} action={action} U={U:.3f}")
    time.sleep(0.01)  # simulate time advance