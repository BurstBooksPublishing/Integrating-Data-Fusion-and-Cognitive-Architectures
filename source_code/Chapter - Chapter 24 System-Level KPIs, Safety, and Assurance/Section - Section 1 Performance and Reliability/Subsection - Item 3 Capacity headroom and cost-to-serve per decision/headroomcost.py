import numpy as np
# params
np.random.seed(1)
lambda_rate = 20.0            # decisions/sec (mean arrival)
service_rate = 30.0           # decisions/sec per capacity unit
capacity_units = 1.0          # number of parallel service units
sim_time = 200.0
C_fixed = 0.10                # $/sec fixed infra
C_compute_per_unit = 0.05     # $/sec per capacity unit
C_net = 0.01                  # $/sec network
C_store = 0.005               # $/sec storage
C_human = 0.02                # $/sec human overhead
# simulate Poisson arrivals and exponential service, M/M/1 approx
mu = service_rate*capacity_units
rho = lambda_rate/mu
assert rho < 1.0, "unstable system"
# analytic mean response (M/M/1)
mean_resp = 1.0/(mu-lambda_rate)
# approximate p95 via gamma approximation (conservative)
p95_resp = mean_resp * 3.0
# cost per decision
served = lambda_rate*sim_time
cost_total = (C_fixed + C_compute_per_unit*capacity_units + C_net + C_store + C_human)*sim_time
cost_per_decision = cost_total / served
# print results (would be logged in production)
print(f"utilization={rho:.2f}, mean_resp={mean_resp:.3f}s, p95~{p95_resp:.3f}s")
print(f"headroom={1-rho:.2f}, cost_per_decision=${cost_per_decision:.4f}")