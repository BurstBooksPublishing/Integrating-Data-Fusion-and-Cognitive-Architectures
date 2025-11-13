import numpy as np
from scipy.optimize import minimize

# example fused state (from L2/L3) and candidate actions (features)
situation = {'risk_model': lambda a: 0.0005 + 0.0001*a}  # safety risk as function of action
actions = np.linspace(0, 10, 21)  # scalar controllable knob (e.g., conveyor speed)

def f_safety(a, s): return s['risk_model'](a)          # safety cost (lower is safer)
def f_prod(a): return - (1.0 * a)                      # production: higher a -> higher throughput
def f_energy(a): return 0.2 * a                        # kWh proxy
def f_emiss(a): return 0.05 * a                        # CO2 proxy

# feasible set: actions with safety below threshold
safety_thresh = 1e-3
feasible = [a for a in actions if f_safety(a, situation) <= safety_thresh]

if not feasible:
    # fallback: reduce action space or trigger manual override
    plan = min(actions, key=lambda a: f_safety(a, situation))  # safest available
else:
    # optimize production within feasible discrete actions (simple selection)
    plan = max(feasible, key=lambda a: -f_prod(a))  # pick action maximizing production

# runtime guard: verify plan still safe at execution time
assert f_safety(plan, situation) <= safety_thresh
print("Selected plan action:", plan)