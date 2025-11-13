import math

# Inputs: component MTBFs (hours) and target system SLO
component_mtbf = [10000.0, 10000.0]  # hours per component
target_slo = 0.999  # desired system availability

# Solve for equal MTTR per component under series composition
# Let A_i = MTBF_i / (MTBF_i + MTTR_i). For equal MTTR assume MTTR_i = x.
def system_avail(x):
    prod = 1.0
    for mtbf in component_mtbf:
        prod *= mtbf / (mtbf + x)
    return prod

# Simple bisection to find x such that system_avail(x) >= target_slo
lo, hi = 1e-6, 1e5
for _ in range(60):
    mid = 0.5*(lo+hi)
    if system_avail(mid) >= target_slo:
        hi = mid
    else:
        lo = mid
mttr_target = hi
print(f"Per-component MTTR target (hours): {mttr_target:.3f}")  # actionable number
print(f"Resulting system availability: {system_avail(mttr_target):.6f}")