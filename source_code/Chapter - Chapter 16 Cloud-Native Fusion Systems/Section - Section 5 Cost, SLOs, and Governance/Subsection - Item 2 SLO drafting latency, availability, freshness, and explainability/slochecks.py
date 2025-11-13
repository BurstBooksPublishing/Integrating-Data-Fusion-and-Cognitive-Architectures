import numpy as np, time

# sample data (ms latencies, seconds ages, booleans for explainability pass)
latencies = np.array([120, 80, 210, 95, 150, 180, 60])       # example latencies
ages = np.array([0.2, 0.3, 1.5, 0.9, 0.4, 0.8, 0.5])        # data ages at decision
explains_ok = np.array([True, True, False, True, True, True, True])  # rationale fidelity pass

# SLO targets
latency_slo = 0.99         # p99 target
latency_target_ms = 200.0
freshness_target_s = 1.0
explain_coverage_target = 0.95

# compute p99
p99 = np.percentile(latencies, 99)
latency_violations = (latencies > latency_target_ms).sum()
total = len(latencies)

# freshness compliance
fresh_ok = (ages <= freshness_target_s).sum()

# explainability coverage
explains_ok_count = explains_ok.sum()

# burn rate (Eq. burn_rate): errors/total divided by allowed error budget
latency_error_rate = latency_violations / total
burn_rate = latency_error_rate / (1 - latency_slo)

print(f"p99 latency = {p99:.1f} ms; violations = {latency_violations}/{total}")
print(f"freshness compliance = {fresh_ok}/{total}; explain coverage = {explains_ok_count}/{total}")
print(f"latency burn rate = {burn_rate:.2f}")  # >1 => error budget overspent