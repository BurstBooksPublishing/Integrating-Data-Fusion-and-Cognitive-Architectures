# Simulate simple evidence-accumulation decision with compute budget
import numpy as np

def simulate_policy(n_trials=1000, budget=5.0, cost_per_step=0.5, true_p=0.7):
    rng = np.random.default_rng(0)
    utilities, latencies = [], []
    successes = 0
    for _ in range(n_trials):
        belief = 0.5
        steps = 0
        # accumulate evidence while under budget
        while steps * cost_per_step < budget:
            obs = rng.random() < true_p  # noisy observation
            belief = (belief * steps + (1 if obs else 0)) / (steps + 1)
            steps += 1
            if belief > 0.75:  # satisficing decision threshold
                break
        decision = belief > 0.5
        success = bool(decision) and true_p > 0.5  # simplified success model
        utility = 1.0 if success else -1.0
        utility -= steps * cost_per_step * 0.1  # penalize compute cost
        utilities.append(utility)
        latencies.append(steps)
        successes += int(success)
    return {
        "mean_utility": np.mean(utilities),
        "success_rate": successes / n_trials,
        "p50_latency": np.percentile(latencies, 50),
        "p95_latency": np.percentile(latencies, 95)
    }

if __name__ == "__main__":
    print(simulate_policy())