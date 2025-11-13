import numpy as np

# simple generative SCM components
def sensor_reading(true_presence, noise_scale, drop_prob=0.0):
    if np.random.rand() < drop_prob:
        return None  # sensor dropout
    return true_presence + np.random.normal(scale=noise_scale)

def fusion_decision(s1, s2, thresh=0.5):
    # naive fusion: average available sensors, abstain if both missing
    vals = [v for v in (s1,s2) if v is not None]
    if not vals: return False  # conservative: no detection => no stop
    return (np.mean(vals) > thresh)

def run_counterfactual(n_trials=10000, intervention=None):
    # intervention: dict e.g. {'sensor2_noise': 1.5, 'sensor1_drop': 0.2}
    rng = np.random.RandomState(0)
    violations = 0
    for _ in range(n_trials):
        true_presence = 1.0 if rng.rand() < 0.1 else 0.0  # scene prior
        s1 = sensor_reading(true_presence,
                            noise_scale=0.2 if intervention is None else intervention.get('sensor1_noise',0.2),
                            drop_prob=0.0 if intervention is None else intervention.get('sensor1_drop',0.0))
        s2 = sensor_reading(true_presence,
                            noise_scale=0.3 if intervention is None else intervention.get('sensor2_noise',0.3),
                            drop_prob=0.0 if intervention is None else intervention.get('sensor2_drop',0.0))
        decision = fusion_decision(s1, s2, thresh=0.5)
        # violation: pedestrian present but safe-stop not initiated
        if true_presence > 0.5 and not decision:
            violations += 1
    return violations / n_trials

# baseline and counterfactual
baseline_risk = run_counterfactual(n_trials=5000)
cf_risk = run_counterfactual(n_trials=5000, intervention={'sensor2_drop': 0.8})
print("Baseline violation rate:", baseline_risk)
print("Counterfactual (sensor2 dropout) rate:", cf_risk)