import numpy as np

# simulate neural detector: returns hazard probability at each timestep
def neural_detect(true_hazard, p_hit=0.9, p_false=0.05):
    if true_hazard:
        return np.random.rand() < p_hit  # true positive with noise
    else:
        return np.random.rand() < p_false  # false positive

# symbolic alarm rule: raises alarm if detector true for two consecutive frames
def symbolic_alarm(det_seq):
    for t in range(1, len(det_seq)):
        if det_seq[t-1] and det_seq[t]:
            return t  # alarm time index
    return None

# property: if true hazard occurs at time 0, alarm must occur by deadline
def check_property(num_steps=5, deadline=2, trials=10000):
    success = 0
    failures = []
    for i in range(trials):
        # simulate single episode: hazard at t=0 with 50% chance
        hazard = np.random.rand() < 0.5
        det_seq = [neural_detect(hazard if t==0 else False) for t in range(num_steps)]
        alarm_t = symbolic_alarm(det_seq)
        sat = (not hazard) or (alarm_t is not None and alarm_t <= deadline)
        if sat:
            success += 1
        else:
            if len(failures) < 5:
                failures.append({'det_seq': det_seq, 'alarm_t': alarm_t})
    p_hat = success / trials
    return p_hat, failures

if __name__ == "__main__":
    p, counter = check_property(trials=20000)  # empirical check
    print(f"Empirical p_sat = {p:.4f}")         # report probability
    print("Example failures:", counter)         # compact counterexamples