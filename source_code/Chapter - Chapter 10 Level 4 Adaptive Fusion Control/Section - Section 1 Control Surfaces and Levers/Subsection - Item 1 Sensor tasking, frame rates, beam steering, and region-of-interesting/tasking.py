import numpy as np

# simulate tracks: [id, trace_cov, salience, last_update]
tracks = np.array([
    (1, 2.5, 0.9, 0.1),
    (2, 5.0, 0.7, 0.5),
    (3, 1.2, 0.2, 1.2),
], dtype=[('id','i4'),('trace','f4'),('sal','f4'),('age','f4')])

# cost model: cost per frame rate unit (arbitrary)
cost_per_fps = 1.0
budget = 5.0  # total compute/bandwidth budget
max_fps = 10
min_fps = 1

# utility parameters
alpha, beta, gamma = 1.0, 2.0, 0.5

def expected_info_gain(trace, fps):
    # diminishing returns: higher fps reduces trace roughly inversely
    return trace - trace / (1.0 + 0.1*fps)

alloc = {int(t['id']): min_fps for t in tracks}  # start allocations
remaining = budget - sum(cost_per_fps * f for f in alloc.values())

# greedy loop: pick best marginal utility per cost
while remaining >= cost_per_fps * 1e-6:
    best = None
    best_ratio = 0.0
    for t in tracks:
        tid = int(t['id'])
        current_f = alloc[tid]
        if current_f >= max_fps:
            continue
        new_f = current_f + 1
        delta_info = expected_info_gain(t['trace'], new_f) - expected_info_gain(t['trace'], current_f)
        delta_U = alpha * delta_info + beta * t['sal']/(1.0 + t['age']) - gamma * cost_per_fps
        ratio = delta_U / cost_per_fps
        if ratio > best_ratio:
            best_ratio = ratio
            best = tid
    if best is None or best_ratio <= 0:
        break
    alloc[best] += 1
    remaining -= cost_per_fps

print("Allocations (fps):", alloc)
# Next steps: convert fps to sensor commands, compute beam steering to each ROI.