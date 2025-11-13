import numpy as np
# simple track record and measurement; real system has lists/IDs
track_state = np.array([10., 2.])        # position, velocity
P = np.array([[4., 0.],[0., 1.]])        # covariance
H = np.array([[1., 0.]])                 # measurement model
z = np.array([12.0])                     # measurement
R = np.array([[2.0]])                    # meas noise
S = H @ P @ H.T + R                      # innovation covariance
v = z - H @ track_state                  # innovation
d2 = float(v.T @ np.linalg.inv(S) @ v)   # Mahalanobis distance
threshold = 9.21                         # chi2(1,0.99)
stable_count = 3
track_meta = {'stable': 0, 'provenance': []}
if d2 < threshold:
    track_meta['stable'] += 1
    track_meta['provenance'].append(f"gate_ok d2={d2:.2f}")
else:
    track_meta['stable'] = 0
    track_meta['provenance'].append(f"gate_fail d2={d2:.2f}")

# promotion decision
if track_meta['stable'] >= stable_count:
    cognitive_queue = [{'state': track_state.copy(),
                        'cov': P.copy(),
                        'provenance': track_meta['provenance'].copy()}]
else:
    cognitive_queue = []

print("d2", d2, "stable", track_meta['stable'], "promoted", bool(cognitive_queue))