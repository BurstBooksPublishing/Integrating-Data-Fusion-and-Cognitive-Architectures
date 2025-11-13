import numpy as np

def conservative_distance(mean_pos, cov_pos, ego_pos, z=3.09):
    # mean and covariance of relative position vector
    rel = mean_pos - ego_pos
    grad = rel / np.linalg.norm(rel)  # linearize distance w.r.t. pos
    var_d = grad @ cov_pos @ grad.T    # projected variance
    mu_d = np.linalg.norm(rel)
    return mu_d - z * np.sqrt(max(var_d, 0.0))  # eq (2)

def stop_distance(velocity, reaction_time=0.5, decel=3.0):
    # simple braking model with margin
    return velocity * reaction_time + velocity**2/(2*decel) + 0.5

# Example fused belief
mean_ped = np.array([20.0, 1.5])     # meters ahead, lateral offset
cov_ped = np.array([[0.9,0.0],[0.0,0.6]])  # position covariance
ego_pos = np.array([0.0, 0.0])
v = 12.0  # m/s

d_cons = conservative_distance(mean_ped, cov_ped, ego_pos)  # compute conservative margin
d_stop = stop_distance(v)

if d_cons < d_stop:
    action = "BRAKE"   # conservative execution: slow down or stop
else:
    action = "PROCEED" # planner may continue
print(f"conservative_gap={d_cons:.2f}m stop_req={d_stop:.2f}m -> {action}")