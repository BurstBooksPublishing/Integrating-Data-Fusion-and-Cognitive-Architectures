import time, numpy as np
# simple Gaussian belief for a single target (mean, var)
belief_var = 4.0
sensors = {
    'cam': {'cost':1.0, 'sigma':2.0},    # lower info (high sigma), low cost
    'lidar': {'cost':3.0, 'sigma':0.8},  # high info, higher cost
    'radar': {'cost':2.0, 'sigma':1.2}
}
last_choice = None
dwell_seconds = 2.0
last_switch_time = 0.0
budget = 4.0

def expected_entropy_reduction(var_prior, sensor_sigma):
    # Gaussian posterior var = (1/var_prior + 1/sigma^2)^{-1}
    var_post = 1.0 / (1.0/var_prior + 1.0/(sensor_sigma**2))
    H_prior = 0.5 * np.log(2*np.pi*np.e*var_prior)
    H_post  = 0.5 * np.log(2*np.pi*np.e*var_post)
    return H_prior - H_post

def score_sensor(name, info):
    eig = expected_entropy_reduction(belief_var, info['sigma'])
    return eig / info['cost']

def select_sensors(now=time.time()):
    global last_choice, last_switch_time
    # enforce dwell: keep last selection if within dwell window
    if last_choice and now - last_switch_time < dwell_seconds:
        return [last_choice]
    # greedy fill under budget
    candidates = sorted(sensors.items(), key=lambda kv: -score_sensor(*kv))
    chosen, rem = [], budget
    for name, info in candidates:
        if info['cost'] <= rem:
            chosen.append(name); rem -= info['cost']
    # update dwell state if changed
    if chosen != [last_choice]:
        last_choice = chosen[0] if chosen else None
        last_switch_time = now
    return chosen

if __name__ == "__main__":
    print("Selected:", select_sensors())  # demonstration run