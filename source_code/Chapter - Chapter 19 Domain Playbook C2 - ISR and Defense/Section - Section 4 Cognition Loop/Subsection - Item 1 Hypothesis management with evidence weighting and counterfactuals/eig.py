import numpy as np
from scipy.stats import norm, entropy

# simple two-hypothesis model H0,H1 with Gaussian evidence models
priors = np.array([0.5, 0.5])                 # P(H0), P(H1)
# likelihood params: (mu, sigma) for each hypothesis per sensor
lik_params = {'AIS': [(0.0,1.0),(2.0,1.0)],
              'SAR': [(0.0,1.5),(1.5,1.0)]}

def loglik(e, sensor, h):
    mu, s = lik_params[sensor][h]
    return norm.logpdf(e, mu, s)

def update_posteriors(obs_list, sensors, weights):
    # obs_list: list of observed values; sensors aligned; weights per observation
    logodds = np.log(priors[1]/priors[0])
    for e, s, w in zip(obs_list, sensors, weights):
        llr = loglik(e, s, 1) - loglik(e, s, 0)
        logodds += w * llr
    p1 = 1/(1+np.exp(-logodds))
    return np.array([1-p1, p1])

def montecarlo_eig(current_obs, current_sensors, weights, probe_sensor,
                    probe_cost, nsamps=200):
    base_post = update_posteriors(current_obs, current_sensors, weights)
    # sample possible probe outcomes under predictive mixture
    # predictive: sum_h P(E'|h) P(h)
    samples = []
    for h, ph in enumerate(base_post):
        mu, s = lik_params[probe_sensor][h]
        samples.append(np.random.normal(mu, s, size=nsamps) * ph)
    samp_concat = np.concatenate(samples)
    kl_vals = []
    for eprime in samp_concat:
        new_obs = current_obs + [eprime]
        new_sensors = current_sensors + [probe_sensor]
        new_weights = weights + [1.0]               # assume probe reliability 1.0
        new_post = update_posteriors(new_obs, new_sensors, new_weights)
        kl_vals.append(entropy(new_post, base_post))
    return np.mean(kl_vals) / probe_cost

# Example usage
current_obs = [0.2]                             # AIS reading
current_sensors = ['AIS']
weights = [0.3]                                 # AIS discounted for spoof risk
score = montecarlo_eig(current_obs, current_sensors, weights,
                       probe_sensor='SAR', probe_cost=10.0)
print("EIG per cost:", score)                  # higher is better