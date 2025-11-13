import math, numpy as np

def stopping_distance(v, t_delay, a_max):
    # v: m/s, t_delay: s, a_max: positive m/s^2
    return v * t_delay + v*v / (2.0 * a_max)

def ttc(s_rel, v_rel):
    # return None if no closing
    if v_rel >= 0: return None
    return s_rel / (-v_rel)

def decision_step(track, params):
    # track: dict with s_rel, v_rel, cov (2x2), ego_v
    s, vrel = track['s_rel'], track['v_rel']
    ego_v = track['ego_v']
    t_delay, a_max = params['t_delay'], params['a_max']
    t_crit, alpha = params['t_crit'], params['alpha']

    sd = stopping_distance(ego_v, t_delay, a_max)
    t = ttc(s, vrel)
    # quick probabilistic approx: if t exists, linearize variance
    if t is None:
        prob_critical = 0.0
    else:
        # compute sigma_TTC via delta method
        s_v = np.array([s, vrel])
        grad = np.array([1.0/(-vrel), s/(vrel*vrel)])
        sigma_t = math.sqrt(max(0.0, grad.dot(track['cov']).dot(grad)))
        # assume approx normal
        prob_critical = 0.5 * (1 + math.erf((t_crit - t)/(sigma_t*math.sqrt(2))))

    # fallback policy with hysteresis: hard stop > slow > continue
    if sd > s or prob_critical > alpha:
        return 'HARD_STOP', {'sd': sd, 'ttc': t, 'p': prob_critical}
    if t is not None and t < (t_crit*2):
        return 'SLOW_DOWN', {'sd': sd, 'ttc': t, 'p': prob_critical}
    return 'CONTINUE', {'sd': sd, 'ttc': t, 'p': prob_critical}

# Example usage omitted for brevity.