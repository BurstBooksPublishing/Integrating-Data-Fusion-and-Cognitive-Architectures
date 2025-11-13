import math, collections, random
# Simple hypothesis object
Hyp = collections.namedtuple('Hyp', ['label','params','weight'])

def likelihood(obs, hyp):
    # obs: dict with 'pos','vel','lane_id'; hyp.params: dict
    # lateral error likelihood (Gaussian)
    lat_err = obs['pos'][1] - hyp.params['lane_offset']
    L_lat = math.exp(-0.5*(lat_err/0.3)**2)
    # speed compatibility
    speed_err = obs['vel'] - hyp.params['target_speed']
    L_speed = math.exp(-0.5*(speed_err/1.0)**2)
    # simple gap acceptance for merge/overtake
    if hyp.label in ('merge','overtake'):
        g = obs.get('gap_time', 2.0)
        tau = hyp.params.get('gap_thresh', 1.5)
        L_gap = 1.0/(1.0+math.exp(-(g-tau)/0.3))
    else:
        L_gap = 1.0
    return L_lat * L_speed * L_gap

def update(hyps, obs):
    # Bayesian update and normalize, spawn/prune
    total = 0.0
    new = []
    for h in hyps:
        w = h.weight * likelihood(obs, h)
        new.append(Hyp(h.label, h.params, w))
        total += w
    # normalize
    if total==0:
        # reinitialize uniform fallback
        n = len(new) or 1
        return [Hyp(h.label,h.params,1.0/n) for h in new]
    new = [Hyp(h.label,h.params,h.weight/total) for h in new]
    # prune tiny weights
    return [h for h in new if h.weight>0.01]

def recommend_action(hyps):
    # simple expected-utility over three candidate actions
    actions = ['yield','proceed','evade']
    utils = {a:0.0 for a in actions}
    for h in hyps:
        # utility table heuristic
        for a in actions:
            if a=='yield' and h.label=='yield': u=5
            elif a=='proceed' and h.label=='merge': u=4
            elif a=='evade' and h.label=='overtake': u=3
            else: u=-1
            utils[a] += h.weight * u
    return max(utils.items(), key=lambda kv: kv[1])[0]

# Example usage: initialize hypotheses from scene graph
hyps = [
    Hyp('merge', {'lane_offset':0.0,'target_speed':8.0,'gap_thresh':1.2}, 0.4),
    Hyp('yield', {'lane_offset':-0.3,'target_speed':2.0}, 0.4),
    Hyp('overtake', {'lane_offset':0.1,'target_speed':12.0,'gap_thresh':1.0}, 0.2),
]
obs = {'pos':(10.0, -0.1), 'vel':7.5, 'lane_id':'A', 'gap_time':1.0}
hyps = update(hyps, obs)
action = recommend_action(hyps)
print(action)  # planner uses this suggestion