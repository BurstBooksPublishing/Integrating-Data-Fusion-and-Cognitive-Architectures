import math, json
from typing import List, Dict, Tuple

# Simple event: {'t': float, 'pos': (x,y), 'units': 'm', 'score_vec': [p1,p2,...]}
def euclid(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def validate_scenario(events: List[Dict], v_max: float, eps: float=0.5):
    # Check timestamp monotonicity and generate counterexample on first failure.
    for i in range(len(events)-1):
        t0, t1 = events[i]['t'], events[i+1]['t']
        if t1 < t0:
            return False, {'kind':'time_order','index':i,'t0':t0,'t1':t1}
        dt = t1 - t0
        # Unit check: convert if needed; here we enforce meters.
        if events[i]['units'] != 'm' or events[i+1]['units'] != 'm':
            return False, {'kind':'unit_mismatch','index':i,
                           'units0':events[i]['units'],'units1':events[i+1]['units']}
        # Kinematic check per Eq. (1)
        d = euclid(events[i]['pos'], events[i+1]['pos'])
        if d > v_max*dt + eps:
            return False, {'kind':'kinematic','index':i,'distance':d,'dt':dt,'v_max':v_max}
    # Probabilistic check: last event may carry scenario score vector
    if 'score_vec' in events[-1]:
        vec = events[-1]['score_vec']
        if any(p < -1e-9 or p > 1+1e-9 for p in vec):
            return False, {'kind':'prob_range','vec':vec}
        s = sum(vec)
        if abs(s-1.0) > 1e-6:
            return False, {'kind':'prob_norm','sum':s,'vec':vec}
    return True, {'kind':'ok'}

# Example usage: returns counterexample describing first failure.
if __name__ == '__main__':
    trace = [
      {'t':0.0,'pos':(0.0,0.0),'units':'m'},
      {'t':1.0,'pos':(100.0,0.0),'units':'m'},   # implausible for low v_max
      {'t':2.0,'pos':(200.0,0.0),'units':'m','score_vec':[0.6,0.4]}
    ]
    ok, info = validate_scenario(trace, v_max=10.0)  # 10 m/s max
    print(ok, json.dumps(info))  # integrate print into telemetry or alerting