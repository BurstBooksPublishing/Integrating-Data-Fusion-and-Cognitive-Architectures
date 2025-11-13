import numpy as np
from sklearn.utils import resample

# logs: list of dicts with keys 'noise_bin','var_bin','asr_ok','grasp_ok','asr_conf','grasp_margin'
logs = ...  # load trial logs

def cond_rate(logs, cond_key, cond_val, outcome_key):
    sel = [t for t in logs if t[cond_key]==cond_val]
    vals = np.array([t[outcome_key] for t in sel], dtype=int)
    if len(vals)==0: return np.nan, (np.nan, np.nan)
    rate = vals.mean()
    # bootstrap 95% CI
    boots = [resample(vals, replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rate, (lo, hi)

# example: conditional P(grasp | asr_ok=True)
rate, ci = cond_rate(logs, 'asr_ok', True, 'grasp_ok')
print('P(grasp|ASR_ok)=%.3f CI=[%.3f,%.3f]' % (rate, ci[0], ci[1]))

# estimate joint task success by eq. (1)
def joint_task_success(logs):
    # compute empirical p(n,v)
    bins = {}
    for t in logs:
        bins.setdefault((t['noise_bin'], t['var_bin']), []).append(t)
    total = len(logs)
    s = 0.0
    for (n,v), trials in bins.items():
        p_nv = len(trials)/total
        p_asr = np.mean([t['asr_ok'] for t in trials])
        p_grasp = np.mean([t['grasp_ok'] for t in trials])
        s += p_nv * p_asr * p_grasp
    return s

print('Joint task success:', joint_task_success(logs))