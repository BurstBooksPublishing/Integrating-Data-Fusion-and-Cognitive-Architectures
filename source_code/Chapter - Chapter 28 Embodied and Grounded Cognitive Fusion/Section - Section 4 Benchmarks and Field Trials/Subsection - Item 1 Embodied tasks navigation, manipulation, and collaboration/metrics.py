import csv, math, statistics, sys

# Usage: python summarizer.py trials.csv
fname = sys.argv[1]
trials = []
with open(fname) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # expected columns: task, success (0/1), time_s, resets, violations
        trials.append({
            'task': row['task'],
            'success': int(row['success']),
            'time': float(row['time_s']),
            'resets': int(row['resets']),
            'violations': int(row['violations'])
        })

# aggregate per task
by_task = {}
for t in trials:
    by_task.setdefault(t['task'], []).append(t)

def summarize(group):
    n = len(group)
    succ = sum(g['success'] for g in group) / n
    mean_time = statistics.mean(g['time'] for g in group)
    resets = sum(g['resets'] for g in group) / n
    viol_rate = sum(g['violations'] for g in group) / n
    # composite score with example weights
    alpha, beta, gamma = 0.6, 0.3, 0.1
    tmin, tmax = 5.0, 120.0
    norm_time = max(0.0, min(1.0, (mean_time - tmin) / (tmax - tmin)))
    score = alpha*succ + beta*(1.0 - norm_time) - gamma*viol_rate
    return {'n': n, 'success': succ, 'mean_time': mean_time,
            'resets': resets, 'viol_rate': viol_rate, 'score': score}

for task, group in by_task.items():
    s = summarize(group)
    print(f"Task={task} trials={s['n']} success={s['success']:.3f} "
          f"t={s['mean_time']:.1f}s resets={s['resets']:.2f} "
          f"viol_rate={s['viol_rate']:.2f} score={s['score']:.3f}")
# end of script