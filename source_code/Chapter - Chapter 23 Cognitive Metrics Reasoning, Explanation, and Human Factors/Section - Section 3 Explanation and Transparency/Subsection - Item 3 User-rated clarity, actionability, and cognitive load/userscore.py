import pandas as pd, numpy as np
from random import choices

# read CSV with columns: session_id, prov_id, clarity, actionability, cognitive_load
df = pd.read_csv('user_ratings.csv')  # ratings already normalized to [0,1]

# weights (task-specific)
alpha, beta, gamma = 0.4, 0.4, 0.2

# compute composite score per row
df['S'] = alpha*df['clarity'] + beta*df['actionability'] + gamma*(1.0 - df['cognitive_load'])

# bootstrap mean and 95% CI
def bootstrap_ci(arr, n=1000, p=95):
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    lo = np.percentile(means, (100-p)/2)
    hi = np.percentile(means, 100-(100-p)/2)
    return np.mean(means), lo, hi

mean_S, lo, hi = bootstrap_ci(df['S'].values)
print(f"Mean S={mean_S:.3f}, 95% CI=({lo:.3f}, {hi:.3f})")

# flag low-score sessions for audit (preserve provenance)
threshold = 0.6
to_audit = df[df['S'] < threshold][['session_id','prov_id','S']]
to_audit.to_csv('low_score_audit_queue.csv', index=False)  # fed to human-review pipeline