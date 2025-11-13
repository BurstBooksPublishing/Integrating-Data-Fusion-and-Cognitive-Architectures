import pandas as pd
import numpy as np

# load telemetry: columns ['confidence','correct','action','context_cost']
df = pd.read_csv('telemetry.csv')  # confidence in [0,1], correct:1/0, action: 'accept'/'override'

# binning
B = 10
df['bin'] = np.minimum((df['confidence']*B).astype(int), B-1)
grouped = df.groupby('bin')
n = len(df)

# ECE
ece = (grouped.size()/n * (grouped['correct'].mean() - grouped['confidence'].mean()).abs()).sum()
print('ECE:', ece)

# acceptance rate and conditional error per bin
acc_rate = grouped.apply(lambda g: (g['action']=='accept').mean())
error_if_accepted = grouped.apply(lambda g: g.loc[g['action']=='accept','correct'].apply(lambda x: 1-x).mean())
# replace NaN where no accepts
error_if_accepted = error_if_accepted.fillna(0)

print('Acceptance rates per bin:', acc_rate.values)
print('Error when accepted per bin:', error_if_accepted.values)

# overreliance risk using average cost per context
# R = mean over bins of P(s in bin)*P(accepted|bin)*P(wrong|accepted,bin)*E[C|bin]
p_bin = grouped.size()/n
cost_bin = grouped['context_cost'].mean().fillna(0)
R = (p_bin * acc_rate * error_if_accepted * cost_bin).sum()
print('Estimated overreliance risk per epoch:', R)