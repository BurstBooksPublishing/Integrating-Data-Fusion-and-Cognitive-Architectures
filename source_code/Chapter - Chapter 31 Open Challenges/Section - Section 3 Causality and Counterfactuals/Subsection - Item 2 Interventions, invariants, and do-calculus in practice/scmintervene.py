import numpy as np
import pandas as pd

# SCM parameters
N = 200000
# U ~ Bern(0.5)
U = np.random.binomial(1, 0.5, size=N)
# S2 depends on U (confounder proxy)
S2 = np.random.binomial(1, 0.7*U + 0.2, size=N)
# S1 depends on U and noise
S1 = np.random.binomial(1, 0.6*U + 0.1, size=N)
# Y depends causally on S1 and modestly on U (unobserved)
prob_Y = 0.2 + 0.5*S1 + 0.15*U
Y = np.random.binomial(1, np.clip(prob_Y, 0, 1))

df = pd.DataFrame({'S1': S1, 'S2': S2, 'Y': Y})

def empirical_PY_do_S1(s):
    # Interventional sampling forces S1 = s, leaves U and S2 drawn from their margins.
    # Simulate by resampling U and generating Y with S1 fixed.
    M = 100000
    U_sim = np.random.binomial(1, 0.5, size=M)
    S2_sim = np.random.binomial(1, 0.7*U_sim + 0.2)
    probY = 0.2 + 0.5*s + 0.15*U_sim
    Y_sim = np.random.binomial(1, np.clip(probY, 0, 1))
    return Y_sim.mean()

def observational_PY(S1val):
    return df[df['S1']==S1val]['Y'].mean()

def backdoor_adjusted_PY(s):
    # Sum_z P(Y|S1=s,S2=z) * P(S2=z)
    terms = []
    for z in [0,1]:
        p_y = df[(df['S1']==s) & (df['S2']==z)]['Y'].mean()
        p_z = (df['S2']==z).mean()
        terms.append(p_y * p_z)
    return sum(terms)

for s in [0,1]:
    print(f"S1={s}: observational={observational_PY(s):.3f}, "
          f"interventional={empirical_PY_do_S1(s):.3f}, "
          f"backdoor_adj={backdoor_adjusted_PY(s):.3f}")
# Expected: backdoor_adj ≈ interventional, observational differs due to confounding.