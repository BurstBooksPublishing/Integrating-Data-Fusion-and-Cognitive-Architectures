import numpy as np
import pandas as pd
import dowhy  # causal inference library
from dowhy import CausalModel

# synth data: C (severity), A (escalate), Y (mission outcome metric)
np.random.seed(1)
n = 2000
C = np.random.normal(size=n)                   # fused L2 severity score
A = (0.5*C + 0.2*np.random.normal(size=n))>0   # cognitive escalate decision (binary)
Y = 1.5*A + 0.8*C + 0.5*np.random.normal(size=n) # outcome influenced by A and C

data = pd.DataFrame({'C':C, 'A':A.astype(int), 'Y':Y})
model = CausalModel(data=data, treatment='A', outcome='Y',
                    common_causes=['C'])       # declare confounder from fusion
identified = model.identify_effect()           # check identifiability
estimate = model.estimate_effect(identified,
                                 method_name="backdoor.linear_regression") # estimate
print(estimate.value)                           # causal effect estimate