import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(0)

# Simulate linear DAG: A -> B -> C and A -> C (A: sensor1, B: derived, C: outcome)
n = 2000
A = np.random.normal(size=n)
B = 0.8*A + 0.5*np.random.normal(size=n)
C = 0.6*B + 0.4*A + 0.5*np.random.normal(size=n)

# Fuse by stacking; add small sensor noise and a redundant correlated stream D
D = 0.9*A + 0.2*np.random.normal(size=n)
X = np.vstack([A,B,C,D]).T  # shape (n,4)

# Estimate precision matrix and partial correlations
cov = np.cov(X, rowvar=False)
prec = np.linalg.inv(cov)
rho = -prec / np.sqrt(np.outer(np.diag(prec), np.diag(prec)))  # partial corr matrix
np.fill_diagonal(rho, 1.0)

# Threshold to build undirected skeleton (simple criterion)
thr = 0.05
skeleton = (np.abs(rho) > thr).astype(int)
print("Partial-corr skeleton:\n", skeleton)

# Estimate causal effect of A on C using regression with adjustment set {B} (inferred)
X_adj = B.reshape(-1,1)
model = LinearRegression().fit(np.column_stack([A, X_adj]), C)  # controls for B
print("Estimated A->C coef (adj B):", model.coef_[0])  # effect estimate