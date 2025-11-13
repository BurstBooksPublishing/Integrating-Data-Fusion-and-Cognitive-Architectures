import numpy as np
from scipy.special import logsumexp
def reliability_stats(p, y, bins=10):
    # p: predicted prob vector, y: binary labels (0/1)
    idx = np.argsort(p)
    p_s, y_s = p[idx], y[idx]
    edges = np.quantile(p_s, np.linspace(0,1,bins+1))  # quantile bins
    ece = 0.0; mce = 0.0
    stats = []
    N = len(p)
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (p_s>=lo)&(p_s<=hi) if i==bins-1 else (p_s>=lo)&(p_s
\subsection{Item 2:  Covariance validity tests and NIS/NEES consistency}
The previous subsection described probabilistic calibration tools for outputs, such as reliability diagrams and ECE/MCE. Those tools assess marginal probability alignment; covariance validity tests evaluate the internal consistency of multivariate uncertainty representations across time.

Concept: Covariance validity checks ask whether reported covariances match observed dispersions. Two standard scalar summaries are the normalized innovation squared (\lstinline|NIS|) and the normalized estimation error squared (\lstinline|NEES|). Under the linear–Gaussian model and correct covariances,
\begin{equation}[H]\label{eq:nis}
\text{NIS}_k = \varepsilon_k^\top S_k^{-1}\varepsilon_k,\quad
\varepsilon_k = z_k - H x_{k|k-1},\quad
S_k = H P_{k|k-1} H^\top + R,
\end{equation}
and
\begin{equation}[H]\label{eq:nees}
\text{NEES}_k = e_k^\top P_{k|k}^{-1} e_k,\quad
e_k = x_k - x_{k|k},
\end{equation}
where $z_k$ is the measurement, $x_{k|k-1}$ and $x_{k|k}$ are prior and posterior estimates, $P$ are covariances, and $R$ is measurement noise covariance. Then $\text{NIS}_k\sim\chi^2_m$ and $\text{NEES}_k\sim\chi^2_n$, with $m,n$ the measurement and state dimensions.

Process: implement sample-level and sequence-level tests.
1. Choose test horizon length $N$ and significance level $\alpha$.
2. Compute scalars $\eta_k=\text{NIS}_k$ and $\xi_k=\text{NEES}_k$ for $k=1\ldots N$.
3. Use the chi-square sum distribution: $S_\eta=\sum_k\eta_k\sim\chi^2_{mN}$ and $S_\xi=\sum_k\xi_k\sim\chi^2_{nN}$.
4. Compute two-sided bounds
   \begin{equation*}
   \frac{\chi^2_{mN,\,\alpha/2}}{N}\le \bar\eta \le \frac{\chi^2_{mN,\,1-\alpha/2}}{N},
   \end{equation*}
   where $\bar\eta=S_\eta/N$. If $\bar\eta$ lies outside, the covariances are inconsistent at level $\alpha$.
5. Complement with time-series checks: autocorrelation of $\eta_k$ or running window tests to find nonstationary mismatches.

Example: the following Python snippet computes \lstinline|NIS| and \lstinline|NEES| for a trajectory, and performs chi-square consistency tests. It uses NumPy and SciPy and assumes access to ground-truth states for NEES evaluation.

\begin{lstlisting}[language=Python,caption={Compute NIS/NEES and chi-square consistency test},label={lst:nis_nees}]
import numpy as np
from scipy.stats import chi2

# Inputs: lists/arrays over k of innovations, prior/posterior covariances, and ground truth
eps_list = np.array(eps_list)         # shape (N, m): measurement innovations
S_list   = np.array(S_list)           # shape (N, m, m): innovation covariances
err_list = np.array(err_list)         # shape (N, n): estimation errors (truth - estimate)
Ppost_list = np.array(Ppost_list)     # shape (N, n, n): posterior covariances

N, m = eps_list.shape
n = err_list.shape[1]

# compute per-step NIS/NEES
nis = np.array([eps_list[k].T @ np.linalg.inv(S_list[k]) @ eps_list[k] for k in range(N)])
nees = np.array([err_list[k].T @ np.linalg.inv(Ppost_list[k]) @ err_list[k] for k in range(N)])

# sequence tests
alpha = 0.05
s_nis = nis.sum()
s_nees = nees.sum()

chisq_low_nis  = chi2.ppf(alpha/2,  m*N)
chisq_high_nis = chi2.ppf(1-alpha/2, m*N)
chisq_low_nees = chi2.ppf(alpha/2,  n*N)
chisq_high_nees= chi2.ppf(1-alpha/2, n*N)

# normalize to per-step average
avg_nis = s_nis / N
avg_nees= s_nees / N

print(f"avg NIS={avg_nis:.3f}, CI per-step=[{chisq_low_nis/N:.3f},{chisq_high_nis/N:.3f}]")
print(f"avg NEES={avg_nees:.3f}, CI per-step=[{chisq_low_nees/N:.3f},{chisq_high_nees/N:.3f}]")
# quick diagnostics: trends and autocorrelation (left as exercise)