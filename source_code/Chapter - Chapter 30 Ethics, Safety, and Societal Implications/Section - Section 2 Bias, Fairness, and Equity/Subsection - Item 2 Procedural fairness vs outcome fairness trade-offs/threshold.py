import numpy as np
# synthetic calibrated scores and labels for cohorts A and B
scores = {'A': np.random.rand(1000), 'B': np.random.rand(1000)}
labels = {'A': (np.random.rand(1000) < 0.3).astype(int), # positive rate 30%
          'B': (np.random.rand(1000) < 0.2).astype(int)} # positive rate 20%

def tpr(scores_arr, labels_arr, th):
    preds = (scores_arr >= th).astype(int)
    # avoid divide-by-zero
    pos = labels_arr.sum()
    return preds[labels_arr==1].sum() / pos if pos>0 else 0.0

best = None
for ta in np.linspace(0,1,101):
    for tb in np.linspace(0,1,101):
        tpr_a = tpr(scores['A'], labels['A'], ta)
        tpr_b = tpr(scores['B'], labels['B'], tb)
        gap = abs(tpr_a - tpr_b)
        cost = 1 - ( ( (scores['A']>=ta)==labels['A']).mean()*(0.5) \
                   + ( (scores['B']>=tb)==labels['B']).mean()*(0.5) ) # simple accuracy cost
        # prefer small gap then low cost
        if best is None or (gap < best[0]) or (gap==best[0] and cost
\subsection{Item 3:  Mitigations: reweighting, constraints, and impact assessments}
Building on the trade-offs between procedural and outcome fairness and on subgroup coverage diagnostics, these mitigations are placed into an operational lifecycle. They must link data provenance (L0–L1) to situation and impact scoring (L2–L3) and to governance approvals before deployment.

Mitigation concept: three complementary families
- Pre-processing: rebalance or repair the training input to reduce sample bias without changing model class. Common techniques: importance reweighting, targeted augmentation, and synthetic minority oversampling.
- In-processing: change the learner objective to incorporate fairness constraints or adversarial debiasers. This preserves raw data but enforces downstream parity properties during optimization.
- Post-processing: adjust scores or decisions (thresholding, calibration, rule overrides) after model output to satisfy operational constraints while preserving provenance for audit.

Concrete process (lifecycle-aligned)
1. Audit and hazard analysis at L0–L2
   - Compute subgroup prevalence, detection rates, false-alarm asymmetries, and provenance gaps.
   - Produce an impact hypothesis: which subgroup errors produce mission-level harm?
2. Choose mitigation family by cost-benefit and assurance needs
   - If data collection feasible, prefer pre-processing for stable, interpretable change.
   - For constrained-latency systems, favor post-processing with human-in-loop veto.
3. Formalize objectives and constraints
   - Translate stakeholder fairness goals into measurable constraints C_j(θ) and tolerances δ_j.
4. Implement, simulate, and verify
   - Run closed-loop sims and HIL tests with counterfactual probes and stress suites.
5. Deploy with telemetry, rollback gates, and periodic impact re-assessments.

Mathematical framing
Use weighted empirical risk minimization with explicit constraints. Let g(i) denote group of sample i.
\begin{equation}[H]\label{eq:weighted_risk_constraint}
\min_{\theta}\; \sum_{i=1}^N w_{g(i)}\,\ell(f_\theta(x_i),y_i)\quad\text{s.t.}\quad C(\theta)=\left|\Pr\!\left(\hat{y}=1\mid g=a\right)-\Pr\!\left(\hat{y}=1\mid g=b\right)\right|\le\delta.
\end{equation}
Solve via Lagrangian relaxation when direct constrained solvers are impractical:
\begin{equation}[H]\label{eq:lagrangian}
\min_{\theta}\; \sum_{i} w_{g(i)}\ell(f_\theta(x_i),y_i) + \lambda\,\max\{0,\,C(\theta)-\delta\}.
\end{equation}

Design choices and estimates for weights
- Inverse prevalence: $w_g \propto 1/\hat{p}_{\text{obs}}(g)$ corrects sampling imbalance.
- Impact-weighted: $w_g \propto \text{expected\_harm}(g)$ prioritizes reducing harmful errors.
- Importance reweighting should be validated for variance inflation and overfitting.

Executable example: reweighting + subgroup impact report
\begin{lstlisting}[language=Python,caption={Weighted training and subgroup impact assessment},label={lst:reweight}]
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

# toy data: X, y, group (0/1) arrays
X = torch.randn(1000, 16)               # features
y = (torch.rand(1000) > 0.9).long()     # rare positive events
group = (torch.rand(1000) > 0.7).long() # imbalanced groups

# compute inverse-prevalence weights per group
unique, counts = np.unique(group.numpy(), return_counts=True)
p_obs = dict(zip(unique, counts / len(group)))
weights = torch.tensor([1.0 / p_obs[int(g.item())] for g in group]) 
weights = weights / weights.mean()         # normalize to keep scale

model = nn.Linear(16,1)
opt = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss(reduction='none')

for epoch in range(50):
    logits = model(X).squeeze(1)
    losses = loss_fn(logits, y.float())
    weighted_loss = (weights * losses).mean()   # apply group weights
    opt.zero_grad(); weighted_loss.backward(); opt.step()

# subgroup evaluation: detection rates and false alarm
with torch.no_grad():
    probs = torch.sigmoid(model(X)).numpy()
for g in [0,1]:
    mask = (group.numpy()==g)
    pd = ((probs[mask]>0.5) & (y.numpy()[mask]==1)).sum() / max(1, (y.numpy()[mask]==1).sum())
    pfa = ((probs[mask]>0.5) & (y.numpy()[mask]==0)).sum() / max(1, (y.numpy()[mask]==0).sum())
    print(f"group={g} PD={pd:.3f} PFA={pfa:.3f}")  # actionable subgroup metrics