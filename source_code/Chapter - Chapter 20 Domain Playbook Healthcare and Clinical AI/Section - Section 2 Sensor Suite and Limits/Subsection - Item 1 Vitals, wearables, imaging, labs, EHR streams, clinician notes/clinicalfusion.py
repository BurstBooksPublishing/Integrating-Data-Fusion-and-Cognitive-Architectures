import pandas as pd, numpy as np, re, json, datetime as dt

# synthetic streams (timestamp ISO strings) -- replace with real ingest
bed = pd.DataFrame({'ts':['2025-01-01T10:00:00','2025-01-01T10:00:05'],
                    'hr':[110,112],'var':[4.0,4.0]})
wear = pd.DataFrame({'ts':['2025-01-01T10:00:02'],'hr':[118],'var':[16.0]})
labs = pd.DataFrame({'ts':['2025-01-01T09:58:00'],'lab':['lactate'],'val':[3.2]})
notes = [{"ts":"2025-01-01T10:00:03","text":"Suspected sepsis; start antibiotics."}]

# normalize timestamps
for df in (bed,wear,labs):
    df['ts']=pd.to_datetime(df['ts'])
notes_ts = pd.to_datetime(notes[0]['ts'])

# align to 5-second windows; pick latest measurement per window
window = '5s'
bed_w = bed.set_index('ts').resample(window).last().dropna().reset_index()
wear_w = wear.set_index('ts').resample(window).last().dropna().reset_index()

# nearest-time fusion within 3s tolerance
def fuse_row(r1,r2, tol=pd.Timedelta('3s')):
    if abs(r1.ts - r2.ts) > tol: return None
    w1, w2 = 1.0/r1['var'], 1.0/r2['var']
    hr = (w1*r1['hr'] + w2*r2['hr'])/(w1+w2)
    var = 1.0/(w1+w2)
    return {'ts':r1.ts, 'hr':float(hr), 'var':float(var)}

fused=[]
i=j=0
while i
\subsection{Item 2:  Limits: heterogeneity, missingness, drift from practice changes}
The preceding sensor inventory (vitals, wearables, imaging, labs, EHR streams, clinician notes) defines heterogeneous inputs and temporal regimes that expose limits in fusion and cognition. The following describes those limits, the lifecycle processes to manage them, a concise mathematical test for distributional change, an executable pipeline fragment for detection and triage, and concrete operational trade-offs.

Concept — what the limits are and why they matter
- Heterogeneity: differences in format, unit, sampling rate, semantic schema, and provenance across devices and clinical systems. Heterogeneity causes misalignment in L0–L1 preprocessing and semantic mismatch when lifting to L2 ontologies.
- Missingness: three statistical regimes matter—MCAR (missing completely at random), MAR (missing at random conditioned on observables), and MNAR (missing not at random depending on unobserved factors). Treatment choice affects bias and downstream causal inferences.
- Drift from practice changes: operational shifts include protocol updates, new devices, altered lab panels, or guideline changes. These induce:
  - covariate shift (p(x) changes),
  - label shift (p(y) changes),
  - concept drift (p(y|x) changes).
Each drift type requires different detection and remediation.

Process — lifecycle integration and assurance
1. Ingest and normalize: canonical units, timestamp alignment, identity linkage, and schema mapping. Record provenance fields as first-class metadata.
2. Baseline characterization: maintain population baselines per cohort and per device generation for critical features.
3. Continuous monitoring:
   - Feature-level missingness rates and pattern matrices.
   - Distributional tests between current window and baseline.
   - Model performance tracking (AUC, calibration, NLL) per subgroup.
4. Triage and control:
   - Low-severity: flag, log, and surface to clinician dashboards.
   - Medium: retrain candidate models in shadow mode; activate recalibration.
   - High-severity: canary rollback, gated alerts, or human-in-the-loop veto.
5. Governance: change-control for practice/protocol updates, mandatory annotation of training data with versioned schemas, and silent trial windows before full rollout.

Example — a compact mathematical detector
Use a symmetric divergence to measure distributional change. For discrete binned feature histograms P and Q,
\begin{equation}[H]\label{eq:kl}
D_{\mathrm{KL}}(P\parallel Q)=\sum_i P_i\log\frac{P_i}{Q_i}
\end{equation}
and the population stability index (PSI) for bins i is commonly computed as
PSI = \sum_i (P_i - Q_i)\ln\frac{P_i}{Q_i}.
Thresholds (e.g., PSI>0.2) indicate significant shift requiring investigation. Complement with Kolmogorov–Smirnov for continuous features: sup_x|F_1(x)-F_2(x)|.

Code — streaming detection, missingness, and alerting
\begin{lstlisting}[language=Python,caption={Windowed drift and missingness monitor for clinical streams},label={lst:drift_monitor}]
import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

# load baseline histograms (precomputed) and streaming window df
baseline = pd.read_parquet("baseline_vitals.parquet")  # baseline population
window = pd.read_parquet("current_window.parquet")    # recent arrivals

def missingness_report(df):
    return df.isna().mean().sort_values(ascending=False)  # fraction missing

def psi(expected, actual, bins=10):
    exp_hist, edges = np.histogram(expected.dropna(), bins=bins, density=True)
    act_hist, _ = np.histogram(actual.dropna(), bins=edges, density=True)
    # avoid zeros
    exp_hist += 1e-8; act_hist += 1e-8
    return np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist))

alerts = []
# feature loop
for feat in ["heart_rate","spo2","resp_rate"]:
    miss = missingness_report(pd.concat([baseline[feat], window[feat]], axis=0))
    if miss.loc[feat] > 0.3: alerts.append(f"High missingness {feat}={miss.loc[feat]:.2f}")
    # KS test for distributional change
    stat, p = ks_2samp(baseline[feat].dropna(), window[feat].dropna())
    if stat>0.2 and p<0.01: alerts.append(f"KS shift {feat} stat={stat:.3f}")
    # PSI
    pscore = psi(baseline[feat], window[feat], bins=20)
    if pscore>0.2: alerts.append(f"PSI shift {feat} psi={pscore:.3f}")

# emit structured alert (to telemetry or governance queue)
for a in alerts: print(a)