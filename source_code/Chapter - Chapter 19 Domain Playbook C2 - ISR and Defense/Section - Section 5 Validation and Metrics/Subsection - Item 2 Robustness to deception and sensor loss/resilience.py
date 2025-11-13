import numpy as np, math, random, time
# simple sensor model: likelihood ratios for H1 vs H0
def sensor_llr(true_state, spoof=False, bias=0.0):
    # base LLR distribution: Gaussian around +2 for H1, -2 for H0
    mu = 2.0 if true_state else -2.0
    sample = np.random.normal(mu + (bias if spoof else 0.0), 1.0)
    return sample

def run_trial(T=200):
    # sensors: EO, SAR, ESM
    weights = np.array([0.6, 0.3, 0.1])
    avail = np.ones(3)
    threshold = 0.0
    detections = []
    for t in range(T):
        true = (t>80 and t<160)  # transient event window
        # simulate spoof and dropout
        spoof = (t>100 and t<130)  # EO spoof window
        avail[2] = 0 if (t%25==0) else 1  # ESM intermittent dropout
        llrs = np.array([sensor_llr(true, spoof=spoof, bias=5.0),
                         sensor_llr(true),
                         sensor_llr(true)])
        # apply availability and weights
        llr_total = np.sum(avail * weights * llrs)
        detections.append(llr_total>threshold)
    # compute simple metrics
    detections = np.array(detections)
    pd = detections[81:159].mean()
    far = detections[:80].mean()
    return {'PD':pd, 'FAR':far}

# run ensemble trials
results=[run_trial() for _ in range(200)]
print({k:np.mean([r[k] for r in results]) for k in ['PD','FAR']})