import numpy as np

np.random.seed(0)
T_hours = 8760  # one year
# simulate failure process (Poisson)
fail_rate_baseline = 1/2000.0  # failures per hour
fails_baseline = np.random.poisson(fail_rate_baseline * T_hours)

# alerts: true alerts correlate with failures within lead window
lead_window = 24
true_alerts = fails_baseline  # one true alert per failure (simplify)
false_alerts_baseline = int(true_alerts * 0.43)  # FAR ~ 30% -> FP/(TP+FP)=0.3 => FP ~ 0.43*TP

# after fusion+cognition
fails_post = max(0, int(fails_baseline * 0.77))  # 23% fewer failures prevented by intervention
false_alerts_post = int(false_alerts_baseline * 0.4)  # FAR reduced
tp_actions = fails_post  # assume actions correspond to true failures handled
fp_actions = false_alerts_post

def mtbf_from_fails(total_hours, failures): 
    return total_hours / max(1, failures)

MTBF0 = mtbf_from_fails(T_hours, fails_baseline)
MTBF1 = mtbf_from_fails(T_hours, fails_post)
MTTR = 4.0

def availability(mtbf, mttr): 
    return mtbf / (mtbf + mttr)

FAR0 = false_alerts_baseline / (true_alerts + false_alerts_baseline)
FAR1 = false_alerts_post / (true_alerts + false_alerts_post)
Precision0 = true_alerts / (true_alerts + false_alerts_baseline)
Precision1 = tp_actions / (tp_actions + fp_actions)

print(f"MTBF baseline {MTBF0:.1f} h -> post {MTBF1:.1f} h")
print(f"Availability baseline {availability(MTBF0,MTTR):.5f} -> post {availability(MTBF1,MTTR):.5f}")
print(f"FAR {FAR0:.2f} -> {FAR1:.2f}; Precision {Precision0:.2f} -> {Precision1:.2f}")