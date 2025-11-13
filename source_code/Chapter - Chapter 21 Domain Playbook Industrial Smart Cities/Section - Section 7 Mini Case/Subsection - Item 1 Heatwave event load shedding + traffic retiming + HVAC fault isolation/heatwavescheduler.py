import numpy as np
# simple fused inputs (would come from L0/L1 services)
feeder_load = 0.95          # fraction of rated
transformer_temp = 95.0     # Celsius
queue_lengths = np.array([10, 25, 5])  # vehicles per lane
hvac_fault_conf = 0.8       # [0,1]

# priority weights (tunable by policy)
w_outage = 0.7; w_traffic = 0.2; w_hvac = 0.1

# candidate actions: fractions of noncritical load to shed, signal offsets (s), hvac_isolate bool
candidates = [
    {'shed': 0.03, 'retime': np.array([0,5,0]), 'isolate': False},
    {'shed': 0.05, 'retime': np.array([0,10,0]), 'isolate': False},
    {'shed': 0.05, 'retime': np.array([0,10,0]), 'isolate': True},
]

def impact_score(action):
    # rough models: outage risk ~ f(load_after, temp), traffic harm ~ mean(queue+retime), hvac risk ~ fault_conf if not isolated
    load_after = feeder_load - action['shed']
    outage_risk = max(0, (load_after - 0.9)) * (transformer_temp/100.0)
    traffic_harm = np.mean(queue_lengths + action['retime']/5.0) / 50.0
    hvac_risk = hvac_fault_conf * (0.0 if action['isolate'] else 1.0)
    return w_outage*outage_risk + w_traffic*traffic_harm + w_hvac*hvac_risk

# evaluate and pick best action under safety constraints
best = None; best_score = 1e9
for a in candidates:
    if a['shed'] > 0.1: continue  # safety: no >10% single-step shed
    if a['isolate'] and hvac_fault_conf < 0.5: continue  # avoid false isolation
    score = impact_score(a)
    if score < best_score:
        best_score = score; best = a

# guarded execution: simulate and require operator if outage reduction below threshold
simulated_outage = max(0, (feeder_load-best['shed']-0.9))*(transformer_temp/100.0)
if simulated_outage > 0.05:
    print("Escalate to operator; simulated outage risk high.")  # operator approval needed
else:
    # issue commands to actuators (placeholders)
    print(f"Issuing shed {best['shed']*100:.1f}% , retime {best['retime']}, isolate {best['isolate']}")
    # log action with provenance for audit