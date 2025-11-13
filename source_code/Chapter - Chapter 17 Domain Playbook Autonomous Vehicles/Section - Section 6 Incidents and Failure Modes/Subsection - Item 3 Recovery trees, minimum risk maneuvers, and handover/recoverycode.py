from math import inf
import time

# Example fused posterior P(s|z)
posterior = {'occluded_ped': 0.6, 'static_obstacle': 0.4}

# Candidate actions with cost model C(a,s)
cost = {
    'immediate_stop': {'occluded_ped': 1.0, 'static_obstacle': 0.2},
    'slow_approach':   {'occluded_ped': 2.0, 'static_obstacle': 0.5},
    'lane_shift':      {'occluded_ped': 3.0, 'static_obstacle': 0.7},
}

# Health and assurance thresholds
handover_risk_threshold = 1.5    # if best expected risk > threshold -> handover
min_autonomy_confidence = 0.7    # minimal belief concentration to act
map_staleness_limit = 5.0        # seconds

def expected_risk(action, posterior, cost):
    return sum(posterior[s]*cost[action][s] for s in posterior)

def should_handover(best_risk, posterior, sensor_state):
    # handover if high risk or diffuse posterior or sensor/map faults
    belief_entropy_proxy = 1.0 - max(posterior.values())  # simple dispersion metric
    if best_risk > handover_risk_threshold: return True
    if belief_entropy_proxy > (1-min_autonomy_confidence): return True
    if sensor_state['map_age'] > map_staleness_limit: return True
    if not sensor_state['actuators_ok']: return True
    return False

# simple evaluation
sensor_state = {'map_age': 1.2, 'actuators_ok': True}
best_action, best_risk = None, inf
for a in cost:
    r = expected_risk(a, posterior, cost)
    if r < best_risk:
        best_action, best_risk = a, r

handover = should_handover(best_risk, posterior, sensor_state)
if handover:
    # handover protocol: safe-stop and notify operator (traceable)
    print("Handover: safe-stop, upload trace, notify operator")
else:
    print(f"Execute {best_action} with expected risk {best_risk:.2f}")