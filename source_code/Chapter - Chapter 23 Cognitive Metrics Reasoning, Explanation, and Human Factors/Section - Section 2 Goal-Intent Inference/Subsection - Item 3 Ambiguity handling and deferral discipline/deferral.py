import numpy as np

# posterior over goals (example)
posterior = np.array([0.6, 0.4])          # P(g1), P(g2)
# utilities matrix U[a_index, g_index]
utilities = np.array([[10.0, 0.0],        # U(a1, g1), U(a1, g2)
                      [2.0,  8.0]])       # U(a2, g1), U(a2, g2)

# compute expected utilities per action
eu_per_action = utilities.dot(posterior)  # shape (n_actions,)

best_a = np.argmax(eu_per_action)         # index of best action
best_eu = eu_per_action[best_a]

# deferral parameters (domain-calibrated)
U_defer = 9.0                             # expected utility if deferred
C_wait = 1.0                              # expected waiting cost

# decision
if best_eu < (U_defer - C_wait):
    decision = "DEFER"                    # escalate to human/analysis
else:
    decision = f"ACTION_{best_a+1}"       # commit chosen action

# emit trace (structured)
trace = {
    "posterior": posterior.tolist(),
    "eu_per_action": eu_per_action.tolist(),
    "best_eu": float(best_eu),
    "U_defer": U_defer,
    "C_wait": C_wait,
    "decision": decision
}
print(trace)                               # pipe to telemetry/logging