import numpy as np

# Example goal posterior from inverse planning (sum to 1).
P_g = np.array([0.6, 0.3, 0.1])  # e.g., [maintain, overtake, exit]

# Tactic library and parameter grids (scalar params for simplicity).
tactics = ["keep_lane","gradual_change","assertive_change"]
params = {"keep_lane": [0.0], "gradual_change": [0.2,0.4], "assertive_change":[0.6]}

# Utility matrix U(tactic_index, goal_index, param_index) placeholder.
def utility(t_idx, g_idx, p):
    # combine progress (higher for overtaking), comfort penalty for large p
    progress = [0.1, 1.0, 0.5][g_idx]            # domain-specific
    comfort_penalty = 2.0 * p                   # higher p => less comfort
    compliance = 1.0                             # add domain rule tests separately
    return progress - comfort_penalty + compliance

# Constraint checks: return True if feasible.
def constraints_ok(t, p, env):
    # env contains TTC to nearest vehicle and lane legality
    ttc = env["ttc"]
    lane_open = env["lane_open"]
    # Hard safety: require TTC > 1.5s for assertive maneuvers.
    if t == "assertive_change" and ttc < 1.5:
        return False
    # Lateral-jerk proxy: p should be <= 0.5
    if p > 0.5: 
        return False
    # lane must be open for any lane-change
    if t != "keep_lane" and not lane_open:
        return False
    return True

# Example environment observation
env = {"ttc": 2.0, "lane_open": True}

# Compute EU for all feasible (t,p)
results = []
for ti, t in enumerate(tactics):
    for p in params[t]:
        if not constraints_ok(t,p,env):
            continue
        # expected utility per Eq. (1)
        eu = sum(P_g[g]*utility(ti,g,p) for g in range(len(P_g)))
        results.append((t,p,eu))

# Rank tactics by EU
ranked = sorted(results, key=lambda x: x[2], reverse=True)
for t,p,eu in ranked:
    print(f"tactic={t}, param={p:.2f}, EU={eu:.3f}")  # brief inline comment