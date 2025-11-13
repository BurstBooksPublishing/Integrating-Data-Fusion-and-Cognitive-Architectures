import numpy as np

def tie_break_intent(posteriors, utilities, cost_defer=0.1, margin=0.05):
    """
    posteriors: 1D numpy array over goals (sums to 1)
    utilities: 2D array shape (n_goals, n_actions) utility table
    cost_defer: cost for requesting more info (scalar)
    margin: posterior margin threshold to force inquiry
    returns: (choice, info) where choice is int index or 'DEFER'
    """
    # expected utility per goal (assumes action selection optimal under goal)
    # here we use max over actions as proxy for expected action utility
    eu = utilities.max(axis=1)  # best achievable utility per goal
    # weight by posterior to compare expected-outcome gap
    weighted_eu = posteriors * eu
    g_star = int(weighted_eu.argmax())
    top = weighted_eu[g_star]
    runner_up = weighted_eu[weighted_eu.argsort()[-2]]
    gap = top - runner_up
    # margin-based deferral or cost-aware decision
    if (posteriors.max() - np.partition(posteriors.flatten(), -2)[-2]) < margin:
        return ('DEFER', {'reason':'posterior_margin', 'gap':gap})
    if gap < cost_defer:
        return ('DEFER', {'reason':'utility_gap', 'gap':gap})
    return (g_star, {'gap':gap})
# Example run
if __name__ == "__main__":
    p = np.array([0.38,0.35,0.27])
    U = np.array([[10,7],[9,1],[2,1]])  # rows: goals, cols: actions
    print(tie_break_intent(p, U))  # either goal index or DEFER