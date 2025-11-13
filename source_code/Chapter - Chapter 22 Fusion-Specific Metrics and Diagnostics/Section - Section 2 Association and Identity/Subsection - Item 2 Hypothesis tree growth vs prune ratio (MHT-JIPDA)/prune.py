def prune_hypotheses(hypotheses, target_prune_ratio, safety_min=0.05):
    """
    hypotheses: list of (score, hypothesis_state) tuples, higher score better
    target_prune_ratio: desired fraction to drop (0..1)
    safety_min: minimum fraction to always retain
    """
    # sort descending by score
    hypotheses.sort(key=lambda x: x[0], reverse=True)
    G = len(hypotheses)                       # generated
    retain_goal = max(int((1 - target_prune_ratio) * G),
                      int(safety_min * G))   # enforce safety floor
    # keep top retain_goal
    retained = hypotheses[:retain_goal]
    pruned = hypotheses[retain_goal:]
    return retained, pruned

# Example usage: enforce <= 70% pruning
# retained, pruned = prune_hypotheses(candidate_list, target_prune_ratio=0.7)