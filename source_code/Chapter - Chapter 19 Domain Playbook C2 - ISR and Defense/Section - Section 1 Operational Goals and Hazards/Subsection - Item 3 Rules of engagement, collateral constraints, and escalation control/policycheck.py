def policy_check(action, state, evidence, policy): 
    # compute collateral probability using fused evidence (probabilistic model)
    p_collateral = estimate_collateral_prob(action, state, evidence) 
    # compute required auth level for action
    required_level = policy.required_escalation_level(action, state) 
    # check hard collateral bound
    if p_collateral > policy.collateral_budget: 
        return {"decision":"veto", "reason":"collateral_exceed", "p":p_collateral}
    # check authorization bound and dwell-time rules
    if required_level > policy.current_auth_level: 
        # package compact rationale for human reviewer
        rationale = summarize_rationale(action, state, evidence) 
        request = make_escalation_request(action, rationale, required_level) 
        log_request(request) 
        return {"decision":"escalate", "request_id":request["id"]}
    # soft constraints: attach cost penalty to planner if close to limits
    cost_penalty = policy.soft_penalty(p_collateral) 
    return {"decision":"approve", "penalty":cost_penalty}