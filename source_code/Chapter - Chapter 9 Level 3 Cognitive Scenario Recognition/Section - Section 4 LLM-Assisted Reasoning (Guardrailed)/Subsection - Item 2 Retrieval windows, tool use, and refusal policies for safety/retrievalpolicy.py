import time, math
# simple time-decay weight and greedy window selection
def time_weight(ts, now, tau): return math.exp(-(now - ts)/tau)

def select_window(evidence_list, now, tau, token_budget):
    # evidence_list: [{'ts':..., 'tokens':..., 'relevance':..., 'payload':...}, ...]
    scored = [ (e['relevance']*time_weight(e['ts'], now, tau), e) for e in evidence_list ]
    scored.sort(reverse=True, key=lambda x: x[0])
    selected, tokens = [], 0
    for score, e in scored:
        if tokens + e['tokens'] > token_budget: break
        selected.append(e); tokens += e['tokens']
    return selected

def guarded_llm_invoke(prompt_schema, allowed_functions, deterministic_fallback):
    # call to LLM returns {'ok':bool,'confidence':float,'result':...,'refusal':bool}
    resp = llm_call(prompt_schema, allowed_functions=allowed_functions) # external SDK
    if not resp['ok'] or resp.get('refusal') or resp['confidence'] < 0.6:
        # deterministic fallback (rule engine or human escalation)
        return deterministic_fallback(prompt_schema)
    return resp['result']

# usage
now = time.time()
window = select_window(evidence_buffer, now, tau=300.0, token_budget=2048)
prompt = build_schema_prompt(window)  # enforces schema and provenance tags
result = guarded_llm_invoke(prompt, allowed_functions=['score_scenario'], fallback=rule_engine)