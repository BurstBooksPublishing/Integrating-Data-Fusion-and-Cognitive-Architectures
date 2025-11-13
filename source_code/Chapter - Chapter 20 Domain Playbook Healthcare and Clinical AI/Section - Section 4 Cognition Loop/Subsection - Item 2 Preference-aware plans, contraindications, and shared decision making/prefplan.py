from typing import List, Dict, Callable
# Simple models: plans = list of dicts; evidence_model returns success probability.
def score_plans(plans: List[Dict], evidence_model: Callable, theta: Dict, hard_cons: List[Callable]):
    scored = []
    for p in plans:
        # Hard contraindication check: skip plan if any hard rule triggers.
        if any(c(p) for c in hard_cons):
            continue
        # Estimate outcome probability from L3 model; compute utility from preferences.
        prob_success = evidence_model(p)            # e.g., P(recovery|plan,evidence)
        utility = theta.get("benefit_weight",1.0)*prob_success \
                  - theta.get("burden_weight",0.1)*p.get("burden",0.0)
        # Add soft-penalty for mild contraindications.
        penalty = sum(p.get("soft_flags",[]))*theta.get("soft_penalty",0.5)
        score = utility - penalty
        scored.append({"plan":p,"score":score,"prob":prob_success})
    # Return ranked options plus a brief rationale string.
    return sorted(scored, key=lambda x: x["score"], reverse=True)

# Example usage omitted for brevity; integrate with L3 model and UI.