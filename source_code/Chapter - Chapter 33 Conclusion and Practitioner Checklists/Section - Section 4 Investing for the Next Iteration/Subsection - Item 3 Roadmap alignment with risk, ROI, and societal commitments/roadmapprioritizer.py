#!/usr/bin/env python3
from math import isfinite

# candidate items: name, benefit, cost, fail_prob, societal_score
candidates = [
    ("improve_tracker", 120.0, 40.0, 0.08, 0.2),
    ("LLM_explainers", 200.0, 120.0, 0.12, 0.7),
    ("golden_trace_refresh", 80.0, 20.0, 0.02, 0.1),
]

lambda_risk = 0.9  # risk aversion
mu_soc = 1.5       # societal weight

def utility(B, C, p_f, S, lam=lambda_risk, mu=mu_soc):
    p_f = max(0.0, min(1.0, p_f))  # clamp
    U = (1 - lam * p_f) * B * (1 + mu * S) - C
    return U if isfinite(U) else float("-inf")

ranked = sorted(
    [(name, utility(B,C,p_f,S)) for name,B,C,p_f,S in candidates],
    key=lambda x: x[1], reverse=True
)
for name,score in ranked:
    print(f"{name}: utility={score:.2f}")  # prioritized list