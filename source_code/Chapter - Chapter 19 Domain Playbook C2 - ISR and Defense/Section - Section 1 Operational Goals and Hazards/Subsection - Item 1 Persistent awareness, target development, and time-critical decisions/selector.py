import math
from typing import List, Dict, Tuple

# Example entity record: {'id':'T1','threat':0.8,'entropy':0.3,'t_last':100.0,'I0':1.0,'lambda':0.02,'cost':5.0}
def urgency(entity: Dict, t_now: float) -> float:
    dt = max(0.0, t_now - entity['t_last'])
    I = entity['I0'] * math.exp(-entity['lambda'] * dt)          # Eq. (1)
    certainty = 1.0 - min(1.0, entity['entropy'])               # map entropy->[0,1]
    return entity['threat'] * certainty * I                    # Eq. (3)

def select_targets(entities: List[Dict], t_now: float, budget: float) -> List[Tuple[str,float]]:
    # Greedy by urgency per unit cost
    scored = []
    for e in entities:
        u = urgency(e, t_now)
        score_per_cost = u / max(1e-6, e['cost'])
        scored.append((score_per_cost, u, e['id'], e['cost']))
    scored.sort(reverse=True)  # highest score_per_cost first
    chosen = []
    used = 0.0
    for spc, u, id_, cost in scored:
        if used + cost <= budget:
            chosen.append((id_, u))
            used += cost
    return chosen

# --- Minimal usage example ---
if __name__ == "__main__":
    now = 200.0
    ents = [
        {'id':'T1','threat':0.9,'entropy':0.2,'t_last':190.0,'I0':1.0,'lambda':0.05,'cost':4.0},
        {'id':'T2','threat':0.7,'entropy':0.6,'t_last':150.0,'I0':1.0,'lambda':0.02,'cost':6.0},
        {'id':'T3','threat':0.4,'entropy':0.1,'t_last':199.0,'I0':1.0,'lambda':0.1,'cost':2.0},
    ]
    print(select_targets(ents, now, budget=10.0))  # prints chosen target ids with urgency