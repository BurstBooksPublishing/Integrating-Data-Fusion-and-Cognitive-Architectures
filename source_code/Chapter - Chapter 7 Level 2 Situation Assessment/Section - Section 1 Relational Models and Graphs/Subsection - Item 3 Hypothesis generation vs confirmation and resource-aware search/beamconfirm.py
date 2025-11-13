import heapq, time
from dataclasses import dataclass
from typing import List

@dataclass(order=True)
class Cand:
    score: float  # negative for max-heap via heapq
    hypothesis: str
    p: float      # posterior approx
    u: float      # utility estimate
    cost: float   # confirmation cost

def propose(evidence) -> List[str]:
    # cheap generator: graph motifs or template expansion
    return ["rendezvous","transit","spoof"]

def fast_score(h, evidence):
    # replace with learned/lightweight model; returns (p,u)
    return (0.5 if h=="rendezvous" else 0.3, 10.0 if h=="rendezvous" else 4.0)

def confirm(h, evidence):
    # expensive check (deep model, sensor tasking); returns confirmed boolean
    time.sleep(0.01)                # simulate cost
    return h == "rendezvous"       # placeholder logic

def beam_and_confirm(evidence, budget, beam_width=5):
    # generation
    cands = []
    for h in propose(evidence):
        p,u = fast_score(h, evidence)
        cost = 1.0 if h=="rendezvous" else 0.5  # estimated confirm cost
        # expected net value = p*u - cost; use negative for heapq
        heapq.heappush(cands, Cand(-(p*u - cost), h, p, u, cost))
    # triage: keep beam
    beam = [heapq.heappop(cands) for _ in range(min(beam_width, len(cands)))]
    confirmed = []
    spend = 0.0
    for cand in beam:
        if spend + cand.cost > budget:
            continue  # respect budget
        # VoI decision: simple threshold on expected gain
        expected_gain = cand.p * cand.u - cand.cost
        if expected_gain <= 0:
            continue
        if confirm(cand.hypothesis, evidence):
            confirmed.append(cand.hypothesis)
        spend += cand.cost
    return confirmed

if __name__ == "__main__":
    print(beam_and_confirm({"tracks":"..."} , budget=2.0))  # short demo