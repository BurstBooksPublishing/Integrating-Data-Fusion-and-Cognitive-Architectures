from typing import List, Dict
import math

# simple evidence item: {'id':..., 'score':likelihood_ratio, 'source':..., 'unc':std}
def influence_score(evidence: Dict) -> float:
    # higher likelihood ratio and lower uncertainty increase influence
    return math.log(max(evidence['score'], 1e-6)) / max(evidence['unc'], 1e-3)

def compose_explanation(hypothesis: str, evidence_list: List[Dict], action: str, k: int=3):
    # rank evidence
    ranked = sorted(evidence_list, key=influence_score, reverse=True)
    top = ranked[:k]
    # concise rationale lines
    lines = [f"Recommendation: {action} (based on hypothesis: {hypothesis})."]
    for e in top:
        lines.append(f"- Evidence {e['id']} from {e['source']}: influence={influence_score(e):.2f}, uncert={e['unc']:.2f}")
    # brief counterfactual (show most important missing evidence)
    if len(ranked) < len(evidence_list):
        lines.append(f"If {evidence_list[len(top)]['id']} were absent, consider alternative.")
    return "\n".join(lines)

# example usage
evidence = [
    {'id':'radar_sig1','score':8.0,'source':'RadarA','unc':0.15},
    {'id':'comm_intel','score':2.2,'source':'SIGINT','unc':0.4},
    {'id':'track_speed','score':1.1,'source':'TrackPipe','unc':0.2},
]
print(compose_explanation("hostile_approach", evidence, "engage"))