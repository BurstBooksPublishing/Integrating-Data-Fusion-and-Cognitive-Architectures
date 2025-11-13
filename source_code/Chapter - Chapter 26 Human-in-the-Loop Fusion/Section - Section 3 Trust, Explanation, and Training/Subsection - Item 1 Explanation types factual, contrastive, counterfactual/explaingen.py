from typing import List, Dict, Any
# Simple scoring: weighted sum of evidence features (executable example)
def score(hyp:Dict[str,Any], evidence:Dict[str,float]) -> float:
    # hyp['weights'] maps feature->weight
    return sum(evidence.get(f,0.0)*w for f,w in hyp['weights'].items())

def generate_factual(hyps:List[Dict], evidence:Dict[str,float]) -> Dict:
    # return top hypothesis, score, and contributing features (provenance)
    scores = [(h, score(h,evidence)) for h in hyps]
    top_h, top_s = max(scores, key=lambda x:x[1])
    contrib = {f: evidence.get(f,0.0)*w for f,w in top_h['weights'].items() if evidence.get(f,0.0)!=0.0}
    return {'hypothesis': top_h['name'], 'score': top_s, 'contributions': contrib}

def generate_contrastive(hyps:List[Dict], evidence:Dict[str,float]) -> Dict:
    # compare top two hypotheses and list discriminative features
    scores = sorted([(h, score(h,evidence)) for h in hyps], key=lambda x:x[1], reverse=True)
    (h1,s1),(h2,s2) = scores[0], scores[1]
    # discriminative features: where weighted contributions differ in sign or magnitude
    features = set(h1['weights']).union(h2['weights'])
    diff = {f: (h1['weights'].get(f,0)*evidence.get(f,0) - h2['weights'].get(f,0)*evidence.get(f,0)) for f in features}
    return {'preferred': h1['name'], 'other': h2['name'], 'score_delta': s1-s2, 'feature_differences': diff}

def generate_counterfactual(hyps:List[Dict], evidence:Dict[str,float], intervention:Dict[str,float]) -> Dict:
    # apply intervention (overwrite features) and recompute ranking
    ev2 = evidence.copy(); ev2.update(intervention)
    scores_before = sorted([(h['name'], score(h,evidence)) for h in hyps], key=lambda x:x[1], reverse=True)
    scores_after  = sorted([(h['name'], score(h,ev2)) for h in hyps], key=lambda x:x[1], reverse=True)
    return {'before': scores_before, 'after': scores_after, 'intervention': intervention}

# Example usage (would be invoked by a cognition node)
if __name__ == "__main__":
    hyps = [{'name':'vehicle','weights':{'speed':0.7,'radar_r':0.3}},
            {'name':'bird','weights':{'speed':0.1,'radar_r':-0.5}}]
    evidence = {'speed': 12.0, 'radar_r': 0.8}
    print(generate_factual(hyps,evidence))                  # factual rationale
    print(generate_contrastive(hyps,evidence))              # contrastive rationale
    print(generate_counterfactual(hyps,evidence,{'speed':3})) # counterfactual probe