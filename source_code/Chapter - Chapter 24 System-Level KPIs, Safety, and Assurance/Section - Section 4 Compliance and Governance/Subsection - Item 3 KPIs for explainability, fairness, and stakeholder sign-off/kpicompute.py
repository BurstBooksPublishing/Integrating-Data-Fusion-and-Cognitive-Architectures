import json, math, statistics
from collections import defaultdict
# load persistent decision log (JSON lines) with keys: model, explainer, decision, explainer_decision, group
logs = [json.loads(l) for l in open('decisions.log')]

# explainability: coverage and fidelity
covered = [1 for r in logs if r.get('explainer_decision') is not None]
fidelity_count = sum(1 for r in logs if r.get('explainer_decision')==r.get('decision'))
coverage = sum(covered)/len(logs)
fidelity = fidelity_count/len(logs)

# fairness: subgroup performance gap on accuracy
by_group = defaultdict(lambda: {'total':0,'correct':0})
for r in logs:
    g = r.get('group','__unknown')
    by_group[g]['total'] += 1
    by_group[g]['correct'] += int(r.get('decision')==r.get('label'))

group_acc = {g: v['correct']/v['total'] for g,v in by_group.items()}
overall = sum(v['correct'] for v in by_group.values())/sum(v['total'] for v in by_group.values())
subgroup_gap = max(abs(acc-overall) for acc in group_acc.values())

# sign-off: evidence completeness simple check
required_artifacts = ['metric_card','explanation_samples','canary_report']
present = json.loads(open('release_manifest.json').read()).get('artifacts',[])
evidence_completeness = len(set(required_artifacts)&set(present))/len(required_artifacts)

# emit report
report = {'coverage':coverage,'fidelity':fidelity,'subgroup_gap':subgroup_gap,'evidence_completeness':evidence_completeness}
print(json.dumps(report,indent=2))