import json, math
from collections import defaultdict

# load scenarios: each entry has 'id','slice','weight','p_fail' (0..1)
with open('scenarios.json') as f:
    scenarios = json.load(f)

total_weight = sum(s['weight'] for s in scenarios)
# risk-weighted coverage from Eq. (1)
num = sum(s['weight'] * s['p_fail'] for s in scenarios)
rwc = 1.0 - num / total_weight

# per-slice diagnostics
slice_stats = defaultdict(lambda: {'weight':0.0,'risk':0.0,'count':0})
for s in scenarios:
    sl = s['slice']
    slice_stats[sl]['weight'] += s['weight']
    slice_stats[sl]['risk'] += s['weight'] * s['p_fail']
    slice_stats[sl]['count'] += 1

# produce ranked list of slices by risk density (risk per unit weight)
ranked = sorted(
    [(sl, v['risk']/v['weight'], v['count']) for sl,v in slice_stats.items()],
    key=lambda x: x[1], reverse=True
)

# output (a CI artifact)
print(f"RWC={rwc:.4f}; total_scenarios={len(scenarios)}")
for sl, risk_density, cnt in ranked[:10]:
    print(f"slice={sl} risk_density={risk_density:.3f} scenarios={cnt}")
# test harness can fail the build if RWC below threshold
if rwc < 0.90:
    raise SystemExit("Coverage gate failed: RWC below threshold")