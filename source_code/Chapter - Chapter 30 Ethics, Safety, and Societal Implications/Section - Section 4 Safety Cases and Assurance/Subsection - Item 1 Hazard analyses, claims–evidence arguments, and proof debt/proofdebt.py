# compute_proof_debt.py -- simple lifecycle tool
from typing import List, Dict
def proof_debt(weights: List[float], coverage: List[float]) -> float:
    # weights: claim criticality; coverage: evidence score per claim (0..1)
    assert len(weights) == len(coverage)
    return 1.0 - (sum(w*c for w,c in zip(weights,coverage)) / sum(weights))

# Example hazard record (could come from GSN tool or CSV)
hazard = {
  "id": "H_ped_dusk",
  "claims": ["L0_recall","L2_occlusion","L3_planner_bound"],
  "weights": [5.0, 3.0, 2.0],
  "coverage": [0.7, 0.2, 0.5], # aggregated evidence scores
  "mitigations": [
    ("collect_more_night_data", 5), # cost/priority metric
    ("improve_occlusion_model", 8),
    ("formalize_planner_envelope", 3)
  ]
}

pd = proof_debt(hazard["weights"], hazard["coverage"])
# prioritize mitigations by coverage_gap * claim_weight
gaps = [w*(1-c) for w,c in zip(hazard["weights"], hazard["coverage"])]
# map gaps to mitigation candidates (simple heuristic)
mit_order = sorted(hazard["mitigations"], key=lambda m: m[1])  # smaller cost first
print(f"Hazard {hazard['id']} PD={pd:.2f}, gaps={gaps}, mitigation_order={mit_order}")