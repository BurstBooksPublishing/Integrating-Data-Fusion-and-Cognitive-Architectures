# simple proof-debt calculator for hazards
hazards = {
    "false_escalation": {"severity": 10,
        "evidence": {"unit_tests": (1.0,0.4),  # (completeness, weight)
                     "stress_sim": (0.8,0.35),
                     "shadow_logs": (0.2,0.15),
                     "formal_checks": (0.0,0.1)}},
    "sensor_loss_mode": {"severity": 7,
        "evidence": {"hw_inject": (0.7,0.5),
                     "redundancy_test": (1.0,0.5)}}
}
def compute_proof_debt(hazards_dict):
    pd_total = 0.0
    for h, meta in hazards_dict.items():
        s = meta["severity"]
        deficit = sum((1-c)*w for c,w in meta["evidence"].values())
        pd_total += s * deficit
    return pd_total
print("Proof debt:", compute_proof_debt(hazards))