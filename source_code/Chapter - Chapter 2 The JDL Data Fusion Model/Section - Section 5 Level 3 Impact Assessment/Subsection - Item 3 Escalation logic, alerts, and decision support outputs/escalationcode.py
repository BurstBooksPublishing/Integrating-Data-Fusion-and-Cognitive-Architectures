from dataclasses import dataclass, asdict
from time import time

@dataclass
class Hypothesis:
    id: str
    posterior: float
    U_no_act: float
    U_act: float
    provenance: dict

@dataclass
class Alert:
    hyp_id: str
    severity: str
    confidence: float
    recommendation: str
    rationale: dict
    timestamp: float

# Policy params
C_action = 5.0                # cost of escalation
alpha_auto = 2.0              # margin for auto action
p_quorum = 0.75               # modality quorum threshold
dwell = 3.0                   # seconds hysteresis

# simple in-memory dwell store
_last_state = {}              # hyp_id -> (last_decision, last_time)

def should_escalate(h: Hypothesis, now=None):
    now = now or time()
    deltaEU = h.posterior * (h.U_no_act - h.U_act)
    decision = deltaEU > C_action
    last = _last_state.get(h.id)
    if last and last[0] == decision:
        # maintain previous decision without re-evaluating dwell
        _last_state[h.id] = (decision, now)
        return decision
    # require dwell to flip state to avoid flapping
    if not last:
        _last_state[h.id] = (decision, now)
        return decision
    elapsed = now - last[1]
    if elapsed < dwell:
        return last[0]            # hold previous state
    _last_state[h.id] = (decision, now)
    return decision

def generate_alert(h: Hypothesis, modalities_supporting:int):
    if modalities_supporting < 2 and h.posterior < p_quorum:
        # with insufficient corroboration, produce advisory only
        severity = "advisory"
        rec = "Monitor and collect additional evidence"
    else:
        auto_cond = h.posterior * (h.U_no_act - h.U_act) > C_action * alpha_auto
        if auto_cond:
            severity = "auto-mitigate"
            rec = "Execute mitigation plan A (automated)"
        else:
            severity = "operator"
            rec = "Request operator confirmation for mitigation plan A"
    alert = Alert(
        hyp_id=h.id,
        severity=severity,
        confidence=h.posterior,
        recommendation=rec,
        rationale={"deltaEU": h.posterior*(h.U_no_act - h.U_act), "provenance": h.provenance},
        timestamp=time()
    )
    return alert

# Example usage (would be called in the fusion-cognition loop)
if __name__ == "__main__":
    h = Hypothesis("H1", posterior=0.82, U_no_act=100, U_act=40,
                   provenance={"sensors": ["radar","EO"], "model": "scenario-scorer-v1"})
    if should_escalate(h):
        alert = generate_alert(h, modalities_supporting=2)
        print(asdict(alert))   # deliver to message bus / operator UI