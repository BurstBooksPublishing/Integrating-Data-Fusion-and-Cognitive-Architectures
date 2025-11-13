import copy
import json

class HybridReasoner:                             # stub: replace with real service client
    def decide(self, context):                    # returns decision scalar or label
        # simple rule: obstruction triggers stop, else based on speed
        if context.get("obstruction", False): return "STOP"
        return "SLOW" if context.get("target_speed",0) < 5 else "GO"
    def explain(self, context):                   # returns list of (token,importance)
        # importance matches rule logic; real explainers can be LLM + schema
        if context.get("obstruction", False):
            return [("obstruction", 0.9), ("target_speed", 0.1)]
        return [("target_speed", 0.8), ("map_quality", 0.2)]

def invariance_check(reasoner, base_ctx, permute_fields):
    base_dec = reasoner.decide(base_ctx)
    base_exp = reasoner.explain(base_ctx)
    for f in permute_fields:
        ctx = copy.deepcopy(base_ctx)
        ctx[f] = ctx.get(f, None)                 # permutation or replacement
        dec = reasoner.decide(ctx)
        exp = reasoner.explain(ctx)
        assert dec == base_dec, f"Decision changed on irrelevant field {f}"
        assert json.dumps(exp, sort_keys=True) == json.dumps(base_exp, sort_keys=True), \
            f"Explanation changed for field {f}"
    print("Invariance checks passed.")

if __name__ == "__main__":
    r = HybridReasoner()
    context = {"target_speed": 4, "obstruction": False, "map_quality": 0.7, "sensor_id": "A1"}
    invariance_check(r, context, permute_fields=["sensor_id", "trace_id"])  # irrelevant fields