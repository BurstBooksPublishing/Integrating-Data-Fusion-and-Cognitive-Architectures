# simple trace structure and mock LLM/symbolic checks
trace = [("00:00","vessel_A:heading=090"),
         ("00:10","vessel_B:heading=270"),
         ("00:20","vessel_A:range_to_B=2nm")]

def intervene(trace, edit):  # apply surgical edit
    t,f = edit
    return [x for x in trace if x[0]!=t] + [(t,f)]

def mock_llm_answer(context, question):  # deterministic stub
    # returns structured tuple (claim, confidence)
    if "will collide" in question:
        # simple heuristic: range <1nm implies high collision risk
        for _,f in context:
            if "range_to_B=0.5nm" in f:
                return ("collision_likely",0.92)
        return ("collision_unlikely",0.12)

def symbolic_verify(claim, context):  # rule-based verifier
    for _,f in context:
        if "range_to_B=0.5nm" in f and claim=="collision_likely":
            return True
    return False

# build counterfactual: reduce range at 00:20
cf = intervene(trace, ("00:20","vessel_A:range_to_B=0.5nm"))
ans,conf = mock_llm_answer(cf,"If vessel_A reduces separation, will they collide?")
ok = symbolic_verify(ans,cf)
print(ans,conf,ok)  # demonstrates pipeline: intervention -> LLM -> symbolic check