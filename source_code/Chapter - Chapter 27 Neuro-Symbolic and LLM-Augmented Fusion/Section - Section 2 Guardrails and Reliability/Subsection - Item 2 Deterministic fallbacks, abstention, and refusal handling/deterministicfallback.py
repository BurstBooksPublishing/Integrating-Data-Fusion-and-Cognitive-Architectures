import time, json, random
# simple schema validator (no external deps)
def validate_schema(obj, required_keys):
    return all(k in obj for k in required_keys)

# simulated LLM call with timeout
def call_llm(prompt, timeout=1.0):
    start = time.time()
    # simulate network/compute jitter
    time.sleep(min(0.2, timeout))
    if time.time() - start > timeout:
        raise TimeoutError("LLM timeout")
    # simulated response: structured dict + confidence
    return {"classification": random.choice(["friend","unknown","hostile"]),
            "confidence": random.uniform(0.4, 0.95)}

# deterministic fallback rule engine (conservative)
def deterministic_fallback(sensor_tracks):
    # simple conservative heuristic: require at least two agreeing modalities
    votes = {}
    for t in sensor_tracks:
        votes[t["source"]] = t["label"]
    # return majority or "unknown"
    labels = list(votes.values())
    label = max(set(labels), key=labels.count) if labels else "unknown"
    return {"classification": label, "confidence": 0.75, "source": "fallback"}

# orchestrator
def decide_action(context, thresholds=(0.8, 0.65), alpha=0.6):
    try:
        llm_out = call_llm(context["prompt"], timeout=context.get("llm_timeout",1.0))
    except TimeoutError:
        llm_out = {"error":"timeout","confidence":0.0}
    # validate LLM schema
    if "error" in llm_out or not validate_schema(llm_out, ["classification","confidence"]):
        # immediate fallback on malformed/timeout
        fallback = deterministic_fallback(context.get("sensor_tracks",[]))
        return {"mode":"fallback","result":fallback,"reason":"llm_failure"}
    # aggregate confidences
    c = float(llm_out["confidence"])
    # deterministic integrity check: simple geometric consistency score
    d = 0.9 if context.get("geometry_consistent",False) else 0.4
    c_comb = alpha*c + (1-alpha)*d
    tau_auto, tau_abst = thresholds
    if c_comb >= tau_auto:
        return {"mode":"auto","result":llm_out,"score":c_comb}
    if c_comb >= tau_abst:
        # abstain: run fallback but escalate
        fallback = deterministic_fallback(context.get("sensor_tracks",[]))
        return {"mode":"abstain","result":fallback,"score":c_comb,"escalate":True}
    # refusal: refuse to act, provide rationale
    return {"mode":"refuse","reason":"low_confidence","score":c_comb}

# example run
ctx = {"prompt":"classify contact","llm_timeout":0.5,
       "geometry_consistent":False,
       "sensor_tracks":[{"source":"radar","label":"unknown"},
                        {"source":"eo","label":"unknown"}]}
print(decide_action(ctx))