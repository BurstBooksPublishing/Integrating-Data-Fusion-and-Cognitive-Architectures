import time, requests, logging
from collections import deque

LLM_URL = "https://api.example.com/v1/llm"  # external LLM endpoint
TIMEOUT = 0.8  # seconds allocated to LLM (T_llm budget)
RETRIEVAL_WINDOW = 0.5  # seconds for retrieval before LLM
metrics_hist = deque(maxlen=1000)  # simple latency buffer

def deterministic_rule_engine(context):
    # deterministic fallback: rule-based conservative scenario scorer
    # keep rules simple, auditable, and provenance-tagged
    if "hostile_maneuver" in context.get("events", []):
        return {"scenario":"possible_attack","confidence":0.6,"source":"rule_engine"}
    return {"scenario":"unknown","confidence":0.3,"source":"rule_engine"}

def cached_response(key):
    # lookup previously computed safe responses (memoization)
    # return None if not found
    return None

def call_llm(payload):
    start = time.time()
    try:
        resp = requests.post(LLM_URL, json=payload, timeout=TIMEOUT)
        latency = time.time() - start
        metrics_hist.append(latency)
        resp.raise_for_status()
        return {"ok":True, "result":resp.json(), "latency":latency}
    except Exception as e:
        latency = time.time() - start
        metrics_hist.append(latency)
        logging.warning("LLM call failed or timed out: %s", e)
        return {"ok":False, "error":str(e), "latency":latency}

def reason_with_fallback(context, retrieval_docs):
    # orchestration: retrieval -> LLM -> post-process -> fallback
    # retrieval stage should respect T_retrieval budget externally
    cache_key = f"ctx:{hash(str(context))}"
    cached = cached_response(cache_key)
    if cached:
        return {**cached, "provenance":"cache"}

    payload = {"prompt_schema":"scenario_v1","context":context,"docs":retrieval_docs}
    llm_out = call_llm(payload)
    if llm_out["ok"]:
        # validate schema, sanitize, and return
        return {"scenario": llm_out["result"], "provenance":"llm", "latency":llm_out["latency"]}
    # deterministic fallback path
    rule_out = deterministic_rule_engine(context)
    return {**rule_out, "provenance":"fallback", "latency":llm_out["latency"]}