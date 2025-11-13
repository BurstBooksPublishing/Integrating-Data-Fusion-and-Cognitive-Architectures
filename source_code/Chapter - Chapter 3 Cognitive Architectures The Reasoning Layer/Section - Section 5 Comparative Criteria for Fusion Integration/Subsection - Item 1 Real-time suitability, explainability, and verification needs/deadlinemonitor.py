import asyncio, time, json
# Simple async monitor enforcing deadline tau and fallback policy
async def run_with_deadline(task_coro, tau, fallback_coro, trace_log):
    start = time.time()
    try:
        # run task, but enforce deadline
        result = await asyncio.wait_for(task_coro, timeout=tau)
        elapsed = time.time() - start
        # append brief rationale (IDs, timings, rule hits)
        trace_log.append({"t": start, "elapsed": elapsed, "decision": result["label"]})
        return result
    except asyncio.TimeoutError:
        # record overrun and call deterministic fallback
        trace_log.append({"t": start, "elapsed": tau, "decision": "fallback", "reason": "deadline"})
        return await fallback_coro()  # deterministic quick response

# Example usage elsewhere in pipeline (pseudo-call)
# decision = await run_with_deadline(neuro_reasoner(obs), tau=0.2, fallback_coro=fast_rule_based(obs), trace_log=tracebuf)