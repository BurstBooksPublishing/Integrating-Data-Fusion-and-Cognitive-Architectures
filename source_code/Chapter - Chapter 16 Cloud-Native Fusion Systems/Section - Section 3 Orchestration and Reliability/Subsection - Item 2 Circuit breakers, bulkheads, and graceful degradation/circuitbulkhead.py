import asyncio, time
from collections import deque

class CircuitBreaker:
    def __init__(self, max_fail=5, window=60, reset_after=30):
        self.max_fail = max_fail
        self.window = window
        self.reset_after = reset_after
        self.fail_times = deque()
        self.open_until = 0

    def record_failure(self):
        now = time.time()
        self.fail_times.append(now)
        # purge old failures
        while self.fail_times and now - self.fail_times[0] > self.window:
            self.fail_times.popleft()
        if len(self.fail_times) >= self.max_fail:
            self.open_until = now + self.reset_after

    def is_open(self):
        return time.time() < self.open_until

# bulkhead semaphore for concurrency isolation
bulkhead = asyncio.Semaphore(8)  # limit concurrent cloud LLM calls
cb = CircuitBreaker(max_fail=5, window=60, reset_after=30)

async def cloud_llm_query(prompt):
    # placeholder for real network call
    await asyncio.sleep(0.1)
    return {"summary": "cloud result"}

def local_rule_based_summarizer(prompt):
    # deterministic fallback, fast and auditable
    return {"summary": "local rule summary"}

async def safe_summarize(prompt):
    if cb.is_open():
        return local_rule_based_summarizer(prompt)  # deterministic fallback
    try:
        async with bulkhead:  # bulkhead isolation
            res = await cloud_llm_query(prompt)
            return res
    except Exception:
        cb.record_failure()
        return local_rule_based_summarizer(prompt)

# simple test harness
async def main():
    prompts = ["scene A"]*20
    tasks = [safe_summarize(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    print(results[:3])

if __name__ == "__main__":
    asyncio.run(main())