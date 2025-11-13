import time
from heapq import nlargest

class TokenBucket:
    def __init__(self, rate, burst):
        self.rate = rate         # tokens per second
        self.burst = burst       # max tokens
        self.tokens = burst
        self.ts = time.monotonic()
    def consume(self, amt=1):
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.ts) * self.rate)
        self.ts = now
        if self.tokens >= amt:
            self.tokens -= amt
            return True
        return False

def select_epoch_tasks(queue, capacity):
    """
    queue: iterable of dicts with keys: id, utility, cost, deadline, critical(bool)
    capacity: available compute units this epoch
    returns accepted task ids
    """
    # always accept safety critical tasks first
    accepted = []
    cap = capacity
    for t in queue:
        if t.get("critical", False) and t["cost"] <= cap:
            accepted.append(t["id"]); cap -= t["cost"]
    # build remaining candidates with value density and slack filter
    candidates = []
    now = time.time()
    for t in queue:
        if t["id"] in accepted: continue
        slack = t["deadline"] - now - t.get("est_runtime", t["cost"])
        if slack < 0: continue                    # already infeasible
        density = t["utility"] / max(1e-6, t["cost"])
        candidates.append((density, t))
    # greedy pick highest density while capacity allows
    for _, t in sorted(candidates, key=lambda x: x[0], reverse=True):
        if t["cost"] <= cap:
            accepted.append(t["id"]); cap -= t["cost"]
    return accepted

# Example usage
bucket = TokenBucket(rate=5, burst=10)
incoming = [{"id":"trk1","utility":8,"cost":2,"deadline":time.time()+5},
            {"id":"procH","utility":20,"cost":6,"deadline":time.time()+20,"critical":True}]
if bucket.consume():                    # fast-path rate-check
    accepted = select_epoch_tasks(incoming, capacity=8)
    # pass accepted ids to executor; rejected ones get explicit refusal events