import asyncio, random, time
# Simple token bucket
class TokenBucket:
    def __init__(self, rate, burst):
        self.rate = rate; self.burst = burst
        self.tokens = burst; self.last = time.monotonic()
    def consume(self, n=1):
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.last)*self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n; return True
        return False
    def set_rate(self, r): self.rate = r

# AIMD controller updates token bucket refill rate
class AIMDController:
    def __init__(self, init_rate=10.0, alpha=1.0, gamma=0.5):
        self.rate = init_rate; self.alpha = alpha; self.gamma = gamma
    def on_success(self): self.rate += self.alpha
    def on_congestion(self): self.rate *= self.gamma
    def get_rate(self): return max(0.1, self.rate)

# Sender with retries, jittered backoff, and deadline check
async def send_loop(queue, controller):
    bucket = TokenBucket(controller.get_rate(), burst=controller.get_rate()*2)
    while True:
        item = await queue.get()  # (msg, priority, deadline, idempotent_key)
        msg, deadline = item['payload'], item['deadline']
        now = asyncio.get_event_loop().time()
        if now > deadline:
            # mark stale, optionally escalate to cognition controller
            queue.task_done(); continue
        # update bucket rate periodically
        bucket.set_rate(controller.get_rate())
        sent = False
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts and not sent:
            if not bucket.consume():
                await asyncio.sleep(0.01); continue
            attempt += 1
            try:
                # simulated network send with possible loss
                await network_send(msg)  # raises on simulated loss
                controller.on_success(); sent = True
            except CongestionError:
                controller.on_congestion()
                # jittered exponential backoff
                backoff = (2**attempt) * 0.05 * (1 + random.random()*0.2)
                await asyncio.sleep(backoff)
        if not sent:
            # give up: log bounded failure, attach provenance, and inform L4
            record_failure(msg, attempts=attempt)
        queue.task_done()

# placeholder network primitives
async def network_send(msg):
    await asyncio.sleep(0.01)  # serialize+tx time
    if random.random() < 0.1: raise CongestionError()
class CongestionError(Exception): pass
def record_failure(msg, attempts): pass