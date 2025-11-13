import time, asyncio

class SafetyMonitor:
    def __init__(self, T_s, c_min):
        self.T_s = T_s            # staleness bound (s)
        self.c_min = c_min        # confidence threshold
        self.last = {}            # last update times per token
        self.conf = {}            # last confidence per token
        self.state = "nominal"

    def update(self, token, confidence):
        self.last[token] = time.time()   # record freshness
        self.conf[token] = confidence    # record confidence

    def check_permit(self, token):
        t = time.time() - self.last.get(token, 0.0)
        c = self.conf.get(token, 0.0)
        return (t <= self.T_s) and (c >= self.c_min)

    async def monitor_loop(self):
        while True:
            # evaluate critical tokens and escalate deterministically
            if not self.check_permit("object_track"):
                await self.escalate()
            await asyncio.sleep(0.1)  # watchdog period

    async def escalate(self):
        if self.state == "nominal":
            self.state = "reduced_speed"   # step 1
            # log, increase sampling, notify planner
        elif self.state == "reduced_speed":
            self.state = "conservative_model"  # step 2
        elif self.state == "conservative_model":
            self.state = "minimal_risk_maneuver"  # step 3
            # execute safe-stop and alert operator
# Example usage omitted for brevity