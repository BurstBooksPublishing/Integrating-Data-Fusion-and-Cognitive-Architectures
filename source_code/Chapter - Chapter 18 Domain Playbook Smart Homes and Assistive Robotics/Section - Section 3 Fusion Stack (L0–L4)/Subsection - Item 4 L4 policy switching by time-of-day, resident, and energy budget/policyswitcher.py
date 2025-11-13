import time
import math
from typing import Dict, Callable

# Example policy descriptors with cost and expected utility models.
POLICIES = {
    "high_fidelity": {"energy": 5.0},
    "balanced": {"energy": 2.0},
    "low_power": {"energy": 0.7},
    "privacy_local": {"energy": 1.2},
}

def utility(policy: str, context: Dict) -> float:
    # Simple utility model: increase utility if resident needs assistive help.
    base = {"high_fidelity": 1.0, "balanced": 0.8, "low_power": 0.4, "privacy_local": 0.6}[policy]
    if context.get("resident_awake") and context.get("task_urgent"):
        base += 0.8
    if context.get("time_of_day") == "night":
        base -= 0.2
    return base

class PolicySwitcher:
    def __init__(self, lam=0.5, dwell=30.0, hysteresis=0.1):
        self.lam = lam
        self.dwell = dwell
        self.hysteresis = hysteresis
        self.current = "balanced"
        self.last_switch = time.time()
    def score(self, policy: str, ctx: Dict) -> float:
        return utility(policy, ctx) - self.lam * POLICIES[policy]["energy"]
    def select(self, ctx: Dict) -> str:
        scores = {p: self.score(p, ctx) for p in POLICIES}
        best = max(scores, key=scores.get)
        now = time.time()
        if best != self.current:
            if (scores[best] > scores[self.current] + self.hysteresis
               and (now - self.last_switch) > self.dwell):
                self._switch(best, scores[best], scores[self.current], ctx)
        return self.current
    def _switch(self, new, s_new, s_old, ctx):
        # Guarded enactment: check safety invariant before switching.
        if ctx.get("human_near") and new == "low_power":
            # reject unsafe downgrade
            self._log("reject", new, reason="safety_invariant")
            return
        self._log("switch", new, prev=self.current, delta=s_new - s_old)
        self.current = new
        self.last_switch = time.time()
    def _log(self, event, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {event} {args} {kwargs}")

# Example runtime loop
if __name__ == "__main__":
    switcher = PolicySwitcher(lam=0.6, dwell=15.0, hysteresis=0.05)
    ctx = {"resident_awake": True, "task_urgent": False, "time_of_day": "day", "human_near": False}
    for _ in range(6):
        print("active:", switcher.select(ctx))
        time.sleep(5)
        # simulate battery drain and an urgent task later
        ctx["task_urgent"] = True