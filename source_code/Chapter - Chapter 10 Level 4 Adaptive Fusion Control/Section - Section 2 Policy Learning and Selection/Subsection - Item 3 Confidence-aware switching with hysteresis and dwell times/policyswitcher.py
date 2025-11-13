import time
from typing import Dict

class PolicySwitcher:
    def __init__(self, policies, alpha=0.2, delta_h=0.12, tau_d=5.0):
        self.policies = policies                    # dict name->callable/action
        self.alpha = alpha
        self.delta_h = delta_h
        self.tau_d = tau_d
        self.smoothed: Dict[str,float] = {p:0.5 for p in policies}  # init baseline
        self.last_switch_time = time.monotonic()
        self.active = next(iter(policies))         # start policy
    def update_confidences(self, confidences: Dict[str,float]):
        # EMA smoothing of raw confidences
        for p,c in confidences.items():
            self.smoothed[p] = self.alpha*c + (1-self.alpha)*self.smoothed[p]
    def safety_ok(self, candidate)->bool:
        # placeholder for safety checks, e.g., resource budgets, invariants
        return True
    def maybe_switch(self):
        now = time.monotonic()
        if now - self.last_switch_time < self.tau_d:
            return self.active                        # dwell not expired
        # find best candidate
        best = max(self.smoothed, key=self.smoothed.get)
        if best == self.active:
            return self.active
        if not self.safety_ok(best):
            return self.active
        if self.smoothed[best] - self.smoothed[self.active] > self.delta_h:
            # perform switch and audit
            old = self.active
            self.active = best
            self.last_switch_time = now
            print(f"SWITCH {old} -> {best} at {now:.2f}; "
                  f"smoothed diff={self.smoothed[best]-self.smoothed[old]:.3f}")
        return self.active

# Example usage: policies are placeholders; confidences simulated by external estimator.