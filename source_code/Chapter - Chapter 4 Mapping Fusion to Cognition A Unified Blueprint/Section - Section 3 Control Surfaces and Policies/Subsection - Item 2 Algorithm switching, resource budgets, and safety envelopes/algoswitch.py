import time, math

class Algo:
    def __init__(self,name,cost,utility,requires): 
        self.name, self.cost, self.utility, self.requires = name,cost,utility,requires

# Example algorithms with cost (CPU units), expected utility, and safety prereq
ALGS = [
    Algo("detector_high", cost=30, utility=0.95, requires={"gps_fix":True}),
    Algo("tracker_light", cost=5, utility=0.6, requires={}),
    Algo("filter_pf", cost=15, utility=0.8, requires={}),
]

class Manager:
    def __init__(self, budget_seq, switch_penalty=5, dwell=2.0):
        self.budget_seq = budget_seq         # callable->current budget
        self.penalty = switch_penalty
        self.dwell = dwell
        self.last_switch = 0
        self.current = None

    def safety_ok(self, algo, telemetry):
        # Simple safety envelope: require predicates present in telemetry
        return all(telemetry.get(k)==v for k,v in algo.requires.items())

    def select(self, telemetry):
        B = self.budget_seq()               # available budget
        now = time.time()
        candidates=[]
        for a in ALGS:
            if a.cost <= B and self.safety_ok(a, telemetry):
                score = a.utility
                if self.current and a.name != self.current.name:
                    score -= self.penalty
                candidates.append((score, a))
        if not candidates: return self.failover(telemetry)
        best = max(candidates, key=lambda x:x[0])[1]
        # enforce dwell time
        if self.current and now - self.last_switch < self.dwell:
            return self.current
        if self.current and best.name != self.current.name:
            self.last_switch = now
        self.current = best
        return best

    def failover(self, telemetry):
        # deterministic safe fallback: lowest-cost safe algo
        safe = [a for a in ALGS if self.safety_ok(a, telemetry)]
        return min(safe, key=lambda a:a.cost) if safe else None

# Example runtime loop (pseudo-real)
def budget_provider():
    return 20  # e.g., measured CPU units

m = Manager(budget_provider)
telemetry = {"gps_fix":True}
for _ in range(5):
    a = m.select(telemetry)
    print("Selected:", a.name if a else "NONE")
    time.sleep(1)