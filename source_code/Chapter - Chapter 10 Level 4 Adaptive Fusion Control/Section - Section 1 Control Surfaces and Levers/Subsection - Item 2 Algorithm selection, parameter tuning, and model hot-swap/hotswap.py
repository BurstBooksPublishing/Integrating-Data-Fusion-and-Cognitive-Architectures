import time, threading
from collections import deque
# simple model interface: model.predict(observation)
# registry holds pre-validated candidate models
class HotSwapController:
    def __init__(self, active, registry, dwell_sec=10, beta=1.5):
        self.active = active
        self.registry = registry
        self.dwell_sec = dwell_sec
        self.beta = beta
        self.last_switch = 0
        self.metrics_window = deque(maxlen=200)  # rolling KPIs

    def record_metric(self, metric):  # called by telemetry pipeline
        self.metrics_window.append(metric)

    def estimate_gain(self, candidate):  # shadow-run based estimate
        # run candidate on recent buffered observations (mocked)
        # returns (E_delta_U, sigma_delta_U)
        # here we use stored metric differences as proxy
        diffs = [m['candidate'] - m['active'] for m in self.metrics_window if 'candidate' in m]
        if not diffs:
            return 0.0, 1.0
        import statistics
        return statistics.mean(diffs), statistics.pstdev(diffs)

    def safety_checks(self, candidate):  # assurance hooks
        # check compute budget and safety envelope (mocked)
        return candidate.compute_cost <= self.active.compute_budget and candidate.certified

    def hot_swap(self, candidate):
        now = time.time()
        if now - self.last_switch < self.dwell_sec:
            return False  # enforce dwell
        if not self.safety_checks(candidate):
            return False
        E_delta, sigma = self.estimate_gain(candidate)
        C_switch = candidate.switch_cost
        if E_delta > C_switch + self.beta * sigma:
            # canary: run in shadow for one interval then promote
            candidate.run_shadow_once()
            # compare live metrics (mock quick check)
            E_post, _ = self.estimate_gain(candidate)
            if E_post > 0:
                self.active = candidate  # atomic promote
                self.last_switch = now
                self.log_swap(candidate, E_post)
                return True
        return False

    def log_swap(self, candidate, benefit):
        print(f"Swapped to {candidate.name}; estimated benefit {benefit:.3f}")
# End of controller