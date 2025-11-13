import time
from collections import deque
import math

# simple state per patient/shift
class AlertController:
    def __init__(self, max_per_shift=5, false_window=3600, lambda_decay=0.12):
        self.max_per_shift = max_per_shift
        self.false_history = deque()  # timestamps of recent false alerts
        self.false_window = false_window
        self.lambda_decay = lambda_decay

    def _false_count(self, now):
        # evict old entries
        while self.false_history and now - self.false_history[0] > self.false_window:
            self.false_history.popleft()
        return len(self.false_history)

    def attention(self, now):
        N_false = self._false_count(now)
        return math.exp(-self.lambda_decay * N_false)  # A(N)

    def score(self, model_confidence, corroboration_count):
        # combine confidence and corroboration (simple monotone fusion)
        return 0.6*model_confidence + 0.4*(1 - math.exp(-0.8*corroboration_count))

    def consider_alert(self, model_confidence, corroboration_count, now=None):
        now = now or time.time()
        s = self.score(model_confidence, corroboration_count)
        A = self.attention(now)
        # response probability surrogate; used to gate high-volume alerts
        p_resp = 1.0 / (1.0 + math.exp(-6.0*s - 3.0*A))
        # enforce per-shift cap and require p_resp above threshold
        if corroboration_count >= 2 and p_resp > 0.35 and self.shift_count() < self.max_per_shift:
            self.emit_alert(now, s, corroboration_count)  # escalate
            return True
        return False

    def emit_alert(self, now, s, c):
        # auditing metadata; in real system send to messaging layer
        print(f"ALERT emit time={now:.0f} score={s:.2f} corroboration={c}")
        # placeholder for tracking true/false after outcome known
        # self.log_outcome(is_false)

    def record_false(self, now=None):
        self.false_history.append(now or time.time())

    def shift_count(self):
        # placeholder for per-shift count; integrate with telemetry in prod
        return 0  # stub for demo