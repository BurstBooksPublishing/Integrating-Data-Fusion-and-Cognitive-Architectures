import heapq, time
# Task tuple: (deadline, period, exec_ms, name) -- DM priority by deadline
class Task:
    def __init__(self, deadline, period, exec_ms, name):
        self.deadline, self.period, self.exec_ms, self.name = deadline, period, exec_ms, name
        self.next_release = time.monotonic()
# Simple admission: sum utilizations must stay <= 0.75 (safety headroom)
def util(tasks): return sum(t.exec_ms / t.period for t in tasks)
# Dispatcher: multi-queue: high, medium, low
queues = {"high":[], "med":[], "low":[]}
# Insert tasks after admission check
def admit_and_enqueue(qname, t, active_tasks):
    if util(active_tasks + [t]) > 0.75:
        return False  # reject or defer; trigger graceful-degrade
    heapq.heappush(queues[qname], (t.deadline, t))  # DM: smallest deadline highest priority
    return True
# Runner loop (simplified)
def run_once():
    for q in ("high","med","low"):
        if queues[q]:
            _, task = heapq.heappop(queues[q])
            # execute task atomically for exec_ms (placeholder)
            time.sleep(task.exec_ms/1000.0)
            task.next_release += task.period
            heapq.heappush(queues[q], (task.deadline, task))  # requeue for periodicity
            break