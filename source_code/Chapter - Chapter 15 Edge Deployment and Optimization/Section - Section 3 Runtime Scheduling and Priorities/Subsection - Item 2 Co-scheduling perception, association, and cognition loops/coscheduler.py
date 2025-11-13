import time, heapq, threading
# Task descriptor (period, budget, name, simulated work seconds)
tasks = [
    {"period": 0.05, "budget": 0.02, "name": "perception", "work": 0.015},
    {"period": 0.1,  "budget": 0.03, "name": "association", "work": 0.02},
    {"period": 0.3,  "budget": 0.05, "name": "cognition", "work": 0.04},
]
stop_at = time.time() + 5.0
# Priority: lower period -> higher priority
tasks.sort(key=lambda t: t["period"])
# Scheduler queue entries: (next_release, task_index, seq)
pq = []
for i,t in enumerate(tasks):
    heapq.heappush(pq, (time.time(), i, 0))
misses = {t["name"]: 0 for t in tasks}
last_output = {t["name"]: None for t in tasks}
s_max = 0.25  # staleness contract for cognition inputs
while time.time() < stop_at:
    next_release, idx, seq = heapq.heappop(pq)
    now = time.time()
    if next_release > now:
        time.sleep(next_release - now)  # idle until release
    t = tasks[idx]
    start = time.time()
    # Simulate work but enforce budget (cooperative preemption)
    work = t["work"]
    budget = t["budget"]
    run = min(work, budget)
    time.sleep(run)  # simulated compute
    elapsed = time.time() - start
    # Deadline check: missed if elapsed > period
    if elapsed > t["period"]:
        misses[t["name"]] += 1
    last_output[t["name"]] = time.time()
    # Cognition staleness check using perception and association outputs
    if t["name"] == "cognition":
        p_age = (time.time() - last_output["perception"]) if last_output["perception"] else float("inf")
        a_age = (time.time() - last_output["association"]) if last_output["association"] else float("inf")
        if max(p_age, a_age) > s_max:
            # degrade: skip heavy reasoning or use conservative fallback
            # here we mark a miss for visibility
            misses["cognition"] += 1
    # schedule next release
    heapq.heappush(pq, (next_release + t["period"], idx, seq + 1))
# Print summary
for n,c in misses.items():
    print(f"{n} misses: {c}")