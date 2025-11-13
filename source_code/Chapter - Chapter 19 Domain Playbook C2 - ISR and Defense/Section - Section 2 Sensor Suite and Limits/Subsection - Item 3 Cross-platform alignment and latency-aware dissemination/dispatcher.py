import time
import heapq
from math import exp

# Message descriptor: (id, size_bytes, deadline_s, base_utility, decay_rate)
messages = [
    ("track_alert", 256, 2.0, 1.0, 1.5),
    ("full_image", 5_000_000, 30.0, 2.0, 0.05),
    ("situation_report", 10_000, 10.0, 1.5, 0.5),
]

link_bandwidth = 1_000_000  # bytes/sec available
now = time.time()

def utility_sent_at(base_u, decay, send_delay):
    return base_u * exp(-decay * send_delay)

# Build EDF-like queue weighted by urgency and size-efficiency
pq = []
for mid, size, dl, base_u, decay in messages:
    time_until_deadline = dl
    # heuristic score: urgency per byte
    score = (base_u * exp(-decay * 0.0)) / max(size, 1)
    # negative for min-heap
    heapq.heappush(pq, (-score, mid, size, dl, base_u, decay))

bandwidth_budget = link_bandwidth * 1.0  # one-second budget example
sent = []
while pq and bandwidth_budget > 0:
    _, mid, size, dl, base_u, decay = heapq.heappop(pq)
    # estimate transmission time
    tx_time = size / link_bandwidth
    if tx_time <= bandwidth_budget and tx_time <= dl:
        send_delay = 0.0  # immediate model for simplicity
        u = utility_sent_at(base_u, decay, send_delay)
        sent.append((mid, u, tx_time))
        bandwidth_budget -= tx_time
    else:
        # partial or deferred: send compact notice if available
        compact_size = min(size, 512)  # metadata fallback
        tx_time = compact_size / link_bandwidth
        if tx_time <= bandwidth_budget and tx_time <= dl:
            send_delay = 0.0
            u = utility_sent_at(base_u*0.4, decay, send_delay)  # reduced utility
            sent.append((mid + "_meta", u, tx_time))
            bandwidth_budget -= tx_time
# sent contains chosen transmissions
print("Dispatch plan:", sent)  # in real system, publish and record provenance