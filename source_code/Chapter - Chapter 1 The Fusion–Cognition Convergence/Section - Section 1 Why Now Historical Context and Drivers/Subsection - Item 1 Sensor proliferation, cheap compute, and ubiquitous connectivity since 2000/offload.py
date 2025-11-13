import time, random
# Parameters (tuned per deployment)
bandwidth_bps = 5e6            # 5 Mbps
rtt = 0.05                     # 50 ms RTT
privacy_penalty = 0.02         # seconds equivalent cost for privacy concerns

def tx_time(size_bytes, bps=bandwidth_bps, rtt=rtt):
    return rtt + size_bytes*8.0/bps

def proc_time_edge(payload_complexity):
    return 0.01 + 0.005*payload_complexity    # seconds

def proc_time_cloud(payload_complexity):
    return 0.005 + 0.002*payload_complexity   # faster cloud op

def should_offload(size_bytes, complexity, privacy_penalty=privacy_penalty):
    c_edge = proc_time_edge(complexity)
    c_cloud = tx_time(size_bytes) + proc_time_cloud(complexity) + privacy_penalty
    return c_cloud < c_edge

# Simulated stream
for t in range(20):
    size = random.randint(5_000, 200_000)   # bytes per frame
    complexity = random.uniform(1.0, 10.0)  # abstract work measure
    if should_offload(size, complexity):
        # mock send to cloud (non-blocking in real system)
        action = "offload"
    else:
        # run local fusion + cognition
        action = "local"
    print(f"t={t:02d}: size={size}B comp={complexity:.2f} -> {action}")
    time.sleep(0.01)  # simulate ingest rate