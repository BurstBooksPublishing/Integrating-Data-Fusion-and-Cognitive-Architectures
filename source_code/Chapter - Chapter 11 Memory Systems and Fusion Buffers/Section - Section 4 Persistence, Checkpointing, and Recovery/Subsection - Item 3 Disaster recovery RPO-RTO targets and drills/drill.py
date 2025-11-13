#!/usr/bin/env python3
# simple estimator for checkpoint interval and replay time
def checkpoint_interval(rpo_seconds):
    return max(1, int(rpo_seconds))  # seconds

def estimate_replay_time(ingest_bytes_per_s, r_bytes_per_s, rpo_s, overhead=1.5):
    S = ingest_bytes_per_s * rpo_s
    T_replay = overhead * (S / r_bytes_per_s)
    return S, T_replay

if __name__ == "__main__":
    ingest = 200 * 1024**2  # 200 MB/s
    replay_throughput = 100 * 1024**2  # 100 MB/s
    rpo = 60  # 60 s target
    interval = checkpoint_interval(rpo)
    size, t_replay = estimate_replay_time(ingest, replay_throughput, rpo)
    print(f"checkpoint interval: {interval}s")
    print(f"event volume since snapshot: {size/1024**3:.2f} GiB")
    print(f"estimated replay time: {t_replay:.0f}s")
    # CI can fail if t_replay + restore_time > RTO