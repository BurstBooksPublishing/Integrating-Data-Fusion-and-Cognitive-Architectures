import collections, time, math

# sliding-window params
WINDOW = 60        # seconds
BUCKET = 1         # second

# simple in-memory buckets (use TSDB in production)
buckets = collections.deque()  # (bucket_time, events_list)

def now_sec():
    return int(time.time())

def push_event(evt):
    # evt: dict with keys 't','policy','reward','oracle_reward','action', 'slo_ok'
    t = int(evt['t'])
    if not buckets or buckets[-1][0] != t:
        buckets.append((t, []))
    buckets[-1][1].append(evt)
    # evict old
    cutoff = now_sec() - WINDOW
    while buckets and buckets[0][0] < cutoff:
        buckets.popleft()

def aggregate():
    total_reward = 0.0
    total_oracle = 0.0
    action_sum = 0
    slo_ok = 0
    count = 0
    for _, evs in buckets:
        for e in evs:
            total_reward += e['reward']
            total_oracle += e.get('oracle_reward', e['reward'])  # fallback
            action_sum += 1 if e['action'] else 0
            slo_ok += 1 if e['slo_ok'] else 0
            count += 1
    if count == 0:
        return None
    cum_regret = total_oracle - total_reward
    duty = action_sum / count
    sla_adherence = slo_ok / count
    avg_reward = total_reward / count
    return {'avg_reward': avg_reward, 'cum_regret': cum_regret,
            'duty': duty, 'sla': sla_adherence, 'n': count}

# example event stream ingestion (replace with real telemetry hook)
if __name__ == "__main__":
    # simulate events; in real system ingest from message bus
    for i in range(120):
        t = now_sec()
        evt = {'t': t, 'policy':'p1',
               'reward': max(0, min(1, 0.8 - 0.01*math.sin(i))),
               'oracle_reward': 0.85,      # from simulator/baseline
               'action': (i % 5 != 0),     # action on most ticks
               'slo_ok': (i % 20 != 0)}    # occasional SLA misses
        push_event(evt)
        if i % 10 == 0:
            print(aggregate())
        time.sleep(0.01)               # throttle for demo