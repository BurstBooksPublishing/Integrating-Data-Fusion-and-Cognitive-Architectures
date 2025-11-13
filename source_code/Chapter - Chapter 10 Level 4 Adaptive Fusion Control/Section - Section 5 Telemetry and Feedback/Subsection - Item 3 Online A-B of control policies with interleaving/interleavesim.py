import random, time, statistics
# Two simple policies (could be model inference calls)
def policy_a(state): return 0 if state < 0.5 else 1  # binary action
def policy_b(state): return 1 if random.random() < 0.6 else 0

# Safety check (veto unsafe actions)
def safe_check(action, state):
    # example: forbid action 1 when state below threshold
    return not (action == 1 and state < 0.1)

# Telemetry stores
telemetry = {"assign_A":0, "assign_B":0, "rewards_A":[], "rewards_B":[], "sla_violations":0}

def reward_fn(action, state):
    # synthetic reward with noise
    base = 1.0 if action == (1 if state>0.3 else 0) else 0.0
    return base + random.gauss(0, 0.1)

def run_episode(steps=100, p=0.5):
    cum_regret = 0.0
    for t in range(steps):
        state = random.random()  # contextual feature
        assign_to_A = random.random() < p
        action = policy_a(state) if assign_to_A else policy_b(state)
        if not safe_check(action, state):
            action = 0  # fallback safe action; count as SLA check
            telemetry["sla_violations"] += 1
        r = reward_fn(action, state)
        # log telemetry
        if assign_to_A:
            telemetry["assign_A"] += 1; telemetry["rewards_A"].append(r)
        else:
            telemetry["assign_B"] += 1; telemetry["rewards_B"].append(r)
        # counterfactual best among observed policies (simple oracle)
        r_star = max(
            reward_fn(policy_a(state), state),
            reward_fn(policy_b(state), state)
        )
        cum_regret += (r_star - r)
    return cum_regret

# Run many episodes and report metrics
regrets = [run_episode() for _ in range(50)]
print("avg_regret", statistics.mean(regrets))
print("assignments", telemetry["assign_A"], telemetry["assign_B"])
print("mean_reward_A", statistics.mean(telemetry["rewards_A"]))
print("mean_reward_B", statistics.mean(telemetry["rewards_B"]))
print("sla_violations", telemetry["sla_violations"])