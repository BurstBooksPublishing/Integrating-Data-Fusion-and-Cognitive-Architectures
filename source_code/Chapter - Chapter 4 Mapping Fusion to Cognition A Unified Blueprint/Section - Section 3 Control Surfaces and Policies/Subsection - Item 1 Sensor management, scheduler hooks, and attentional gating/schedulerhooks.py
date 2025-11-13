import asyncio, time, math, logging
logging.basicConfig(level=logging.INFO)

class SensorAction:
    def __init__(self, name, cost, latency, base_salience):
        self.name, self.cost, self.latency, self.base_salience = name, cost, latency, base_salience
        self.last_executed = 0.0  # for dwell/hysteresis

# simple expected info estimator (placeholder for model-based EIG)
def expected_info(action, uncertainty):
    # higher uncertainty raises info; salience weights it
    return action.base_salience * uncertainty

class AttentionalScheduler:
    def __init__(self, budget_per_tick, dwell=2.0, lambda_cost=0.5):
        self.budget_per_tick = budget_per_tick
        self.dwell = dwell  # seconds to wait before re-serving same action
        self.lambda_cost = lambda_cost
        self.pre_execute_hooks = []  # e.g., safety checks
        self.post_execute_hooks = []  # telemetry, provenance

    async def schedule(self, actions, uncertainty_map):
        budget = self.budget_per_tick
        selected = []
        # score actions
        scored = []
        now = time.time()
        for a in actions:
            if now - a.last_executed < self.dwell:
                continue  # attentional gating via dwell
            u = expected_info(a, uncertainty_map.get(a.name, 1.0))
            score = u - self.lambda_cost * a.cost
            scored.append((score, a))
        scored.sort(reverse=True, key=lambda x: x[0])
        # admit while budget allows
        for score, a in scored:
            if score <= 0 or a.cost > budget:
                continue
            # run pre-exec hooks (can veto)
            for hook in self.pre_execute_hooks:
                if not hook(a):
                    logging.info("Pre-hook vetoed %s", a.name); break
            else:
                budget -= a.cost
                selected.append(a)
                a.last_executed = now
        # execute selected actions concurrently
        tasks = [self._execute(a) for a in selected]
        await asyncio.gather(*tasks)
        return selected

    async def _execute(self, action):
        logging.info("Executing %s", action.name)
        await asyncio.sleep(action.latency)  # simulate sensor/comm latency
        for hook in self.post_execute_hooks:
            hook(action)  # record telemetry

# example hooks
def safety_check(action):
    # placeholder: reject high-cost sensors at night, etc.
    return True

def telemetry_hook(action):
    logging.info("Telemetry: %s executed cost=%.2f", action.name, action.cost)

# usage
async def main():
    actions = [SensorAction("SAR_revisit", 4.0, 0.6, 1.5),
               SensorAction("EO_highres", 2.0, 0.2, 1.1),
               SensorAction("Passive_listen", 0.5, 0.05, 0.4)]
    sched = AttentionalScheduler(budget_per_tick=5.0, dwell=1.5)
    sched.pre_execute_hooks.append(safety_check)
    sched.post_execute_hooks.append(telemetry_hook)
    uncertainty = {"SAR_revisit": 2.0, "EO_highres": 0.8}
    await sched.schedule(actions, uncertainty)

if __name__ == "__main__":
    asyncio.run(main())