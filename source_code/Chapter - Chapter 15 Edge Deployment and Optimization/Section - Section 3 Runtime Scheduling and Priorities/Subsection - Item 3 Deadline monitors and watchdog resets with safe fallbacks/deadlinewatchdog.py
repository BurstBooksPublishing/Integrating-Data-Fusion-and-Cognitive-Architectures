import asyncio, time, os, sys
# Simple watchdog: register tasks with expected_period (s), fallback_cb, and restart_cb.
class Watchdog:
    def __init__(self, window=100):
        self.tasks = {}  # name -> metadata
        self.window = window

    def register(self, name, expected_period, fallback_cb, restart_cb):
        self.tasks[name] = {
            "T": expected_period, "last": time.monotonic(),
            "misses": 0, "window_cnt": 0,
            "fallback": fallback_cb, "restart": restart_cb
        }

    async def poke(self, name):  # call from monitored task when it finishes work
        meta = self.tasks[name]
        now = time.monotonic()
        dt = now - meta["last"]
        meta["last"] = now
        meta["window_cnt"] += 1
        if dt > 1.5 * meta["T"]:  # simple deadline check
            meta["misses"] += 1
        # evaluate window
        if meta["window_cnt"] >= self.window:
            miss_rate = meta["misses"] / meta["window_cnt"]
            if miss_rate > 0.01:
                await meta["fallback"]()        # degrade gracefully
            if miss_rate > 0.05:
                await meta["restart"]()         # restart operator
            meta["misses"] = 0; meta["window_cnt"] = 0

    async def monitor_loop(self):
        while True:
            await asyncio.sleep(1.0)  # background tasks like telemetry flushing

# Example fallback and restart handlers
async def degrade_perception():
    print("Fallback: switching to low-rate model")  # replace with real reconfigure

async def restart_process_hard():
    print("Hard restart: exec to respawn")  # preserve logs, telemetry
    await asyncio.sleep(0.1)
    os.execv(sys.executable, [sys.executable] + sys.argv)  # restart current process

# Example monitored periodic task
async def perception_task(wd: Watchdog, name: str):
    while True:
        # simulate variable processing time
        await asyncio.sleep(0.08 + 0.06 * (time.time() % 2))  # sometimes slow
        # publish perception outputs...
        await wd.poke(name)

async def main():
    wd = Watchdog(window=50)
    wd.register("perception", expected_period=0.1,
                fallback_cb=degrade_perception, restart_cb=restart_process_hard)
    asyncio.create_task(wd.monitor_loop())
    asyncio.create_task(perception_task(wd, "perception"))
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())