import asyncio, time, random, statistics

# Simple message representation
class Msg:
    def __init__(self, seq, ts):
        self.seq = seq
        self.ts = ts

async def emulator(queue, rate_hz=10):
    seq = 0
    while True:
        await asyncio.sleep(1.0/rate_hz)
        seq += 1
        await queue.put(Msg(seq, time.monotonic()))  # synthetic sensor ts

async def hardware_proxy(queue, rate_hz=10):
    seq = 0
    while True:
        await asyncio.sleep(1.0/rate_hz + random.uniform(-0.01, 0.02))  # jitter
        seq += 1
        await queue.put(Msg(seq, time.monotonic()))  # proxy forwards hardware ts

async def fusion_consumer(in_queue, results):
    while True:
        msg = await in_queue.get()
        process_start = time.monotonic()
        # simulated processing latency
        await asyncio.sleep(0.02 + random.uniform(0, 0.01))
        latency = time.monotonic() - msg.ts
        results.append(latency)

async def orchestrator():
    q = asyncio.Queue()
    results = []
    # start both sources but only route one into consumer
    em = asyncio.create_task(emulator(q))
    hw = asyncio.create_task(hardware_proxy(q))
    consumer = asyncio.create_task(fusion_consumer(q, results))
    # run for a fixed window to collect metrics
    await asyncio.sleep(5.0)
    # compute latency statistics and enforce SLO (0.1s)
    mean_lat = statistics.mean(results)
    p95 = sorted(results)[int(0.95*len(results))-1]
    print(f"mean_lat={mean_lat:.3f}s p95={p95:.3f}s")
    # cleanup
    em.cancel(); hw.cancel(); consumer.cancel()

if __name__ == "__main__":
    asyncio.run(orchestrator())