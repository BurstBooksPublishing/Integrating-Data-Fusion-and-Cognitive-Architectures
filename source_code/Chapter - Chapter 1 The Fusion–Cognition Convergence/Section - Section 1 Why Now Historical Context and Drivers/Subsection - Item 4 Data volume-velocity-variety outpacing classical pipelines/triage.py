import asyncio, heapq, json, random, time
from collections import deque

# simple salience: rarity*energy*confidence proxy
def salience(msg):
    return msg.get("rarity",1.0)*msg.get("energy",1.0)*msg.get("conf",0.5)

async def ingest(queue):
    # produce synthetic heterogeneous messages
    while True:
        msg = {"ts": time.time(), "type": random.choice(["cam","rad","rf"]),
               "rarity": random.random(), "energy": random.random(),
               "conf": random.random()}
        await queue.put(msg)
        await asyncio.sleep(0.01)  # 100 Hz ingest

async def triage(in_q, forward_q, archive, budget_per_sec=50):
    window = deque()
    count = 0
    t0 = time.time()
    heap = []
    while True:
        msg = await in_q.get()
        s = salience(msg)
        heapq.heappush(heap, (-s, msg))  # max-heap via negation
        count += 1
        # periodic budget enforcement (per-second)
        if time.time() - t0 >= 1.0:
            # forward top-k within budget; compress rest to archive
            k = min(budget_per_sec, len(heap))
            for _ in range(k):
                _, top = heapq.heappop(heap)
                await forward_q.put(top)  # full processing
            # compress remainder
            while heap:
                _, low = heapq.heappop(heap)
                archive.append({"ts": low["ts"], "type": low["type"], "s": salience(low)})
            heap = []
            t0 = time.time()

async def cognitive_consumer(forward_q):
    while True:
        item = await forward_q.get()
        # placeholder: heavy cognition/scene understanding step
        await asyncio.sleep(0.02)  # simulate processing latency
        print("Cognition processed", item["type"], "salience", round(salience(item),2))

async def main():
    in_q = asyncio.Queue()
    forward_q = asyncio.Queue()
    archive = []
    await asyncio.gather(ingest(in_q), triage(in_q, forward_q, archive), cognitive_consumer(forward_q))

if __name__ == "__main__":
    asyncio.run(main())