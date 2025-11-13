import asyncio, time, json, math
FRESHNESS_SEC = 0.25  # freshness budget
CONF_THRESH = 0.6     # minimum confidence to accept

async def producer(q):
    # simulate fused messages with fields: id,tstamp,confidence,payload
    for i in range(10):
        msg = {
            "id": f"trk{i}",
            "capture_ts": time.time() - (0.05 * i),  # simulated latency
            "ingest_ts": time.time(),
            "confidence": max(0.1, 1 - 0.08 * i),
            "payload": {"pos": [i, i*0.5]}
        }
        await q.put(json.dumps(msg))
        await asyncio.sleep(0.05)

async def validator(q):
    while True:
        raw = await q.get()
        msg = json.loads(raw)
        latency = time.time() - msg["capture_ts"]
        stale = latency > FRESHNESS_SEC
        low_conf = msg["confidence"] < CONF_THRESH
        # decision policy: veto if stale or low confidence
        if stale or low_conf:
            print(f"VETO {msg['id']} stale={stale} low_conf={low_conf} lat={latency:.3f}")
            # emit veto control to actuator/control surface (placeholder)
        else:
            print(f"ACCEPT {msg['id']} conf={msg['confidence']:.2f} lat={latency:.3f}")
        q.task_done()

async def main():
    q = asyncio.Queue()
    await asyncio.gather(producer(q), validator(q))

if __name__ == "__main__":
    asyncio.run(main())