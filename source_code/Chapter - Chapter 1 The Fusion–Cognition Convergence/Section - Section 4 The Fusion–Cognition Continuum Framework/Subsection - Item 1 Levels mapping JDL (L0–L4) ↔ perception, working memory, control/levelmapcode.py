import asyncio, random, time, json

# Simple message structure with provenance and uncertainty
def mk_msg(level, payload):
    return {
        "level": level, "ts": time.time(), "payload": payload,
        "uncertainty": random.random(), "provenance": "sim_sensor"
    }

async def perception_l0(out_q):           # L0: feature extraction
    for i in range(5):
        await out_q.put(mk_msg("L0", {"feature": i}))
        await asyncio.sleep(0.02)         # simulate sensor rate

async def perception_l1(in_q, out_q):    # L1: tracking/ID fusion
    tracks = {}
    while True:
        msg = await in_q.get()
        if msg is None: break
        # simple track update/creation
        tid = msg["payload"]["feature"] % 2
        tracks[tid] = {"last": msg, "id": tid}
        await out_q.put(mk_msg("L1", {"track": tracks[tid]}))
        in_q.task_done()

async def working_memory_l2(in_q, out_q): # L2: situation building
    buffer = []
    while True:
        msg = await in_q.get()
        if msg is None: break
        buffer.append(msg)
        if len(buffer) >= 2:
            situation = {"events": [m["payload"] for m in buffer[-2:]]}
            await out_q.put(mk_msg("L2", situation))
        in_q.task_done()

async def evaluator_l3(in_q, out_q):     # L3: impact assessment
    while True:
        msg = await in_q.get()
        if msg is None: break
        score = 1.0 - msg["uncertainty"]    # simple utility
        await out_q.put(mk_msg("L3", {"situation": msg["payload"], "score": score}))
        in_q.task_done()

async def controller_l4(in_q):            # L4: policy selection
    while True:
        msg = await in_q.get()
        if msg is None: break
        policy = "alert" if msg["payload"]["score"]>0.5 else "monitor"
        # action with traceable rationale
        print(json.dumps({"action":policy,"trace":msg}, default=str))
        in_q.task_done()

async def main():
    q0,q1,q2,q3 = asyncio.Queue(), asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
    producers = [
        perception_l0(q0),
        perception_l1(q0,q1),
        working_memory_l2(q1,q2),
        evaluator_l3(q2,q3),
        controller_l4(q3)
    ]
    tasks = [asyncio.create_task(p) for p in producers]
    await q0.join(); await q1.join(); await q2.join(); await q3.join()
    # teardown
    for q in (q0,q1,q2,q3): await q.put(None)
    await asyncio.gather(*tasks)

asyncio.run(main())  # run pipeline