#!/usr/bin/env python3
# demo of pipeline vs actor topology for small fusion chain.
import sys, time, multiprocessing as mp, asyncio, random

MSG = {"ts": None, "payload": None}

def sensor_proc(out_q):                    # pipeline sensor process
    for i in range(20):
        m = dict(MSG); m["ts"]=time.time(); m["payload"]=i
        out_q.put(m); time.sleep(0.01)     # 100 Hz source
    out_q.put(None)

def filter_proc(in_q, out_q):               # pipeline filter
    while True:
        m = in_q.get()
        if m is None: out_q.put(None); break
        m["payload"] *= 2                   # cheap transform
        out_q.put(m)

def tracker_proc(in_q):                     # pipeline tracker (consumer)
    while True:
        m = in_q.get()
        if m is None: break
        # simple tracking placeholder
        print("TRACKER got", m["payload"], "latency", time.time()-m["ts"])

async def actor_sensor(mbox):               # actor sensor (async)
    for i in range(20):
        await mbox.put(dict(MSG, ts=time.time(), payload=i))
        await asyncio.sleep(0.01)

async def actor_filter(inbox, outbox):
    while True:
        m = await inbox.get()
        if m is None: await outbox.put(None); break
        m["payload"] *= 2
        await outbox.put(m)

async def actor_tracker(inbox):
    while True:
        m = await inbox.get()
        if m is None: break
        print("ACTOR TRACKER", m["payload"], "latency", time.time()-m["ts"])

def run_pipeline():
    q1 = mp.Queue(); q2 = mp.Queue()
    ps = [mp.Process(target=sensor_proc, args=(q1,)),
          mp.Process(target=filter_proc, args=(q1,q2,)),
          mp.Process(target=tracker_proc, args=(q2,))]
    for p in ps: p.start()
    for p in ps: p.join()

async def run_actors():
    q1 = asyncio.Queue(); q2 = asyncio.Queue()
    await asyncio.gather(actor_sensor(q1), actor_filter(q1,q2), actor_tracker(q2))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv)>1 else "pipeline"
    if mode=="pipeline": run_pipeline()
    else: asyncio.run(run_actors())