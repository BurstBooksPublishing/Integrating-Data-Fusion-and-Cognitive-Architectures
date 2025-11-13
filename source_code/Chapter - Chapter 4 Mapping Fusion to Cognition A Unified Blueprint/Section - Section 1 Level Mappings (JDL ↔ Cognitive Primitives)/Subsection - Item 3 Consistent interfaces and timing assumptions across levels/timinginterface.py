import asyncio, time, heapq
# Simple message structure: {'src':str,'ts':float,'seq':int,'payload':...,'deadline':float}
class StreamBuffer:
    def __init__(self, name, max_stale=0.2):
        self.name = name
        self.buf = []           # min-heap by timestamp
        self.max_stale = max_stale
    def push(self, msg):
        heapq.heappush(self.buf, (msg['ts'], msg))
    def watermark(self):
        # watermark = smallest ts present minus max_stale
        if not self.buf: return time.time() - self.max_stale
        return self.buf[0][0] - self.max_stale
    def pop_until(self, t):
        out=[]
        while self.buf and self.buf[0][0] <= t:
            out.append(heapq.heappop(self.buf)[1])
        return out

async def fusion_loop(streams, fusion_callback, poll=0.05):
    while True:
        now=time.time()
        # check deadlines and prune
        for s in streams.values():
            # drop any messages older than their declared deadline
            s.buf = [(ts,m) for (ts,m) in s.buf if now - m['ts'] <= m.get('deadline', 1.0)]
            heapq.heapify(s.buf)
        # compute global watermark (min of stream watermarks)
        w = min(s.watermark() for s in streams.values())
        # pop items up to watermark and aggregate
        window=[]
        for s in streams.values():
            window.extend(s.pop_until(w))
        if window:
            # short-circuit: if any payload violates freshness, flag for re-check
            if any(now - m['ts'] > m.get('deadline',1.0) for m in window):
                # send to evaluator with flag (soft real-time path)
                await fusion_callback({'status':'stale','items':window})
            else:
                # normal promotion to L2
                await fusion_callback({'status':'ok','items':window})
        await asyncio.sleep(poll)

# Example fusion consumer
async def consumer(msg):
    print("FUSED:", msg['status'], "count", len(msg['items']))

# Demo
async def demo():
    s1=StreamBuffer('camera', max_stale=0.05)
    s2=StreamBuffer('radar', max_stale=0.02)
    streams={'cam':s1,'radar':s2}
    # producers
    async def produce(s, interval):
        seq=0
        while True:
            seq+=1
            m={'src':s.name,'ts':time.time(),'seq':seq,'payload':None,'deadline':0.2}
            s.push(m)
            await asyncio.sleep(interval)
    await asyncio.gather(produce(s1,0.03), produce(s2,0.04), fusion_loop(streams, consumer))

if __name__=='__main__':
    asyncio.run(demo())