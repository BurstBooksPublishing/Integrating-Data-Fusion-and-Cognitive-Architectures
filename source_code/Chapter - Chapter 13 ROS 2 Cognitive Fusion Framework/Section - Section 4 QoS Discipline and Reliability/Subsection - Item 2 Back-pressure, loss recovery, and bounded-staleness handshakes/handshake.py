# Minimal runnable example. Run two tasks: producer and consumer.
import asyncio, time, collections, random

S_MAX = 0.5          # seconds: bounded staleness
RATE = 50            # msgs/sec
B = 25               # buffer capacity (Eq.1)
REXMIT_WINDOW = 0.4  # max retransmit latency

# shared channel (in-memory for demo)
channel = asyncio.Queue()

async def producer():
    seq = 0
    sent_log = {}  # seq -> (t, payload)
    credit = B
    while True:
        # wait for credit (back-pressure)
        while credit <= 0:
            # simulate receiving a credit heartbeat
            await asyncio.sleep(0.01)
            credit = min(B, credit + 1)
        # produce
        payload = {"seq": seq, "t": time.time(), "data": f"track{seq}"}
        sent_log[seq] = payload
        await channel.put(("DATA", payload))
        seq += 1
        credit -= 1
        await asyncio.sleep(1.0 / RATE)
        # handle incoming control messages (ACK/NACK/credit)
        try:
            while True:
                typ, msg = channel.get_nowait()
                if typ == "ACK":
                    # drop log up to acked seq
                    ack = msg["ack"]
                    for s in list(sent_log):
                        if s <= ack:
                            sent_log.pop(s, None)
                elif typ == "NACK":
                    for r in msg["ranges"]:
                        for s in range(r[0], r[1]+1):
                            # retransmit only if within staleness
                            if s in sent_log and time.time() - sent_log[s]["t"] <= S_MAX:
                                await channel.put(("DATA", sent_log[s]))
                elif typ == "CREDIT":
                    credit = min(B, credit + msg["add"])
        except asyncio.QueueEmpty:
            pass

async def consumer():
    expected = 0
    recv_buf = {}
    while True:
        typ, msg = await channel.get()
        if typ == "DATA":
            s = msg["seq"]
            recv_buf[s] = msg
            # detect gap
            if s > expected:
                # NACK missing range
                await channel.put(("NACK", {"ranges": [(expected, s-1)]}))
            # deliver contiguous
            while expected in recv_buf:
                # check staleness
                if time.time() - recv_buf[expected]["t"] > S_MAX:
                    # drop and advance; signal possible degrade
                    recv_buf.pop(expected)
                    expected += 1
                    continue
                # process (placeholder)
                # print(f"process seq {expected}")
                recv_buf.pop(expected)
                expected += 1
                # ack in windowed fashion
                if expected % 10 == 0:
                    await channel.put(("ACK", {"ack": expected-1}))
            # send credits if buffer low
            free = B - len(recv_buf)
            if free > 0:
                await channel.put(("CREDIT", {"add": min(free, 5)}))

async def main():
    await asyncio.gather(producer(), consumer())

if __name__ == "__main__":
    asyncio.run(main())