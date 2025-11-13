import asyncio, time, uuid

class Broker:
    def __init__(self):
        self.topics = {}                   # topic -> list of queues
        self.services = {}                 # service -> handler coroutine

    async def publish(self, topic, msg):
        for q in self.topics.get(topic, []):
            await q.put(msg)               # back-pressure via queue size

    def subscribe(self, topic, max_queue=10):
        q = asyncio.Queue(max_queue)
        self.topics.setdefault(topic, []).append(q)
        return q

    def register_service(self, name, handler):
        self.services[name] = handler

    async def request(self, service, payload, timeout=1.0):
        handler = self.services.get(service)
        if handler is None:
            raise RuntimeError("no-service")
        return await asyncio.wait_for(handler(payload), timeout)

# example handlers and orchestrator
async def fusion_node(broker, topic):
    q = broker.subscribe(topic)
    while True:
        msg = await q.get()
        # small processing; attach provenance
        msg['processing_time'] = time.time()
        # forward fused track to cognitive topic
        await broker.publish('tracks', msg)

async def cognitive_service(payload):
    await asyncio.sleep(0.05)            # scoring cost
    return {'score': 0.87, 'id': payload['id']}  # deterministic response

async def orchestrate_task(coro, deadline):
    try:
        return await asyncio.wait_for(coro, timeout=deadline)
    except asyncio.TimeoutError:
        return {'status':'timed_out'}

async def main():
    b = Broker()
    b.register_service('score', cognitive_service)
    # start a fusion node
    asyncio.create_task(fusion_node(b, 'sensor'))
    # publish sensor messages
    for i in range(50):
        msg = {'id': str(uuid.uuid4()), 'ts': time.time()}
        await b.publish('sensor', msg)
        # synchronous control query example
        resp = await b.request('score', msg, timeout=0.2)
        # schedule heavy analysis with deadline
        res = await orchestrate_task(asyncio.sleep(0.1, result={'ok':True}), deadline=0.15)
        print(i, resp, res)

asyncio.run(main())