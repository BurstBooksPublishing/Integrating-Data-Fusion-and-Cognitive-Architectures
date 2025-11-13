import heapq, time
from collections import defaultdict

class Event:
    def __init__(self, eid, event_time, payload):
        self.eid = eid               # entity id
        self.t = event_time          # event-time (seconds)
        self.payload = payload

class WatermarkBuffer:
    def __init__(self, partitions, allowed_lateness):
        self.allowed_lateness = allowed_lateness
        self.last_seen = {p: float('-inf') for p in partitions}   # per-partition progress
        self.buffers = defaultdict(list)  # per-entity min-heaps by event-time
        self.cognitive_callback = None

    def set_cognitive_callback(self, cb):
        self.cognitive_callback = cb

    def ingest(self, partition, event):    # event arrives from a partition
        self.last_seen[partition] = max(self.last_seen[partition], event.t)
        heapq.heappush(self.buffers[event.eid], (event.t, event))  # order by event-time

    def compute_watermark(self):
        # global watermark is min of per-partition maxima minus allowed lateness
        min_max = min(self.last_seen.values())
        return min_max - self.allowed_lateness

    def flush(self):
        W = self.compute_watermark()
        for eid, heap in list(self.buffers.items()):
            updated = False
            while heap and heap[0][0] <= W:
                _, ev = heapq.heappop(heap)
                # process event and produce correction if needed
                corr = self._process_and_maybe_correct(eid, ev)
                if corr and self.cognitive_callback:
                    self.cognitive_callback(corr)  # send correction to working memory
                updated = True
            if not heap:
                del self.buffers[eid]

    def _process_and_maybe_correct(self, eid, event):
        # placeholder: apply smoothing/update to track state
        # return correction record when prior cognitive outputs must be amended
        # For demo, return a dict as correction.
        return {'entity': eid, 'time': event.t, 'payload': event.payload, 'type': 'update'}

# --- Example usage ---
def cognitive_process(corr):              # receive correction in cognitive layer
    print("Cognitive correction:", corr)

buf = WatermarkBuffer(partitions=['cam','radar','ais'], allowed_lateness=0.5)
buf.set_cognitive_callback(cognitive_process)
buf.ingest('cam', Event('A', 10.0, {'pos':(1,2)}))
buf.ingest('radar', Event('A', 9.8, {'vel':0.5}))
buf.flush()  # may or may not emit depending on last_seen values