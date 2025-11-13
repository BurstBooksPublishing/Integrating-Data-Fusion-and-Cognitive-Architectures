import time, heapq, random
# Simple association manager for streaming frames.
class AssociationManager:
    def __init__(self, latency_budget=0.2):
        self.latency_budget = latency_budget  # seconds
        self.deferred = []  # heap of (expiry_time, hypothesis)
        self.tracks = {}    # track_id -> state
    def process_frame(self, detections):
        tnow = time.time()
        # compute naive greedy associations (score = higher is better)
        for d in detections:
            # simulate two candidate scores
            sA, sB = d['scoreA'], d['scoreB']
            margin = abs(sA - sB)
            if margin < 0.15:  # ambiguous -> defer
                expiry = tnow + self.latency_budget
                hyp = {'det': d, 'scores':(sA,sB), 'created':tnow}
                heapq.heappush(self.deferred, (expiry, hyp))
            else:
                # immediate commit to best score
                winner = 'A' if sA> sB else 'B'
                self.commit(d, winner)
    def tick(self):
        # call periodically to flush expired deferred hypotheses
        tnow = time.time()
        while self.deferred and self.deferred[0][0] <= tnow:
            _, hyp = heapq.heappop(self.deferred)
            d = hyp['det']; sA,sB = hyp['scores']
            # simple rule: choose by max score, tie-break toward stability
            choice = 'A' if sA >= sB else 'B'
            self.commit(d, choice)
    def commit(self, detection, track_label):
        # update track and emit association result (placeholder)
        print(f"Committed det {detection['id']} -> track {track_label}")
# demo
am = AssociationManager(latency_budget=0.1)
for fid in range(5):
    # create random ambiguous and clear detections
    dets = []
    for i in range(3):
        sA = random.uniform(0,1)
        sB = sA + random.uniform(-0.2,0.2)
        dets.append({'id':f'{fid}-{i}','scoreA':sA,'scoreB':sB})
    am.process_frame(dets)
    time.sleep(0.03)
    am.tick()
# final flush
time.sleep(0.2); am.tick()