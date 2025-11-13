import heapq, time, numpy as np

class SkewTolerantBuffer:
    def __init__(self, skew_window_sec=0.1):
        self.buffers = {}           # per-sensor min-heaps
        self.skew_window = skew_window_sec
        self.watermark = 0.0

    def push(self, sensor_id, timestamp, mean, cov):
        # store tuples; allow OOSM insertion
        heap = self.buffers.setdefault(sensor_id, [])
        heapq.heappush(heap, (timestamp, mean, cov))

    def _pop_up_to(self, sensor_id, t):
        heap = self.buffers.get(sensor_id, [])
        out = []
        while heap and heap[0][0] <= t:
            out.append(heapq.heappop(heap))
        return out

    def advance_watermark(self, now):
        # conservative watermark: now - skew_window
        self.watermark = now - self.skew_window

    def _propagate(self, mean, cov, dt):
        # simple 2D constant velocity: state [x,y,vx,vy]
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        q = 0.1  # process noise scalar
        G = np.array([[0.5*dt*dt],[0.5*dt*dt],[dt],[dt]])
        Q = q * (G @ G.T)
        return F @ mean, F @ cov @ F.T + Q

    def collect_for_epoch(self, epoch_time):
        # gather propagated observations ≤ watermark, align to epoch_time
        fused = []
        for sid in list(self.buffers.keys()):
            obs = self._pop_up_to(sid, self.watermark)
            for ts, mean, cov in obs:
                dt = epoch_time - ts
                pm, pc = self._propagate(mean, cov, dt)
                fused.append((sid, epoch_time, pm, pc))
        return fused

# Usage example (not part of unit test)
if __name__ == "__main__":
    b = SkewTolerantBuffer(skew_window_sec=0.05)
    now = time.time()
    # push two sensors with slight skew
    b.push("cam", now-0.08, np.zeros(4), np.eye(4)*0.01)
    b.push("lidar", now-0.02, np.array([1,0,0,0]), np.eye(4)*0.05)
    b.advance_watermark(now)
    fused = b.collect_for_epoch(now)
    print(fused)  # fused aligned observations