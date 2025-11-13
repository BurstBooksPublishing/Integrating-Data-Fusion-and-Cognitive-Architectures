import time, math, sqlite3, random
# Perception: simulated fused detection with covariance and score
def perceive():
    return {'id': random.randint(0,1000), 'pos': (10.0,5.0),
            'cov': [[0.5,0],[0,0.5]], 'score': random.random(),
            'ts': time.time()}
# Working memory: fixed-lag store (seconds)
class WorkingMemory:
    def __init__(self, lag=2.0): self.lag=lag; self.buf=[]
    def write(self, item): self.buf.append(item); self._evict()
    def _evict(self): cutoff=time.time()-self.lag; self.buf=[b for b in self.buf if b['ts']>=cutoff]
    def query(self): return list(self.buf)
# Controller: select action by simple utility (score minus uncertainty penalty)
def select_action(hypotheses, tau=0.5):
    U=[h['score']-0.1*(h['cov'][0][0]+h['cov'][1][1]) for h in hypotheses]
    # softmax selection (Eq. 1)
    ex=[math.exp(u/tau) for u in U]; s=sum(ex)
    probs=[e/s for e in ex]
    return random.choices(hypotheses, weights=probs, k=1)[0]
# Run loop
wm=WorkingMemory(lag=1.5)
for _ in range(10):
    d=perceive(); wm.write(d)
    chosen=select_action(wm.query())
    print('Chosen id', chosen['id'])
    time.sleep(0.2)