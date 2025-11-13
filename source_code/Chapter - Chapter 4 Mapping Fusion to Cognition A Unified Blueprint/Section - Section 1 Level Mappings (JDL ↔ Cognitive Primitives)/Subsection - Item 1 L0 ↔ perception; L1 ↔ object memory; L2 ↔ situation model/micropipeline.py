import math,random
# L0: synthetic detections with covariance
def generate_detections(n=3):
    for i in range(n):
        x,y = 10*i + random.uniform(-0.5,0.5), 0.5*i + random.uniform(-0.2,0.2)
        yield {'id':None,'z':(x,y),'R':[[0.25,0],[0,0.25]],'t':0.0}

# L1: minimal 2D constant-velocity Kalman track
class KFTrack:
    def __init__(self, z):
        self.x = [z[0], 0.0, z[1], 0.0]           # px,vx,py,vy
        self.P = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    def predict(self, dt=1.0):
        F = [[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]]
        # simple predict (matrix multiply omitted for brevity)
        self.x = [self.x[0]+dt*self.x[1], self.x[1], self.x[2]+dt*self.x[3], self.x[3]]
    def update(self, z):
        # simple correction toward measurement (pseudo-KF for clarity)
        alpha=0.6
        self.x[0] = alpha*z[0] + (1-alpha)*self.x[0]
        self.x[2] = alpha*z[1] + (1-alpha)*self.x[2]

# L2: build situation edges by proximity and velocity alignment
def build_situation(tracks):
    nodes=[{'id':i,'pos':(t.x[0],t.x[2]),'vel':(t.x[1],t.x[3])} for i,t in enumerate(tracks)]
    edges=[]
    for a in nodes:
        for b in nodes:
            if a['id']>=b['id']: continue
            dx=b['pos'][0]-a['pos'][0]; dy=b['pos'][1]-a['pos'][1]
            dist=math.hypot(dx,dy)
            # relation: "close_and_follow" if within 2.0 m and velocity aligned
            vel_dot = a['vel'][0]*b['vel'][0]+a['vel'][1]*b['vel'][1]
            if dist<2.0 and vel_dot>0.0:
                edges.append((a['id'],b['id'],'close_and_follow'))
    return nodes,edges

# run pipeline
dets=list(generate_detections(4))
tracks=[KFTrack(d['z']) for d in dets]   # init L1 from L0
for t in tracks: t.predict(1.0)
# pretend we get new measurements and update
for i,t in enumerate(tracks):
    t.update((t.x[0]+random.uniform(-0.2,0.2), t.x[2]+random.uniform(-0.1,0.1)))
nodes,edges = build_situation(tracks)
print('nodes',nodes)
print('edges',edges)  # L2 situation hypotheses as labeled relations