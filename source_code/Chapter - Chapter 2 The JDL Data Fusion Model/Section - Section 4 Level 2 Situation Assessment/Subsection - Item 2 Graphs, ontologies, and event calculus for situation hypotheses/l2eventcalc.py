# Simple L2 reasoning pipeline (pure Python)
from math import hypot, log
from collections import defaultdict, namedtuple

Track = namedtuple('Track',['id','times','poses'])  # poses: list of (x,y)
Event = namedtuple('Event',['name','t','params'])
Fluent = namedtuple('Fluent',['name','args'])

# Example tracks (two vehicles converging)
tracks = [
  Track('A',[0,1,2],[ (0,0),(1,0),(2,0) ]),
  Track('B',[0,1,2],[ (5,0),(3,0),(2.2,0) ]),
]

# Ontology: allowed relations (simple)
ontology = {'Vehicle': ['approaching','rendezvous']}

# Numeric->graph: proximity edges when distance < d_thresh at same time
d_thresh = 1.0
events = []
for ta in tracks:
  for tb in tracks:
    if ta.id>=tb.id: continue
    for t,pa,pb in zip(ta.times,ta.poses,tb.poses):
      d = hypot(pa[0]-pb[0], pa[1]-pb[1])
      if d < d_thresh:
        events.append(Event('close_approach', t, {'a':ta.id,'b':tb.id,'dist':d}))

# Event-calculus rules: initiates Rendezvous when sustained close_approach for k frames
required_frames = 2
close_counts = defaultdict(int)
for e in sorted(events, key=lambda x:(x.params['a'],x.params['b'],x.t)):
  key = (e.params['a'], e.params['b'])
  close_counts[key] += 1
  if close_counts[key] >= required_frames:
    # spawn fluent and hypothesis
    fluent = Fluent('Rendezvous',key)
    # simple scoring: prior + evidence (each close_approach multiplies likelihood)
    prior = -2.0  # log prior
    score = prior + sum(log(0.7) for _ in range(close_counts[key]))  # evidence model
    print(f"Hypothesis: {fluent} at t={e.t}, score={score:.2f}, provenance=close_count={close_counts[key]}")