import numpy as np
from collections import deque

# Simulated tracks: (t, x, y, vx, vy)
tracks = [
    (0, 0.0, 0.0, 1.0, 0.0),
    (1, 1.0, 0.1, 1.0, 0.0),
    (2, 2.0, 0.1, 0.5, 0.0),
    (3, 2.5, 0.1, 0.0, 0.0),  # slows near another actor
]

# Fixed-lag buffer for temporal abstraction (lag=2 timesteps)
lag = 2
buffer = deque(maxlen=lag)

# Simple semantic lift: proximity event when speed < threshold and near point
def detect_event(frame):
    t,x,y,vx,vy = frame
    speed = np.hypot(vx,vy)
    near_point = np.hypot(x-2.5,y-0.0) < 0.5
    return {'time':t, 'type':'approach_candidate', 'speed':speed, 'near':near_point}

# Intent hypotheses with priors
hypotheses = ['meet','passby']
priors = {'meet':0.5,'passby':0.5}

# Likelihood models (toy): if near and slow => more likely 'meet'
def likelihood(event, H):
    if H=='meet':
        return 0.9 if event['near'] and event['speed']<0.6 else 0.2
    else:
        return 0.8 if (not event['near'] or event['speed']>=0.6) else 0.1

post = priors.copy()
for frame in tracks:
    ev = detect_event(frame)
    buffer.append(ev)                     # temporal buffer
    # Use buffered evidence for fixed-lag update
    if len(buffer)==lag:
        for H in hypotheses:
            # Multiply likelihoods across buffer, then renormalize
            L = np.prod([likelihood(e,H) for e in buffer])
            post[H] *= L
        # normalize posterior
        s = sum(post.values())
        for H in post: post[H] /= s
        print(f"t={frame[0]} posterior={post}")
# End of pipeline