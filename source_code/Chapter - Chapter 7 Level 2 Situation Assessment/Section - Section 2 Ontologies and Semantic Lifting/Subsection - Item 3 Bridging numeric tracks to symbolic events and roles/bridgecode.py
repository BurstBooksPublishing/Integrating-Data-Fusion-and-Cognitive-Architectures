#!/usr/bin/env python3
import math
from datetime import datetime

# Example tracks (id, x, y, vx, vy, last_seen)
tracks = [
    {"id":"T1","x":0.0,"y":0.0,"vx":1.0,"vy":0.2,"t":0.0},
    {"id":"T2","x":50.0,"y":2.0,"vx":-1.2,"vy":-0.1,"t":0.0},
]

def relative_features(a,b):
    dx = b["x"]-a["x"]; dy = b["y"]-a["y"]
    dvx = b["vx"]-a["vx"]; dvy = b["vy"]-a["vy"]
    d = math.hypot(dx,dy)
    closing_rate = -(dx*dvx + dy*dvy)/d if d>0 else 0.0
    # heading difference proxy via velocity angle difference
    ha = math.atan2(a["vy"],a["vx"]); hb = math.atan2(b["vy"],b["vx"])
    dtheta = abs((ha-hb+math.pi)%(2*math.pi)-math.pi)
    return {"d":d,"closing_rate":closing_rate,"dtheta":dtheta}

# Simple logistic scorer (weights tuned by domain knowledge)
w = {"d": -0.06, "closing_rate": 0.5, "dtheta": -0.8}
b = -0.2
def score_event(feat):
    s = w["d"]*feat["d"] + w["closing_rate"]*feat["closing_rate"] + w["dtheta"]*feat["dtheta"] + b
    p = 1.0/(1.0+math.exp(-s))
    return p

def make_turtle_assertion(a,b,p):
    ts = datetime.utcnow().isoformat() + "Z"
    ev_id = f"ex:evt_{a['id']}_{b['id']}_{int(1000*p)}"
    turtle = []
    turtle.append(f"{ev_id} a ex:Rendezvous ;")
    turtle.append(f'  ex:subject "{a["id"]}" ;')
    turtle.append(f'  ex:object "{b["id"]}" ;')
    turtle.append(f'  ex:confidence "{p:.3f}"^^xsd:double ;')
    turtle.append(f'  prov:generatedAtTime "{ts}"^^xsd:dateTime .')
    return "\n".join(turtle)

# Run pairwise checks and emit assertions above threshold
THRESH = 0.6
for i in range(len(tracks)):
    for j in range(i+1,len(tracks)):
        a,b = tracks[i],tracks[j]
        feat = relative_features(a,b)
        p = score_event(feat)
        if p > THRESH:
            print(make_turtle_assertion(a,b,p))
        else:
            # small diagnostic output for pipeline monitoring
            print(f"# No assertion: pair {a['id']},{b['id']} p={p:.3f}")