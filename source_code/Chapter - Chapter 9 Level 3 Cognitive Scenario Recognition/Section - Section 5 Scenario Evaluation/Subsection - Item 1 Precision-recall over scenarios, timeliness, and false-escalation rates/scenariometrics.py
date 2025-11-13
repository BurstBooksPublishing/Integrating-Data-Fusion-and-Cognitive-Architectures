import numpy as np
import pandas as pd

# Example ground truth: id,start,end,label
gt = pd.DataFrame([
  (1, 0.0, 30.0, "rendezvous"),
  (2, 100.0, 140.0, "loiter"),
], columns=["id","start","end","label"])

# Predictions: id,detect_time,start_est,end_est,label,escalated,score
pred = pd.DataFrame([
  (10, 5.0, 2.0, 28.0, "rendezvous", True, 0.92),
  (11, 110.0, 108.0, 130.0, "loiter", False, 0.75),
  (12, 50.0, 48.0, 55.0, "rendezvous", True, 0.40),  # false alarm
], columns=["pid","detect","pstart","pend","label","escalated","score"])

def overlap(a_start,a_end,b_start,b_end):
    return max(0.0, min(a_end,b_end) - max(a_start,b_start))

alpha = 0.3  # minimum overlap fraction
tp=fp=fn=0
delays=[]
escalations=0
false_escal=0

for _,p in pred.iterrows():
    matched=None
    for _,g in gt.iterrows():
        o = overlap(p.pstart,p.pend,g.start,g.end)
        frac = o / (g.end - g.start)
        if p.label==g.label and frac>=alpha:
            matched=g
            break
    if matched is not None:
        tp+=1
        delays.append(p.detect - matched.start)
    else:
        fp+=1
    if p.escalated:
        escalations+=1
        if matched is None:
            false_escal+=1

fn = len(gt) - tp
precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
mean_delay = np.mean(delays) if delays else np.nan
false_escal_rate = false_escal / escalations if escalations>0 else 0.0

print(dict(precision=precision, recall=recall, f1=f1,
           mean_delay=mean_delay, false_escal_rate=false_escal_rate))