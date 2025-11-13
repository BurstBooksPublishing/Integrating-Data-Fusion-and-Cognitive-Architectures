import numpy as np
from scipy.optimize import linear_sum_assignment

def cost_matrix(tracks, meas, motion_sigma=1.0, appearance_weight=0.5):
    # tracks: list of (x,y,vx,vy,feat) ; meas: list of (x,y,feat)
    n, m = len(tracks), len(meas)
    C = np.full((n, m), 1e6)
    for i,t in enumerate(tracks):
        tx,ty,vx,vy,tf = t
        pred = (tx+vx, ty+vy)
        for j,mo in enumerate(meas):
            mx,my,mf = mo
            d2 = (pred[0]-mx)**2 + (pred[1]-my)**2
            motion_cost = d2/(2*motion_sigma**2)
            app_cost = -np.dot(tf, mf)  # higher dot => lower cost
            C[i,j] = motion_cost + appearance_weight*app_cost
    return C

def expand_beam(hypotheses, tracks, meas, beam_width=10):
    new_hyps = []
    C = cost_matrix(tracks, meas)
    # solve base assignment as proposal (others found by heuristics here)
    row,col = linear_sum_assignment(C)
    base_score = C[row, col].sum()
    new_hyps.append((base_score, col.tolist()))  # simple proposal
    # naive additional proposals: try nearest-measure for each track
    for i in range(min(beam_width-1, len(tracks))):
        alt = []
        for t_idx in range(len(tracks)):
            j = np.argmin(C[t_idx])  # best for this track
            alt.append(j)
        score = sum(C[t, alt[t]] for t in range(len(tracks)))
        new_hyps.append((score, alt))
    # prune by score (lower is better)
    new_hyps.sort(key=lambda x: x[0])
    return new_hyps[:beam_width]

# Usage sketch: maintain window of frames, call expand_beam, commit after N frames.