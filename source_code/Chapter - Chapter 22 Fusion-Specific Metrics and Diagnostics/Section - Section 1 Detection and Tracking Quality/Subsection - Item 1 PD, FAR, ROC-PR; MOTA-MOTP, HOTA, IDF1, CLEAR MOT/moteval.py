import numpy as np
from scipy.optimize import linear_sum_assignment

def eval_tracks(gt_frames, det_frames, max_dist=2.0):
    # gt_frames, det_frames: dict frame->list of (id, x, y, score)
    TP=FP=FN=IDSW=0
    sum_dist=matches=0
    id_map_prev={}  # frame->gt_id : det_id mapping history
    id_map={}       # current mapping

    frames = sorted(set(gt_frames)|set(det_frames))
    for f in frames:
        gt = gt_frames.get(f, [])
        det = det_frames.get(f, [])
        if not gt and not det:
            continue
        G=np.array([[g[1], g[2]] for g in gt])
        D=np.array([[d[1], d[2]] for d in det])
        if G.size==0:
            FP += len(det); id_map_prev = id_map.copy(); id_map.clear(); continue
        if D.size==0:
            FN += len(gt); id_map_prev = id_map.copy(); id_map.clear(); continue
        cost = np.linalg.norm(G[:,None,:]-D[None,:,:], axis=2)
        row,col = linear_sum_assignment(cost)
        assigned = set()
        id_map = {}
        for r,c in zip(row,col):
            if cost[r,c] <= max_dist:
                TP += 1
                sum_dist += cost[r,c]; matches += 1
                gt_id = gt[r][0]; det_id = det[c][0]
                id_map[gt_id] = det_id
                # ID switch detection
                prev = id_map_prev.get(gt_id)
                if prev is not None and prev != det_id:
                    IDSW += 1
                assigned.add(c)
            else:
                # assignment too far => treat as FN/FP
                pass
        FN += len(gt) - sum([1 for r in row if cost[r, col[list(row).index(r)]]<=max_dist])
        FP += len(det) - len(assigned)
        id_map_prev = id_map.copy()

    PD = TP / (TP + FN) if (TP+FN)>0 else 0.0
    FAR = FP / len(frames) if frames else 0.0
    MOTA = 1.0 - (FN + FP + IDSW) / (sum(len(gt_frames[f]) for f in gt_frames) or 1)
    MOTP = (sum_dist / matches) if matches>0 else np.nan
    # approximate IDF1 using IDTP/IDFP/IDFN ~ TP/(TP+FP+FN) scaled (simple proxy)
    IDF1 = (2*TP) / (2*TP + FP + FN) if (2*TP+FP+FN)>0 else 0.0
    return dict(PD=PD, FAR=FAR, MOTA=MOTA, MOTP=MOTP, IDF1=IDF1, IDSW=IDSW)

# Example usage with synthetic frames
gt = {1: [(1,0,0)], 2: [(1,1,0)]}         # frame-> [(gt_id,x,y)]
det = {1: [(11,0.1,0.2,0.9)], 2: [(11,1.5,0.2,0.6)]}  # det_id, x,y,score
print(eval_tracks(gt, det))