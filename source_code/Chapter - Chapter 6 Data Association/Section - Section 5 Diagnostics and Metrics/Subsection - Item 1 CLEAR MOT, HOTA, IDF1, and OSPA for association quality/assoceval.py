import numpy as np
from scipy.optimize import linear_sum_assignment
# toy distance (centroid euclidean)
def dist(a,b): return np.linalg.norm(a-b, axis=1)
def iou_cost(boxesA, boxesB): # placeholder: use centroid distance
    A = np.array([[(x[0]+x[2])/2,(x[1]+x[3])/2] for x in boxesA])
    B = np.array([[(x[0]+x[2])/2,(x[1]+x[3])/2] for x in boxesB])
    C = np.linalg.norm(A[:,None,:]-B[None,:,:],axis=2)
    return C
def ospa_set(X,Y,p=1,c=100.0):
    m,n = len(X), len(Y); if_m = max(m,n)
    if if_m==0: return 0.0
    Xp = np.array(X); Yp = np.array(Y)
    # cost matrix padded
    C = np.zeros((if_m,if_m))
    if m>0 and n>0:
        D = np.linalg.norm(Xp[:,None,:]-Yp[None,:,:],axis=2)
        D = np.minimum(D,c)
        C[:m,:n] = D
    C[m:, :] = c; C[:, n:] = c
    row,col = linear_sum_assignment(C)
    return (np.sum(C[row,col]**p)/if_m)**(1.0/p)
# evaluator for sequence: frames is list of (gt_boxes, gt_ids, pred_boxes, pred_ids)
def evaluate_sequence(frames, ospa_p=1, ospa_c=100.0):
    TP=FP=FN=IDSW=0; motp_sum=0.0; gt_total=0
    prev_map = {} # pred_id -> gt_id last seen
    id_tp = 0; id_fp = 0; id_fn = 0
    ospa_vals=[]
    for gt_boxes,gt_ids,pred_boxes,pred_ids in frames:
        gt_total += len(gt_ids)
        if len(gt_boxes)==0 and len(pred_boxes)==0:
            ospa_vals.append(0.0); continue
        C = iou_cost(gt_boxes,pred_boxes)
        r,c = linear_sum_assignment(C)
        matched = []
        for i,j in zip(r,c):
            if C[i,j] < 50.0: # threshold for match (centroid dist)
                TP += 1
                motp_sum += C[i,j]
                matched.append((i,j))
                # ID bookkeeping
                if prev_map.get(pred_ids[j], None) not in (None, gt_ids[i]):
                    IDSW += 1
                if prev_map.get(pred_ids[j], None) != gt_ids[i]:
                    prev_map[pred_ids[j]] = gt_ids[i]
                if prev_map[pred_ids[j]] == gt_ids[i]:
                    id_tp += 1
            else:
                # treat as unmatched
                pass
        FP += len(pred_boxes) - len(matched)
        FN += len(gt_boxes) - len(matched)
        # simple id_fp/id_fn counts (approx)
        id_fp += max(0, len(pred_boxes) - id_tp)
        id_fn += max(0, len(gt_boxes) - id_tp)
        ospa_vals.append(ospa_set([b[:2] for b in gt_boxes],[b[:2] for b in pred_boxes],p=ospa_p,c=ospa_c))
    mota = 1.0 - (FN + FP + IDSW) / max(1, gt_total)
    motp = motp_sum / max(1, TP)
    idf1 = 2*id_tp / max(1, 2*id_tp + id_fp + id_fn)
    return {'MOTA':mota,'MOTP':motp,'IDF1':idf1,'IDSW':IDSW,'OSPA_mean':np.mean(ospa_vals)}