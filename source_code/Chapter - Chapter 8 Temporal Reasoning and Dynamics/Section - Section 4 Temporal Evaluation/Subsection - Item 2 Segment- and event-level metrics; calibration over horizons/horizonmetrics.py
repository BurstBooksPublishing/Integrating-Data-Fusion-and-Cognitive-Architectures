import numpy as np
import pandas as pd

# preds/gt: DataFrame with columns ['t_start','t_end','label','conf'] for segments/events.
# horizons: list of (name, min_s, max_s)
def iou(a_start,a_end,b_start,b_end):
    inter = max(0, min(a_end,b_end)-max(a_start,b_start))
    union = max(a_end,b_end)-min(a_start,b_start)
    return inter/union if union>0 else 0.0

def match_events(preds, gt, delta=1.0):
    preds = preds.sort_values('t_start').copy()
    gt = gt.sort_values('t_start').copy()
    matched,p_idx,g_idx = [],set(),set()
    for i,p in preds.iterrows():
        # find closest unmatched gt within delta and same label
        candidates = gt[(~gt.index.isin(g_idx)) & (np.abs(gt['t_start']-p['t_start'])<=delta) & (gt['label']==p['label'])]
        if not candidates.empty:
            j = candidates.iloc[(np.abs(candidates['t_start']-p['t_start'])).argmin()].name
            matched.append((i,j))
            p_idx.add(i); g_idx.add(j)
    return matched, p_idx, g_idx

def horizon_metrics(preds, gt, horizons, delta=1.0, bins=10):
    results={}
    N = len(preds)
    for name, mn, mx in horizons:
        p_slice = preds[(preds['t_start']>=mn)&(preds['t_start']=mn)&(gt['t_start']0 else 0.0
        rec  = tp/(tp+fn) if tp+fn>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
        # ECE
        if len(p_slice)>0:
            bins_edges = np.linspace(0,1,bins+1)
            inds = np.digitize(p_slice['conf'], bins_edges, right=True)-1
            ece = 0.0
            for b in range(bins):
                sel = p_slice[inds==b]
                if len(sel)==0: continue
                acc = np.mean([1 if i in p_idx else 0 for i in sel.index])
                conf = sel['conf'].mean()
                ece += (len(sel)/len(p_slice))*abs(acc-conf)
        else:
            ece = np.nan
        results[name] = dict(precision=prec,recall=rec,f1=f1,ece=ece,tp=tp,fp=fp,fn=fn)
    return results

# Example usage would load preds/gt and call horizon_metrics; omitted for brevity.