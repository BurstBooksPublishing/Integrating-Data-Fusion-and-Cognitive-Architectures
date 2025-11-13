from datetime import datetime
import numpy as np

# Example inputs: lists of dicts with 'id','label','time','escalate' (bool)
# times as ISO strings for readability.
def parse_time(t): return datetime.fromisoformat(t)

def match_scenarios(gt, sys, time_tol_s=30):
    matched_sys_ids = set()
    TP,FP,FN=0,0,0
    latencies=[]
    for g in gt:
        gt_time=parse_time(g['time'])
        found=False
        for s in sys:
            if s['id'] in matched_sys_ids: continue
            if s['label']!=g['label']: continue
            if abs((parse_time(s['time'])-gt_time).total_seconds())<=time_tol_s:
                TP+=1
                matched_sys_ids.add(s['id'])
                # latency measured from evidence time (gt_time) to system decision
                latencies.append((parse_time(s['time'])-gt_time).total_seconds())
                found=True
                break
        if not found: FN+=1
    FP = len(sys)-len(matched_sys_ids)
    return {'TP':TP,'FP':FP,'FN':FN,'latencies':latencies}

def escalation_metrics(gt, sys):
    # Align by matched label and time tolerance as above for escalation table
    # For brevity, assume one-to-one matching by label and nearest time
    E_TP=E_FP=E_FN=E_TN=0
    for g in gt:
        # find nearest system entry with same label
        candidates=[s for s in sys if s['label']==g['label']]
        if not candidates:
            if g['escalate']: E_FN+=1
            else: E_TN+=1
            continue
        s=min(candidates, key=lambda x: abs((parse_time(x['time'])-parse_time(g['time'])).total_seconds()))
        if s['escalate'] and g['escalate']: E_TP+=1
        elif s['escalate'] and not g['escalate']: E_FP+=1
        elif not s['escalate'] and g['escalate']: E_FN+=1
        else: E_TN+=1
    return {'E_TP':E_TP,'E_FP':E_FP,'E_FN':E_FN,'E_TN':E_TN}

# Example usage is omitted; plug real traces into these functions.