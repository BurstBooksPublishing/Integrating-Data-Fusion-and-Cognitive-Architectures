import numpy as np
from scipy.stats import chi2

# synthetic example inputs: arrays per frame
# meas_pos: list of (N_t,2) arrays of true measurement positions
# track_pos: list of (M_t,2) arrays of track estimates at same frames
# assoc: list of (assoc_pairs) where assoc_pairs is list of (track_idx, meas_idx, d2)
# true_labels: list per frame of ground-truth target ids for measurements

def gating_hit_rate(meas_pos, assoc, true_matches, k=2, alpha=0.01):
    gamma = chi2.ppf(1-alpha, k)
    hits = 0
    total = 0
    for t, true_idx_map in enumerate(true_matches):  # mapping meas_idx -> true target id
        for m_idx in range(len(meas_pos[t])):
            total += 1
            # find d2 for that measurement if associated to its true track
            pair = next((p for p in assoc[t] if p[1]==m_idx and p[2] <= gamma), None)
            if pair is not None:
                hits += 1
    return hits / total if total else 0.0

def reid_and_switches(assoc_history, gt_labels_history):
    # assoc_history: list per frame of (track_idx, meas_idx)
    id_correct = 0; assoc_count = 0
    switches = {}
    coalesce_count = 0; frames = len(assoc_history)
    prev_assign = {}
    for t in range(frames):
        assigns = {}
        for track_idx, meas_idx in assoc_history[t]:
            gt = gt_labels_history[t][meas_idx]
            assigns[track_idx] = gt
            assoc_count += 1
            if gt == assigns[track_idx]:
                id_correct += 1
        # switches: compare prev_assign track->gt mapping
        for tr, gt in assigns.items():
            if tr in prev_assign and prev_assign[tr] != gt:
                switches[tr] = switches.get(tr,0)+1
        prev_assign = assigns
        # coalescence: any duplicated gt among tracks?
        if len(set(assigns.values())) < len(assigns.values()):
            coalesce_count += 1
    reid_acc = id_correct / assoc_count if assoc_count else 0.0
    avg_switch_freq = sum(switches.values()) / max(1,len(switches))
    coalescence = coalesce_count / frames
    return reid_acc, avg_switch_freq, coalescence

# Example usage would load real logs and call these functions.