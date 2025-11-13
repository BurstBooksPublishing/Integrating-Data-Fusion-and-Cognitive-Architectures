import numpy as np
from collections import defaultdict

# frame_assignments: list of tuples (frame, track_id, gt_id)
# Example synthetic data
frame_assignments = [
    (0, 10, 1),(1,10,1),(2,10,1),(3,11,1),(4,11,1), # GT 1 fragmented (tracks 10->11)
    (0,20,2),(1,20,2),(2,20,2),(3,20,2)             # GT 2 single track 20
]

# build sequences
gt_frames = defaultdict(list)
track_frames = defaultdict(list)
for frame, track, gt in frame_assignments:
    gt_frames[gt].append((frame, track))
    track_frames[track].append((frame, gt))

# purity per track
purity = {}
for track, seq in track_frames.items():
    frames, gts = zip(*sorted(seq))
    purity[track] = np.mean([1 if gt==gts[0] else 0 for gt in gts])  # mode-based reference

# fragmentation per ground-truth
frag = {}
for gt, seq in gt_frames.items():
    seq_sorted = sorted(seq)
    # count contiguous segments by track id changes with frame continuity
    segments = 0
    prev_track, prev_frame = None, None
    for frame, track in seq_sorted:
        if track != prev_track or (prev_frame is not None and frame != prev_frame + 1):
            segments += 1
        prev_track, prev_frame = track, frame
    frag[gt] = max(0, segments - 1)

print("Purity:", purity)  # per-track purity
print("Fragmentation:", frag)  # per-GT fragmentation