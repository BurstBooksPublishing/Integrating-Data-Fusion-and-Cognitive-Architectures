#!/usr/bin/env python3
# Simple safety daemon for testbeds. Uses filesystem metrics for portability.
import asyncio, json, shutil, time, os
METRICS_PATH = "metrics.json"            # metrics produced by fusion/cog nodes
CHECKPOINT_DIR = "checkpoints"           # directory with named snapshots
CURRENT_SYMLINK = "current_checkpoint"   # active checkpoint link
KILL_FLAG = "kill.lock"                  # presence means hard-kill enforced

# parameters: quorum fraction and dwell samples
ALPHA = 0.67
DWELL = 3
POLL = 0.5

async def read_metrics():
    try:
        with open(METRICS_PATH,"r") as f:
            return json.load(f)
    except Exception:
        return {}  # treat missing as benign until enough detectors fire

def detectors_from(metrics):
    # detectors return True when they see an anomaly (example heuristics)
    d = []
    # NEES check
    nees = metrics.get("nees", 0.0)
    d.append(nees > 12.0)                 # threshold per calibration
    # track_entropy check
    entropy = metrics.get("track_entropy", 0.0)
    d.append(entropy > 1.5)
    # scenario_confidence check (low is bad)
    conf = metrics.get("scenario_confidence", 1.0)
    d.append(conf < 0.4)
    return d

def quorum_required(n):
    return int((ALPHA * n) // 1 + (1 if (ALPHA*n) % 1 > 0 else 0))

def atomic_swap_checkpoint(target_name):
    target = os.path.join(CHECKPOINT_DIR, target_name)
    if not os.path.exists(target):
        return False
    tmp = CURRENT_SYMLINK + ".tmp"
    if os.path.islink(tmp): os.unlink(tmp)
    os.symlink(os.path.abspath(target), tmp)   # create new link
    os.replace(tmp, CURRENT_SYMLINK)           # atomic replace
    return True

async def monitor_loop():
    consecutive = 0
    n_detectors = 3
    q = quorum_required(n_detectors)
    while True:
        metrics = await read_metrics()
        dets = detectors_from(metrics)
        hits = sum(1 for x in dets if x)
        if hits >= q:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= DWELL:
            # enforce soft kill then rollback to last known-good
            open(KILL_FLAG,"w").close()            # create kill flag
            # choose rollback target from metrics (policy) or recent version
            target = metrics.get("recommended_checkpoint","golden_v1")
            ok = atomic_swap_checkpoint(target)
            # brief log to stdout (replace with signed audit in prod)
            print(f"{time.strftime('%T')} safety: quorum={hits}/{n_detectors} dwell={consecutive} rollback={target} ok={ok}")
            # give operators time to inspect before clearing
            await asyncio.sleep(2.0)
            # in a real setup, restart services and maintain kill until cleared
            os.remove(KILL_FLAG)
            consecutive = 0
        await asyncio.sleep(POLL)

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # ensure a default symlink exists
    if not os.path.exists(CURRENT_SYMLINK):
        atomic_swap_checkpoint("golden_v1")
    asyncio.run(monitor_loop())