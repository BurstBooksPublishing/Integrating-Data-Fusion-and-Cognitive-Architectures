#!/usr/bin/env python3
import time, json, os
from multiprocessing import Process, Event

SNAP = "snapshot.json"

class StateActor:
    def __init__(self, stop_evt):
        self.stop = stop_evt
        self.state = {"counter":0,"tracks":{}}  # simple state

    def load(self):
        if os.path.exists(SNAP):
            with open(SNAP,"r") as f: self.state = json.load(f)

    def snapshot(self):
        with open(SNAP+".tmp","w") as f:
            json.dump(self.state,f)
        os.replace(SNAP+".tmp", SNAP)   # atomic replace

    def run(self):
        self.load()
        while not self.stop.is_set():
            # simulate fusion update: increment counter and pretend to add track
            self.state["counter"] += 1
            tid = f"t{self.state['counter']}"
            self.state["tracks"][tid] = {"pos": [self.state["counter"],0], "ts": time.time()}
            if self.state["counter"] % 5 == 0:
                self.snapshot()  # periodic checkpoint
            time.sleep(0.5)    # processing cadence

def actor_process(stop_evt):
    a = StateActor(stop_evt)
    a.run()

def upgrade_once():
    stop_evt = Event()
    p = Process(target=actor_process, args=(stop_evt,))
    p.start()
    time.sleep(3)           # let actor run and create snapshots
    # begin upgrade: request quiesce, snapshot, spawn new actor, then stop old
    stop_evt.set()          # simple quiesce signal
    p.join(timeout=2)
    # spawn replacement which will load snapshot
    stop_evt2 = Event()
    p2 = Process(target=actor_process, args=(stop_evt2,))
    p2.start()
    time.sleep(3)
    stop_evt2.set()
    p2.join()

if __name__ == "__main__":
    upgrade_once()   # single upgrade demonstration