#!/usr/bin/env python3
import json, os, tempfile
from typing import Dict

LOG_PATH = "events.log"         # append-only JSONL
CHK_PATH = "checkpoint.json"    # atomic snapshot of state + last_seq

class CrashOnlyService:
    def __init__(self):
        self.state = {}          # in-memory store
        self.last_seq = 0
        self._load_checkpoint()
        self._replay()

    def _load_checkpoint(self):
        if not os.path.exists(CHK_PATH): return
        with open(CHK_PATH,'r') as f:
            chk = json.load(f)
        self.state = chk.get("state", {})
        self.last_seq = chk.get("last_seq", 0)

    def append_event(self, seq: int, payload: Dict):
        # Persist event before acknowledging. Caller ensures seq monotonic.
        with open(LOG_PATH,'a') as f:
            f.write(json.dumps({"seq": seq, "payload": payload}) + "\n")
            f.flush(); os.fsync(f.fileno())

    def _replay(self):
        if not os.path.exists(LOG_PATH): return
        with open(LOG_PATH,'r') as f:
            for line in f:
                evt = json.loads(line)
                seq = evt["seq"]
                if seq <= self.last_seq:
                    continue                        # idempotent skip
                self._apply(evt["payload"])
                self.last_seq = seq
                # optional: checkpoint incrementally for bounded RTO
                self._write_checkpoint()

    def _apply(self, payload: Dict):
        # Deterministic state transform; keep side-effects out or transactional.
        key = payload["key"]
        val = payload["value"]
        self.state[key] = val

    def _write_checkpoint(self):
        tmp_fd, tmp_path = tempfile.mkstemp(dir='.')
        with os.fdopen(tmp_fd,'w') as f:
            json.dump({"last_seq": self.last_seq, "state": self.state}, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_path, CHK_PATH)     # atomic rename
# --- simple usage ---
if __name__ == "__main__":
    svc = CrashOnlyService()
    svc.append_event(1, {"key":"a","value":1})
    svc.append_event(2, {"key":"b","value":2})
    # simulate next restart
    svc2 = CrashOnlyService()
    print(svc2.state)  # {'a':1,'b':2}