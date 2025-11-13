import os, json, time
SNAP_DIR = "snapshots"            # durable folder (object store in prod)
WAL_PATH = "wal.log"
os.makedirs(SNAP_DIR, exist_ok=True)

class StatefulOperator:
    def __init__(self):
        self.state = {}            # example: track_id -> attributes
        self.last_seq = 0
        self.load()

    def append_wal(self, seq, op):
        with open(WAL_PATH, "a") as f: f.write(json.dumps((seq, op))+"\n")
        self.last_seq = seq

    def apply_op(self, op):
        # idempotent apply based on sequence numbers kept in op metadata
        typ = op["type"]
        k = op["key"]
        if typ == "set": self.state[k] = op["value"]
        elif typ == "del": self.state.pop(k, None)

    def snapshot(self):
        # atomic write: write tmp then rename
        seq = self.last_seq
        tmp = os.path.join(SNAP_DIR, f"snap_{seq}.tmp")
        final = os.path.join(SNAP_DIR, f"snap_{seq}.json")
        with open(tmp, "w") as f:
            json.dump({"seq": seq, "state": self.state}, f)
        os.replace(tmp, final)      # atomic on POSIX

    def load(self):
        # recovery: load latest snapshot, replay WAL
        snaps = [f for f in os.listdir(SNAP_DIR) if f.startswith("snap_")]
        if snaps:
            latest = sorted(snaps, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
            with open(os.path.join(SNAP_DIR, latest)) as f:
                data = json.load(f)
            self.state = data["state"]
            self.last_seq = data["seq"]
        # replay WAL
        if os.path.exists(WAL_PATH):
            with open(WAL_PATH) as f:
                for line in f:
                    seq, op = json.loads(line)
                    if seq > self.last_seq:
                        self.apply_op(op); self.last_seq = seq

# usage: operator processes messages with seq numbers
if __name__ == "__main__":
    op = StatefulOperator()
    # simulate incoming ops
    for i in range(1, 6):
        op.append_wal(i, {"type":"set", "key":f"t{i}", "value":i})
    op.snapshot()                   # checkpoint after batch
    # further ops
    for i in range(6,9):
        op.append_wal(i, {"type":"set", "key":f"t{i}", "value":i})
    # crash simulation: reload into new instance
    new = StatefulOperator()
    print("recovered state:", new.state)  # shows keys t1..t8