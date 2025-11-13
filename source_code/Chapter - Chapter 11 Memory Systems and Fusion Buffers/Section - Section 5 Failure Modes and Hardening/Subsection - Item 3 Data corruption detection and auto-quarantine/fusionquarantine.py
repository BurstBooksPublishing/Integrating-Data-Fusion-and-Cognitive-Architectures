import hashlib, json, sqlite3, time, queue, threading
# Simple streaming stats (Welford)
class OnlineStat:
    def __init__(self):
        self.n=0; self.mean=0.0; self.M2=0.0
    def update(self,x):
        self.n+=1
        delta=x-self.mean
        self.mean += delta/self.n
        self.M2 += delta*(x-self.mean)
    def var(self): return (self.M2/(self.n-1)) if self.n>1 else 0.0
    def std(self): return (self.var()**0.5)
# Validator with quarantine DB
class FusionBufferValidator:
    def __init__(self, db_path='|quarantine.db|', z_thresh=6.0):
        self.q = queue.Queue()                     # processing queue
        self.stats = {}                            # per-field stats
        self.z_thresh = z_thresh
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS quarantine
            (id TEXT PRIMARY KEY, payload TEXT, reason TEXT, checksum TEXT, ts REAL)''')
    def checksum(self, payload):
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()
    def schema_ok(self, rec):
        return isinstance(rec, dict) and 'seq' in rec and 'ts' in rec
    def validate(self, rec):
        raw = json.dumps(rec, sort_keys=True)
        cs = self.checksum(raw)
        if not self.schema_ok(rec):
            return False, 'schema'
        # sequence monotonicity check (simple example)
        if rec['seq']<0: return False, 'seq_bad'
        # statistical checks
        for k,v in rec.items():
            if isinstance(v,(int,float)):
                st = self.stats.setdefault(k, OnlineStat())
                if st.n>1:
                    z = abs((v - st.mean) / (st.std() or 1e-9))
                    if z>self.z_thresh:
                        return False, f'stat_z:{k}'
                st.update(float(v))
        return True, cs
    def quarantine(self, rec, reason, checksum):
        raw = json.dumps(rec, sort_keys=True)
        self.conn.execute('INSERT OR REPLACE INTO quarantine VALUES (?,?,?,?,?)',
                          (checksum, raw, reason, checksum, time.time()))
        self.conn.commit()
    def ingest(self, rec):
        ok, info = self.validate(rec)
        if ok is True:
            self.q.put(rec)                # forward to fusion pipeline
        else:
            self.quarantine(rec, info, info if isinstance(info,str) else info)
# Example usage omitted for brevity.