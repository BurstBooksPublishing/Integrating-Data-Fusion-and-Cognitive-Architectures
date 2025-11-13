import time
from threading import Lock

class MaterializedCache:
    def __init__(self, refresh_func, stale_bound=0.5):
        self._store = {}               # key -> (value, ts)
        self._lock = Lock()
        self._refresh = refresh_func   # callable(key) -> (value, ts)
        self._stale_bound = stale_bound

    def write(self, key, value):
        ts = time.time()
        with self._lock:
            self._store[key] = (value, ts)
        return ts  # return timestamp for session use

    def read(self, key, session_last_write_ts=None):
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
        if entry:
            value, ts = entry
            if now - ts <= self._stale_bound:
                return value  # bounded-stale satisfied
        # stale or missing: synchronous refresh
        value, ts = self._refresh(key)   # blocking fetch from authoritative source
        with self._lock:
            self._store[key] = (value, ts)
        # enforce read-your-writes for session that just wrote locally
        if session_last_write_ts and session_last_write_ts > ts:
            # session has newer local write; return session view instead
            with self._lock:
                return self._store.get(key)[0]
        return value

# Example refresh function (authoritative source)
def fetch_authoritative(key):
    return ("authoritative-"+key, time.time())