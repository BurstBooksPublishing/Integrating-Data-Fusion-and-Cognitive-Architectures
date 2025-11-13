import time, threading
from collections import OrderedDict

# Simple thread-safe TTL LRU cache
class ViewCache:
    def __init__(self, maxsize=1000):
        self.lock = threading.Lock()
        self.data = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        with self.lock:
            if key not in self.data: return None
            value, expiry = self.data.pop(key)
            if expiry and expiry < time.time(): return None
            self.data[key] = (value, expiry)      # refresh position
            return value

    def set(self, key, value, ttl=None):
        expiry = time.time() + ttl if ttl else None
        with self.lock:
            if key in self.data: self.data.pop(key)
            self.data[key] = (value, expiry)
            if len(self.data) > self.maxsize:
                self.data.popitem(last=False)   # evict LRU

# Example sources
tracks = { 't1': {'entity_id':'t1','pos':(10,5),'last_seen':1620000000},
           't2': {'entity_id':'t2','pos':(20,8),'last_seen':1620000020} }
ontology = {'vehicle': {'threat':0.2}, 'person': {'threat':0.9}}
ontology_version = 1

cache = ViewCache(maxsize=100)

def compute_view(track, ontology_map, ont_ver):
    # deterministic join logic; inexpensive example
    cls = 'vehicle' if track['pos'][0] > 15 else 'person'
    enriched = dict(track)
    enriched.update({'class': cls, 'threat': ontology_map[cls]['threat'],
                     'ont_ver': ont_ver})
    return enriched

def view_key(entity_id, time_bucket, ont_ver):
    return (entity_id, time_bucket, ont_ver)

def lookup_view(entity_id, timestamp, ttl=2.0):
    tb = int(timestamp // 5)   # coarse time bucket
    key = view_key(entity_id, tb, ontology_version)
    v = cache.get(key)
    if v: return v              # cache hit
    # cache miss: fetch sources and compute
    track = tracks.get(entity_id)
    if not track: return None
    view = compute_view(track, ontology, ontology_version)
    cache.set(key, view, ttl=ttl)
    return view

# Usage example (simulate requests)
now = 1620000024
print(lookup_view('t2', now))   # miss then cached
print(lookup_view('t2', now+1)) # hit