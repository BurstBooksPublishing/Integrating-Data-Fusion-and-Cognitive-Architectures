# Minimal on-device pipeline: simulate frames, detect, extract embedding, store metadata with TTL
import time, sqlite3, hmac, hashlib, os
import numpy as np
from PIL import Image, ImageFilter

DB="edge_store.db"
KEY=b"device_secret_key"  # secure key in enclave in practice
# init DB
conn=sqlite3.connect(DB); c=conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, room TEXT, conf REAL, emb BLOB, sig BLOB, ttl REAL)")
conn.commit()

def detect_and_embed(frame):
    # dummy detector: thresholded mean pixel; embedding: small vector
    mean=np.array(frame).mean()/255.0
    conf=float(mean)  # proxy confidence
    emb=(np.random.RandomState(int(time.time()*1e6)).randn(8)).astype(np.float32).tobytes()
    return conf, emb

def sign_record(ts, room, conf, emb):
    msg=f"{ts}:{room}:{conf}".encode()+emb
    return hmac.new(KEY,msg,hashlib.sha256).digest()

def store_event(room, conf, emb, ttl_seconds=3600.0):
    ts=time.time()
    sig=sign_record(ts,room,conf,emb)
    c.execute("INSERT INTO events(ts,room,conf,emb,sig,ttl) VALUES(?,?,?,?,?,?)",
              (ts,room,conf,emb,sig,ts+ttl_seconds))
    conn.commit()

def purge_expired():
    now=time.time()
    c.execute("DELETE FROM events WHERE ttllower conf
    img=img.filter(ImageFilter.GaussianBlur(radius=1))  # redaction step example
    conf,emb=detect_and_embed(img)
    # selective retention: keep only if confidence above threshold or safety flag
    if conf>0.15:
        store_event("kitchen",conf,emb,ttl_seconds=60.0)  # short TTL by default
    purge_expired()
    time.sleep(1)
conn.close()