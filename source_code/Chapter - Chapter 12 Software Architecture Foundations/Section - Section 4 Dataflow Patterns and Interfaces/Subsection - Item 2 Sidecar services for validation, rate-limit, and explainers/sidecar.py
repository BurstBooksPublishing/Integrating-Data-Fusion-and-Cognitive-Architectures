from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ValidationError
import time, asyncio

app = FastAPI()

# Pydantic schema for an object track message
class Track(BaseModel):
    id: int
    x: float
    y: float
    cov: list  # covariance flattened

# Token-bucket state
RATE = 10.0  # tokens/sec
CAP = 50.0
tokens = CAP
last_ts = time.time()
lock = asyncio.Lock()

async def consume(tokens_needed=1.0):
    global tokens, last_ts
    async with lock:
        now = time.time()
        tokens = min(CAP, tokens + RATE*(now-last_ts))
        last_ts = now
        if tokens >= tokens_needed:
            tokens -= tokens_needed
            return True
        return False

@app.post("/validate")
async def validate(track: dict):
    # schema check
    try:
        t = Track(**track)               # raises ValidationError if bad
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # provenance/time checks (simple)
    if "timestamp" in track and track["timestamp"] > time.time()+1.0:
        raise HTTPException(status_code=400, detail="timestamp future")
    # rate control
    if not await consume():
        raise HTTPException(status_code=429, detail="rate_limited")
    # statistical check placeholder (e.g., NIS test) omitted for brevity
    return {"status":"ok","id":t.id}

@app.post("/explain")
async def explain(payload: dict):
    # produce compact rationale: rule hits, evidence pointers, confidence
    rationale = {
        "rule_hits":["schema_ok"],                 # example
        "evidence":[payload.get("source_uri")],   # pointer to raw input
        "confidence":0.87
    }
    return rationale

# Run with: uvicorn this_module:app --port 8000