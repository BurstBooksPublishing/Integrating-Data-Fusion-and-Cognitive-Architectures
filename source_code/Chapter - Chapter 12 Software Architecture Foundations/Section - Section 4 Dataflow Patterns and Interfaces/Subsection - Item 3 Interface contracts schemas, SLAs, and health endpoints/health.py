from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import math, uvicorn

app = FastAPI()

# Schema: track ID, position, covariance, ts, provenance, schema_version
class Track(BaseModel):
    id: str
    x: float
    y: float
    cov00: float = Field(..., ge=0)  # inline validation
    timestamp: datetime
    provenance: str
    schema_version: str

# SLA parameters (could be loaded from config)
UPDATE_RATE_PER_SEC = 1.0  # lambda
T_MAX = 2.0                 # max freshness seconds

def stale_probability(lambda_per_s: float, T: float) -> float:
    return math.exp(-lambda_per_s * T)  # Poisson no-arrival prob

@app.post("/ingest/track")
def ingest_track(track: Track):
    # sidecar-like validation happens via Pydantic
    # enforce provenance and version policy
    if not track.schema_version.startswith("v"):
        raise HTTPException(status_code=400, detail="Bad schema_version")
    return {"status": "accepted", "id": track.id}

@app.get("/health")
def health():
    p_stale = stale_probability(UPDATE_RATE_PER_SEC, T_MAX)
    # derive state thresholds for example
    state = "UP" if p_stale < 0.1 else ("DEGRADED" if p_stale < 0.5 else "DOWN")
    return {
        "state": state,
        "stale_probability": p_stale,
        "sli": {"update_rate": UPDATE_RATE_PER_SEC, "t_max": T_MAX},
        "schema_registry": {"track": "v1.2.0"}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # run for local testing