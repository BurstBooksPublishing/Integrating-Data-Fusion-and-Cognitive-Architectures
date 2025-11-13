from fastapi import FastAPI, Query
from typing import List
import datetime, json

app = FastAPI()

# Mock stores: replace with real clients (Kafka consumer, object store, OLAP DB).
EVENT_STORE = {}   # dict[(entity_id, time)] -> event

@app.get("/audit/trace")  # signed bundle for compliance review
def audit_trace(entity_id: str, start: str, end: str):
    # parse window, fetch events from materialized view
    s = datetime.datetime.fromisoformat(start); e = datetime.datetime.fromisoformat(end)
    events = [v for k,v in EVENT_STORE.items()
              if k[0]==entity_id and s <= k[1] <= e]
    # attach manifest with schema and signatures (placeholder)
    manifest = {"entity": entity_id, "schema_version": "v1.2", "signed": True}
    return {"manifest": manifest, "events": events}

@app.get("/replay/stream")  # deterministic replay in ingest order
def replay_stream(entity_id: str, cursor: int = 0, limit: int = 100):
    # cursor is monotonic commit_seq to ensure deterministic ordering
    ordered = sorted([ (v['commit_seq'],v) for v in EVENT_STORE.values()
                       if v['entity']==entity_id and v['commit_seq']>=cursor ])
    return {"events": [v for _,v in ordered[:limit]]}

@app.post("/counterfactual/apply")  # return event stream after intervention
def counterfactual_apply(entity_id: str, start: str, end: str,
                         remove_types: List[str] = Query([])):
    s = datetime.datetime.fromisoformat(start); e = datetime.datetime.fromisoformat(end)
    events = [v for v in EVENT_STORE.values()
              if v['entity']==entity_id and s <= v['time'] <= e]
    # intervention: drop specified event types, maintain ordering and provenance
    cf = [v for v in events if v['type'] not in remove_types]
    # include metadata for downstream simulation (diffs, rationale)
    return {"original_count": len(events), "counterfactual_count": len(cf), "events": cf}