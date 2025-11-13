from datetime import datetime, timedelta
# simple registry of compute nodes with region tags
COMPUTE_NODES = {"eu-node-1": "EU", "us-node-1": "US", "edge-device-1": "EU"}

def policy_allows(data_meta, purpose, node):
    # data_meta: dict with keys 'region','purposes','expiry','provenance_sig'
    now = datetime.utcnow()
    # check purpose membership and expiry
    if purpose not in data_meta["purposes"]: return False
    if now >= data_meta["expiry"]: return False
    # check residency: node must be in same legal region
    node_region = COMPUTE_NODES.get(node)
    if node_region is None: return False
    if node_region != data_meta["region"]: return False
    # provenance_sig validation stub (attestation required in real system)
    return verify_signature(data_meta["provenance_sig"])

def verify_signature(sig):
    # placeholder for cryptographic attestation check
    return sig == "valid-sig"

def route_and_execute(data, data_meta, purpose):
    # choose nodes that satisfy residency, prefer low-latency
    candidates = [n for n,r in COMPUTE_NODES.items() if r==data_meta["region"]]
    for node in candidates:
        if policy_allows(data_meta, purpose, node):
            # execute in-region; in production this invokes remote tasking
            return execute_on_node(node, data, purpose)
    return None  # deny or trigger fallback

def execute_on_node(node, data, purpose):
    # placeholder: run fusion/cognition on selected node
    return {"node": node, "result": f"processed for {purpose}"}

# example usage
meta = {"region":"EU","purposes":{"maritime-safety"},"expiry":datetime.utcnow()+timedelta(hours=2),
        "provenance_sig":"valid-sig"}
print(route_and_execute("track_stream", meta, "maritime-safety"))