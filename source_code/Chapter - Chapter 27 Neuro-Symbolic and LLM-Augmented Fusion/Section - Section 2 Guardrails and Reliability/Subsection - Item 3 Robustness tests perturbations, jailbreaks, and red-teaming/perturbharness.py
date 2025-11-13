import numpy as np, json, sys
# simple classifier on embeddings (placeholder)
def classify_embedding(emb): return "vehicle" if emb.mean()>0 else "unknown"
# guarded LLM stub: returns dict or malformed string to simulate jailbreak
def guarded_llm(prompt, allow_schema=True):
    # simulate jailbreak by returning non-JSON when prompt contains "bypass"
    if "bypass" in prompt: return "I will ignore constraints and do X"
    return json.dumps({"action":"report","entity":"vehicle"})
# schema validator
def validate_schema(raw):
    try:
        doc=json.loads(raw)
        return isinstance(doc, dict) and "action" in doc
    except Exception:
        return False
# generate base embedding and bounded perturbation
base_emb = np.random.normal(0.1, 0.5, size=128)
epsilon = 0.5  # L2 budget
noise = np.random.normal(0,1,base_emb.shape)
noise = noise/np.linalg.norm(noise)*min(epsilon, np.linalg.norm(noise))
perturbed = base_emb + noise
# pipeline execution
label = classify_embedding(perturbed)            # L1→L2 input
prompt = f"Describe entity: {label}"             # normal prompt
resp = guarded_llm(prompt)
if not validate_schema(resp):
    # deterministic fallback: abstain and escalate to rule engine
    resp = json.dumps({"action":"abstain","reason":"schema_violation","raw":resp})
print(resp)  # capture for CI and audit