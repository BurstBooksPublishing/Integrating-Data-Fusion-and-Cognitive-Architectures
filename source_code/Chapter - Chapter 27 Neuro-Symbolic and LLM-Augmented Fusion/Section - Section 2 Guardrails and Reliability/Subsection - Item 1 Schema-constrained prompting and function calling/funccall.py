import json
import openai      # pip install openai
import jsonschema  # pip install jsonschema

# Domain schema: situation hypothesis with provenance and confidence.
SITUATION_SCHEMA = {
  "type": "object",
  "required": ["id", "scenario_type", "confidence", "evidence"],
  "properties": {
    "id": {"type": "string"},
    "scenario_type": {"type": "string", "enum": ["rendezvous","pursuit","loitering"]},
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "evidence": {"type": "array", "items": {"type": "string"}},
    "timestamp": {"type": "string", "format": "date-time"}
  },
  "additionalProperties": False
}

# Function declaration exposed to the LLM (OpenAI function-calling style).
functions = [{
  "name": "create_situation_hypothesis",
  "description": "Produce a structured situation hypothesis per schema",
  "parameters": SITUATION_SCHEMA
}]

prompt = (
  "Given fused tracks and event graph, return a single situation hypothesis.\n"
  "Conform exactly to the provided JSON schema; do not add fields.\n"
  "Include evidence item IDs and a confidence score."
)

resp = openai.ChatCompletion.create(
  model="gpt-4o-mini",  # example model
  messages=[{"role":"system","content":"You are a constrained reasoning assistant."},
            {"role":"user","content":prompt}],
  functions=functions,
  function_call={"name":"create_situation_hypothesis"},
  temperature=0.0
)

# Parse function call result and validate.
payload = json.loads(resp.choices[0].message.get("function_call", {}).get("arguments","{}"))
try:
  jsonschema.validate(instance=payload, schema=SITUATION_SCHEMA)
  # Accept and record the hypothesis.
  print("VALID HYPOTHESIS:", payload)
except jsonschema.ValidationError as e:
  # Deterministic fallback: rule-based template ensures safe structure.
  fallback = {
    "id": "fb-"+resp.id,
    "scenario_type": "loitering",
    "confidence": 0.0,              # conservative default
    "evidence": ["no_valid_evidence"],
    "timestamp": "1970-01-01T00:00:00Z"
  }
  print("FALLBACK HYPOTHESIS:", fallback)