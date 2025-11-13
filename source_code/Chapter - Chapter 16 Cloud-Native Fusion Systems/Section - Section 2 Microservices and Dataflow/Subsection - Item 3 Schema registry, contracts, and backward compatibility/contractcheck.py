# Simple contract validator: checks required fields presence and nullability.
import json
from typing import Dict, List

def load_schema(path: str) -> Dict:
    with open(path) as f: return json.load(f)

def field_map(schema: Dict) -> Dict[str, Dict]:
    return {f['name']: f for f in schema.get('fields', [])}

def is_compatible(new_schema: Dict, required_fields: List[Dict]) -> bool:
    new_fields = field_map(new_schema)
    for req in required_fields:
        name = req['name']
        if name not in new_fields: return False
        nf = new_fields[name]
        # allow nullable (union with "null") or default provided
        if 'default' not in nf and not (isinstance(nf.get('type'), list) and 'null' in nf['type']):
            return False
        # basic type check (could be extended with promotion rules)
        if req.get('type') and nf.get('type') != req['type']:
            return False
    return True

if __name__ == "__main__":
    # consumer requires timestamp, track_id, pose (nullable allowed)
    consumer_req = [
        {'name':'t_obs','type':'long'},
        {'name':'track_id','type':'string'},
        {'name':'pose'}  # type flexible if default/null present
    ]
    new = load_schema('track_v2.json')   # producer schema file
    ok = is_compatible(new, consumer_req)
    print("Compatible:", ok)             # CI fails build if False