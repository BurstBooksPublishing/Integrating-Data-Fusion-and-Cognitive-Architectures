# CI job: compare descriptors between v1 and v2 (requires both package versions installed)
from fusion_msgs_v1.msg import Track as TrackV1
from fusion_msgs_v2.msg import Track as TrackV2
from typing import Dict

def fields_map(msg_cls) -> Dict[str,str]:
    return dict(msg_cls._fields_and_field_types)  # rosidl_runtime_py introspection

v1_fields = fields_map(TrackV1)
v2_fields = fields_map(TrackV2)

# simple compatibility predicate: every v1 field must exist in v2 with compatible type
incompatible = []
for name, typ in v1_fields.items():
    if name not in v2_fields:
        incompatible.append((name, "missing"))
    elif v2_fields[name] != typ:
        incompatible.append((name, f"type_mismatch {typ} -> {v2_fields[name]}"))

if incompatible:
    for f, reason in incompatible:
        print(f"Incompatibility: {f}: {reason}")
    raise SystemExit(2)  # fail CI
print("Descriptor-level compatibility: OK")