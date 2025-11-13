#!/usr/bin/env python3
import os, sys, json, time, hashlib, random, subprocess
# optional libraries
try:
    import numpy as np
except Exception:
    np = None
try:
    import torch
except Exception:
    torch = None

# 1) Seed control
SEED = 20251107
random.seed(SEED)
if np is not None:
    np.random.seed(SEED)
if torch is not None:
    torch.manual_seed(SEED)
    # deterministic flags where supported
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# 2) Environment capture
def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode().strip()
    except Exception:
        return "UNKNOWN"

git_hash = run("git rev-parse HEAD")
pip_freeze = run("pip freeze")

# 3) Artifact checksums (example: model.pt, dataset.zip)
artifacts = []
for fname in ["model.pt", "dataset.zip"]:
    if os.path.exists(fname):
        h = hashlib.sha256()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        artifacts.append({"path": fname, "sha256": h.hexdigest(), "size": os.path.getsize(fname)})
    else:
        artifacts.append({"path": fname, "sha256": None, "size": None, "note": "missing"})

# 4) Manifest write and local "registry" publish
manifest = {
    "timestamp": time.time(),
    "seed": SEED,
    "git_commit": git_hash,
    "pip_freeze": pip_freeze,
    "artifacts": artifacts,
    "platform": {"python": sys.version.split()[0], "os": os.name},
}
os.makedirs("artifact_registry", exist_ok=True)
manifest_path = os.path.join("artifact_registry", f"manifest-{int(time.time())}.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print("Wrote manifest:", manifest_path)
# optional: CI should sign the manifest and upload to remote registry here