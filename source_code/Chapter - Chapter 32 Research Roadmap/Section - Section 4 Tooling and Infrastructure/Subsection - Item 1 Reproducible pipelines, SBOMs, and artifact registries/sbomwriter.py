#!/usr/bin/env python3
# Simple: compute SHA256, write SBOM JSON, place files in registry dir.
import hashlib, json, os, sys, time
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def build_sbom(model_path, deps, build_env):
    model_hash = sha256_file(model_path)
    sbom = {
        "sbom_spec": "spdx-lite-1.0",
        "artifact": Path(model_path).name,
        "artifact_hash": model_hash,
        "dependencies": deps,            # list of {"name","version","hash"}
        "build_env": build_env,         # canonical toolchain and OS info
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    # content-addressable id (simple canonical JSON)
    id_input = json.dumps(sbom, sort_keys=True).encode('utf-8')
    sbom_id = hashlib.sha256(id_input).hexdigest()
    sbom["artifact_id"] = sbom_id
    return sbom

def register(model_path, sbom, registry_dir="artifact_registry"):
    os.makedirs(registry_dir, exist_ok=True)
    # write model copy (immutable store) and SBOM
    model_dst = Path(registry_dir) / f"{sbom['artifact_id']}_{Path(model_path).name}"
    sbom_dst = Path(registry_dir) / f"{sbom['artifact_id']}.sbom.json"
    if not model_dst.exists():
        # copy file (stream to avoid memory pressure)
        with open(model_path,'rb') as src, open(model_dst,'wb') as dst:
            for chunk in iter(lambda: src.read(8192), b''):
                dst.write(chunk)
    with open(sbom_dst,'w', encoding='utf-8') as f:
        json.dump(sbom, f, indent=2, sort_keys=True)
    print("Registered:", sbom['artifact_id'])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: sbom_writer.py ")
        sys.exit(1)
    model = sys.argv[1]
    # Example deps/build_env; replace with real resolver in production
    deps = [{"name":"numpy","version":"1.24.0","hash":"sha256:..."}]
    build_env = {"os":"ubuntu-22.04","python":"3.10.8","builder":"ci-runner-7"}
    sbom = build_sbom(model, deps, build_env)
    register(model, sbom)