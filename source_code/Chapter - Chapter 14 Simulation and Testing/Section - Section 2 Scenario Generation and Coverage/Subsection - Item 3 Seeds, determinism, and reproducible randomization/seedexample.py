import json, hashlib, datetime
import numpy as np
from numpy.random import SeedSequence, Generator, PCG64

# Base seed and version (pin these externally)
base_seed = 123456789  # stable integer for reproducibility
version = "sim-v1.2.0"

# Spawn independent streams for actors and sensors
ss = SeedSequence(base_seed)
# spawn( ) allocates child SeedSequences deterministically
child_seqs = ss.spawn(4)                # 0:env,1:actor1,2:actor2,3:sensor
rngs = [Generator(PCG64(cs)) for cs in child_seqs]

# Example use: actor draws and sensor noise
env_rng, actor1_rng, actor2_rng, sensor_rng = rngs
actor1_action = actor1_rng.choice(["stop","go","yield"])  # deterministic given seed
sensor_noise = sensor_rng.normal(0, 0.05, size=(640,480))  # reproducible array

# Compute scenario identifier
params = {"actors":2,"weather":"rain","time":"dusk"}
param_hash = hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
scenario_id = hashlib.sha256(f"{version}|{base_seed}|{param_hash}".encode()).hexdigest()

# Persist metadata for replay and auditing
meta = {"scenario_id":scenario_id,"version":version,"base_seed":base_seed,
        "spawn_map":["env","actor1","actor2","sensor"],
        "timestamp":datetime.datetime.utcnow().isoformat()+"Z"}
with open(f"meta_{scenario_id}.json","w") as f:
    json.dump(meta,f,indent=2)