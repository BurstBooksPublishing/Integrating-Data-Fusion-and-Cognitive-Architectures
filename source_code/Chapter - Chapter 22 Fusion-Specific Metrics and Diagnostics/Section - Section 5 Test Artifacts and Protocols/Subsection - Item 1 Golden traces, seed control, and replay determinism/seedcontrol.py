import os, json, time, platform, subprocess
import random, numpy as np
import torch

# 1) Fix seeds for Python, NumPy, Torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 2) Enforce framework deterministic flags
torch.use_deterministic_algorithms(True)           # raise if non-det op
torch.backends.cudnn.benchmark = False

# 3) Capture RNG states for later exact restoration
rng_state = {
  "python_random": random.getstate(),
  "numpy_random": np.random.get_state(),
  "torch_cpu": torch.get_rng_state().tolist()
}
if torch.cuda.is_available():
    rng_state["torch_cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]

# 4) Environment snapshot
env = {
  "time": time.time(),
  "platform": platform.platform(),
  "uname": platform.uname()._asdict(),
  "python": platform.python_version(),
  "git_commit": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
}

# 5) Persist trace metadata
with open("golden_meta.json","w") as f:
    json.dump({"seed":seed, "rng_state": rng_state, "env": env}, f, default=str)
# sensor data recorded separately (e.g., ros2 bag or raw files)