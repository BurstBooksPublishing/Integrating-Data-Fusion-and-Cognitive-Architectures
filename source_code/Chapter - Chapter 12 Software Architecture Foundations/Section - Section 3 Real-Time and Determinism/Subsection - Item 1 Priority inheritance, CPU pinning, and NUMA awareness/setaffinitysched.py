import os, subprocess, ctypes, logging
from ctypes import c_int, Structure, POINTER, byref

logging.basicConfig(level=logging.INFO)
pid = 0  # current process

# Set CPU affinity to cores 2-3 (example; adjust per machine)
cores = {2,3}
try:
    os.sched_setaffinity(pid, cores)             # bind process to cores
    logging.info("Affinity set to %s", cores)
except PermissionError:
    logging.warning("Insufficient privilege to change affinity")

# Request SCHED_FIFO priority 50 (root required)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
SCHED_FIFO = 1
class SchedParam(Structure):
    _fields_ = [("sched_priority", c_int)]
param = SchedParam(50)
res = libc.sched_setscheduler(0, SCHED_FIFO, byref(param))  # 0 => self
if res != 0:
    err = ctypes.get_errno()
    logging.warning("sched_setscheduler failed: %d", err)
else:
    logging.info("SCHED_FIFO set at priority %d", param.sched_priority)

# Optional: bind to NUMA node 0 for memory locality (requires numactl)
try:
    subprocess.check_call(["numactl","--cpunodebind=0","--membind=0","--", "/bin/true"])
    logging.info("NUMA binding supported")
except (FileNotFoundError, subprocess.CalledProcessError):
    logging.info("numactl not available or binding failed; ensure first-touch allocator policy")