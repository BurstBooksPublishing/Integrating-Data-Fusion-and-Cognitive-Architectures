import subprocess, shutil, time, os
# simple manager for simulator-backed experiments
class FaultInjectionManager:
    def __init__(self, sim_cmd, state_dir, snapshot_root, golden_trace, monitor_fn):
        self.sim_cmd = sim_cmd            # command to run simulator
        self.state_dir = state_dir        # simulator state path
        self.snapshot_root = snapshot_root
        self.golden_trace = golden_trace  # path for golden-trace comparator
        self.monitor_fn = monitor_fn      # callable -> (ok, metric)
        self.proc = None

    def start_sim(self):
        self.proc = subprocess.Popen(self.sim_cmd)  # start simulator
        time.sleep(0.5)                              # wait for warmup

    def take_snapshot(self, name):
        dst = os.path.join(self.snapshot_root, name)
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(self.state_dir, dst)         # filesystem checkpoint
        return dst

    def inject_fault(self, fault_cb, *args, **kwargs):
        fault_cb(*args, **kwargs)                    # domain-specific injector

    def rollback(self, snapshot):
        if self.proc: self.proc.terminate()
        if os.path.exists(self.state_dir): shutil.rmtree(self.state_dir)
        shutil.copytree(snapshot, self.state_dir)    # restore state
        self.start_sim()                             # restart simulator

    def monitor_and_run(self, snapshot, timeout=30, threshold=0.1):
        start=time.time()
        while time.time()-startthreshold:
                self.rollback(snapshot)             # automated safety rollback
                return False, metric
            time.sleep(0.1)
        return True, metric
# usage would wire in simulator command, monitor, and injector functions