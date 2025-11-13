from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SensorFrame:
    name: str
    timestamp: float
    data: Any
    meta: Dict[str,Any] = None

class Simulator(ABC):
    @abstractmethod
    def reset(self, seed: int=0): ...
    @abstractmethod
    def step(self, dt: float): ...
    @abstractmethod
    def read_sensor(self, name: str) -> SensorFrame: ...
    @abstractmethod
    def capabilities(self) -> Dict[str,Any]: ...

# Simple in-process physics sim (placeholder)
class SimplePhysicsSim(Simulator):
    def __init__(self, cfg): self.state = 0.0; self.cfg = cfg
    def reset(self, seed: int=0): self.state = 0.0
    def step(self, dt: float): self.state += dt*1.0  # simple integrator
    def read_sensor(self, name: str):
        return SensorFrame(name, 0.0, {"pos": self.state})
    def capabilities(self): return {"sensors":["pose"], "fidelity": 0.4}

# Adapter for an external simulator like CARLA (simplified stub)
class CarlaAdapter(Simulator):
    def __init__(self, cfg):
        import random
        self.client = "carla_client_stub"  # replace with actual client init
        self.cfg = cfg
    def reset(self, seed: int=0): pass  # map to CARLA reset
    def step(self, dt: float): pass    # map to CARLA tick
    def read_sensor(self, name: str):
        # convert CARLA sensor output to SensorFrame
        return SensorFrame(name, 0.0, {"image": b"\x89PNG..."})
    def capabilities(self): return {"sensors":["camera","lidar"], "fidelity": 0.9}

def load_simulator(cfg):
    back = cfg.get("sim_backend","simple")
    if back == "carla": return CarlaAdapter(cfg)
    return SimplePhysicsSim(cfg)

# Usage: fusion/cognition code calls only Simulator methods.