from pydantic import BaseModel, Field, ValidationError
import asyncio, json, time, math, logging
# Canonical models
class CanonicalTrack(BaseModel):
    id: str
    x: float; y: float
    vx: float; vy: float
    cov: list  # 4x4 row-major covariance
    provenance: dict

# Simulated legacy message consumer (replace with DDS/ROS2 subscription)
async def legacy_consumer(queue):
    # example legacy payloads
    legacy = {"track_id":"A1","pos":[100.0,5.0],"speed":-12.0,"heading_deg":180}
    await queue.put(legacy)

def translate_legacy(legacy):
    # map legacy sign convention: speed is negative for forward
    speed = -legacy["speed"]
    theta = math.radians(legacy["heading_deg"])
    vx = speed * math.cos(theta); vy = speed * math.sin(theta)
    # default covariance if missing (sensor model)
    sigma_pos = 5.0; sigma_vel = 2.0
    cov = [
        sigma_pos**2,0,0,0,
        0,sigma_pos**2,0,0,
        0,0,sigma_vel**2,0,
        0,0,0,sigma_vel**2
    ]
    return CanonicalTrack(
        id=legacy["track_id"],
        x=legacy["pos"][0], y=legacy["pos"][1],
        vx=vx, vy=vy, cov=cov,
        provenance={"source":"radar_v1","ts":time.time(),"confidence":0.8}
    )

async def acl_loop():
    q = asyncio.Queue()
    await legacy_consumer(q)
    legacy = await q.get()
    try:
        track = translate_legacy(legacy)  # validate via pydantic
    except ValidationError as e:
        logging.error("translation failed: %s", e)
        return
    # publish to canonical bus (here, print JSON)
    print(track.json())

if __name__ == "__main__":
    asyncio.run(acl_loop())