import asyncio
import aioredis
import json
from aiohttp import web

async def extract_features(obs):
    # stateless: compute features from one observation
    return {"centroid": (obs["x"], obs["y"]), "speed": obs.get("v", 0.0)}

async def update_track(redis, key, features):
    # stateful: merge into per-key track history (bounded)
    hist_key = f"track:{key}"
    # push recent feature snapshot (trim to last N)
    await redis.lpush(hist_key, json.dumps(features))
    await redis.ltrim(hist_key, 0, 9)  # keep last 10 entries
    # maintain last-seen timestamp for eviction diagnostics
    await redis.hset("track_meta", key, features.get("ts", 0))

async def handle_event(request):
    data = await request.json()
    # deterministic routing key (e.g., sensor id or track id)
    key = data.get("id") or f"sensor:{data['sensor']}"
    features = await extract_features(data)           # stateless call
    # update state asynchronously
    await update_track(request.app["redis"], key, features)
    return web.json_response({"key": key, "features": features})

async def init_app():
    app = web.Application()
    app.add_routes([web.post("/event", handle_event)])
    app["redis"] = await aioredis.from_url("redis://localhost")
    return app

if __name__ == "__main__":
    web.run_app(asyncio.run(init_app()), port=8080)
# Run with a local Redis; this minimal service illustrates separation.