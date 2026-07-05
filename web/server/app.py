"""Telemetry server for browser-based Pokemon Red RL training.

Ingests StreamWrapper coordinate broadcasts from training environments on
/broadcast and fans them out to browser dashboards subscribed on /live.
Serves the frontend and the Kanto map assets from the engine repo.

Run from the repo root:
    .venv/bin/uvicorn web.server.app:app --port 8000
"""

import asyncio
import json
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_DIR / "web" / "frontend"
ENGINE_PKG = REPO_DIR.parent / "pokemonred_puffer" / "pokemonred_puffer"

# Fall back to this repo's copies if the engine repo is not checked out.
MAP_IMAGE = ENGINE_PKG / "kanto_map_dsv.png"
MAP_DATA = ENGINE_PKG / "map_data.json"
if not MAP_DATA.exists():
    MAP_DATA = REPO_DIR / "v2" / "map_data.json"

# Ring buffer of recent coordinate batches so a browser that connects
# mid-run immediately sees trails instead of a blank map.
REPLAY_SIZE = 500

app = FastAPI(title="pokerl-web telemetry")

live_clients: set[WebSocket] = set()
replay: deque[str] = deque(maxlen=REPLAY_SIZE)


async def fan_out(message: str) -> None:
    dead = []
    for client in live_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead.append(client)
    for client in dead:
        live_clients.discard(client)


@app.websocket("/broadcast")
async def broadcast(ws: WebSocket) -> None:
    """Ingest endpoint for StreamWrapper (one connection per environment)."""
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            message = json.dumps({"ts": time.time(), **payload})
            replay.append(message)
            await fan_out(message)
    except WebSocketDisconnect:
        pass


@app.websocket("/live")
async def live(ws: WebSocket) -> None:
    """Fan-out endpoint for browser dashboards."""
    await ws.accept()
    for message in list(replay):
        await ws.send_text(message)
    live_clients.add(ws)
    try:
        while True:
            # Browsers don't send anything; this keeps the socket open and
            # detects disconnects.
            await ws.receive_text()
    except WebSocketDisconnect:
        live_clients.discard(ws)


@app.get("/api/status")
async def status() -> dict:
    return {"live_clients": len(live_clients), "replay_batches": len(replay)}


@app.get("/assets/kanto_map.png")
async def kanto_map() -> FileResponse:
    return FileResponse(MAP_IMAGE)


@app.get("/api/map-data")
async def map_data() -> FileResponse:
    return FileResponse(MAP_DATA)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
