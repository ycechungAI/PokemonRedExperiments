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
from pydantic import BaseModel

from .runner import RunManager

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
runs = RunManager()
# Wall-clock time of the newest ingested batch; feeds the run-health watchdog.
last_batch_at: float | None = None

# Training metrics (SPS, agent steps, losses, reward-component stats), pushed
# by the engine's local metrics sink (see cleanrl_puffer.py's
# _push_local_metrics — an engine patch, this repo has no wandb/tensorboard
# account for the engine's own logging to go to). One snapshot every ~5s;
# history is kept for a simple in-page chart, not full offline analysis.
METRICS_HISTORY_SIZE = 720  # ~1 hour at one point per 5s
latest_metrics: dict | None = None
metrics_history: deque[dict] = deque(maxlen=METRICS_HISTORY_SIZE)


@app.on_event("shutdown")
def _stop_training_on_shutdown() -> None:
    # Don't orphan a training subprocess if the server is stopped.
    if runs.is_running():
        runs.stop()


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
    global last_batch_at
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            last_batch_at = time.time()
            message = json.dumps({"ts": last_batch_at, **payload})
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


# --- training run control (one-click Train / Stop) -------------------------


class StartRequest(BaseModel):
    command: str = "train"
    device: str | None = None
    instances: int | None = None
    num_envs: int | None = None
    num_workers: int | None = None
    env_batch_size: int | None = None
    total_timesteps: int | None = None
    wrappers_name: str | None = None
    reward_name: str | None = None
    debug: bool = False


@app.get("/api/config/defaults")
async def config_defaults() -> dict:
    """Current engine config values + choices, to populate the launch form."""
    from omegaconf import OmegaConf

    from .local_train import ENGINE_CONFIG

    cfg = OmegaConf.load(ENGINE_CONFIG)
    return {
        "device": cfg.train.get("device"),
        "num_envs": cfg.train.get("num_envs"),
        "num_workers": cfg.train.get("num_workers"),
        "total_timesteps": cfg.train.get("total_timesteps"),
        "wrappers": list(cfg.wrappers.keys()),
        "rewards": list(cfg.rewards.keys()),
        "default_wrappers_name": "stream_only",
        "default_reward_name": "baseline.ObjectRewardRequiredEventsMapIdsFieldMoves",
    }


@app.get("/api/train/status")
async def train_status() -> dict:
    return {**runs.status(), **runs.health(last_batch_at)}


@app.post("/api/train/start")
async def train_start(req: StartRequest) -> dict:
    global latest_metrics
    try:
        # Clear stale metrics from a previous run before the new one's first
        # push arrives, so the panel doesn't show old numbers as "current".
        latest_metrics = None
        metrics_history.clear()
        return runs.start(
            command=req.command,
            device=req.device,
            instances=req.instances,
            num_envs=req.num_envs,
            num_workers=req.num_workers,
            env_batch_size=req.env_batch_size,
            total_timesteps=req.total_timesteps,
            wrappers_name=req.wrappers_name,
            reward_name=req.reward_name,
            debug=req.debug,
        )
    except (RuntimeError, ValueError) as exc:
        return {"error": str(exc), **runs.status()}


@app.post("/api/train/stop")
async def train_stop() -> dict:
    return runs.stop()


@app.get("/api/runs")
async def list_runs(limit: int = 50) -> dict:
    return {"runs": runs.registry.list_runs(limit)}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    record = runs.registry.get_run(run_id)
    return record or {"error": "run not found"}


# --- training metrics (SPS, losses, reward stats) --------------------------


class MetricsPayload(BaseModel):
    global_step: int
    epoch: int
    sps: float
    uptime: float
    stats: dict
    losses: dict


@app.post("/api/metrics/ingest")
async def metrics_ingest(payload: MetricsPayload) -> dict:
    global latest_metrics
    entry = {"ts": time.time(), **payload.model_dump()}
    latest_metrics = entry
    metrics_history.append(entry)
    return {"ok": True}


@app.get("/api/metrics/latest")
async def metrics_latest() -> dict:
    return latest_metrics or {}


@app.get("/api/metrics/history")
async def metrics_history_endpoint() -> dict:
    return {"points": list(metrics_history)}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
