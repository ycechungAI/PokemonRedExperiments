# Plan: Browser-Based Training UI for Pokemon Red RL

Goal: use [pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer) (checked out at `../pokemonred_puffer`) as the training engine for this repo's Pokemon Red RL experiments, controlled and observed entirely from the browser through a clean web interface.

## Reality check: what "runs in the browser" means

Training itself cannot execute inside a browser tab. The training stack is PyTorch (CUDA/MPS), PyBoy (SDL + CPython), and PufferLib (multiprocessing vectorization) — none of which run under Pyodide/WASM. So the architecture splits into:

| Piece | Where it runs |
|---|---|
| Training loop (`pokemonred_puffer.train`) | Local machine / remote GPU box |
| Control + telemetry server (FastAPI) | Same machine as training |
| Dashboard UI (map, screens, charts, run control) | **Browser** |
| Pretrained-agent demo (ONNX policy + WASM Game Boy) | **Browser, fully client-side** (stretch goal) |

The browser is the *only* surface the user touches: open `http://localhost:8000`, pick a config, press Train, watch agents explore Kanto live.

## Why pokemonred_puffer as the engine

- It is the maintained follow-up to this repo's `baselines/` and `v2/` scripts (same authors, same env lineage) and trains dramatically faster via PufferLib vectorization.
- It is already structured as a library: policies, rewards, and wrappers are plug-in classes selected in `config.yaml`, so a web server can compose runs without code edits.
- It already ships the hook we need: `pokemonred_puffer/wrappers/stream_wrapper.py` broadcasts agent coordinates over a WebSocket every ~300 steps (same protocol as `v2/stream_agent_wrapper.py` here, which powers [pokerl-map-viz](https://github.com/pwhiddy/pokerl-map-viz)). Repointing its `ws_address` from `wss://transdimensional.xyz/broadcast` to our local server gives us live map telemetry for free.
- Typer CLI (`python -m pokemonred_puffer.train train`) is easy to drive as a managed subprocess, and wandb/tensorboard hooks already exist for metrics.

## Architecture

```
┌────────────────────────── Browser ──────────────────────────┐
│  Dashboard SPA (web/frontend)                                │
│  • Run control (start/stop/pause, config form)               │
│  • Live Kanto map — agent dots over kanto_map_dsv.png        │
│  • Live env screens (throttled JPEG frames)                  │
│  • Reward / event / exploration charts                       │
│  • Run history & checkpoint browser                          │
└───────────────▲ REST (control)  ▲ WebSocket (telemetry) ─────┘
                │                 │
┌───────────────┴─────────────────┴────────────────────────────┐
│  web/server — FastAPI + uvicorn (localhost:8000)              │
│  • POST /runs → spawn `python -m pokemonred_puffer.train`     │
│  • DELETE /runs/{id} → graceful stop, checkpoint saved        │
│  • GET /runs/{id}/metrics → parsed stats                      │
│  • /broadcast (WS, ingest) ← StreamWrapper from each env      │
│  • /live (WS, fan-out) → connected browsers                   │
└───────────────▲───────────────────────────────────────────────┘
                │ subprocess + WS
┌───────────────┴───────────────────────────────────────────────┐
│  Training: pokemonred_puffer (../pokemonred_puffer)            │
│  config.yaml: wrappers → stream_wrapper.StreamWrapper          │
│  ws_address: ws://localhost:8000/broadcast                     │
└────────────────────────────────────────────────────────────────┘
```

### Telemetry protocol (already exists, reuse as-is)

`StreamWrapper` sends JSON: `{"metadata": {user, env_id, color, extra}, "coords": [[x, y, map_n], ...]}`. The server ingests on `/broadcast`, tags with run id, and fans out to `/live` subscribers. Frontend converts `(x, y, map_n)` to global map pixels using `map_data.json` / `global_map.py` (present in both repos) and draws fading dot trails — same technique as pokerl-map-viz, which we can borrow rendering code from.

### Run control

The server owns a run registry (SQLite): id, config snapshot, PID, wandb/tensorboard paths, checkpoint dir, status. Start = write a merged `config.yaml` overlay to the run dir and spawn the trainer with `cwd=../pokemonred_puffer`. Stop = SIGTERM (the trainer checkpoints on exit). Metrics come from the wandb local dir or tensorboard event files, parsed server-side and pushed over `/live` so the dashboard needs no wandb account.

### Frontend

Vite + React (or Svelte — decide at kickoff), TypeScript, no heavy state library. Four views:

1. **Train** — config form generated from `config.yaml` schema (env flags, reward weights, num_envs, total steps), start button, autotune helper.
2. **Live map** — full-Kanto canvas with per-env colored trails, zoom/pan; env screen tiles in a sidebar.
3. **Metrics** — reward components, episode length, badges/events, exploration counts (uPlot or Chart.js).
4. **Runs** — history table, checkpoint download, resume-from-checkpoint.

### Stretch: true in-browser demo mode

Export a trained policy to ONNX, run it with `onnxruntime-web` against a WASM Game Boy core (wasmboy / binjgb) reading the same RAM addresses for observations. Zero-install "watch the agent play in your tab" page, deployable to GitHub Pages (users supply their own ROM file). Inference-only — this never replaces server-side training.

## Phases

**Phase 0 — Foundation (repo plumbing)**
Verify `pip install -e ../pokemonred_puffer` on this machine (Python 3.10/3.11, `red.gb` ROM in place); confirm `train --help`, `autotune`, and a short `--debug` run work. Scaffold `web/server` and `web/frontend`.

**Phase 1 — Telemetry pipeline (see it)**
FastAPI `/broadcast` ingest + `/live` fan-out; config overlay pointing StreamWrapper at localhost; browser map page rendering live coordinate trails over the Kanto map. *Milestone: watch a local training run explore Kanto in the browser.*

**Phase 2 — Run control (drive it)**
Run registry, start/stop/resume endpoints, config form UI, log tail view. *Milestone: full train lifecycle without touching a terminal.*

**Phase 3 — Metrics & screens (understand it)**
Tensorboard/wandb event parsing → charts; throttled env screen frames (add a small frame-publisher wrapper next to StreamWrapper); run comparison. *Milestone: replace tensorboard for day-to-day monitoring.*

**Phase 4 — Polish & stretch**
Auth token for non-localhost use, mobile layout, ONNX in-browser demo, one-command launcher (`python -m web` starts server + opens browser).

## Risks / open questions

- **PufferLib pin**: `pokemonred_puffer` pins a fork (`thatguy11325/PufferLib@1.0`); macOS install may need workarounds (this repo's v2 already documents macOS-specific requirements).
- **Frame streaming volume**: dozens of envs × 144×160 frames must be throttled/subsampled server-side; coords-only is the default, screens opt-in.
- **Metrics source**: wandb offline dir vs tensorboard event files — pick whichever parses more robustly during Phase 3 spike.
- **Where web code lives**: this repo (`web/`) keeps the experiment history and viz assets together; pokemonred_puffer stays an untouched engine dependency (it explicitly invites use as a library).
