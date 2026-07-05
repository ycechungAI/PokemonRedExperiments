# ToDo — Browser Training UI (see [Plan.md](Plan.md))

## Phase 0 — Foundation
- [x] Verify Python 3.10/3.11 venv and `pip install -e ../pokemonred_puffer` — done 2026-07-05: Python 3.11.15 (pyenv) venv at `.venv/`, pufferlib 1.0.1 fork built OK, pyboy 2.7.0, torch 2.12.1. Note: the editable install doesn't expose the package (upstream `packages.find` config), so all engine commands must run with `cwd=../pokemonred_puffer`.
- [x] Place/verify ROM: `red.gb` (sha1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`) copied from `PokemonRed.gb` into `../pokemonred_puffer`
- [x] Short training run confirmed end to end (4 envs, serial, CPU): 4.1k agent steps, 4 PPO epochs, ~340 SPS, 4.3M-param model, clean exit. macOS caveat: `--vectorization multiprocessing` crashes (spawn can't pickle the env-creator closure) — use `serial` on Mac or patch the trainer to fork.
- [ ] Fix/patch multiprocessing on macOS (fork start method or module-level env creator) so training can scale past serial
- [ ] Run `python -m pokemonred_puffer.train autotune` after the multiprocessing fix and record the recommended `num_envs` for this machine
- [ ] Scaffold `web/server` (FastAPI + uvicorn) and `web/frontend` (Vite + TypeScript)
- [ ] Decide frontend framework (React vs Svelte) and charting lib (uPlot vs Chart.js)

## Phase 1 — Telemetry pipeline
- [ ] FastAPI WebSocket `/broadcast` ingest endpoint (accepts existing StreamWrapper JSON: `{metadata, coords}`)
- [ ] `/live` fan-out WebSocket for browser subscribers, tagged by run id
- [ ] Config overlay that sets `stream_wrapper.StreamWrapper` `ws_address` → `ws://localhost:8000/broadcast`
- [ ] Frontend live map: render `kanto_map_dsv.png`, convert `(x, y, map_n)` → global pixels via `map_data.json` (port logic from `global_map.py` / pokerl-map-viz)
- [ ] Per-env colored trails with fade; zoom/pan
- [ ] Milestone: watch a live local training run in the browser

## Phase 2 — Run control
- [ ] Run registry (SQLite): id, config snapshot, PID, status, checkpoint dir
- [ ] `POST /runs` — write merged config to run dir, spawn trainer subprocess (`cwd=../pokemonred_puffer`)
- [ ] `DELETE /runs/{id}` — SIGTERM for graceful checkpoint-and-exit; hard kill fallback
- [ ] Resume-from-checkpoint endpoint
- [ ] Config form UI generated from `config.yaml` (env flags, reward weights, train hyperparams)
- [ ] Live stdout/stderr log tail in the UI
- [ ] Milestone: full train lifecycle without a terminal

## Phase 3 — Metrics & screens
- [ ] Spike: parse wandb offline dir vs tensorboard event files; pick one
- [ ] Push parsed metrics over `/live`; charts for reward components, episode stats, badges/events, exploration
- [ ] Frame-publisher wrapper (throttled JPEG env screens, opt-in per run)
- [ ] Env screen tile grid in the UI
- [ ] Run comparison view (overlay metrics from two runs)
- [ ] Milestone: tensorboard no longer needed day-to-day

## Phase 4 — Polish & stretch
- [ ] One-command launcher: start server, open browser tab
- [ ] Auth token when binding to non-localhost
- [ ] Mobile-friendly layout
- [ ] Stretch: export policy to ONNX; in-browser demo with onnxruntime-web + WASM Game Boy (user-supplied ROM), deployable to GitHub Pages
- [ ] Docs: update README web section with real usage instructions and screenshots
