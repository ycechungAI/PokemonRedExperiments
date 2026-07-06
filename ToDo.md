# ToDo — Browser Training UI (see [Plan.md](Plan.md))

## Phase 0 — Foundation
- [x] Verify Python 3.10/3.11 venv and `pip install -e ../pokemonred_puffer` — done 2026-07-05: Python 3.11.15 (pyenv) venv at `.venv/`, pufferlib 1.0.1 fork built OK, pyboy 2.7.0, torch 2.12.1. Note: the editable install doesn't expose the package (upstream `packages.find` config), so all engine commands must run with `cwd=../pokemonred_puffer`.
- [x] Place/verify ROM: `red.gb` (sha1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`) copied from `PokemonRed.gb` into `../pokemonred_puffer`
- [x] Short training run confirmed end to end (4 envs, serial, CPU): 4.1k agent steps, 4 PPO epochs, ~340 SPS, 4.3M-param model, clean exit.
- [x] Fix multiprocessing on macOS — **final fix 2026-07-05 (evening): picklable env creator + spawn.** The fork-based approaches (forcing `fork` on darwin + `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`) were a band-aid and still failed *intermittently*: forked workers died instantly from a signal (zombie `<defunct>` children, no traceback, no crash report — the env var suppresses the ObjC abort but forking after torch/SDL/Metal load is fundamentally unreliable), while pufferlib's main process spin-polled its semaphores at 100% CPU forever, looking like a hung run at SPS 0. Real fix in engine `train.py`: `make_env_creator` now returns a `functools.partial` of a module-level `_create_env` (classes resolved from string paths inside the worker), which pickles by reference — so the default macOS `spawn` start method just works and the fork-safety env var is no longer needed (kept harmlessly by launchers). Verified: 24 envs / 2 spawned workers, batches flowing to the map in under a minute. Still required on Mac: `train.device: cpu` and `early_stop` disabled (wall-clock limits assume GPU speed).
- [x] Ran `autotune` on this M4 (2026-07-05, idle machine): env-stepping throughput peaks at ~4,370 SPS with **num_envs: 120 / num_workers: 10** (env_batch_size 36 is valid: 36 % (120/10) == 0). Higher env counts regress (180→3.1k, 360→2.9k); throughput scales with workers up to the 10-core count. (An earlier contended run peaked at 90 envs / ~3,571 SPS; on an idle machine 120 wins by ~6%.) Applied to engine `config.yaml` (stock 288/24 was a GPU-box config). Note: autotune SPS is raw env stepping (empty wrappers, no NN); real training SPS is lower but the optimal counts hold. Optional cpu-vs-mps A/B still open (mps likely marginal — emulator-bound, tiny 4.3M net).
- [x] Scaffold `web/server` (FastAPI + uvicorn) and `web/frontend` — server (`app.py`) + vanilla-TS frontend built and running.
- [x] Decide frontend framework — went with **no-framework vanilla JS + Canvas** for the live map (zero build step, direct canvas control). Revisit React/Svelte + uPlot if the metrics/run-control views (Phase 2/3) need it.

## Phase 1 — Telemetry pipeline
- [x] FastAPI WebSocket `/broadcast` ingest endpoint (accepts existing StreamWrapper JSON: `{metadata, coords}`) — verified 2026-07-05: 22 batches ingested from a live CPU run.
- [x] `/live` fan-out WebSocket for browser subscribers, with a replay ring buffer so late joiners see trails — verified: subscriber received 500-coord batches.
- [x] Config overlay that sets `stream_wrapper.StreamWrapper` `ws_address` → `ws://localhost:8000/broadcast` — done via `web/server/local_train.py` (loads engine config, overlays ws_address on every wrapper set + optional device/num_envs, writes a merged run config, launches with the fork-safety env var). Engine repo left untouched.
- [x] Frontend live map: render `kanto_map_dsv.png`, convert `(x, y, map_n)` → global pixels via `map_data.json` — `app.js` mapping verified to match engine `global_map.py` (gx = x + map_x + PAD, gy = y + map_y + PAD).
- [x] Per-env colored trails with fade; zoom/pan — implemented in `app.js` (TTL fade, hashed per-agent colors, cursor-anchored zoom).
- [x] Milestone: watch a live local training run in the browser — closed 2026-07-05: 24-env run started via `POST /api/train/start`, per-agent colored trails visually confirmed rendering on the Kanto map canvas (12k points / 72 batches, header showing live status + run uptime).
- [x] Env-ID collision fixed 2026-07-05: spawned workers each re-import RedGymEnv and get their own shared-memory env_id counter, so bare env_ids repeat across workers and the map merged different envs into one "agent" (dots appearing off any walkable path). StreamWrapper metadata now sends `"{pid}-{env_id}"`, giving every env a unique trail/color.
- [x] **Run health watchdog** — done 2026-07-05: `RunManager.health()` classifies the active run
      as ok / warming / degraded from two signals: zombie (defunct) children of the trainer pid
      (psutil — a dead spawn worker lingers unreaped while pufferlib keeps polling) and telemetry
      silence (`last_batch_at` tracked in `app.py` on every `/broadcast` ingest, compared against
      the run's own start time so a previous run's batches don't count). Grace windows are
      deliberately generous — 15 min for warmup (8 instances measured ~8 min to first batch on CPU)
      and 10 min for stalls (envs pause ~5 min during each PPO train phase) — so healthy quiet
      spells don't false-alarm; this exact pattern (trainer at 217% CPU, workers idle, 4-min
      silence, then batches resume) was observed healthy on a 218-min run. Surfaced via
      `GET /api/train/status` (`health`, `health_reason`) and shown orange in the header
      (`⚠ no data for N min…` / `N worker process(es) died…`). The watchdog originally motivated
      by: fork-unsafe workers (fixed via spawn), then a PyBoy 2.7 empty-bag slice crash (guarded
      in the engine checkout). Note: the running server picks this up on its next restart.

### How to run the live pipeline

Terminal 1 — telemetry server + dashboard at http://localhost:8000:
```sh
.venv/bin/uvicorn web.server.app:app --port 8000
```
Terminal 2 — training that streams to the local server (add `--debug` for a quick single-env run):
```sh
.venv/bin/python -m web.server.local_train train
```

## Phase 2 — Run control
- [x] **One-click Train / Stop from the browser** — `web/server/runner.py` `RunManager` spawns the
      trainer subprocess (config overlay + fork-safety env var, `cwd=../pokemonred_puffer`) and
      stops it via SIGTERM→SIGKILL on the process group (kills forked workers, no orphans).
      Endpoints: `POST /api/train/start`, `POST /api/train/stop`, `GET /api/train/status`.
      Frontend: Train/Stop button + status in the header. Verified end-to-end 2026-07-05
      (start → running pid, graceful stop returncode -15, 0 orphans).
- [x] Live stdout/stderr log tail in the UI — captured per run under `web/server/.runs/<id>/train.log`,
      surfaced via `status().log` and a toggle-able log panel in the dashboard.
- [x] SQLite run registry (`web/server/registry.py`): one row per run — id, command, params,
      status (running/stopped/exited/failed), started/ended, returncode, log path, and the engine
      checkpoint dir (`exp_id`, discovered best-effort by watching `../pokemonred_puffer/runs/`).
      `RunManager` records start/stop and reconciles natural exits. Endpoints `GET /api/runs`,
      `GET /api/runs/{id}`; a **runs history drawer** in the UI. Verified: run recorded while
      running, persisted as `stopped` with returncode after stop.
      (Still single active run — concurrent/queued runs would build on this.)
- [x] Config form UI — a **launch-settings drawer** (device, an **instances slider 1–12
      defaulting to 12**, total_timesteps, wrappers set, reward set, debug), pre-filled from
      `GET /api/config/defaults`. The instances slider drives num_workers and `derive_env_layout`
      computes a *valid* env trio (num_envs = instances×12, env_batch_size = num_envs) that always
      satisfies the engine's divisibility rules (num_envs % workers == 0, batch % (envs/worker) == 0)
      — a free worker count against the fixed 120/36 would otherwise be invalid for most values.
      Values flow through `build_local_config` (train overrides) + CLI args (`-w`/`-r`). Verified for
      all 1–12 and end-to-end (instances=2 → 24 envs/2 workers/batch 24, graceful stop, no orphans).
      (Curated high-value knobs, not every env flag / reward weight — those can be added later.)
- [ ] Resume-from-checkpoint — **blocked by engine design**: `setup()` generates a fresh
      `exp_id` (uuid) every run and `train()` never loads a prior model (only the `evaluate`/rollout
      path takes `model_path`). True resume needs an engine patch (reuse a run's `exp_id` dir + load
      its `model_*.pt` into the policy), which conflicts with keeping the engine untouched. Deferred.
- [x] Milestone: **full train lifecycle without a terminal** — configure, start, watch (map + log),
      stop, and review history all from the page.

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
