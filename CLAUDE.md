# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

Train RL agents to play Pokemon Red (PyBoy emulator + PPO). This fork's active goal is a **browser-based training experience**: the training engine is the sibling repo `../pokemonred_puffer`, and this repo hosts the experiments plus a web dashboard (`web/`) to launch, control, and watch training from the browser. Read [Plan.md](Plan.md) for the architecture and [ToDo.md](ToDo.md) for current status before starting work.

## Repo layout

- `baselines/` — original SB3 training scripts (`red_gym_env.py`, `run_baseline_parallel_fast.py`). Legacy; keep working but don't extend.
- `v2/` — improved SB3 version (`baseline_fast_v2.py`, `red_gym_env_v2.py`). Also legacy relative to the puffer engine.
- `v2/stream_agent_wrapper.py` — WebSocket coordinate broadcaster; its JSON protocol (`{metadata, coords: [[x, y, map_n], ...]}`) is the telemetry contract the web UI ingests.
- `visualization/`, `VisualizeProgress.ipynb` — offline map/progress visualization.
- `web/` — (being built) FastAPI control/telemetry server + browser dashboard. New work goes here.
- `../pokemonred_puffer` — training engine (separate repo, also a working directory). Library-style: policies/rewards/wrappers are plug-ins selected in its `config.yaml`. Treat it as a dependency — prefer config overlays and wrappers in *this* repo over editing engine code.

## Key commands

```sh
# Engine install (Python 3.10–3.11 only; pufferlib is pinned to a fork)
pip install -e ../pokemonred_puffer

# Engine training (run from ../pokemonred_puffer; needs red.gb there)
python -m pokemonred_puffer.train autotune   # find num_envs first
python -m pokemonred_puffer.train train
python -m pokemonred_puffer.train --config config.yaml --debug  # quick test

# On macOS, launch via the wrapper instead (sets the fork-safety env var):
./train_macos.sh train          # ./train_macos.sh {train|autotune|...} [flags]

# Legacy SB3 training (from v2/; needs ../PokemonRed.gb)
python baseline_fast_v2.py
```

## Constraints & gotchas

- **ROM**: `PokemonRed.gb` here / `red.gb` in the engine repo, sha1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`. Never commit the ROM or suggest downloading it; users supply their own legally.
- **Python**: engine requires `>=3.10,<3.12`. macOS needs SDL and has its own requirements file for v2 (`v2/macos_requirements.txt`).
- **Training cannot run in-browser.** Anything "browser" means the dashboard/telemetry layer or the ONNX inference demo — do not attempt Pyodide/WASM training.
- **StreamWrapper default endpoint** is the public `wss://transdimensional.xyz/broadcast` (shared community map). Local web-UI work must repoint it to the local server via config, not by editing the public default.
- **macOS training launch**: the engine forces the `fork` start method on darwin (`train.py`), which avoids the old spawn/pickle error but crashes worker processes once torch/PyBoy/SDL have initialized the ObjC runtime (`+[Swift.__SharedStringStorage initialize] ... Crashing instead`; dashboard stays at SPS 0). Fix: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` before launching. Use **`./train_macos.sh`** (repo root), which sets that env var and runs the engine — pass the same subcommands/flags (`./train_macos.sh train`, `./train_macos.sh train --debug`). Separately, the engine's `config.yaml` must set `train.device: cpu` (or `mps`) on Mac; `cuda` raises `AssertionError: Torch not compiled with CUDA enabled`. Verified 2026-07-05 with Python 3.11.15 (SPS ~27 on cpu via `--debug`).
- Memory addresses for game state live in `baselines/memory_addresses.py` and in the engine's `environment.py`; coordinate mapping in `global_map.py` + `map_data.json` (duplicated in both repos — keep in sync if touched).
- Session output dirs (`session_*`, `runs/`) and checkpoints are large; never commit them.

## Conventions

- Web server code: FastAPI, type-hinted, no framework magic; frontend: Vite + TypeScript.
- Engine-side customization: add wrappers/rewards as new classes referenced from `config.yaml` (see engine README "Making Changes") rather than modifying its core files.
- Update [ToDo.md](ToDo.md) checkboxes as tasks complete; substantive design changes go in [Plan.md](Plan.md).
