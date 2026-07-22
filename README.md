# Train RL agents to play Pokemon Red

### 🌐 New! Browser-Based Training with pokemonred_puffer — see [Web Training](#web-training) below
### New 10-19-24! Updated & Simplified V2 Training Script - See V2 below
### New 1-29-24! - [Multiplayer Live Training Broadcast](https://github.com/pwhiddy/pokerl-map-viz/)  🎦 🔴 [View Here](https://pwhiddy.github.io/pokerl-map-viz/)
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py)  

See how in [Training Broadcast](#training-broadcast) section

## Watch the Video on Youtube! 

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)

## Web Training

This fork has a **working browser interface for training**: the fast
[pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer) engine runs natively on
your machine while everything you touch lives in the browser — configure and launch runs,
stop them, watch agents explore a live Kanto map with colored per-agent trails, and track SPS /
losses / reward-component metrics, with no terminal or tensorboard required.

How it fits together:

1. **Engine** — `pokemonred_puffer` (cloned as a sibling directory) trains with
   PufferLib-vectorized PyBoy environments; policies, rewards, and wrappers are plug-ins chosen
   in `config.yaml`.
2. **Telemetry** — its built-in `StreamWrapper` broadcasts agent coordinates over WebSocket (the
   same protocol behind the [community map](https://pwhiddy.github.io/pokerl-map-viz/)); a local
   FastAPI server (`web/server/app.py`) ingests it on `/broadcast` and fans it out to browser
   tabs on `/live`, with a replay buffer so a tab that connects mid-run still sees trails.
3. **Run control** — the same server can start/stop the engine as a subprocess
   (`web/server/runner.py`), so there's no need to open a second terminal: a "Train"/"Stop"
   button in the header drives the whole lifecycle, with a live log tail and a SQLite-backed run
   history.
4. **Metrics** — a small engine patch pushes SPS, PPO losses, and reward-component stats to the
   same local server once per epoch (no wandb account or tensorboard needed), rendered as a
   numeric panel + SPS sparkline in a drawer.
5. **Stretch** — a fully in-browser demo mode (ONNX policy + WASM Game Boy) for watching
   pretrained agents with zero install. Training itself always runs natively — PyTorch and PyBoy
   don't run in a browser tab.

Read the full architecture in **[Plan.md](Plan.md)**, track progress in **[ToDo.md](ToDo.md)**,
and see **[CLAUDE.md](CLAUDE.md)** for contributor/agent guidance.

### Quickstart

```sh
git clone https://github.com/drubinstein/pokemonred_puffer ../pokemonred_puffer
pip install -e ../pokemonred_puffer   # Python 3.10–3.11
# copy your legally obtained ROM to ../pokemonred_puffer/red.gb

python -m venv .venv && .venv/bin/pip install -e ../pokemonred_puffer fastapi uvicorn websockets
.venv/bin/uvicorn web.server.app:app --port 8000
```

Open `http://localhost:8000` — everything from here is in the browser:

- **config** drawer — device, an *instances* slider (parallel PyBoy workers; the hint under the
  slider tells you whether a given count still leaves the machine usable for other work),
  total timesteps, wrapper/reward set, and a debug toggle for a fast single-env smoke test.
- **Train / Stop** button — launches the engine as a subprocess wired to this server and stops it
  cleanly (graceful `SIGTERM`, escalating to `SIGKILL` if it doesn't exit in time — no orphaned
  worker processes either way).
- **log** drawer — tails the trainer's stdout/stderr live.
- **runs** drawer — history of past runs (params, status, checkpoint dir) from a local SQLite
  registry.
- **metrics** drawer — SPS, agent steps, epoch, a sparkline, and the full PPO loss / reward-stat
  tables, refreshed every few seconds.
- The header itself flags a **degraded run** (a dead worker process, or telemetry gone quiet far
  longer than a normal PPO train-phase pause) instead of silently sitting at "running" forever.

On CPU, expect the map to stay blank for anywhere from tens of seconds to a few minutes after
pressing Train — every env boots its own PyBoy emulator, and the header shows a "warming up"
state during that window rather than looking stuck. It's also normal for the map to go quiet for
a few minutes at a time once training starts: PPO alternates between stepping envs and running a
training pass on the collected batch, and envs pause during the latter.

### macOS

Two macOS-specific things are needed (already applied if you're using this repo's engine
checkout, or run `./train_macos.sh` for the plain CLI path — see below):

1. **Spawn-safe workers.** macOS uses the `spawn` multiprocessing start method, and forking
   instead (even with `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`) intermittently kills worker
   processes silently, leaving the run stuck at `SPS 0`. Our engine checkout patches
   `train.py` so the env creator is picklable and spawn just works.
2. **Device.** There's no CUDA on a Mac, so set `train.device: cpu` (or `mps`) in the
   engine's `config.yaml`; `cuda` raises `AssertionError: Torch not compiled with CUDA enabled`.

If you'd rather drive the engine from the CLI instead of the browser, a wrapper script in this
repo runs it for you:

```sh
./train_macos.sh train          # ./train_macos.sh {train|autotune|...} [flags]
./train_macos.sh train --debug  # quick single-env CPU smoke test
```

Note: on CPU, startup is slow — every env boots its own PyBoy emulator and `torch.compile`
(`compile: True` in `config.yaml`) adds a one-time compile that can stall the first train step
for minutes; this repo's config overlay sets `compile: False` for that reason. Run `autotune`
first to pick a sane `num_envs` for raw env-stepping throughput — but note that on a shared
machine (the one you're also using for other work), RAM is usually the tighter constraint than
CPU: a 65536-sample GPU-scale PPO batch can use more memory than all the env workers combined
and push the machine into swap. The web UI's launch form already applies a CPU-scale batch size
and caps the trainer's thread count so it shares the machine reasonably; if you're driving the
engine directly via CLI/`config.yaml`, consider similarly reducing `train.batch_size` /
`train.minibatch_size` from their GPU-box defaults and capping `OMP_NUM_THREADS`.

## Running the Pretrained Model Interactively 🎮  
🐍 Python 3.10+ is recommended. Other versions may work but have not been tested.   
You also need to install ffmpeg and have it available in the command line.

### Windows Setup
Refer to this [Windows Setup Guide](windows-setup-guide.md)

### For AMD GPUs
Follow this [guide to install pytorch with ROCm support](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html)

### Linux / MacOS

V2 is now recommended over the original version. You may follow all steps below but replace `baselines` with `v2`.

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 
2. Move into the `baselines/` directory:  
 ```cd baselines```  
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.
For V2 MacOS users should use ```macos_requirements.txt``` instead of ```requirements.txt```
4. Run:  
```python run_pretrained_interactive.py```
  
Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

## Training the Model 🏋️ 

<img src="/assets/grid.png?raw=true" height="156">

### pokemonred_puffer (recommended)

The fastest way to train is the [pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer) engine — see [Web Training](#web-training) above.

### V2

- Trains faster and with less memory
- Reaches Cerulean
- Streams to map by default
- Other improvements

Replaces the frame KNN with a coordinate based exploration reward, as well as some other tweaks.
1. Previous steps but in the `v2` directory instead of `baselines`
2. Run:
```python baseline_fast_v2.py```

## Tracking Training Progress 📈

### Training Broadcast
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py) on your environment like this:
```python
env = StreamWrapper(
            env, 
            stream_metadata = { # All of this is part is optional
                "user": "super-cool-user", # choose your own username
                "env_id": id, # environment identifier
                "color": "#0033ff", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
```

Hack on the broadcast viewing client or set up your own local stream with this repo:  
  
https://github.com/pwhiddy/pokerl-map-viz/

### Local Metrics
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

## Static Visualization 🐜
Map visualization code can be found in `visualization/` directory.

## Follow up work  
 
Check out our follow up projects & papers!  
  
### [Pokemon Red via Reinforcement Learning 🔗](https://arxiv.org/abs/2502.19920)
```  
  @misc{pleines2025pokemon,
    title={Pokemon Red via Reinforcement Learning},
    author={Marco Pleines and Daniel Addis and David Rubinstein and Frank Zimmer and Mike Preuss and Peter Whidden},
    year={2025},
    eprint={2502.19920},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
  }
```
### [Pokemon RL Edition 🔗](https://drubinstein.github.io/pokerl/)
### [PokeGym 🔗](https://github.com/PufferAI/pokegym)

## Supporting Libraries
Check out these awesome projects!
### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>
