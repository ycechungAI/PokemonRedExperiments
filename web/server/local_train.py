"""Launch the pokemonred_puffer engine wired to the local telemetry server.

The engine's ``--config`` flag *replaces* the base config rather than merging,
and its StreamWrapper defaults to the public ``wss://transdimensional.xyz``
endpoint. So to point telemetry at our local FastAPI server we load the engine's
own ``config.yaml``, overlay a few local-only values (the WebSocket address, plus
optional device / env-count overrides), write the merged result to a run config,
and launch the trainer against it — leaving the engine repo untouched, per
CLAUDE.md.

Usage (from this repo root):
    .venv/bin/python -m web.server.local_train train
    .venv/bin/python -m web.server.local_train train --debug
    .venv/bin/python -m web.server.local_train autotune --num-envs 8

Anything after the engine subcommand is forwarded to the engine CLI verbatim.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

REPO_DIR = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_DIR.parent / "pokemonred_puffer"
ENGINE_CONFIG = ENGINE_DIR / "config.yaml"
PYTHON = REPO_DIR / ".venv" / "bin" / "python"

# Where the merged run config is written (gitignored).
RUN_CONFIG = REPO_DIR / "web" / "server" / ".run_config.yaml"

STREAM_WRAPPER_KEY = "stream_wrapper.StreamWrapper"

# config.train keys the web form is allowed to override.
TRAIN_OVERRIDE_KEYS = ("device", "num_envs", "num_workers", "env_batch_size", "total_timesteps")


def build_local_config(ws_address: str, train_overrides: Optional[dict[str, Any]] = None) -> Path:
    """Load the engine config, overlay local values, and write a run config.

    ``train_overrides`` maps ``config.train`` keys (a subset of
    ``TRAIN_OVERRIDE_KEYS``) to values; None values are ignored. Returns the
    path to the written merged config.
    """
    if not ENGINE_CONFIG.exists():
        sys.exit(f"error: engine config not found at {ENGINE_CONFIG}")

    config = OmegaConf.load(ENGINE_CONFIG)

    # Point every StreamWrapper (across all wrapper sets) at the local server,
    # and upload more often than the public-map default (500 steps) so the
    # dashboard shows first dots quickly — on CPU each env only steps ~10x/s,
    # so 500 steps meant several minutes of blank map after boot.
    for wrapper_set in config.wrappers.values():
        for entry in wrapper_set:
            if STREAM_WRAPPER_KEY in entry:
                entry[STREAM_WRAPPER_KEY]["ws_address"] = ws_address
                entry[STREAM_WRAPPER_KEY]["upload_interval"] = 150

    for key, value in (train_overrides or {}).items():
        if value is None:
            continue
        if key not in TRAIN_OVERRIDE_KEYS:
            raise ValueError(f"unsupported train override: {key}")
        config.train[key] = value

    RUN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, RUN_CONFIG)
    return RUN_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        help="engine subcommand: train | autotune | debug | evaluate",
    )
    parser.add_argument(
        "--ws-address",
        default="ws://localhost:8000/broadcast",
        help="StreamWrapper target (default: the local telemetry server)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override train.device (e.g. cpu, mps). Defaults to whatever config.yaml has.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override train.num_envs. Defaults to config.yaml.",
    )
    args, engine_args = parser.parse_known_args()

    if not PYTHON.exists():
        sys.exit(f"error: {PYTHON} not found — create the venv and install the engine first")

    run_config = build_local_config(
        args.ws_address, {"device": args.device, "num_envs": args.num_envs}
    )
    print(f"[local_train] merged run config: {run_config}", file=sys.stderr)
    print(f"[local_train] streaming telemetry to: {args.ws_address}", file=sys.stderr)

    env = dict(os.environ)
    # macOS: the engine forces the fork start method, unsafe once torch/PyBoy/SDL
    # have initialized the ObjC runtime. See train_macos.sh / README macOS section.
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    cmd = [
        str(PYTHON),
        "-m",
        "pokemonred_puffer.train",
        args.command,
        "--config",
        str(run_config),
        *engine_args,
    ]
    # Engine commands must run from the engine directory (relative red.gb, states).
    proc = subprocess.run(cmd, cwd=ENGINE_DIR, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
