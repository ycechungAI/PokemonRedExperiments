#!/usr/bin/env bash
# macOS launcher for the pokemonred_puffer training engine.
#
# The engine forces the "fork" multiprocessing start method (train.py), which
# crashes on macOS once torch/PyBoy/SDL have initialized the ObjC/Swift runtime
# in the parent ("+[Swift.__SharedStringStorage initialize] ... Crashing
# instead"). This env var disables that crash-on-fork guard; workers only run
# headless Python, so it is safe here.
#
# Usage:
#   ./train_macos.sh                 # = train
#   ./train_macos.sh train --debug   # quick CPU smoke test
#   ./train_macos.sh autotune
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="${REPO_DIR}/../pokemonred_puffer"
PYTHON="${REPO_DIR}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: ${PYTHON} not found — create the venv and 'pip install -e ../pokemonred_puffer' first" >&2
    exit 1
fi
if [[ ! -f "${ENGINE_DIR}/red.gb" ]]; then
    echo "error: ${ENGINE_DIR}/red.gb not found — the engine needs the ROM in its own directory" >&2
    exit 1
fi

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
cd "$ENGINE_DIR"
exec "$PYTHON" -m pokemonred_puffer.train "${@:-train}"
