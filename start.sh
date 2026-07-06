#!/usr/bin/env bash
# One-command launcher for the browser training UI.
#
# Starts the local telemetry/control server, opens the dashboard in your
# browser, and kicks off a training run at the given instance count — no
# manual curl/uvicorn steps needed.
#
# Usage:
#   ./start.sh              # 4 instances (default)
#   ./start.sh 8            # 8 instances (= 8 workers x 12 envs/worker)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${REPO_DIR}/.venv/bin/python"
UVICORN="${REPO_DIR}/.venv/bin/uvicorn"
PORT=8000
HOST="127.0.0.1"
INSTANCES="${1:-4}"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: ${PYTHON} not found — create the venv and 'pip install -e ../pokemonred_puffer' first" >&2
    exit 1
fi

LOG_FILE="${REPO_DIR}/web/server/.start_server.log"
mkdir -p "$(dirname "$LOG_FILE")"

if curl -sf "http://${HOST}:${PORT}/api/status" > /dev/null 2>&1; then
    echo "server already running at http://${HOST}:${PORT}"
else
    echo "starting telemetry server on http://${HOST}:${PORT} (log: ${LOG_FILE})"
    cd "$REPO_DIR"
    nohup "$UVICORN" web.server.app:app --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
    disown

    for _ in $(seq 1 60); do
        if curl -sf "http://${HOST}:${PORT}/api/status" > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
    if ! curl -sf "http://${HOST}:${PORT}/api/status" > /dev/null 2>&1; then
        echo "error: server did not come up in time — check ${LOG_FILE}" >&2
        exit 1
    fi
fi

URL="http://${HOST}:${PORT}"
if command -v open > /dev/null 2>&1; then
    open "$URL"
elif command -v xdg-open > /dev/null 2>&1; then
    xdg-open "$URL"
else
    echo "open ${URL} in your browser"
fi

echo "starting training: ${INSTANCES} instance(s)"
curl -sf -X POST "http://${HOST}:${PORT}/api/train/start" \
    -H "Content-Type: application/json" \
    -d "{\"instances\": ${INSTANCES}}"
echo
echo "training started — watch progress at ${URL}"
echo "stop it from the browser, or: curl -X POST ${URL}/api/train/stop"
