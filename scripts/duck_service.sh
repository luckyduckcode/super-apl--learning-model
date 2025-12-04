#!/bin/bash
# DuckChat Service Launcher (Linux/macOS)
# Starts the Duck Chat API server with external model and auto-reindex
# Usage: ./duck_service.sh [--skip-reindex]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${REPO_ROOT}/deploy/duckchat_engine.json"
ADAPTER="testmylora"
HOST="0.0.0.0"
PORT="5000"

# Parse command-line args
SKIP_REINDEX=""
for arg in "$@"; do
    if [ "$arg" = "--skip-reindex" ]; then
        SKIP_REINDEX="--skip-reindex"
    fi
done

echo ""
echo "========================================"
echo "Duck Chat Service Launcher"
echo "========================================"
echo "Config: $CONFIG"
echo "Adapter: $ADAPTER"
echo "Host/Port: $HOST:$PORT"
echo ""

# Activate venv if present
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "[*] Virtual environment activated"
fi

# Launch bootstrap
cd "$REPO_ROOT"
python scripts/duck_server_bootstrap.py \
  --config "$CONFIG" \
  --adapter "$ADAPTER" \
  --host "$HOST" \
  --port "$PORT" \
  $SKIP_REINDEX

# On exit, display status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[*] Service exited cleanly"
else
    echo "[!] Service exited with error $EXIT_CODE"
fi
exit $EXIT_CODE
