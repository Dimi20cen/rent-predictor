#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="rentpredictor"
IMAGE="rentpredictor:latest"
PORT_BIND="127.0.0.1:8501:8501"
SKIP_PULL="${SKIP_PULL:-0}"

cd "$ROOT_DIR"

if [[ "$SKIP_PULL" != "1" ]]; then
  echo "[deploy] Pulling latest code (ff-only)..."
  git pull --ff-only origin master
fi

echo "[deploy] Building image: $IMAGE"
docker build -t "$IMAGE" .

echo "[deploy] Replacing container: $APP_NAME"
docker rm -f "$APP_NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$APP_NAME" \
  --restart unless-stopped \
  -p "$PORT_BIND" \
  "$IMAGE"

echo "[deploy] Container status"
docker ps --filter "name=^/${APP_NAME}$"

echo "[deploy] Recent logs"
docker logs --tail 60 "$APP_NAME"
