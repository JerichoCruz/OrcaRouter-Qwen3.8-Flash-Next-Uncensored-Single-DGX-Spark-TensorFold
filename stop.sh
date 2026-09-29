#!/usr/bin/env bash
# Stop the server that ./start.sh started: give it STOP_TIMEOUT seconds (default 30) to finish, then remove the container.
# Usage: ./stop.sh      Env: CONTAINER_NAME (see scripts/config.sh), STOP_TIMEOUT
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "No container named $CONTAINER_NAME: nothing to stop"
  exit 0
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == true ]]; then
  log "Stopping $CONTAINER_NAME (up to ${STOP_TIMEOUT:-30}s)"
  docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null
fi
docker rm -f "$CONTAINER_NAME" >/dev/null
log "Stopped and removed $CONTAINER_NAME; its GPU memory is free again"
