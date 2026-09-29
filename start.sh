#!/usr/bin/env bash
# Start Ash's quant (Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) with TensorFold on one DGX Spark, end to end:
# checks prerequisites, replaces any previous container, launches `tensorfold serve` on port 8888,
# waits until the OpenAI API answers, then runs a smoke test.
#
# Usage: ./start.sh [extra tensorfold serve args]
#   ./start.sh                         # scripts/config.sh defaults: 4 streams x 262,144 tokens, int8 KV, --ple-on-ssd
#   ./start.sh --parallel 8 --context 172000
#   PARALLEL=3 KV_DTYPE=bf16 ./start.sh  # 256k at full KV precision
# Env: TENSORFOLD_* (passed to the server), SERVED_NAME, PORT, HOST, PARALLEL, CONTEXT, KV_DTYPE, PLE_ON_SSD, MTP_DRAFTS, MTP_CONFIDENCE (see scripts/config.sh), CONTAINER_NAME, IMAGE, FOREGROUND=1 (stay attached), WAIT_TIMEOUT (seconds, default 1800),
#      HF_HUB_OFFLINE=0 (let TensorFold reach the Hub; default serves from the local cache only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
EXTRA_ARGS=("$@")
# Defaults from scripts/config.sh, each skipped when the same flag is given on the command line.
has_arg() { [[ " ${EXTRA_ARGS[*]:-} " == *" $1"* ]]; }
DEFAULTS=()
has_arg --name     || DEFAULTS+=(--name "$SERVED_NAME")
has_arg --parallel || DEFAULTS+=(--parallel "$PARALLEL")
has_arg --context  || DEFAULTS+=(--context "$CONTEXT")
has_arg --kv-dtype || DEFAULTS+=(--kv-dtype "$KV_DTYPE")
[[ "$PLE_ON_SSD" == 1 ]] && ! has_arg --ple-on-ssd && DEFAULTS+=(--ple-on-ssd)
has_arg --mtp-drafts     || DEFAULTS+=(--mtp-drafts "$MTP_DRAFTS")
has_arg --mtp-confidence || DEFAULTS+=(--mtp-confidence "$MTP_CONFIDENCE")
EXTRA_ARGS=("${DEFAULTS[@]}" "${EXTRA_ARGS[@]}")

# ---------------------------------------------------------------- prerequisites
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing, run scripts/prepare.sh first"
ls -d "$(model_cache_dir)"/snapshots/*/ >/dev/null 2>&1 || die "$MODEL_ID not in $HF_CACHE, run scripts/prepare.sh first"
mkdir -p "$KERNEL_CACHE"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "Removing previous container $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi

# The weights alone take ~81 GB of the Spark's unified memory.
avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
(( avail_gb >= 95 )) || warn "only ${avail_gb} GB memory available; other GPU workloads may need stopping (docker ps)"

# TensorFold's own switches from the environment (TENSORFOLD_*, e.g. TENSORFOLD_MTP_STOP) reach the server too.
ENV_ARGS=()
while IFS='=' read -r name _; do ENV_ARGS+=(-e "$name"); done < <(env | grep -E '^TENSORFOLD_[A-Z0-9_]+=' || true)

# ---------------------------------------------------------------- launch
log "Starting $CONTAINER_NAME: tensorfold serve $MODEL_ID on $HOST:$PORT ${EXTRA_ARGS[*]:-}"
docker run -d --name "$CONTAINER_NAME" \
  --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN="$(hf_token)" \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "${ENV_ARGS[@]}" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$KERNEL_CACHE":/cache \
  "$IMAGE" \
  tensorfold serve "$MODEL_ID" --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}" >/dev/null

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  trap 'log "Stopping $CONTAINER_NAME"; docker rm -f "$CONTAINER_NAME" >/dev/null' INT TERM
  docker logs -f "$CONTAINER_NAME"
  exit 0
fi

# ---------------------------------------------------------------- wait for ready
log "Waiting for the server (first start compiles CUDA kernels, then loads ~81 GB of weights)..."
docker logs -f "$CONTAINER_NAME" > >(sed 's/^/    /') 2>&1 &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT

URL="http://127.0.0.1:$PORT"
start=$SECONDS
until curl -sf "$URL/v1/models" >/dev/null 2>&1; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != true ]]; then
    kill $LOGS_PID 2>/dev/null || true
    die "container exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")); see: docker logs $CONTAINER_NAME"
  fi
  (( SECONDS - start < WAIT_TIMEOUT )) || { kill $LOGS_PID; die "not ready after ${WAIT_TIMEOUT}s; see: docker logs $CONTAINER_NAME"; }
  sleep 5
done
kill $LOGS_PID 2>/dev/null || true
log "Server ready after $((SECONDS - start))s"

# ---------------------------------------------------------------- smoke test
SERVED=$(curl -s "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
log "Model name: $SERVED"
log "Smoke test: chat completion"
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$(python3 - "$SERVED" <<'EOF'
import json, sys
print(json.dumps({"model": sys.argv[1], "max_tokens": 128,
                  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}]}))
EOF
)" | python3 -c '
import json, sys
r = json.load(sys.stdin)
print(r["choices"][0]["message"]["content"])
print("usage:", r.get("usage"))
' || warn "smoke test failed; the server is still running"

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
cat <<EOF

  Endpoint : http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
  Logs     : docker logs -f $CONTAINER_NAME
  Stop     : docker rm -f $CONTAINER_NAME
EOF
