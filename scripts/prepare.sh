#!/usr/bin/env bash
# Prepare everything needed to serve Ash's quant (Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) with TensorFold
# on one DGX Spark:
#   1. preflight checks (docker, GPU runtime, disk space)
#   2. pull NVIDIA's PyTorch container and bake TensorFold into a local image
#   3. download the checkpoint into the Hugging Face cache (~113 GB, resumable)
#   4. verify the checkpoint with `tensorfold info`
# Safe to re-run: every step skips work that is already done. Pass --rebuild to rebuild the image.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."     # the repository root
source ./scripts/config.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$KERNEL_CACHE"
exec 9>"$KERNEL_CACHE/.prepare.lock"
flock -n 9 || die "another prepare.sh is already running (it holds the download locks); wait for it or stop it: pgrep -af prepare.sh"
log "Preflight checks"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
command -v nvidia-smi >/dev/null && nvidia-smi -L || warn "nvidia-smi not found on host"
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime; --gpus all may fail"

mkdir -p "$HF_CACHE/hub" "$KERNEL_CACHE/torch_extensions" "$KERNEL_CACHE/triton"

if [[ ! -d "$(model_cache_dir)/snapshots" ]]; then
  free_gb=$(df -BG --output=avail "$HF_CACHE" | tail -1 | tr -dc '0-9')
  (( free_gb >= MIN_FREE_GB )) || die "only ${free_gb} GB free under $HF_CACHE, need ~${MIN_FREE_GB} GB"
  log "Disk: ${free_gb} GB free under $HF_CACHE"
fi

# ---------------------------------------------------------------- 2. image
# Local fixes in ./patches (unified diffs against site-packages, applied with patch -p0) are baked into the image.
# The image is rebuilt when they change; the TensorFold install layer stays cached, so that takes seconds.
mkdir -p patches
PATCHES_HASH=$(cat patches/*.patch 2>/dev/null | sha256sum | cut -c1-12)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
if [[ $REBUILD -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1 || [[ "$built_hash" != "$PATCHES_HASH" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && [[ $REBUILD -eq 0 ]] || { log "Pulling base image $BASE_IMAGE"; docker pull "$BASE_IMAGE"; }

  log "Building $IMAGE (TensorFold $TF_VERSION, patches $PATCHES_HASH: $(ls patches/*.patch 2>/dev/null | xargs -rn1 basename | paste -sd' ' || true))"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker build "${nocache[@]}" -t "$IMAGE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_SPEC="git+${TF_REPO}@${TF_VERSION}" \
    --build-arg PATCHES_HASH="$PATCHES_HASH" \
    -f - patches <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG TF_SPEC
RUN pip install --no-cache-dir --upgrade "${TF_SPEC}" && tensorfold --version
COPY . /opt/tf-patches
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python -c "import tensorfold.cuda.reply_text"
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH}
ENV HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
DOCKERFILE
else
  log "Image $IMAGE already built with patches $PATCHES_HASH (use --rebuild to force)"
fi
docker run --rm --gpus all "$IMAGE" tensorfold --version 2>/dev/null | tail -1

# Run a command inside the image with the HF cache and kernel cache mounted.
tf_run() {
  docker run --rm --gpus all --ipc=host --network host \
    -e HF_TOKEN="$(hf_token)" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$KERNEL_CACHE":/cache \
    "$IMAGE" "$@"
}

# ---------------------------------------------------------------- 3. download
log "Downloading $MODEL_ID into $HF_CACHE/hub (resumes if interrupted)"
if command -v hf >/dev/null; then
  # Host CLI: resumable, parallel, writes the standard HF cache layout.
  hf download "$MODEL_ID" --cache-dir "$HF_CACHE/hub"
else
  warn "host 'hf' CLI not found, downloading from inside the container"
fi
# `tensorfold pull` is the documented path; with the files already cached it only checks/completes them.
tf_run tensorfold pull "$MODEL_ID"

snapshot=$(ls -d "$(model_cache_dir)"/snapshots/*/ 2>/dev/null | head -1)
[[ -n "$snapshot" ]] || die "no snapshot found under $(model_cache_dir)"
log "Checkpoint: $snapshot ($(du -shL "$snapshot" | cut -f1))"

# ---------------------------------------------------------------- 4. verify
log "Verifying checkpoint with tensorfold info"
tf_run tensorfold info "$MODEL_ID"

log "Done. Start the server with ./start.sh (port $PORT)."
log "The first start compiles CUDA kernels for GB10 (a few minutes); they are cached in $KERNEL_CACHE."
