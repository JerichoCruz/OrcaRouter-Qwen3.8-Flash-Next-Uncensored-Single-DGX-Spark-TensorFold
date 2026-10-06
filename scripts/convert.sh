#!/usr/bin/env bash
# Make MODEL_DIR from SOURCE_ID (orcarouter/Qwen3.8-Flash-Next-Uncensored, BF16) on the DGX Spark:
#   1. preflight checks (the image, a Hugging Face token, disk, no server holding the GPU)
#   2. download SOURCE_ID into the Hugging Face cache (~336 GB, resumable)
#   3. convert it on the GPU with tools/convert_flash_next.py into $MODEL_DIR.partial: MLX affine 4-bit in groups of
#      32, the MTP head and the vision tower kept (~114 GB)
#   4. check the result (tools/check_flash_next.py layout: the tensor names, shapes and dtypes of Vontra's conversion
#      of the official weights; then `tensorfold info`), move it to MODEL_DIR and mark it converted
# scripts/prepare.sh runs this by itself when MODEL_DIR is not converted yet. The source stays in the cache (remove
# it with `hf cache delete` or by deleting its models--* directory under $HF_CACHE/hub once you are happy).
# SOURCE_ID is gated: accept its terms on Hugging Face once, and put a token in $HF_CACHE/token (or HF_TOKEN).
#
# --download-only: step 2 alone, safe next to another running server (e.g. a vLLM one): the download runs in a
#   container capped at DOWNLOAD_MEMORY (default 3g; its page cache counts against the cap, so it cannot crowd out
#   the server's memory), and no GPU is used. Then stop that server and run scripts/convert.sh (or ./start.sh).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."     # the repository root
source ./scripts/config.sh

DOWNLOAD_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --download-only) DOWNLOAD_ONLY=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------- 1. preflight
[[ -n "$MODEL_DIR" ]] || die "MODEL_DIR is empty: set it to the directory to convert into"
# the end replaces MODEL_DIR: only an earlier conversion (it has .converted) or nothing
[[ ! -e "$MODEL_DIR" || -f "$MODEL_DIR/.converted" ]] ||
  die "$MODEL_DIR exists and is not a conversion of this script (no .converted): move it away or set MODEL_DIR"
mkdir -p "$KERNEL_CACHE" "$HF_CACHE/hub" "$(dirname "$MODEL_DIR")"
exec 7>"$KERNEL_CACHE/.convert.lock"
flock -n 7 || die "another convert.sh is already running; wait for it or stop it: pgrep -af convert.sh"
log "Preflight checks"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: run scripts/prepare.sh (it builds or pulls it, then runs this)"
[[ -n "${HF_TOKEN:-}" || -s "$HF_CACHE/token" ]] ||
  die "no Hugging Face token: $SOURCE_ID is gated. Accept its terms at https://huggingface.co/$SOURCE_ID, then put a token in $HF_CACHE/token (or export HF_TOKEN)"
if [[ $DOWNLOAD_ONLY == 0 ]]; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; then
    die "the server ($CONTAINER_NAME) holds the GPU memory: ./stop.sh first"
  fi
  # the conversion holds a few source tensors, their fp32 copies on the GPU and an output shard: ~20 GiB at most
  avail_gib=$(free -g | awk '/^Mem:/ {print $7}')
  (( avail_gib >= CONVERT_MIN_GIB )) || die "only ${avail_gib} GiB memory available, the conversion needs ~${CONVERT_MIN_GIB}: stop the other GPU workloads (docker ps) first, or use --download-only meanwhile"
fi
source_dir() { ls -d "$HF_CACHE/hub/models--${SOURCE_ID//\//--}"/snapshots/*/ 2>/dev/null | head -1; }
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
need=$CONVERT_FREE_GB
[[ -n "$(source_dir)" ]] && need=$((CONVERT_FREE_GB - 336 > 0 ? CONVERT_FREE_GB - 336 : 0))   # a partial download counts as none
have=$(free_gb "$HF_CACHE")
(( have >= need )) || die "only ${have} GB free under $HF_CACHE, need ~${need} GB (CONVERT_FREE_GB)"
log "Disk: ${have} GB free under $HF_CACHE"

# The image with the HF cache mounted; a token only by name, and only when set (else the token file in the cache).
token=(); [[ -n "${HF_TOKEN:-}" ]] && token=(-e HF_TOKEN)
in_image() {
  docker run --rm --ipc=host --network host "${token[@]}" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$PWD/tools":/tools:ro \
    "$@"
}

# ---------------------------------------------------------------- 2. download
# In the image, not with a host `hf`: the container's memory cap bounds the download's page cache and buffers, so a
# server running meanwhile keeps its memory.
log "Downloading $SOURCE_ID into $HF_CACHE/hub (~336 GB, resumes if interrupted; memory capped at $DOWNLOAD_MEMORY)"
in_image --memory "$DOWNLOAD_MEMORY" --entrypoint python "$IMAGE" -c \
  "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], max_workers=4)" "$SOURCE_ID"
src=$(source_dir)
[[ -n "$src" ]] || die "no snapshot of $SOURCE_ID under $HF_CACHE/hub"
revision=$(basename "$src")
log "Source: $src ($(du -shL "$src" | cut -f1))"
if [[ $DOWNLOAD_ONLY == 1 ]]; then
  log "Downloaded. Next: stop the server holding the GPU, then scripts/convert.sh (or ./start.sh, which runs it)"
  exit 0
fi

# ---------------------------------------------------------------- 3. convert
partial="$MODEL_DIR.partial"
rm -rf "$partial"
mkdir -p "$partial"
log "Converting into $partial (MLX affine 4-bit, groups of 32, MTP head and vision tower kept)"
in_image --gpus all -v "$partial":/out --entrypoint python "$IMAGE" \
  /tools/convert_flash_next.py "/root/.cache/huggingface/hub/models--${SOURCE_ID//\//--}/snapshots/$revision" /out

# ---------------------------------------------------------------- 4. check
log "Checking the layout against Vontra's conversion of the official weights"
in_image -v "$partial":/out:ro --entrypoint python "$IMAGE" /tools/check_flash_next.py layout /out
log "Checking with tensorfold info"
in_image -v "$partial":/out:ro --entrypoint tensorfold "$IMAGE" info /out
rm -rf "$MODEL_DIR"
mv "$partial" "$MODEL_DIR"
printf 'source=%s revision=%s converted=%s\n' "$SOURCE_ID" "$revision" "$(date -u +%FT%TZ)" > "$MODEL_DIR/.converted"
log "Done: $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1)) from $SOURCE_ID@${revision:0:12}"
