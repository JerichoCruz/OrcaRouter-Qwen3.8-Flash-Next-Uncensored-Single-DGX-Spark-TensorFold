# Shared settings for scripts/prepare.sh and start.sh. Any value can be overridden from the environment,
# e.g. `PORT=9000 ./start.sh` or `TF_VERSION=v0.3.6.1 scripts/prepare.sh`.

MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"   # Ash's quant: MLX 4-bit, g32, with MTP head
TF_VERSION="${TF_VERSION:-v0.3.6.2}"                              # >= v0.3.6.1 required; 0.3.6.2 adds --kv-dtype
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE="${IMAGE:-tensorfold-qwen38:${TF_VERSION}}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"

SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"   # the model id clients see in /v1/models and replies (tensorfold --name)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Serving defaults (each is skipped when the same flag is passed to ./start.sh). All streams share one
# memory pool (~104 GiB budget on a 128 GB Spark, 75 GiB of it weights), so window x streams x KV bytes
# must fit. 4 streams x 262,144 tokens at int8 KV is ~97 GiB. Other fits at 262k: 3 streams bf16,
# 8 streams int4 (~103 GiB, tight). 8 streams at int8 top out near 172k.
PARALLEL="${PARALLEL:-4}"          # requests decoded together (CUDA "auto" = 1 at a time)
CONTEXT="${CONTEXT:-262144}"       # prompt + reply window per request (the native 256k)
KV_DTYPE="${KV_DTYPE:-int8}"       # bf16 | int8 | int4
PLE_ON_SSD="${PLE_ON_SSD:-1}"      # 1: read the 29.8 GiB n-gram tables from SSD, leaving that RAM to the KV cache
# MTP drafting: at most MTP_DRAFTS drafts a round, a chain stopping before a draft under MTP_CONFIDENCE.
# Swept 2026-09-29 (identical output in every arm): 6/0.60 beat the stock 6/0.30 by ~3% on
# prose and ~4% on code, the best balance of both; 4/0.50, 3/0.30 and 7/0.75 matched it on prose but not on code.
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"
# TensorFold switches from patches/ (measured 2026-09-29); start.sh passes every TENSORFOLD_* variable through.
# 4,096-row prompt chunks (patch 0007): prefill +2-5% from 3k tokens, short prompts unchanged, +0.94 GiB at startup.
export TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS:-4096}"
# Prompt-lookup drafts ahead of MTP (patch 0008): +6% on replies that repeat the prompt, prose/code unchanged. 0: off.
export TENSORFOLD_MTP_COPY="${TENSORFOLD_MTP_COPY:-1}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# Persists compiled CUDA kernels (torch extensions + triton) so only the first start pays the compile.
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38}"

MIN_FREE_GB="${MIN_FREE_GB:-130}"   # the checkpoint is ~113 GB

log()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(basename "$0")" "$*"; }
warn() { printf '\033[1;33m[%s] WARN:\033[0m %s\n' "$(basename "$0")" "$*" >&2; }
die()  { printf '\033[1;31m[%s] ERROR:\033[0m %s\n' "$(basename "$0")" "$*" >&2; exit 1; }

hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then echo "$HF_TOKEN"
  elif [[ -f "$HF_CACHE/token" ]]; then cat "$HF_CACHE/token"
  fi
}

model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
