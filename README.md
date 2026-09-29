# Qwen3.8 Flash Next on one DGX Spark (TensorFold)

Serve **Qwen3.8 Flash Next** from a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API, with
**4 concurrent requests at the full 262,144-token context**. It runs
[TensorFold](https://github.com/ashhart/TensorFold) v0.3.6.2 in NVIDIA's PyTorch container, plus a small set of
patches that make prompt processing about **1.7x faster** without changing a single output token.

- Checkpoint: [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP)
  (MLX 4-bit, group size 32, with the MTP draft head)
- API model id: `Qwen3.8-Flash-Next`
- Two commands: `scripts/prepare.sh` once, then `./start.sh`

## Performance

One DGX Spark, the defaults in [`scripts/config.sh`](scripts/config.sh) (4 streams x 262,144 tokens, int8 KV cache, n-gram tables read
from SSD, MTP drafting), measured through the OpenAI API.

**Decode, prose**

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 62.4 tok/s | 62.4 tok/s | 152 ms |
| 2 | 90.5 tok/s | 46.3 tok/s | 257 ms |
| 4 | 106.7 tok/s | 28.9 tok/s | 436 ms |

**Prefill**

| Prompt | Tokens | Prefill speed | Time to first token |
| ---: | ---: | ---: | ---: |
| 8k | 8,229 | 2,503 tok/s | 3.29 s |
| 16k | 16,425 | 2,520 tok/s | 6.52 s |
| 32k | 32,806 | 2,499 tok/s | 13.13 s |
| 64k | 65,575 | 2,414 tok/s | 27.17 s |
| 128k | 131,110 | 2,200 tok/s | 59.60 s |

Against unpatched TensorFold v0.3.6.2 with the same settings, prefill went from ~1,350-1,490 tok/s to ~2,340-2,480
tok/s (3k-50k-token prompts), a ~195k-token prompt from ~208 s to ~97 s, and single-request decode rose ~4%.
Every reply stayed byte-identical.

## Requirements

- A DGX Spark (or another GB10 system with 128 GB unified memory) with nothing else large on the GPU: the server
  budgets ~104 GiB, 75 GiB of it weights.
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~130 GB free disk under `~/.cache/huggingface` for the first download (the checkpoint is ~106 GB).
- Optional: the `hf` CLI on the host (faster, resumable download) and a Hugging Face token in
  `~/.cache/huggingface/token` or `HF_TOKEN`.

## Quick start

```bash
git clone <this repo> && cd <this repo>
scripts/prepare.sh   # builds the image (TensorFold + patches/), downloads and checks the checkpoint
./start.sh           # starts the container on port 8888 and waits until the API answers
```

The first start compiles the CUDA kernels for the GB10 (a few minutes, cached in `~/.cache/tensorfold-qwen38`);
later starts load the weights in ~2.5 minutes. `start.sh` ends with a smoke test and prints the endpoint.

```bash
curl -s http://<spark-address>:8888/v1/models

curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 256
}'
```

Any OpenAI client works with `base_url = "http://<spark-address>:8888/v1"` and the model `Qwen3.8-Flash-Next`.
Streaming, tool calls (typed parameters, e.g. arrays come back as JSON arrays) and reasoning content are supported.

Operations:

```bash
docker logs -f qwen38-flash-next-tf     # server log
docker rm -f qwen38-flash-next-tf       # stop
curl -s http://<spark-address>:8888/health   # busy flag and live token totals
```

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh) and can be overridden from the environment
(`PARALLEL=2 ./start.sh`) or with `tensorfold serve` flags (`./start.sh --context 131072`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `4` | requests decoded together |
| `CONTEXT` | `262144` | prompt + reply window per request |
| `KV_DTYPE` | `int8` | `bf16`, `int8` or `int4` KV cache |
| `PLE_ON_SSD` | `1` | read the 29.8 GiB n-gram tables from SSD instead of RAM, leaving that memory to the KV cache |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | `6` / `0.60` | at most 6 MTP drafts a round; a chain stops before a draft under 60% |
| `SERVED_NAME` | `Qwen3.8-Flash-Next` | the model id in `/v1/models` and in replies |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `TENSORFOLD_PREFILL_ROWS` | `4096` | rows per prompt chunk (patch 0007); `2048` is TensorFold's default |
| `TENSORFOLD_MTP_COPY` | `1` | prompt-lookup drafts for text that repeats the prompt (patch 0008); `0` turns them off |

Any `TENSORFOLD_*` variable in the environment is passed into the container.

**Memory.** All streams share one pool, so windows x streams x KV bytes must fit. The default fits in 97.8 GiB of the
~103 GiB budget. Other combinations that fit at 262k: 3 streams with bf16 KV, or 8 streams with int4 KV (tight). With
8 streams at int8 the window tops out near 172k. The server refuses a setting that does not fit, and names one that
does.

## What the patches change

`scripts/prepare.sh` bakes every `patches/*.patch` into the image (unified diffs against TensorFold's site-packages, applied
with `patch -p0`) and rebuilds the image automatically when the patches change.

| Patch | Change | Effect |
| --- | --- | --- |
| `0001-cuda-typed-tool-parameters` | tool-call arguments are decoded by the tool's JSON schema | arrays, numbers and objects arrive typed ([upstream #75](https://github.com/ashhart/TensorFold/pull/75)) |
| `0002-cuda-live-token-counters` | `/health` reports live token totals | monitoring ([upstream #79](https://github.com/ashhart/TensorFold/pull/79)) |
| `0003-flash-next-ssd-read-ahead` | a prompt chunk's n-gram rows are read from SSD while the GPU processes the previous chunk | multi-chunk prefill +50% |
| `0004-flash-next-ssd-native-reader` | those reads run on a C++ thread pool outside the Python GIL | short prompts' time to first token -35%, 3k-12k prefill +10-40% on top, decode +4% |
| `0005-flash-next-qsa-tiled-select` | the sparse-attention block selection no longer spills registers past 128k tokens | 149k-token prompts 25% faster; decode at 149k context +19% |
| `0006-cuda-stream-draft-stats` | `drafted` / `accepted` counts in concurrent requests' stats | observability |
| `0007-flash-next-prefill-rows` | configurable prompt chunk size (port of [#40](https://github.com/ashhart/TensorFold/pull/40)) | +2-5% at 4,096 rows |
| `0008-flash-next-copy-drafts` | drafts copied from earlier text when the reply repeats the prompt | +6% on quoting and editing replies |

**Outputs are unchanged.** Every patch changes speed only: drafts are verified against the model's own keyed samples,
and the prefill changes read the same bytes and select the same attention blocks. This was checked by comparing
reply hashes (sampled and greedy, prompts up to 149k tokens) against unpatched TensorFold, and with a ~195k-token
needle-in-a-haystack test. Any request can also be sent with `"draft": false` to get TensorFold's serial,
one-token-at-a-time reference.

## Checks

The scripts in `tools/` talk to the running server (`PORT` env var, default 8888):

| Script | What it does |
| --- | --- |
| `tools/bench.py [label]` | prefill at ~0.85k / 3.2k / 12.6k / 50k tokens (fresh random prompts) and a short decode check |
| `tools/needle.py` | hides a passphrase in a ~195k-token prompt and checks the model returns it |
| `tools/toolcheck.py` | makes a tool call with an array parameter and checks it comes back as a JSON array |

## Repository layout

```
start.sh      start the server
scripts/      prepare.sh (image + checkpoint) and config.sh (all settings)
patches/      patches baked into the image
tools/        benchmark and checks
```

## License

MIT, see [`LICENSE`](LICENSE). TensorFold is MIT-licensed too; the model weights carry their own license.

## Credits

- [TensorFold](https://github.com/ashhart/TensorFold) by ashhart: the inference engine.
- [Vontra](https://huggingface.co/Vontra): the MLX 4-bit checkpoint with the MTP head.
- MovieMaker93: the original prompt chunk size change ([TensorFold #40](https://github.com/ashhart/TensorFold/pull/40)),
  ported here as patch 0007.
