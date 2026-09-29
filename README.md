<h1 align="center">Qwen3.8 Flash Next on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve **Qwen3.8 Flash Next** from a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API, with
**5 concurrent requests at the full 262,144-token context**. It runs
[TensorFold](https://github.com/ashhart/TensorFold) v0.3.6.2 in NVIDIA's PyTorch container, plus a small set of
patches that make prompt processing about **1.7x faster** without changing a single output token.

- Checkpoint: [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP)
  (MLX 4-bit, group size 32, with the MTP draft head)
- API model id: `Qwen3.8-Flash-Next`
- KV pool: **1,310,720 tokens** (5 streams x 262,144, int8 KV cache, ~23.4 GiB), 25% more than 4 streams
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## Performance

One DGX Spark, int8 KV cache, n-gram tables read from SSD and MTP drafting, measured through the OpenAI API. The 5
concurrent requests row is from the current default (5 streams x 262,144 tokens); the other rows and the prefill table
were measured with 4 streams x 262,144 tokens.

**Decode, prose**

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 62.4 tok/s | 62.4 tok/s | 152 ms |
| 2 | 90.5 tok/s | 46.3 tok/s | 257 ms |
| 4 | 106.7 tok/s | 28.9 tok/s | 436 ms |
| 5 | 119.3 tok/s | 27.0 tok/s | 528 ms |

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

- A DGX Spark (or another GB10 system with 128 GB unified memory) with nothing else large on the GPU: the default
  setting needs ~115 GiB free when the server starts (see [KV pool and memory](#kv-pool-and-memory)).
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~160 GB free disk on a fresh machine: ~125 GB for the checkpoint download under `~/.cache/huggingface`
  (~114 GB) and ~35 GB for the image under Docker's root (~24 GB); `scripts/prepare.sh` checks both.
- Optional: the `hf` CLI on the host (faster, resumable download) and a Hugging Face token in
  `~/.cache/huggingface/token` or `HF_TOKEN`.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
./start.sh
```

That is all. The first run sets everything up (see below): it pulls the prebuilt image (~11 GB) and downloads the
~106 GiB checkpoint, then compiles the CUDA kernels for the GB10 (a few minutes, once). Later starts take ~2.5 minutes to load the
weights. `start.sh` shows each step, the server's log and the loading progress, runs a smoke test, prints
`Qwen3.8-Flash-Next is now LIVE! on port 8888` with the endpoint, and returns you to the shell.

```bash
curl -s http://<spark-address>:8888/v1/models

curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

Any OpenAI client works with `base_url = "http://<spark-address>:8888/v1"` and the model `Qwen3.8-Flash-Next`.
Streaming, tool calls (typed parameters, e.g. arrays come back as JSON arrays) and reasoning content are supported.
The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`.

```bash
./start.sh restart                            # restart it, e.g. after changing a setting
./stop.sh                                     # stop the server and free the GPU memory
docker logs -f qwen38-flash-next-tf           # server log
curl -s http://<spark-address>:8888/health    # busy flag and live token totals
```

## What `start.sh` and `scripts/prepare.sh` do

**`./start.sh`** works in five steps, each shown as it runs:

1. **Setup:** runs `scripts/prepare.sh` whenever the setup is not ready: on the first run, after the patches change,
   or with another model or image. It compares what `prepare.sh` last left ready with the current settings, so later
   starts skip it instantly.
2. **Checks:** the arguments (with TensorFold's own parser, in a throwaway container), the previous server, the
   port and the free memory.
3. **Launch:** `tensorfold serve` with the settings from `scripts/config.sh`.
4. **Loading:** the server's log as it comes, and every 15 s the elapsed time and how much of the startup estimate is
   on the GPU. If the server stops, the last log lines and the reason are shown.
5. **Smoke test:** one chat completion, then the LIVE message and the endpoint.

If the server is already running, `./start.sh` says so and leaves it alone; `./start.sh restart` stops it and
starts it again. It stops the server only after the setup and the argument check pass, so a typo leaves the running
server alone and the server is down only while it restarts. Stopping cuts off requests still running (`stop.sh` warns
when there are any). Extra arguments go to `tensorfold serve` after the defaults, so they win
(`./start.sh restart --context 131072`); `./start.sh --help` lists the options. `FOREGROUND=1 ./start.sh` stays
attached to the server's log and exits with its exit code (for a systemd unit).

**`scripts/prepare.sh`** does the one-time setup, and is safe to re-run (each step skips work already done):

1. Preflight: Docker, the NVIDIA runtime, disk space.
2. The image `tensorfold-qwen38:v0.3.6.2`: TensorFold v0.3.6.2 with every `patches/*.patch` applied, on NVIDIA's
   PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`). It first tries the matching prebuilt image from GitHub
   Container Registry (`ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:v0.3.6.2-<patches hash>`,
   ~11 GB); if that tag is not there (e.g. after you change `patches/`), or with `PULL=0`, it builds the image
   locally instead (a few minutes).
3. Downloads the checkpoint into `~/.cache/huggingface` (resumable).
4. Verifies the checkpoint with `tensorfold info`.

Run it yourself to download ahead of time or to rebuild the image from scratch:

```bash
scripts/prepare.sh             # set up without starting the server
scripts/prepare.sh --rebuild   # rebuild the image from scratch
PREPARE=1 ./start.sh restart   # force prepare.sh, then restart; PREPARE=0 skips the check
```

After changing `patches/`, `scripts/publish-image.sh` pushes the new image to GitHub Container Registry
(`latest` and `v0.3.6.2-<patches hash>`).

## KV pool and memory

TensorFold gives every stream its own cache for a full window, so the KV pool is streams x window:

| | Default |
| --- | ---: |
| Streams (`PARALLEL`) | 5 |
| Window per stream (`CONTEXT`, the model's native maximum) | 262,144 tokens |
| **KV pool** | **1,310,720 tokens** (4 streams: 1,048,576) |
| KV precision (`KV_DTYPE`) | int8 (an fp16 scale per 32 values) |
| Memory a stream, allocated (server log) | 4,799 MiB: the KV cache, the sparse-attention index and the stream's own buffers |
| **Memory for the pool, allocated** | **~23.4 GiB** (5 x 4,799 MiB) |

The server reports these at every start: `5 streams of 262144 prompt/reply tokens (4799 MiB a stream)` and
`startup estimate 102.60 GiB within 103.64 GiB` (the budget varies a little from start to start).

Where the memory goes at the default setting (TensorFold's startup estimate):

| | GiB |
| --- | ---: |
| Model weights (the 29.8 GiB of n-gram tables stay on the SSD with `PLE_ON_SSD=1`) | 75.2 |
| Stream caches (5 x 4.49, the context-sized part) | 22.5 |
| Fixed buffers (DeltaNet states, decode windows, prompt-chunk scratch, 8 saved prompt states) | 4.9 |
| **Startup estimate** | **102.6** |

TensorFold's budget is the free memory at start (`MemAvailable`) minus a host reserve of a tenth of RAM (12.2 GiB),
so ~103-104 GiB on an otherwise idle Spark. The reserve covers what the estimate leaves out (CUDA context, workspaces,
the Python process) and the host itself: on the Spark's unified memory, running out tends to freeze the machine
rather than fail an allocation. At the default setting the host kept at least 9.7 GiB free through a 195k-token
prompt and 5 concurrent long requests.

Other settings that fit the same budget (TensorFold's own estimate):

| Setting | KV pool | Estimate | Note |
| --- | ---: | ---: | --- |
| `PARALLEL=4` (int8) | 1,048,576 | 97.8 GiB | more headroom |
| `PARALLEL=5` (int8, default) | 1,310,720 | 102.6 GiB | |
| `PARALLEL=6 CONTEXT=220000` (int8) | 1,320,000 | ~103 GiB | shorter windows, one more stream |
| `PARALLEL=6 KV_DTYPE=int4` | 1,572,864 | 97.7 GiB | int4 changes outputs slightly; quality not measured here |
| `PARALLEL=8 KV_DTYPE=int4 CONTEXT=250000` | 2,000,000 | ~103 GiB | tight |
| `PARALLEL=3 KV_DTYPE=bf16` | 786,432 | 102.1 GiB | full-precision KV |

A setting that does not fit is refused at startup, before any weights load, with a message naming a window that
fits.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh) and can be overridden from the environment
(`PARALLEL=4 ./start.sh`) or with `tensorfold serve` flags (`./start.sh --context 131072`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `5` | requests decoded together (streams) |
| `CONTEXT` | `262144` | prompt + reply window per stream |
| `KV_DTYPE` | `int8` | `bf16`, `int8` or `int4` KV cache |
| `PLE_ON_SSD` | `1` | read the 29.8 GiB n-gram tables from SSD instead of RAM, leaving that memory to the KV cache |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | `6` / `0.60` | at most 6 MTP drafts a round; a chain stops before a draft under 60% |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | default sampling (Qwen's thinking-mode values); a request's own values win |
| `THINKING` | `1` | open a think block by default; `0` answers directly unless a request asks to think |
| `SERVED_NAME` | `Qwen3.8-Flash-Next` | the model id in `/v1/models` and in replies |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `TENSORFOLD_PREFILL_ROWS` | `4096` | rows per prompt chunk (patch 0007); `2048` is TensorFold's default |
| `TENSORFOLD_MTP_COPY` | `1` | prompt-lookup drafts for text that repeats the prompt (patch 0008; needs `PARALLEL` >= 2); `0` turns them off |
| `PREPARE` | `auto` | `start.sh` runs `scripts/prepare.sh` when needed; `1` always, `0` never |
| `PULL` | `1` | `prepare.sh` tries the prebuilt image first; `0` always builds locally |
| `STOP_TIMEOUT` | `30` | seconds `stop.sh` gives the server to shut down before removing it |

Any `TENSORFOLD_*` variable in the environment is passed into the container (`TENSORFOLD_NO_UPDATE_CHECK=1`, the
default, stops TensorFold asking GitHub for a newer release at each start). Less common settings are described in
`scripts/config.sh`: `MODEL_ID`, `TF_VERSION`, `TF_REPO`, `BASE_IMAGE` (the patches are made for TensorFold v0.3.6.2;
after changing any of these run `scripts/prepare.sh --rebuild`), `IMAGE`, `CONTAINER_NAME`, `GHCR_IMAGE`, `HF_CACHE`
(default `$HF_HOME` or `~/.cache/huggingface`), `KERNEL_CACHE`, `MIN_FREE_GB`, `IMAGE_FREE_GB`. `start.sh` also takes
`FOREGROUND=1`, `WAIT_TIMEOUT` (seconds, default 1800) and `HF_HUB_OFFLINE=0` (let the server reach Hugging Face; by
default it serves from the local cache only).

### Thinking and sampling

By default the model thinks before it answers, with Qwen's recommended thinking-mode sampling: temperature 1.0,
top_p 0.95, top_k 20. TensorFold has no min_p, presence penalty or repetition penalty, which is the same as
min_p 0.0, presence_penalty 0.0 and repetition_penalty 1.0; requests that send those fields are served as if they
had not. Per request:

- `temperature`, `top_p`, `top_k` and `seed` override the defaults (`temperature: 0` decodes greedily).
- `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking, and
  `"chat_template_kwargs": {"reasoning_effort": "low"}` (or `"xhigh"`) sets Qwen's reasoning effort; without it the
  template's default (medium) applies. A top-level OpenAI-style `reasoning_effort` field is ignored.
- The reasoning comes back in `reasoning_content`, the answer in `content`.

## What the patches change

`scripts/prepare.sh` bakes every `patches/*.patch` into the image (unified diffs against TensorFold's site-packages,
applied with `patch -p0`), and `start.sh` rebuilds or re-pulls the image by itself when the patches change.

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

The scripts in `tools/` talk to the running server (`API_URL`, default `http://127.0.0.1:8888`; or just `PORT`),
from this machine or another one (`API_URL=http://<spark-address>:8888 tools/bench.py`):

| Script | What it does |
| --- | --- |
| `tools/bench.py [label]` | prefill at ~0.85k / 3.2k / 12.6k / 50k tokens (fresh random prompts) and a short decode check |
| `tools/needle.py` | hides a passphrase in a ~195k-token prompt and checks the model returns it |
| `tools/toolcheck.py` | makes a tool call with an array parameter and checks it comes back as a JSON array |

## Repository layout

```
start.sh      set up (first run) and start the server
stop.sh       stop it
scripts/      prepare.sh (image + checkpoint), config.sh (all settings), publish-image.sh (push the image to GHCR),
              banner.sh (start.sh's banner)
patches/      patches baked into the image
tools/        benchmark and checks
.github/      issue and pull request templates, GitHub Sponsors
CREDITS.md    who and what this builds on
```

## License

MIT, see [`LICENSE`](LICENSE), which also carries TensorFold's MIT notice for the patches. The model weights, downloaded from Hugging Face and not
part of this repository, are under the Qwen Community License 1.0.

**Third-party software in the image.** The prebuilt image (and the one `scripts/prepare.sh` builds) is based on
NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA
software in it is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start (it shows in `start.sh`'s output); by pulling or running the image you
accept them. The MIT license above
covers this repository's scripts and patches only.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)), [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
by Qwen, and [Vontra's MLX 4-bit checkpoint](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP), with a
prompt-chunk change by MovieMaker93 ([TensorFold #40](https://github.com/ashhart/TensorFold/pull/40)). The full list,
including the runtime stack and licenses, is in [`CREDITS.md`](CREDITS.md).
