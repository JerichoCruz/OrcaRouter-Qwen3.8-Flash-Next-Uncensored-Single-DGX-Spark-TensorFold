#!/usr/bin/env python3
"""Check tools/convert_flash_next.py against Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP, which was converted from the
official BF16 checkpoint (Qwen/Qwen3.8-Flash-Next). Reads only what it compares, with HTTP range requests.

  sample          convert slices of the official BF16 tensors (an expert stack, attention, GDN, n-gram table,
                  embeddings, the MTP head, routers, convs, the vision tower) and compare them byte for byte with
                  Vontra's; exit 1 unless every one matches
  layout OUT_DIR  compare a converted checkpoint's tensor names, shapes and dtypes and config with Vontra's

Usage: tools/check_flash_next.py sample | layout OUT_DIR    (HF_TOKEN or ~/.cache/huggingface/token if needed)
"""
from __future__ import annotations

import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_flash_next import QUANT, convert_tensor, rename  # noqa: E402

SOURCE = os.environ.get("SOURCE_REPO", "Qwen/Qwen3.8-Flash-Next")
REFERENCE = os.environ.get("REFERENCE_REPO", "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
ITEM = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "I64": 8, "U8": 1}
# source tensor -> leading rows to compare (None: all of it)
SAMPLES = {
    "model.language_model.layers.3.mlp.experts.gate_up_proj": 2,
    "model.language_model.layers.3.mlp.experts.down_proj": 2,
    "model.language_model.layers.3.mlp.gate.weight": 16,
    "model.language_model.layers.3.mlp.shared_expert_gate.weight": None,
    "model.language_model.layers.3.mlp.shared_expert.down_proj.weight": 64,
    "model.language_model.layers.3.self_attn.q_proj.weight": 256,
    "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight": 64,
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": 256,
    "model.language_model.layers.0.linear_attn.conv1d.weight": None,
    "model.language_model.layers.0.linear_attn.A_log": None,
    "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight": None,
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": 4096,
    "model.language_model.layers.1.ple.conv1d.weight": None,
    "model.language_model.embed_tokens.weight": 256,
    "lm_head.weight": 256,
    "mtp.fc_hidden.weight": 256,
    "mtp.layers.0.mlp.experts.gate_up_proj": 1,
    "model.visual.patch_embed.proj.weight": None,
    "model.visual.blocks.0.attn.qkv.weight": 64,
}


def token() -> str | None:
    found = os.environ.get("HF_TOKEN")
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    return found or (path.read_text().strip() if path.is_file() else None)


def fetch(repo: str, path: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {token()}"} if token() else {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end - 1}"
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


_headers: dict = {}


def header(repo: str, shard: str) -> tuple[dict, int]:
    if (repo, shard) not in _headers:
        n = struct.unpack("<Q", fetch(repo, shard, 0, 8))[0]
        _headers[repo, shard] = (json.loads(fetch(repo, shard, 8, 8 + n)), 8 + n)
    return _headers[repo, shard]


def weight_map(repo: str) -> dict:
    return json.loads(fetch(repo, "model.safetensors.index.json"))["weight_map"]


def read(repo: str, wmap: dict, name: str, rows: int | None):
    """The first ``rows`` of a tensor (along its first axis) as a torch tensor, its dtype as stored."""

    import torch

    shard = wmap[name]
    head, base = header(repo, shard)
    meta = head[name]
    shape = list(meta["shape"])
    lo, hi = meta["data_offsets"]
    if rows is not None and rows < shape[0]:
        row = (hi - lo) // shape[0]
        hi, shape[0] = lo + rows * row, rows
    raw = bytearray(fetch(repo, shard, base + lo, base + hi))
    kind = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U32": torch.int32,
            "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8}[meta["dtype"]]
    t = torch.frombuffer(raw, dtype=kind).reshape(shape)
    return t, meta["dtype"]


def sample() -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_map, ref_map = weight_map(SOURCE), weight_map(REFERENCE)
    print(f"[check] source {SOURCE}, reference {REFERENCE}, quantizing on {device}")
    differing = 0
    for name, rows in SAMPLES.items():
        t, _ = read(SOURCE, src_map, name, rows)
        for key, value in convert_tensor(name, t, device=device):
            if key not in ref_map:
                print(f"  {key}: MISSING in the reference")
                differing += 1
                continue
            ref, dtype = read(REFERENCE, ref_map, key, value.shape[0] if value.ndim else None)
            mine = (value.view(torch.int32) if value.dtype == torch.uint32 else value).clone()
            if tuple(ref.shape) != tuple(mine.shape):
                print(f"  {key}: shape {tuple(mine.shape)} != reference {tuple(ref.shape)}")
                differing += 1
                continue
            a, b = mine.reshape(mine.numel(), 1).view(torch.uint8), ref.reshape(ref.numel(), 1).view(torch.uint8)
            differ = int((a != b).any(-1).sum())
            differing += bool(differ)
            print(f"  {key} {dtype} {list(mine.shape)}: "
                  + ("identical" if not differ else f"{differ} of {mine.numel()} elements DIFFER"))
    print("[check] " + ("every sampled tensor matches Vontra byte for byte" if not differing
                        else f"{differing} tensor(s) differ from Vontra"))
    return 0 if not differing else 1


def layout(out: Path) -> int:
    from safetensors import safe_open

    ref_map = weight_map(REFERENCE)
    ours_map = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    problems = []
    missing, extra = sorted(set(ref_map) - set(ours_map)), sorted(set(ours_map) - set(ref_map))
    problems += [f"missing {k}" for k in missing] + [f"extra {k}" for k in extra]
    ours = {}
    for shard in sorted(set(ours_map.values())):
        with safe_open(str(out / shard), framework="pt") as f:
            for k in f.keys():
                s = f.get_slice(k)
                ours[k] = (s.get_dtype(), list(s.get_shape()))
    # A quantized tensor's weight, scales and biases must be in the same output shard: TensorFold's n-gram table
    # reads a shard's header and classifies it as MLX only when the .scales key is in the same file, so splitting a
    # triple across shards makes it look like a layout mix.
    shard_of = {k: v for k, v in ours_map.items()}
    for k in ours_map:
        if k.endswith(".weight"):
            base = k[:-len(".weight")]
            companions = [base + ".scales", base + ".biases"]
            if all(c in shard_of for c in companions) and shard_of.get(base + ".scales") != shard_of[k]:
                problems.append(f"{base}: weight/scales/biases split across shards "
                                f"({shard_of[k]} vs {shard_of[base + '.scales']})")
    for shard in sorted(set(ref_map.values())):
        head, _ = header(REFERENCE, shard)
        for k, meta in head.items():
            if k != "__metadata__" and k in ours and ours[k] != (meta["dtype"], meta["shape"]):
                problems.append(f"{k}: {ours[k]} != reference {(meta['dtype'], meta['shape'])}")
    config = json.loads((out / "config.json").read_text())
    for key in ("quantization", "quantization_config"):
        if config.get(key) != QUANT:
            problems.append(f"config.json {key} = {config.get(key)}")
    if config.get("model_type") != "qwen4_exp":
        problems.append(f"config.json model_type = {config.get('model_type')}")
    mtp = sum(".mtp." in k for k in ours_map)
    vision = sum(k.startswith("vision_tower.") for k in ours_map)
    for p in problems[:50]:
        print("  " + p)
    print(f"[check] {len(ours_map)} tensors (reference {len(ref_map)}), {mtp} MTP, {vision} vision: "
          + ("layout matches the reference" if not problems else f"{len(problems)} problem(s)"))
    return 0 if not problems else 1


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        sys.exit(sample())
    if len(sys.argv) == 3 and sys.argv[1] == "layout":
        sys.exit(layout(Path(sys.argv[2])))
    sys.exit(__doc__)


if __name__ == "__main__":
    assert rename("lm_head.weight") == "language_model.lm_head.weight"
    main()
