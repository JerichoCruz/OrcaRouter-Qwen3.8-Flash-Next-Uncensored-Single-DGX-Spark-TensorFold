#!/usr/bin/env python3
"""Convert a BF16 Qwen3.8 Flash Next checkpoint (Hugging Face layout, e.g. orcarouter/Qwen3.8-Flash-Next-Uncensored
or Qwen/Qwen3.8-Flash-Next) into the layout TensorFold's Flash Next CUDA engine reads: MLX affine 4-bit in groups of
32, the MTP head and the vision tower kept. The tensor names, shapes, dtypes and the quantized set match
Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP (3,747 index entries); tools/check_flash_next.py compares the two.

  - model.language_model.X -> language_model.model.X, lm_head -> language_model.lm_head, mtp.X -> language_model.mtp.X,
    model.visual.X -> vision_tower.X
  - mlp.experts.gate_up_proj [E, 2I, D] -> mlp.switch_mlp.gate_proj + up_proj [E, I, D] (gate first),
    mlp.experts.down_proj -> mlp.switch_mlp.down_proj
  - every language-model linear and embedding (n-gram tables, lm_head and the MTP head too) is quantized; routers
    (mlp.gate), norms, conv weights, the GDN A_log / dt_bias, integer tables and the vision tower stay as stored
  - conv1d weights [C, 1, K] -> [C, K, 1]; the vision patch embedding goes channels-last [O, T, H, W, C]

Usage (in a container with torch + safetensors and the GPU, see scripts/convert.sh):
  tools/convert_flash_next.py SRC_DIR OUT_DIR [--device cuda] [--shard-gib 5]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

BITS, GROUP = 4, 32
QUANT = {"group_size": GROUP, "bits": BITS, "mode": "affine"}
# what goes with the weights, copied as they are (missing ones are skipped)
SIDE_FILES = ("chat_template.jinja", "generation_config.json", "merges.txt", "preprocessor_config.json",
              "processor_config.json", "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json",
              "vocab.json", "LICENSE")


def _round(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(x.abs() + 0.5)                # halves away from zero, as MLX's kernels do


def quantize(w: torch.Tensor):
    """MLX affine quantization of the last axis in groups of GROUP: (uint32 words, bf16 scales, bf16 biases).

    Per group (fp32), the end of larger magnitude is made exactly representable: w ~ q * scale + bias with q in
    [0, 2^bits - 1], 32 / bits values to a word, low bits first. Byte for byte what MLX wrote for Vontra's
    checkpoint (tools/check_flash_next.py sample). Every division is tensor by tensor: on CUDA, PyTorch turns a
    division by a Python number into a multiplication by its reciprocal, which rounds differently."""

    shape = w.shape
    g = w.reshape(-1, shape[-1] // GROUP, GROUP).float()
    w_max, w_min = g.amax(-1, keepdim=True), g.amin(-1, keepdim=True)
    n_bins = torch.full_like(w_max, float((1 << BITS) - 1))
    neg = w_min.abs() > w_max.abs()
    scale = torch.clamp((w_max - w_min) / n_bins, min=1e-7)
    scale = torch.where(neg, scale, -scale)
    edge = torch.where(neg, w_min, w_max)
    q0 = _round(edge / scale)
    scale = torch.where(q0 != 0, edge / q0, scale)
    bias = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    q = torch.minimum(torch.clamp(_round((g - bias) / scale), min=0), n_bins).to(torch.int64)
    per = 32 // BITS
    q = q.reshape(-1, shape[-1] // per, per)
    shifts = torch.arange(0, 32, BITS, device=q.device, dtype=torch.int64)
    words = (q << shifts).sum(-1)                                   # < 2^32: exact in int64
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).view(torch.uint32)
    lead = shape[:-1]
    return (words.reshape(*lead, shape[-1] // per), scale.to(torch.bfloat16).reshape(*lead, shape[-1] // GROUP),
            bias.to(torch.bfloat16).reshape(*lead, shape[-1] // GROUP))


def rename(name: str) -> str | None:
    """The target name of a source tensor (before any expert split); None for a name this converter does not know."""

    for src, dst in (("model.language_model.", "language_model.model."), ("lm_head.", "language_model.lm_head."),
                     ("mtp.", "language_model.mtp."), ("model.visual.", "vision_tower.")):
        if name.startswith(src):
            return dst + name[len(src):]
    return None


def is_quantized(name: str, t: torch.Tensor) -> bool:
    """Whether the target tensor ``name`` is stored 4-bit (else as it is)."""

    if not name.startswith("language_model.") or not t.is_floating_point() or t.ndim < 2:
        return False
    if ".conv1d." in name or name.endswith(".mlp.gate.weight"):       # depthwise convs, routers
        return False
    return name.endswith(".weight") or ".switch_mlp." in name


def convert_tensor(name: str, t: torch.Tensor, *, device: str = "cpu"):
    """Yield the target (name, tensor) pairs for one source tensor (on the CPU, ready to save)."""

    out = rename(name)
    if out is None:
        raise ValueError(f"unknown tensor {name!r}: not a Qwen3.8 Flash Next BF16 checkpoint?")
    parts = []
    if out.endswith(".mlp.experts.gate_up_proj"):
        base = out[:-len("experts.gate_up_proj")] + "switch_mlp."
        half = t.shape[-2] // 2
        parts = [(base + "gate_proj.weight", t[..., :half, :]), (base + "up_proj.weight", t[..., half:, :])]
    elif out.endswith(".mlp.experts.down_proj"):
        parts = [(out[:-len("experts.down_proj")] + "switch_mlp.down_proj.weight", t)]
    elif ".conv1d.weight" in out and t.ndim == 3:
        parts = [(out, t.reshape(t.shape[0], t.shape[2], 1))]          # [C, 1, K] -> [C, K, 1]
    elif out == "vision_tower.patch_embed.proj.weight" and t.ndim == 5:
        parts = [(out, t.permute(0, 2, 3, 4, 1))]                      # [O, C, T, H, W] -> [O, T, H, W, C]
    else:
        parts = [(out, t)]
    for key, value in parts:
        if is_quantized(key, value):
            base = key[:-len(".weight")]
            if value.shape[-1] % GROUP:
                raise ValueError(f"{name}: last dimension {value.shape[-1]} is not a multiple of {GROUP}")
            w, s, b = quantize(value.to(device))
            yield base + ".weight", w.cpu()
            yield base + ".scales", s.cpu()
            yield base + ".biases", b.cpu()
        else:
            yield key, value.contiguous()


def target_config(src: dict) -> dict:
    config = dict(src)
    config["quantization"] = dict(QUANT)
    config["quantization_config"] = dict(QUANT)
    return config


def convert(src: Path, out: Path, *, device: str, shard_bytes: int) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    out.mkdir(parents=True, exist_ok=True)
    weight_map, pending, pending_bytes, written, total = {}, {}, 0, [], 0
    started = time.time()

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        part = out / f"part-{len(written) + 1:05d}.safetensors"
        save_file(pending, str(part), metadata={"format": "mlx"})
        written.append((part, list(pending)))
        pending, pending_bytes = {}, 0

    for i, shard in enumerate(shards, 1):
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                # Emit one source tensor's outputs as a unit: converting a tensor yields its weight, scales and
                # biases, and TensorFold needs those (e.g. an n-gram shard's triple) in the same output file, so a
                # flush that lands in the middle of one would split them. (A single tensor's outputs are always far
                # under shard_bytes, so this only ever delays a flush.)
                items = list(convert_tensor(name, f.get_tensor(name), device=device))
                group = sum(value.numel() * value.element_size() for _, value in items)
                if pending and group <= shard_bytes and pending_bytes + group > shard_bytes:
                    flush()
                for key, value in items:
                    nbytes = value.numel() * value.element_size()
                    pending[key] = value
                    pending_bytes += nbytes
                    total += nbytes
        print(f"[convert] {i}/{len(shards)} {shard}  {total / 2**30:.1f} GiB out  {time.time() - started:.0f} s",
              flush=True)
    flush()

    count = len(written)
    for n, (part, keys) in enumerate(written, 1):
        final = f"model-{n:05d}-of-{count:05d}.safetensors"
        part.rename(out / final)
        weight_map.update({k: final for k in keys})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    config = target_config(json.loads((src / "config.json").read_text()))
    (out / "config.json").write_text(json.dumps(config, indent=4) + "\n")
    for side in SIDE_FILES:
        if (src / side).is_file():
            shutil.copy2(src / side, out / side)
    print(f"[convert] done: {len(weight_map)} tensors in {count} shards, {total / 2**30:.2f} GiB, "
          f"{time.time() - started:.0f} s -> {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", type=Path, help="BF16 checkpoint directory (a Hugging Face snapshot)")
    p.add_argument("out", type=Path, help="output directory (created)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--shard-gib", type=float, default=5.0, help="target output shard size (GiB)")
    a = p.parse_args()
    if a.out.exists() and any(a.out.glob("*.safetensors")):
        sys.exit(f"{a.out} already has safetensors files; remove them or choose another directory")
    convert(a.src, a.out, device=a.device, shard_bytes=int(a.shard_gib * 2**30))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
