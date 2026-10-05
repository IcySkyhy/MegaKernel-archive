#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle
"""Layer-streamed Qwen3.8 Transformers reference and Lean-logit parity gate.

The published checkpoint is larger than half of a DGX Spark's unified memory. Loading a complete
Transformers model beside the Lean CUDA copy would therefore turn parity checking into an OOM
hazard. This runner keeps the official Transformers arithmetic but owns only one decoder layer at
a time: safetensors remain memory mapped, the current layer moves to the selected device, and the
layer is released before the next one is constructed.

Reference outputs are raw little-endian float32 logits in position-major order. A JSON sidecar
records the checkpoint, token IDs, shape, argmaxes, and SHA-256 so a later Lean run can compare
against the durable reference without loading Transformers again.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)


MIB = 1024 * 1024


def parse_tokens(raw: str) -> list[int]:
    try:
        tokens = [int(part) for part in raw.split(",") if part]
    except ValueError as error:
        raise argparse.ArgumentTypeError("tokens must be comma-separated integers") from error
    if not tokens:
        raise argparse.ArgumentTypeError("at least one token is required")
    return tokens


def process_memory_mib() -> dict[str, float]:
    wanted = {"VmRSS:": "rss", "RssAnon:": "anon", "RssFile:": "file", "VmHWM:": "hwm"}
    result = {name: float("nan") for name in wanted.values()}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key = line.split(maxsplit=1)[0]
            if key in wanted:
                result[wanted[key]] = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointReader:
    """Resolve tensors through the HF weight map with layer-scoped shard mappings."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        with index_path.open() as handle:
            self.weight_map = json.load(handle)["weight_map"]
    def tensor(self, name: str) -> torch.Tensor:
        try:
            filename = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor is missing: {name}") from error
        with safe_open(self.model_dir / filename, framework="pt", device="cpu") as shard:
            return shard.get_tensor(name)

    def prefixed_state(self, prefix: str) -> dict[str, torch.Tensor]:
        selected = [name for name in self.weight_map if name.startswith(prefix)]
        state = {}
        for filename in sorted({self.weight_map[name] for name in selected}):
            with safe_open(self.model_dir / filename, framework="pt", device="cpu") as shard:
                for name in selected:
                    if self.weight_map[name] == filename:
                        state[name.removeprefix(prefix)] = shard.get_tensor(name)
        if not state:
            raise KeyError(f"checkpoint contains no tensors below prefix: {prefix}")
        return state


def release_device() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def progress(label: str, device: torch.device, started: float) -> None:
    allocated = torch.cuda.memory_allocated(device) / MIB if device.type == "cuda" else 0.0
    memory = process_memory_mib()
    print(
        f"{label}: elapsed_s={time.monotonic() - started:.1f} "
        f"rss_mib={memory['rss']:.1f} anon_mib={memory['anon']:.1f} "
        f"file_mib={memory['file']:.1f} hwm_mib={memory['hwm']:.1f} "
        f"device_allocated_mib={allocated:.1f}",
        flush=True,
    )


def load_layer(reader: CheckpointReader, config, layer_index: int, device: torch.device):
    prefix = f"model.language_model.layers.{layer_index}."
    state = reader.prefixed_state(prefix)
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, layer_index)
    layer.load_state_dict(state, strict=True, assign=True)
    layer = layer.to(device=device, dtype=torch.bfloat16).eval()
    del state
    return layer


@torch.inference_mode()
def streamed_reference(model_dir: Path, tokens: list[int], device: torch.device) -> np.ndarray:
    started = time.monotonic()
    outer_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config = outer_config.text_config
    config._attn_implementation = "eager"
    if max(tokens) >= config.vocab_size or min(tokens) < 0:
        raise ValueError(f"token IDs must lie in [0, {config.vocab_size})")

    reader = CheckpointReader(model_dir)
    token_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    embedding = reader.tensor("model.language_model.embed_tokens.weight").to(
        device=device, dtype=torch.bfloat16
    )
    hidden = F.embedding(token_ids, embedding)
    del embedding
    release_device()
    progress("embedding", device, started)

    positions = torch.arange(len(tokens), dtype=torch.long, device=device).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(config, device=device).to(device)
    position_embeddings = rotary(hidden, positions)
    del rotary

    causal = torch.full(
        (len(tokens), len(tokens)),
        torch.finfo(hidden.dtype).min,
        dtype=hidden.dtype,
        device=device,
    ).triu(diagonal=1)[None, None]

    for layer_index, layer_type in enumerate(config.layer_types):
        layer = load_layer(reader, config, layer_index, device)
        layer_mask = causal if layer_type == "full_attention" else None
        hidden = layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=positions,
        )
        del layer
        release_device()
        progress(f"layer {layer_index + 1}/{config.num_hidden_layers}", device, started)

    with torch.device("meta"):
        final_norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    final_norm.load_state_dict(
        {"weight": reader.tensor("model.language_model.norm.weight")},
        strict=True,
        assign=True,
    )
    final_norm = final_norm.to(device=device, dtype=torch.bfloat16).eval()
    hidden = final_norm(hidden)
    del final_norm

    lm_head = reader.tensor("lm_head.weight").to(device=device, dtype=torch.bfloat16)
    logits = F.linear(hidden, lm_head).float().squeeze(0).cpu().numpy()
    del lm_head, hidden
    release_device()
    progress("lm_head", device, started)
    return np.asarray(logits, dtype="<f4", order="C")


def write_reference(path: Path, logits: np.ndarray, model_dir: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logits.tofile(path)
    metadata = {
        "model_dir": str(model_dir.resolve()),
        "tokens": tokens,
        "shape": list(logits.shape),
        "dtype": "float32-le",
        "argmax": logits.argmax(axis=-1).astype(int).tolist(),
        "sha256": sha256(path),
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote Transformers logits: {path} ({path.stat().st_size} bytes)", flush=True)
    print(f"wrote Transformers metadata: {path.with_suffix(path.suffix + '.json')}", flush=True)


def read_logits(path: Path, positions: int, vocabulary: int) -> np.ndarray:
    logits = np.fromfile(path, dtype="<f4")
    expected = positions * vocabulary
    if logits.size != expected:
        raise ValueError(f"{path}: expected {expected} float32 values, found {logits.size}")
    return logits.reshape(positions, vocabulary)


def compare_logits(
    reference: np.ndarray,
    lean_path: Path,
    relative_l2_tolerance: float,
    require_top1: bool,
) -> None:
    lean = read_logits(lean_path, *reference.shape)
    if not np.isfinite(lean).all():
        raise RuntimeError("Lean logits contain NaN or infinity")
    difference = lean - reference
    relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1.0e-30))
    reference_top1 = reference.argmax(axis=-1)
    lean_top1 = lean.argmax(axis=-1)
    report = {
        "max_abs": float(np.abs(difference).max()),
        "mean_abs": float(np.abs(difference).mean()),
        "relative_l2": relative_l2,
        "per_position_max_abs": np.abs(difference).max(axis=-1).astype(float).tolist(),
        "reference_argmax": reference_top1.astype(int).tolist(),
        "lean_argmax": lean_top1.astype(int).tolist(),
        "top1_equal": bool(np.array_equal(reference_top1, lean_top1)),
    }
    print(json.dumps(report, indent=2), flush=True)
    if relative_l2 > relative_l2_tolerance:
        raise RuntimeError(
            f"Lean/Transformers relative L2 {relative_l2:.8g} exceeds {relative_l2_tolerance:.8g}"
        )
    if require_top1 and not report["top1_equal"]:
        raise RuntimeError("Lean and Transformers greedy token IDs differ")
    print("Lean Qwen3.8 logits match the layer-streamed Transformers reference", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=parse_tokens)
    parser.add_argument("--reference-output", required=True, type=Path)
    parser.add_argument("--lean-logits", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-reference", action="store_true")
    parser.add_argument("--relative-l2-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--require-top1", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if args.reuse_reference:
        metadata_path = args.reference_output.with_suffix(args.reference_output.suffix + ".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata["model_dir"] != str(model_dir) or metadata["tokens"] != args.tokens:
            raise RuntimeError("cached Transformers reference metadata does not match model/tokens")
        reference = read_logits(args.reference_output, *metadata["shape"])
        if sha256(args.reference_output) != metadata["sha256"]:
            raise RuntimeError("cached Transformers reference checksum mismatch")
        print(f"reusing Transformers logits: {args.reference_output}", flush=True)
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA reference requested but torch.cuda.is_available() is false")
        reference = streamed_reference(model_dir, args.tokens, device)
        write_reference(args.reference_output, reference, model_dir, args.tokens)

    if args.lean_logits is not None:
        compare_logits(
            reference,
            args.lean_logits,
            args.relative_l2_tolerance,
            args.require_top1,
        )


if __name__ == "__main__":
    main()
