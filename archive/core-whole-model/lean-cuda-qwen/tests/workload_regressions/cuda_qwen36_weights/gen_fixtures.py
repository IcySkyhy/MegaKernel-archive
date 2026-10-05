#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
"""Regenerate the synthetic safetensors fixtures for the cuda_qwen36_weights suite.

Fixtures are checked in so the gate never needs Python; this script documents how they
were made and regenerates them byte-identically:

    python3 gen_fixtures.py

Tensor payloads use a deterministic pattern — byte j of the tensor with seed s is
(j*131 + s*17 + 7) % 256 — which Weights.lean recomputes for the byte-exactness gates.
Uses only the Python standard library (no numpy).
"""
import json
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

DTYPE_SIZE = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "F16": 2, "BF16": 2,
              "I32": 4, "F32": 4, "I64": 8, "F64": 8}


def pattern(nbytes, seed):
    return bytes(((j * 131 + seed * 17 + 7) % 256 for j in range(nbytes)))


def tensor_bytes(dtype, shape):
    nbytes = DTYPE_SIZE[dtype]
    for dim in shape:
        nbytes *= dim
    return nbytes


def write_safetensors(path, tensors, metadata=None):
    """tensors: list of (name, dtype, shape, seed). Returns the manifest entry."""
    header = {}
    blobs = []
    offset = 0
    manifest = []
    for name, dtype, shape, seed in tensors:
        nbytes = tensor_bytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + nbytes]}
        blobs.append(pattern(nbytes, seed))
        manifest.append({"name": name, "dtype": dtype, "shape": list(shape),
                         "bytes": nbytes, "seed": seed})
        offset += nbytes
    if metadata:
        header["__metadata__"] = metadata
    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for blob in blobs:
            f.write(blob)
    return manifest


# Tiny gate configuration from docs/QWEN36_MEGAKERNEL.md (hidden 256, 4 layers
# lin/lin/lin/full, 2x64 key / 4x64 value heads, 4/2 attention heads x 64,
# intermediate 512, vocab 512).
HIDDEN = 256
KEY_DIM = 2 * 64       # linear_num_key_heads * key_head_dim
VALUE_DIM = 4 * 64     # linear_num_value_heads * value_head_dim
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
NUM_V_HEADS = 4
HEAD_V = 64
Q_ROWS = 2 * 4 * 64    # q_proj emits the output gate in its second half
KV_ROWS = 2 * 64
O_COLS = 4 * 64
HEAD_DIM = 64
INTER = 512
VOCAB = 512


def tiny_model_tensors(seed_start):
    """The complete tensor set (names + shapes) of the tiny configuration, all BF16."""
    ts = [("model.language_model.embed_tokens.weight", [VOCAB, HIDDEN])]
    for layer in range(4):
        p = f"model.language_model.layers.{layer}."
        ts.append((p + "input_layernorm.weight", [HIDDEN]))
        if layer == 3:  # i % 4 == 3 -> full_attention
            ts += [
                (p + "self_attn.q_proj.weight", [Q_ROWS, HIDDEN]),
                (p + "self_attn.k_proj.weight", [KV_ROWS, HIDDEN]),
                (p + "self_attn.v_proj.weight", [KV_ROWS, HIDDEN]),
                (p + "self_attn.o_proj.weight", [HIDDEN, O_COLS]),
                (p + "self_attn.q_norm.weight", [HEAD_DIM]),
                (p + "self_attn.k_norm.weight", [HEAD_DIM]),
            ]
        else:
            ts += [
                (p + "linear_attn.in_proj_qkv.weight", [CONV_DIM, HIDDEN]),
                (p + "linear_attn.in_proj_z.weight", [VALUE_DIM, HIDDEN]),
                (p + "linear_attn.in_proj_b.weight", [NUM_V_HEADS, HIDDEN]),
                (p + "linear_attn.in_proj_a.weight", [NUM_V_HEADS, HIDDEN]),
                (p + "linear_attn.conv1d.weight", [CONV_DIM, 1, 4]),
                (p + "linear_attn.A_log", [NUM_V_HEADS]),
                (p + "linear_attn.dt_bias", [NUM_V_HEADS]),
                (p + "linear_attn.norm.weight", [HEAD_V]),
                (p + "linear_attn.out_proj.weight", [HIDDEN, VALUE_DIM]),
            ]
        ts += [
            (p + "post_attention_layernorm.weight", [HIDDEN]),
            (p + "mlp.gate_proj.weight", [INTER, HIDDEN]),
            (p + "mlp.up_proj.weight", [INTER, HIDDEN]),
            (p + "mlp.down_proj.weight", [HIDDEN, INTER]),
        ]
    ts += [("model.language_model.norm.weight", [HIDDEN]),
           ("lm_head.weight", [VOCAB, HIDDEN])]
    return [(name, "BF16", shape, seed_start + i) for i, (name, shape) in enumerate(ts)]


def main():
    os.makedirs(os.path.join(FIXTURES, "sharded"), exist_ok=True)
    os.makedirs(os.path.join(FIXTURES, "complete"), exist_ok=True)

    manifest = {}

    # Single file: one tensor per supported float dtype plus a __metadata__ block.
    manifest["tiny.safetensors"] = write_safetensors(
        os.path.join(FIXTURES, "tiny.safetensors"),
        [("w_qkv_f32", "F32", [8, 16], 1),
         ("w_gate_bf16", "BF16", [4, 32], 2),
         ("w_up_f16", "F16", [16], 3)],
        metadata={"format": "pt"})

    # Two-shard directory resolved through model.safetensors.index.json.
    shard1 = [("tensor.alpha", "F32", [2, 4], 4)]
    shard2 = [("tensor.beta", "BF16", [3, 5], 5),
              ("tensor.gamma", "F16", [7], 6)]
    manifest["sharded/model-00001-of-00002.safetensors"] = write_safetensors(
        os.path.join(FIXTURES, "sharded", "model-00001-of-00002.safetensors"), shard1)
    manifest["sharded/model-00002-of-00002.safetensors"] = write_safetensors(
        os.path.join(FIXTURES, "sharded", "model-00002-of-00002.safetensors"), shard2)
    total = sum(tensor_bytes(dt, sh) for _, dt, sh, _ in shard1 + shard2)
    index = {
        "metadata": {"total_size": total},
        "weight_map": {
            "tensor.alpha": "model-00001-of-00002.safetensors",
            "tensor.beta": "model-00002-of-00002.safetensors",
            "tensor.gamma": "model-00002-of-00002.safetensors",
        },
    }
    with open(os.path.join(FIXTURES, "sharded", "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    # Complete tiny-model checkpoint: every tensor the registry expects, nothing else.
    manifest["complete/tiny_model.safetensors"] = write_safetensors(
        os.path.join(FIXTURES, "complete", "tiny_model.safetensors"),
        tiny_model_tensors(seed_start=10))

    manifest["pattern"] = "byte j of tensor with seed s = (j*131 + s*17 + 7) % 256"
    with open(os.path.join(FIXTURES, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("fixtures regenerated under", FIXTURES)


if __name__ == "__main__":
    main()
