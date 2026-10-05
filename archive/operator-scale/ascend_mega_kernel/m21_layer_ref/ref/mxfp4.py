"""MXFP4 (E2M1 + E8M0, group 32) dequantization for the Qwen3.8 checkpoint.

The checkpoint's own spec is authoritative here, because vLLM main has no
``ascend`` quant method (docs/14 §11 item 6):

  CKPT:README.quant.md:4
      - 元素: FP4 E2M1，可表示值 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}，饱和到 ±6
  CKPT:README.quant.md:5
      - group_size: 32（沿 in_features 分组）
  CKPT:README.quant.md:6
      - scale: per-block OCP E8M0（uint8, byte = exponent + 127），`scale = 2**(floor(log2(max_abs)) - 2)`
  CKPT:README.quant.md:7
      - packing: uint8-nibble-lohi（低 nibble = 偶数 in 索引，高 nibble = 奇数）
  CKPT:README.quant.md:33-35
      权重布局：量化张量以原名存 packed uint8（最后维减半），scale 存
      `<name>.weight_scale`（或 `<name>.weight` -> `<name>.weight_scale`）。
      3D expert 张量 `[E, out, in]` -> packed `[E, out, ceil(in/2)]` + scale `[E, out, in/32]`。

The nibble ordering / E2M1 code table / E8M0 bias also match this repo's existing
numpy golden (R:tools/golden/moe_block_ref.py:57-99,175-180) and m13's device-side
checker (R:m13_moe_layer/check_ref.py:114-122), which is an independent confirmation.

NOTE (re-quantization only, not dequantization): the repo golden's *packer*
(R:tools/golden/moe_block_ref.py:111-124) uses `scale = 2**ceil(log2(amax/6))`,
which differs from the checkpoint/hardware `floor(log2(amax)) - 2` whenever
`frac(log2(amax)) > log2(6) - 2`. That disagreement is already recorded in
R:m13_moe_layer/README.md:146-153 and is unrelated to reading the checkpoint.

NOTE on the `file:line` anchors below: the line number is a LOOKUP HINT against
the pinned base commit shown above; line numbers drift when upstream moves.
The stable reference is the SYMBOL. `python3 tools/check_anchors.py --verbose`
prints, for every anchor, which file it resolves to and which symbol it lands in
(it also fails if an anchor is ambiguous or out of range), and
`selfcheck.py` section I runs that check. Project rule 2026-09-26.

vLLM anchors for the *layout* of the packed expert tensors:
  vllm/model_executor/layers/fused_moe/routed_experts.py:1106-1111
      (fused_mapping: gate_up_proj shard 0 -> w1 = gate, shard 1 -> w3 = up)
  vllm/model_executor/layers/fused_moe/routed_experts.py:936-941
      experts_shard = fused_weight.chunk(2, dim=1)[expert_id]   # dim 1 = N
"""

from __future__ import annotations

import numpy as np

# OCP MX E2M1 positive values for codes 0..7 (sign is bit 3).
# Matches R:tools/golden/moe_block_ref.py:57-59.
E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

GROUP_SIZE = 32


def e2m1_decode(codes: np.ndarray) -> np.ndarray:
    """Decode E2M1 nibbles (uint8 0..15) to float32."""
    codes = np.asarray(codes, dtype=np.uint8)
    sign = np.where(codes & 0x8, -1.0, 1.0).astype(np.float32)
    return sign * E2M1_POS[codes & 0x7]


def e8m0_decode(scale_bytes: np.ndarray) -> np.ndarray:
    """Decode E8M0 scale bytes to float32: value = 2 ** (byte - 127)."""
    b = np.asarray(scale_bytes, dtype=np.int32)
    return np.exp2((b - 127).astype(np.float32))


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """uint8 [..., K/2] -> codes uint8 [..., K], low nibble = even index."""
    packed = np.asarray(packed, dtype=np.uint8)
    n, half = packed.shape[-2], packed.shape[-1]
    codes = np.empty(packed.shape[:-1] + (half * 2,), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F  # CKPT:README.quant.md:7 (lohi)
    codes[..., 1::2] = packed >> 4
    return codes


def dequant_mxfp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize one MXFP4 matrix.

    Args:
      packed: uint8 [N, K/2] (nibble-lohi encoding of the K axis).
      scale:  uint8 [N, K/32] (E8M0 per 32-element group along K).

    Returns:
      float32 [N, K].
    """
    packed = np.asarray(packed, dtype=np.uint8)
    scale = np.asarray(scale, dtype=np.uint8)
    k = packed.shape[-1] * 2
    codes = unpack_nibbles(packed)
    vals = e2m1_decode(codes).reshape(packed.shape[0], k // GROUP_SIZE, GROUP_SIZE)
    sc = e8m0_decode(scale).astype(np.float32)
    return (vals * sc[:, :, None]).reshape(packed.shape[0], k)


def dequant_mxfp4_torch(packed_bits, scale_bits, torch):
    """Torch-native equivalent of ``dequant_mxfp4`` (CPU, fp32 out).

    Kept torch-native so the harness does not copy GBs through numpy; the
    arithmetic is identical (exact e2m1 table lookup, exact power-of-two scale).
    """
    codes = torch.empty(
        packed_bits.shape[:-1] + (packed_bits.shape[-1] * 2,), dtype=torch.uint8
    )
    codes[..., 0::2] = packed_bits & 0x0F
    codes[..., 1::2] = packed_bits >> 4
    table = torch.tensor(E2M1_POS, dtype=torch.float32)
    sign = torch.where(codes & 0x8 != 0, -1.0, 1.0)
    mag = table[(codes & 0x7).long()]
    vals = (sign * mag).to(torch.float32)
    n, k = vals.shape
    vals = vals.reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    sc = torch.exp2((scale_bits.to(torch.int32) - 127).to(torch.float32))
    return (vals * sc[:, :, None]).reshape(n, k)


def e2m1_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round-to-nearest-evil-free quantizer used only for re-quantization tests.

    Returns (codes uint8 [.., K] with sign in bit 3, saturation flag).
    Rounds to nearest, ties away from zero (CKPT:README.quant.md:8).
    """
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    mag = np.abs(x)
    # nearest of E2M1_POS, ties away from zero
    idx = np.searchsorted(E2M1_POS, mag, side="left")
    idx = np.clip(idx, 1, 7)
    lower = E2M1_POS[idx - 1]
    upper = E2M1_POS[idx]
    take_upper = (mag - lower) >= (upper - mag)
    mag_q = np.where(take_upper, upper, lower)
    mag_q = np.where(mag >= E2M1_POS[-1], E2M1_POS[-1], mag_q)
    code = np.where(take_upper, idx, idx - 1).astype(np.uint8)
    code = np.where(mag >= E2M1_POS[-1], 7, code).astype(np.uint8)
    code = np.where(sign < 0, code | 0x8, code)
    sat = mag > E2M1_POS[-1]
    return code.astype(np.uint8), sat
