"""MXFP4 pack/unpack utilities and a float-reference golden for the Qwen3.8
(Qwen4Exp) MoE block vertical slice.

Target model: Qwen3.8-Flash-Next-MXFP4 (model_type ``qwen4_exp``).
The MoE block covered by this reference is the vertical slice

    router -> top-k routing -> token rearrange -> per-expert MXFP4 gate_up GEMM
    -> SwiGLU -> per-expert MXFP4 down GEMM -> weighted combine
    (+ sigmoid-gated shared expert)

plus the layer-chain norm goldens (m6 semantics: ``rmsnorm_ref``) that bracket it:

    x_res -> m6#1 -> x_norm1 -> <MoE block above> -> moe_output -> m6#2 -> y_final + res2

Activation quantization (w4a4) is *not* part of this float reference — the
quantization-aware reference lives in :mod:`qaware_ref` and shares the quantizers
below.

Layout conventions (verified against the HF checkpoint safetensors headers,
see tools/golden/README.md):
  * A linear weight W is logically ``[N, K]`` (out_features x in_features),
    row-major, K contiguous.
  * Packed weight: uint8 ``[N, K // 2]``. Byte ``j`` of row ``n`` holds two
    adjacent K elements in "lohi" order: low nibble = element ``2j``
    (lower k), high nibble = element ``2j + 1``.
  * Scales: uint8 ``[N, K // 32]``, one E8M0 byte per group of 32 consecutive
    K elements of the row. E8M0 is bias-127: scale value = ``2**(byte - 127)``.
  * E2M1 codes (sign-magnitude, exponent bias 1, finite only, no NaN/inf):
    code 0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6.
  * Group scale (this directory's *synthetic* weights): ``s = 2**ceil(log2(amax / 6))``
    per (row, k-group) so that ``max|w / s| <= 6`` (6 is the largest E2M1 value) — the
    "legacy ceil" rule.  Real checkpoint weights arrive pre-packed by the model author's
    CPU RTN tool and are consumed byte-for-byte.  The *activation* side follows the
    official ops-nn sequence (OCP/floor) — see :func:`quantize_ocp`.
  * Activations / router weights are bf16; the reference computes everything
    in float32 and rounds outputs to bf16 (round-to-nearest-even).

Note on round trips: dequantization is a fixed point of the
quantize->dequantize cycle — ``unpack(pack(unpack(p, s))) == unpack(p, s)``
exactly (values, not bytes). Re-packing dequantized weights may pick a
smaller scale when a group's amax lands exactly on an E2M1 level boundary
(e.g. 3*s -> re-pack with s/2 and doubled codes); the dequantized values are
unchanged, but the packed bytes can differ.

Routing semantics follow Qwen3-Next / Qwen4Exp (vLLM ``Qwen3NextSparseMoeBlock``,
``norm_topk_prob`` defaults to True):
  * logits = x @ router_weight.T (float32)
  * scores = softmax(logits) over all experts (float32)
  * top-k by score, sorted by descending score; ties broken by lower expert id
  * routing weights renormalized to sum to 1 per token
  * shared expert output = sigmoid(x @ shared_gate_w.T) * MLP_shared(x),
    added to the weighted expert sum.

Device fp32 FTZ is modelled: the hardware flushes every subnormal fp32 result
(``|v| < 2**-126``) to zero — docs/05 §6.1, user ruling "fp32 次正规 FTZ 是硬件模式，
不在规避范围".  Without it a deep-tail router (``logit <= -87``) keeps *distinct*
subnormal softmax scores and selects a **different** top-k set than the device
(M42: 20/4097 rows at real scale).  See :func:`router_topk` for the three device
materialization points (FTZ#1..#3) and ``tools/golden/README.md`` §"FTZ 建模".

Pure numpy (2.5.1), no torch. Python 3.12.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# E2M1 / E8M0 numeric fundamentals
# ---------------------------------------------------------------------------

# Positive E2M1 values for codes 0..7 (sign bit handled separately).
# OCP MX E2M1 is finite-only: max normal = 1.1b * 2^2 = 6.0.
E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_MAX = 6.0


def e2m1_decode(codes: np.ndarray) -> np.ndarray:
    """Decode E2M1 nibbles (uint8, 0..15) to float32 values."""
    codes = np.asarray(codes, dtype=np.uint8)
    sign = np.where(codes & 0x8, -1.0, 1.0).astype(np.float32)
    mag = E2M1_POS[codes & 0x7]
    return sign * mag


def e2m1_encode(values: np.ndarray) -> np.ndarray:
    """Encode float32 values to E2M1 codes (uint8, 0..15).

    Nearest-value quantization with round-to-nearest-even on the code LSB for
    exact ties. The sign bit is preserved independently (including -0.0).
    """
    v = np.asarray(values, dtype=np.float32)
    sign = np.where(np.signbit(v), 8, 0).astype(np.uint8)
    a = np.abs(v.astype(np.float64))
    # insertion points of |v| into the sorted positive levels
    idx = np.searchsorted(E2M1_POS, a, side="left")
    idx_hi = np.clip(idx, 0, len(E2M1_POS) - 1)
    idx_lo = np.clip(idx - 1, 0, len(E2M1_POS) - 1)
    d_hi = np.abs(E2M1_POS[idx_hi] - a)
    d_lo = np.abs(a - E2M1_POS[idx_lo])
    # tie -> even code (code LSB == 0)
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & ((idx_hi & 1) == 0))
    code = np.where(pick_hi, idx_hi, idx_lo).astype(np.uint8)
    return sign | code


def e8m0_decode(scale_bytes: np.ndarray) -> np.ndarray:
    """Decode E8M0 scale bytes (uint8) to float32 powers of two.

    E8M0 is exponent-only with bias 127: value = 2**(byte - 127).
    Byte 0xFF is reserved for NaN in OCP MX; it is not produced here.
    """
    b = np.asarray(scale_bytes, dtype=np.uint32).astype(np.int32)
    return np.exp2((b - 127).astype(np.float32))


def e8m0_encode_pow2(exponents: np.ndarray) -> np.ndarray:
    """Encode integer base-2 exponents to E8M0 bytes (bias 127)."""
    e = np.asarray(exponents)
    b = e.astype(np.int64) + 127
    if np.any((b < 0) | (b > 254)):
        raise ValueError("E8M0 exponent out of representable range [-127, 127]")
    return b.astype(np.uint8)


def group_scale_exp(amax: np.ndarray) -> np.ndarray:
    """Exponent of the MXFP4 group scale: e = ceil(log2(amax / 6)).

    amax is a float32 array of per-group absolute maxima. Returns integer
    exponents such that scale = 2**e and max|q| <= 6 after quantization.
    amax == 0 maps to e = -127 (smallest E8M0), which only scales zeros.
    """
    a = np.asarray(amax, dtype=np.float64)
    e = np.zeros(a.shape, dtype=np.int64)
    nonzero = a > 0.0
    e[nonzero] = np.ceil(np.log2(a[nonzero] / E2M1_MAX))
    e[nonzero] = np.clip(e[nonzero], -127, 127)
    e[~nonzero] = -127
    return e


# ---------------------------------------------------------------------------
# Activation-side quantization: the official sequence is authoritative
# ---------------------------------------------------------------------------
#
# User ruling (2026-09-26, relayed by the tower): the activation quantization spec is
# "whatever the official implementation does".  The authoritative sequence is the
# official ops-nn kernel
#
#   /workspace/ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/
#       add_rms_norm_dynamic_mx_quant_common.h
#   (same shape as opp built-in .../ops_nn/ascendc/dynamic_mx_quant/arch35/
#    dynamic_mx_quant_tail_axis.h)
#
# which is what `npu_dynamic_mx_quant(dst_type=float4_e2m1fn_x2, round_mode="round")`
# / op type `DynamicMxQuant` runs, and which Qwen3.8's own quantization_config
# references (docs/13-mx-quant-primitives.md §1/§4, `config.json`
# `quantization_config.npu_reference`).  It is a three-stage block quantizer:
#
#   (1) group max in the *exponent domain*  (And 0x7F80 x2 + Max + ReduceDataBlock<MAX>)
#   (2) E8M0 scale + halfScale              (Sub / ShiftRights / Select)
#   (3) multiply in bf16 + fp4 Cast + pack
#
# and it is OCP semantics: `scale = 2**(floor(log2(amax)) - emax)` — a *floor*, not a
# ceil/round (three independent official sources agree, docs/13 §4.2).
# `quantize_ocp` below transcribes it statement by statement (rule id "floor", which is
# what the tower's ruling calls "OCP/floor"); `selfcheck.py` proves the transcription is
# byte-identical to the independently written `m13_moe_layer/check_ref.py` quantizer
# (itself byte-verified against the NPU) for finite, non-degenerate groups, and pins the
# degenerate corners (zero/denormal groups -> all-zero codes; ±Inf/NaN groups -> E8M0
# NaN byte 0xFF and all-zero codes).
#
# The "ceil" rule (`2**ceil(log2(amax/6))`, nearest/tie-even data codes) is **legacy**:
# it is the rule of the m3 scalar quantizer that docs/12 §6 item C replaced, and of this
# directory's own synthetic weight packing.  It is kept as an explicitly labeled legacy
# option so the pre-ruling goldens stay reproducible — it is *not* the reference for
# grading a device kernel.
#
# Note on weights: the checkpoint's packed weights were produced by the model author's
# CPU RTN tool (`cpu_rtn_provenance.tool = qwen3.8-flash-next-cpu-rtn-mxfp4`, per the
# tower's ruling) and we **consume those bytes as given** — no re-packing, so no rule of
# ours is involved.  `pack_mxfp4` (ceil) only describes how this directory's *synthetic*
# test weights are generated.

# official constants (add_rms_norm_dynamic_mx_quant_common.h:71-98)
FP4_E2M1_BF16_MAX_EXP = 0x0100  # emax for e2m1 (= 6.0's exponent field << 7)
MAX_EXP_FOR_BF16 = 0x7F80  # expMaskBF16
SHR_NUM_FOR_BF16 = 7
BF16_EXP_BIAS = 0x7F00
HALF_SCALE_NAN = 0x7F81  # NAN_CUSTOMIZATION (halfScale for non-finite groups)
FP8_NAN = 0x00FF  # E8M0 NaN byte (scale for non-finite groups)

SCALE_RULES = ("floor", "ceil")

SCALE_RULE_DOC = {
    "floor": ("the official ops-nn sequence (OCP, docs/13 §4): "
              "shared = clamp(bf16 exponent field of amax, >= emax=0x0100) - emax; "
              "e8m0 byte = shared >> 7 i.e. scale = 2**(floor(log2(amax)) - 2); "
              "non-finite group -> byte 0xFF, degenerate (zero/denormal) group -> byte 0 "
              "with halfScale 0.  Default and authoritative."),
    "ceil": ("LEGACY: e8m0 byte = 127 + ceil(log2(amax / 6)), i.e. "
             "scale = 2**ceil(log2(amax/6)) — the replaced m3 scalar quantizer rule and "
             "this directory's synthetic weight-packing rule; kept for reproducing "
             "pre-ruling goldens, not for grading device kernels."),
}

SCALE_RULE_ROUNDING = {
    "floor": "CAST_ROUND on the final fp4 cast (:755-758): round half away from zero, saturating at ±6",
    "ceil": "nearest E2M1 level, ties to even code (legacy)",
}


def exp_field(amax: np.ndarray) -> np.ndarray:
    """bf16 biased exponent field of ``amax`` (uint16 0..255), RNE-rounded to bf16 first.

    Convenience for the finite-value path of the official sequence; the quantizer itself
    maxes the masked fields bitwise (:func:`quantize_ocp`), which also covers Inf/NaN.
    """
    bits = f32_to_bf16_bits(np.asarray(amax, dtype=np.float32))
    return ((bits >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int16)


def e8m0_bytes_floor(amax: np.ndarray) -> np.ndarray:
    """E8M0 bytes for finite ``amax``: clip(bf16 exponent field - 2, 0, 254).

    The finite-group path of the official sequence (steps 4-6); see :func:`quantize_ocp`
    for the complete transcription (non-finite / degenerate groups included).
    """
    b = exp_field(amax).astype(np.int32) - 2
    return np.clip(b, 0, 254).astype(np.uint8)


def e8m0_bytes_ceil(amax: np.ndarray) -> np.ndarray:
    """LEGACY E8M0 bytes: 127 + ceil(log2(amax / 6))."""
    return e8m0_encode_pow2(group_scale_exp(amax))


def scale_rule_bytes(amax: np.ndarray, rule: str) -> np.ndarray:
    """E8M0 bytes for finite ``amax`` under ``rule`` ∈ SCALE_RULES (finite-group path)."""
    if rule == "floor":
        return e8m0_bytes_floor(amax)
    if rule == "ceil":
        return e8m0_bytes_ceil(amax)
    raise ValueError(f"unknown scale rule {rule!r}; expected {SCALE_RULES}")


def e2m1_encode_away(values: np.ndarray) -> np.ndarray:
    """Encode float32 values to E2M1 codes with round-half-away-from-zero + saturation.

    This is the data-side rounding of the device CAST_ROUND path (m5/m2): ties go away
    from zero (so exactly-tied magnitudes pick the larger level) and anything beyond 6
    saturates to code 7 (= 6.0).  The sign bit is preserved independently.
    """
    v = np.asarray(values, dtype=np.float32)
    sign = np.where(np.signbit(v), 8, 0).astype(np.uint8)
    a = np.abs(v.astype(np.float64))
    idx = np.clip(np.searchsorted(E2M1_POS, a, side="left"), 0, len(E2M1_POS) - 1)
    lo = np.clip(idx - 1, 0, len(E2M1_POS) - 1)
    d_hi = E2M1_POS[idx] - a
    d_lo = a - E2M1_POS[lo]
    pick_hi = d_hi <= d_lo  # ties (and saturation, d_hi < 0) go to the larger level
    code = np.where(pick_hi, idx, lo).astype(np.uint8)
    return sign | code


def _pack_codes(codes: np.ndarray) -> np.ndarray:
    """lohi-pack ``[rows, K]`` E2M1 codes into ``[rows, K//2]`` bytes."""
    lo = codes[:, 0::2].astype(np.uint16)
    hi = codes[:, 1::2].astype(np.uint16)
    return (lo | (hi << 4)).astype(np.uint8)


def _activation_rows(x: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Validate an activation block and view it as ``[rows, groups, 32]``.

    The device quantizer loads a **bf16** tensor, so the input is bf16-rounded (RNE) first
    — a no-op for the dataset's activations (already on the bf16 grid), but it keeps the
    reference faithful for arbitrary float32 callers.
    """
    x = f32_to_bf16(np.asarray(x, dtype=np.float32))
    if x.ndim != 2:
        raise ValueError(f"expected [rows, K] activations, got shape {x.shape}")
    rows, k = x.shape
    if k % GROUP_SIZE != 0:
        raise ValueError(f"K must be a multiple of {GROUP_SIZE}, got {k}")
    return x, rows, k, x.reshape(rows, k // GROUP_SIZE, GROUP_SIZE)


def quantize_ocp(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The official ops-nn MXFP4 quantization sequence (OCP), line by line.

    Authority: ``add_rms_norm_dynamic_mx_quant_common.h`` — ``MxQuantComputeMaxExpOCP``,
    ``MxQuantComputeScaleOCP`` and the bf16 branch of ``MxQuantComputeDataFP4``; see
    docs/13-mx-quant-primitives.md §4/§5 for the 15-step diff against our device kernels.
    Line numbers below are **hints into the ops-nn copy of that header at the time of
    writing and drift with the code** — anchor by symbol, not by range (the packaged CANN
    copy of the same header numbers differently: the three functions there start at
    :274 / :337 / :688).  Every statement below carries the official line it transcribes
    as a hint (constants from :71-98), also ops-nn:
    expMask 0x7F80 / emax 0x0100 / SHR 7 / bf16 bias 0x7F00 / scale NaN 0x00FF /
    halfScale NaN 0x7F81.

    Returns ``(packed [rows, K//2], scale uint8 [rows, K//32])``.

    Corners matter and are reproduced here (M28 §6): a finite group whose exponent field
    clamps to emax (all-zero / bf16-denormal magnitudes) gets halfScale 0, i.e. **all
    codes 0**; a group containing ±Inf or NaN gets the E8M0 NaN byte 0xFF and halfScale
    0x7F81, i.e. also all codes 0 (``Mul(±Inf, NaN) = NaN`` -> cast -> 0).

    Status of the ``:413`` halfScale override on the device side (checked 2026-09-26):
    when this reference was written the device kernels missed it and returned ±6 for the
    Inf corner.  It is now present in **all three kernels that carry this sequence** --
    ``m2_mxfp4_quant.asc`` / ``m5_swiglu_quant.asc`` / ``m13_moe_layer.asc``, each in its
    ``MxQuantComputeScale`` (``nanRegTensor = Duplicate(NAN_CUSTOMIZATION = 0x7F81)`` +
    ``Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmp)``; grep each file for
    ``0x7F81``) -- so reference and device agree on the Inf corner (M32 逐字节 + M50 的
    ``quant-inf`` 40 判定项，见 m13_moe_layer/README §5.4 与 docs/13 §6，设备/参考日志
    ``m13_moe_layer/evidence/m50_quant_inf_after.log``、``m50_quant_ref_inf_after.log``）。
    The archived ``tools/golden/evidence/*.log`` still print the pre-fix sentence; they are
    kept as the record of that moment, not as current fact.  ``selfcheck.py``'s note used to
    do the same at runtime (it was reprinted into every new log); since M56 it instead
    re-checks the device side live (``grep 0x7F81`` in the three kernels) and prints the
    counts, so it can no longer go stale on its own.
    """
    x, rows, k, x3 = _activation_rows(x)

    # (1) group max in the exponent domain (``MxQuantComputeMaxExpOCP``; ops-nn copy :308-355,
    #     a hint that drifts): And 0x7F80 x2 + Max + ReduceDataBlock<MAX>
    maxexp = np.max(f32_to_bf16_bits(x3) & np.uint16(MAX_EXP_FOR_BF16), axis=2).astype(np.int32)
    # (3) non-finite flag (:401): Compare NE(vdMaxExp, expMask=0x7F80)
    nonfinite = maxexp == MAX_EXP_FOR_BF16
    # (4) clamp the lower bound (:402-403): Compare LE + Select(vdMaxExp, 0x0100, ...)
    shared = np.maximum(maxexp, FP4_E2M1_BF16_MAX_EXP) - FP4_E2M1_BF16_MAX_EXP  # (5) Sub (:404)
    # (6) E8M0 byte (:405) + (7) non-finite -> 0xFF (:406)
    byte = np.where(nonfinite, FP8_NAN, shared >> SHR_NUM_FOR_BF16).astype(np.uint8)
    # (9) halfScale = bf16(0x7F00 - shared) (:412); (10a) non-finite -> 0x7F81 (:413);
    # (11) shared == 0 -> 0 (:414).  (10b/10c are unreachable: shared is a multiple of
    # 0x80 and 0x7F00 = 254*0x80 is never produced.)
    half_bits = (BF16_EXP_BIAS - shared).astype(np.uint16)
    half_bits = np.where(nonfinite, np.uint16(HALF_SCALE_NAN), half_bits).astype(np.uint16)
    half_bits = np.where(shared == 0, np.uint16(0), half_bits).astype(np.uint16)
    half = bf16_bits_to_f32(half_bits)[:, :, None]
    # (12) bf16-domain multiply (:753-754); (14) fp4 cast with the round mode trait
    # (:755-758, castTraitRM<round_mode="round"> = CAST_ROUND, tie away from zero).
    # Inf * NaN = NaN is intentional here (the official sequence lets the cast turn it
    # into 0), hence the suppressed invalid-value warning.
    with np.errstate(invalid="ignore"):
        h = x3 * half
    codes = e2m1_encode_away(h)
    codes = np.where(np.isnan(h), np.uint8(0), codes)  # NaN -> 0 on the fp4 cast
    return _pack_codes(codes.reshape(rows, k)), byte


def quantize_ceil_legacy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Legacy activation quantization: ``scale = 2**ceil(log2(amax/6))`` + tie-even codes.

    This is the rule of the m3 scalar quantizer (replaced by the m5 VEC path in
    docs/12 §6 item C) and of :func:`pack_mxfp4`'s synthetic weight packing.  It is kept
    as an explicitly labeled legacy option next to the official OCP sequence — usable to
    reproduce the pre-ruling goldens, not to grade a device kernel.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    x, rows, k, x3 = _activation_rows(x)
    scale_bytes = e8m0_bytes_ceil(np.max(np.abs(x3), axis=2))
    scale_f = e8m0_decode(scale_bytes).astype(np.float32)[:, :, None]
    codes = e2m1_encode(x3 / scale_f)
    return _pack_codes(codes.reshape(rows, k)), scale_bytes


def quantize_activations(x: np.ndarray, rule: str = "floor"):
    """MXFP4-quantize activation rows ``[rows, K]`` (K % 32 == 0) under ``rule``.

    Returns ``(packed uint8 [rows, K//2], scale uint8 [rows, K//32])`` with the same
    lohi nibble order and group-32-along-K layout as :func:`pack_mxfp4`, so a kernel
    can consume golden weights and golden activations with one unpack routine.

    * ``rule="floor"`` (**default — the authoritative one**): the official ops-nn
      sequence (:func:`quantize_ocp`), i.e. OCP semantics
      ``scale = 2**(floor(log2(amax)) - emax)`` with the data cast in CAST_ROUND; this is
      what ``npu_dynamic_mx_quant(dst_type=float4_e2m1fn_x2, round_mode="round")`` runs,
      and it is byte-identical to our device kernels for finite, non-degenerate groups.
    * ``rule="ceil"``: legacy (see :func:`quantize_ceil_legacy`).
    """
    if rule == "floor":
        return quantize_ocp(x)
    if rule == "ceil":
        return quantize_ceil_legacy(x)
    raise ValueError(f"unknown activation scale rule {rule!r}; expected {SCALE_RULES}")

# ---------------------------------------------------------------------------
# MXFP4 pack / unpack
# ---------------------------------------------------------------------------


def pack_mxfp4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack a float32 weight ``[N, K]`` into MXFP4 (packed nibbles + E8M0 scales).

    Returns (packed, scales):
      packed: uint8 ``[N, K // 2]``, lohi nibble order along K
      scales: uint8 ``[N, K // 32]``, one E8M0 byte per 32-element K group
    Requires K % 32 == 0.
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected [N, K] weight, got shape {w.shape}")
    n, k = w.shape
    if k % 32 != 0:
        raise ValueError(f"K must be a multiple of 32 (group size), got {k}")
    groups = k // 32
    w3 = w.reshape(n, groups, 32)
    amax = np.max(np.abs(w3), axis=2)
    exps = group_scale_exp(amax)
    scales = e8m0_encode_pow2(exps)  # [N, K//32]
    scale_f = np.exp2(exps.astype(np.float32))  # [N, K//32]
    q = e2m1_encode(w3 / scale_f[:, :, None])  # [N, groups, 32] codes 0..15
    q = q.reshape(n, k)
    lo = q[:, 0::2].astype(np.uint16)
    hi = q[:, 1::2].astype(np.uint16)
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed, scales


def unpack_mxfp4(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Unpack MXFP4 (packed nibbles + E8M0 scales) to float32 ``[N, K]``.

    packed: uint8 ``[N, K // 2]`` (lohi), scales: uint8 ``[N, K // 32]``.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("packed and scales must be 2-D [N, K//2] / [N, K//32]")
    n, kh = packed.shape
    k = kh * 2
    if scales.shape != (n, k // 32):
        raise ValueError(
            f"scales shape {scales.shape} does not match packed shape {packed.shape}"
        )
    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    vals = e2m1_decode(codes).reshape(n, k // 32, 32)
    scale_f = e8m0_decode(scales).astype(np.float32)[:, :, None]
    return (vals * scale_f).reshape(n, k)


# ---------------------------------------------------------------------------
# bf16 helpers (round-to-nearest-even), bit-exact without ml_dtypes
# ---------------------------------------------------------------------------


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Convert float32 array to bf16 bit patterns (uint16, RNE)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    nan_mask = (b & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(nan_mask, np.uint32(0x7FC0), rounded)
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Convert bf16 bit patterns (uint16) to float32 array."""
    b = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return b.view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 -> float32 round trip (value now exactly representable)."""
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


# ---------------------------------------------------------------------------
# Binary tensor I/O with JSON descriptors
# ---------------------------------------------------------------------------

# semantic dtype name -> (numpy dtype for storage, kind)
DTYPES = {
    "bf16": (np.uint16, "bf16"),
    "fp32": (np.float32, "fp32"),
    "int32": (np.int32, "int32"),
    "uint8": (np.uint8, "uint8"),
}


def _element_size(dtype_name: str) -> int:
    return np.dtype(DTYPES[dtype_name][0]).itemsize


def strides_of(shape) -> list[int]:
    """Row-major (C order) strides in elements."""
    strides = []
    acc = 1
    for d in reversed(shape):
        strides.append(acc)
        acc *= d
    return list(reversed(strides))


def write_bin(path: Path, array: np.ndarray, dtype_name: str) -> dict:
    """Write a tensor to ``path`` as raw little-endian data.

    For dtype "bf16", ``array`` is a float32 numpy array which is first rounded
    to bf16 (RNE) and stored as uint16 bit patterns. Other dtypes store the
    numpy array as-is (must match the declared dtype). Returns the descriptor
    dict that is also written to ``path`` + ".json" (including the sha256 of the
    stored bytes, so a dataset can be verified without shipping every bin).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = DTYPES[dtype_name][1]
    if kind == "bf16":
        stored = f32_to_bf16_bits(array)
    else:
        stored = np.ascontiguousarray(array, dtype=DTYPES[dtype_name][0])
    stored.tofile(path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    desc = {
        "tensor": path.name,
        "shape": list(stored.shape),
        "dtype": dtype_name,
        "strides": strides_of(stored.shape),
        "elem_size": _element_size(dtype_name),
        "byte_order": "little",
        "size_bytes": int(stored.nbytes),
        "sha256": digest,
    }
    with open(str(path) + ".json", "w") as f:
        json.dump(desc, f, indent=2)
        f.write("\n")
    return desc


def read_bin(path: Path, desc: dict | None = None) -> np.ndarray:
    """Read a tensor bin back. For bf16 returns float32 values."""
    path = Path(path)
    if desc is None:
        with open(str(path) + ".json") as f:
            desc = json.load(f)
    dtype_name = desc["dtype"]
    stored = np.fromfile(path, dtype=DTYPES[dtype_name][0])
    shape = desc["shape"]
    if stored.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {int(np.prod(shape))} elems, got {stored.size}")
    stored = stored.reshape(shape)
    if DTYPES[dtype_name][1] == "bf16":
        return bf16_bits_to_f32(stored)
    return stored


# ---------------------------------------------------------------------------
# MoE block shapes (Qwen3.8-Flash-Next-MXFP4 text config)
# ---------------------------------------------------------------------------

HIDDEN = 2560  # hidden_size, K of gate_up / N of down
MOE_INTERMEDIATE = 640  # moe_intermediate_size
SHARED_INTERMEDIATE = 640  # shared_expert_intermediate_size
GROUP_SIZE = 32

# Router / shared-expert-gate GEMVs are computed in float32 on bf16 inputs.
RENORM_TOPK = True  # norm_topk_prob default in Qwen3-Next/Qwen4Exp


def expert_weight_shapes(e: int) -> dict:
    """MXFP4 packed tensor shapes for ``e`` routed experts (checkpoint layout)."""
    return {
        "experts.gate_up_proj": (e, 2 * MOE_INTERMEDIATE, HIDDEN // 2),
        "experts.gate_up_proj.weight_scale": (e, 2 * MOE_INTERMEDIATE, HIDDEN // GROUP_SIZE),
        "experts.down_proj": (e, HIDDEN, MOE_INTERMEDIATE // 2),
        "experts.down_proj.weight_scale": (e, HIDDEN, MOE_INTERMEDIATE // GROUP_SIZE),
    }


def shared_weight_shapes() -> dict:
    """MXFP4 packed tensor shapes for the shared expert (checkpoint layout)."""
    return {
        "shared_expert.gate_proj": (SHARED_INTERMEDIATE, HIDDEN // 2),
        "shared_expert.gate_proj.weight_scale": (SHARED_INTERMEDIATE, HIDDEN // GROUP_SIZE),
        "shared_expert.up_proj": (SHARED_INTERMEDIATE, HIDDEN // 2),
        "shared_expert.up_proj.weight_scale": (SHARED_INTERMEDIATE, HIDDEN // GROUP_SIZE),
        "shared_expert.down_proj": (HIDDEN, SHARED_INTERMEDIATE // 2),
        "shared_expert.down_proj.weight_scale": (HIDDEN, SHARED_INTERMEDIATE // GROUP_SIZE),
    }


# ---------------------------------------------------------------------------
# Device fp32 FTZ (flush subnormal results to zero)
# ---------------------------------------------------------------------------

# Smallest positive fp32 normal, 2**-126.  The device hardware flushes every fp32
# arithmetic result with |value| < 2**-126 (subnormal) to +0 — docs/05 §6.1, user
# ruling 2026-09-26: "fp32 次正规 FTZ 是硬件模式，不在规避范围" (hardware behaviour,
# the device is *right*, the reference must model it).
FTZ_MIN_NORMAL = np.float32(1.1754944e-38)


def ftz_f32(v: np.ndarray) -> np.ndarray:
    """Device fp32 FTZ: flush subnormal results (``|v| < 2**-126``) to ``+0``.

    Applied at every point where the device *materializes* an fp32 arithmetic result
    on the router path (see :func:`router_topk`, FTZ#1..#3).  A reference that keeps
    1e-38..1e-45 instead is not "more precise" than the device: it is a different
    value, and in the router top-k it is a different *pairwise ordering* (M42 finding:
    20/4097 rows of the real-weight m=4097 dump selected a different top-10 set).
    """
    v = np.asarray(v, dtype=np.float32)
    return np.where(np.abs(v) < FTZ_MIN_NORMAL, np.float32(0.0), v).astype(np.float32)


# ---------------------------------------------------------------------------
# Router (Qwen3.8 top-k routing semantics)
# ---------------------------------------------------------------------------


def router_topk(
    x: np.ndarray, router_weight: np.ndarray, top_k: int, renorm: bool = RENORM_TOPK
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Qwen3.8 router: softmax over all experts, top-k, renormalize.

    x: float32 ``[m, hidden]`` (bf16 activations upcast), router_weight:
    float32 ``[E, hidden]`` (bf16 weight upcast). Returns
    (logits ``[m, E]``, topk_ids ``[m, k]`` int32, topk_weights ``[m, k]`` f32).
    Ids are sorted by descending score; exact ties break to the lower expert id
    (stable argsort on -scores).

    **Device fp32 FTZ is modelled here** (docs/05 §6.1, user ruling: subnormal FTZ is
    hardware behaviour, the device is the correct side).  The device router (m7/m22:
    max-shift Sub -> Exp -> Sort32/merge on the ``e_i`` -> renorm Div) materializes
    exactly three fp32 results, and each one is flushed when subnormal:

      FTZ#1  max-shift ``logits - max`` — Reg ``Sub`` result stored to UB before ``Exp``
      FTZ#2  ``exp()`` result ``e_i`` — Reg ``Exp`` result; the vector the sort compares
      FTZ#3  top-k renorm ``Div`` result (``renorm=True`` only) — the device's *only*
             division (the softmax ``/sum`` below has no device counterpart: it cancels
             against the renorm, so the kernel sorts/renormalizes the raw ``e_i``)

    FTZ#2 is the operative one: deep-tail logits (``logit <= -87``, i.e. ``exp`` below
    2**-126) otherwise keep distinct subnormal scores in the reference, while the device
    sees them all as 0 and breaks the tie by *lower expert id* — a different top-k set.
    FTZ#1 / FTZ#3 are inert on every dataset in this directory (no fp32 result lands in
    the subnormal band; see README "FTZ 建模" and evidence/router_ftz_device_repro.log),
    but they are modelled because they are the other two fp32 materialization points of
    the same device path.
    """
    logits = x @ router_weight.T
    logits = logits - logits.max(axis=1, keepdims=True)
    logits = ftz_f32(logits)  # FTZ#1
    scores = np.exp(logits)
    scores = ftz_f32(scores)  # FTZ#2 (operative; the sort/ties below depend on it)
    # The softmax division below is a *reference* convenience only — the device sorts
    # e_i and renormalizes them directly (per-row positive scale: order-preserving, so
    # it cannot change the ids).  No FTZ is applied to it because the device never
    # materializes it; the device's counterpart division is FTZ#3.
    scores = scores / scores.sum(axis=1, keepdims=True)
    order = np.argsort(-scores, axis=1, kind="stable")  # desc, ties -> lower id
    topk_ids = order[:, :top_k].astype(np.int32)
    rows = np.arange(x.shape[0])[:, None]
    topk_weights = scores[rows, topk_ids].astype(np.float32)
    if renorm:
        topk_weights = topk_weights / topk_weights.sum(axis=1, keepdims=True)
        topk_weights = ftz_f32(topk_weights)  # FTZ#3
    return logits.astype(np.float32), topk_ids, topk_weights.astype(np.float32)


def moe_permute(topk_ids: np.ndarray, num_experts: int):
    """Token rearrangement (group-by-expert permute, moe_align_block_size style).

    topk_ids: int32 ``[m, k]``. Returns dict with:
      perm_src_token int32 ``[m*k]``: source token index per sorted slot
      perm_expert    int32 ``[m*k]``: expert id per sorted slot
      expert_token_counts int32 ``[E]``
      expert_offsets int32 ``[E+1]``: CSR offsets into the sorted slots
    Sorted by (expert id, token index) — the canonical grouped order.
    """
    m, k = topk_ids.shape
    flat_ids = topk_ids.reshape(-1)
    src_token = np.tile(np.arange(m, dtype=np.int32)[:, None], (1, k)).reshape(-1)
    order = np.argsort(flat_ids, kind="stable")  # by expert, then (token, slot)
    perm_expert = flat_ids[order].astype(np.int32)
    perm_src_token = src_token[order].astype(np.int32)
    counts = np.bincount(flat_ids, minlength=num_experts).astype(np.int32)
    offsets = np.zeros(num_experts + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    return {
        "perm_src_token": perm_src_token,
        "perm_expert": perm_expert,
        "expert_token_counts": counts,
        "expert_offsets": offsets,
    }


# ---------------------------------------------------------------------------
# Float golden for the full MoE block slice
# ---------------------------------------------------------------------------


def _silu(v: np.ndarray) -> np.ndarray:
    return v / (1.0 + np.exp(-v))


# ---------------------------------------------------------------------------
# m6 Add+RMSNorm reference (layer-chain golden)
# ---------------------------------------------------------------------------

RMSNORM_EPS = np.float32(1e-6)


def rmsnorm_ref(
    x: np.ndarray,
    gamma: np.ndarray,
    residual: np.ndarray | None = None,
    eps: np.float32 = RMSNORM_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """m6 semantics: ``xAdd = f32(x) + f32(residual)``; ``y = bf16((xAdd*rstd)*gamma)``.

    Follows m6_rmsnorm/check_ref.py (``rstd = 1/sqrt(mean(xAdd^2) + 1e-6)``, fp32
    accumulation). Returns ``(res_out fp32 [m, hidden], y bf16-grid fp32 [m, hidden])``;
    ``res_out`` is the fp32 residual exit (``xAdd``), which is bit-exact.
    """
    xf = np.ascontiguousarray(x, dtype=np.float32)
    gf = np.ascontiguousarray(gamma, dtype=np.float32)
    x_add = xf if residual is None else (xf + np.ascontiguousarray(residual, np.float32))
    ss = np.sum(x_add * x_add, axis=1, dtype=np.float32)
    rstd = (np.float32(1.0) / np.sqrt(ss / np.float32(x_add.shape[1]) + eps)).astype(np.float32)
    y = f32_to_bf16((x_add * rstd[:, None]) * gf[None, :])
    return np.ascontiguousarray(x_add, dtype=np.float32), y


def _expert_forward(x_t: np.ndarray, w_gate_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    """gate_up GEMM -> SwiGLU -> down GEMM for tokens x_t ``[t, hidden]``.

    w_gate_up: dequantized ``[2*inter, hidden]``, w_down: ``[hidden, inter]``.
    Returns ``[t, hidden]``.
    """
    inter = w_down.shape[1]
    gate_up = x_t @ w_gate_up.T  # [t, 2*inter]
    gate = gate_up[:, :inter]
    up = gate_up[:, inter:]
    return (_silu(gate) * up) @ w_down.T


def reference_moe_block(
    x: np.ndarray,
    router_weight: np.ndarray,
    shared_gate_weight: np.ndarray,
    experts_packed: dict,
    shared_packed: dict,
    top_k: int,
    renorm: bool = RENORM_TOPK,
) -> dict:
    """Compute the full MoE block slice in float32.

    x: bf16-rounded float32 ``[m, hidden]`` activations.
    router_weight: bf16-rounded f32 ``[E, hidden]``.
    shared_gate_weight: bf16-rounded f32 ``[1, hidden]``.
    experts_packed: {"experts.gate_up_proj": packed, "...weight_scale": scales, ...}
    shared_packed: {"shared_expert.gate_proj": packed, "...": ...}
    top_k: number of routed experts per token.

    Returns dict of float32 intermediates:
      router_logits, topk_ids, topk_weights, perm_*, expert_token_counts,
      expert_offsets, x_sorted, expert_outputs (list per expert or None),
      routed_output, shared_output, moe_output.
    """
    m = x.shape[0]
    logits, topk_ids, topk_weights = router_topk(x, router_weight, top_k, renorm)
    perm = moe_permute(topk_ids, router_weight.shape[0])
    x_sorted = x[perm["perm_src_token"]]

    # --- routed experts: dequant -> gate_up GEMM -> SwiGLU -> down GEMM ------
    gu_packed = experts_packed["experts.gate_up_proj"]
    gu_scales = experts_packed["experts.gate_up_proj.weight_scale"]
    dn_packed = experts_packed["experts.down_proj"]
    dn_scales = experts_packed["experts.down_proj.weight_scale"]
    num_experts = gu_packed.shape[0]
    inter = dn_packed.shape[2] * 2  # K of down == intermediate size

    routed = np.zeros((m, HIDDEN), dtype=np.float32)
    expert_outputs: list[np.ndarray | None] = [None] * num_experts
    for e in range(num_experts):
        sel = perm["perm_expert"] == e
        if not np.any(sel):
            continue
        x_e = x_sorted[sel]
        w_gu = unpack_mxfp4(gu_packed[e], gu_scales[e])  # [2*inter, hidden]
        w_dn = unpack_mxfp4(dn_packed[e], dn_scales[e])  # [hidden, inter]
        out_e = _expert_forward(x_e, w_gu, w_dn)  # [t_e, hidden]
        expert_outputs[e] = out_e
        for pos, slot in enumerate(np.nonzero(sel)[0]):
            t = int(perm["perm_src_token"][slot])
            j = int(np.nonzero(topk_ids[t] == e)[0][0])
            routed[t] += topk_weights[t, j] * out_e[pos]

    # --- shared expert (sigmoid-gated), MXFP4 weights -----------------------
    w_sg = unpack_mxfp4(shared_packed["shared_expert.gate_proj"],
                        shared_packed["shared_expert.gate_proj.weight_scale"])
    w_su = unpack_mxfp4(shared_packed["shared_expert.up_proj"],
                        shared_packed["shared_expert.up_proj.weight_scale"])
    w_sd = unpack_mxfp4(shared_packed["shared_expert.down_proj"],
                        shared_packed["shared_expert.down_proj.weight_scale"])
    s_gate = x @ w_sg.T
    s_up = x @ w_su.T
    s_h = _silu(s_gate) * s_up
    shared_mlp = s_h @ w_sd.T  # [m, hidden]
    shared_output = (1.0 / (1.0 + np.exp(-(x @ shared_gate_weight.T)))) * shared_mlp

    moe_output = routed + shared_output
    return {
        "router_logits": logits,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        **perm,
        "x_sorted": x_sorted,
        "expert_outputs": expert_outputs,
        "routed_output": routed,
        "shared_output": shared_output,
        "moe_output": moe_output,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

MANIFEST_VERSION = 2


def _gen_weight_f32(rng: np.random.Generator, n: int, k: int, std: float = 0.03):
    return rng.standard_normal((n, k), dtype=np.float32) * np.float32(std)


def build_case(m: int, num_experts: int, top_k: int, seed: int) -> dict:
    """Deterministic inputs + MXFP4 weights of one dataset group.

    The RNG draw order is frozen (x, router weight, shared gate, per-expert gate_up,
    per-expert down, shared gate/up/down, then the M26 additions x_res / gamma1 /
    gamma2) — new tensors are appended so already published bins stay byte-identical.
    """
    rng = np.random.default_rng(seed)

    def bf16(v):
        return f32_to_bf16(v)

    case = {
        "m": m,
        "num_experts": num_experts,
        "top_k": top_k,
        "seed": seed,
    }

    # ---- activations & router weights (bf16 in the model) ------------------
    case["x"] = bf16(rng.standard_normal((m, HIDDEN), dtype=np.float32))
    case["router_weight"] = bf16(_gen_weight_f32(rng, num_experts, HIDDEN, std=0.02))
    case["shared_gate_weight"] = bf16(_gen_weight_f32(rng, 1, HIDDEN, std=0.02))

    # ---- routed expert weights (MXFP4) -------------------------------------
    e_shapes = expert_weight_shapes(num_experts)
    experts_packed = {}
    for name, shape in e_shapes.items():
        if name.endswith("weight_scale"):
            continue
        n, k = shape[1], shape[2] * 2
        packed = np.empty(shape, dtype=np.uint8)
        scales = np.empty(e_shapes[name + ".weight_scale"], dtype=np.uint8)
        for e in range(num_experts):
            p, s = pack_mxfp4(_gen_weight_f32(rng, n, k))
            packed[e], scales[e] = p, s
        experts_packed[name] = packed
        experts_packed[name + ".weight_scale"] = scales

    # ---- shared expert weights (MXFP4, checkpoint layout: separate gate/up) -
    s_shapes = shared_weight_shapes()
    shared_packed = {}
    for name, shape in s_shapes.items():
        if name.endswith("weight_scale"):
            continue
        n, k = shape[0], shape[1] * 2
        p, s = pack_mxfp4(_gen_weight_f32(rng, n, k))
        shared_packed[name] = p
        shared_packed[name + ".weight_scale"] = s

    # ---- M26 layer-chain additions (appended draws: keep old bins stable) --
    # x_res = layer residual input (m6#1 input, zero residual at m6#1);
    # gamma1/gamma2 = pre/post-MoE RMSNorm weights (m6#1 / m6#2).
    case["x_res"] = bf16(rng.standard_normal((m, HIDDEN), dtype=np.float32))
    case["gamma1"] = bf16(np.float32(1.0) + np.float32(0.5) *
                          rng.random(HIDDEN, dtype=np.float32))          # ∈ [1.0, 1.5)
    case["gamma2"] = bf16(np.float32(1.0) + np.float32(0.5) *
                          rng.random(HIDDEN, dtype=np.float32))
    case["experts_packed"] = experts_packed
    case["shared_packed"] = shared_packed
    return case


def generate_dataset(
    outdir: Path,
    m: int,
    num_experts: int,
    top_k: int,
    seed: int,
    rules: tuple[str, ...] = ("floor", "ceil"),
) -> dict:
    """Generate one deterministic dataset (inputs + float golden + quant-aware golden).

    Writes all bins with .json descriptors plus a manifest.json. Returns the manifest
    dict. Shapes keep the real model K/N (hidden=2560, moe_intermediate=640,
    shared_expert_intermediate=640); only the expert count and top_k are shrunk
    (512/10 -> num_experts/top_k) — except for the ``real`` group, which keeps 512/10.

    Besides the float golden, one quant-aware golden group per activation e8m0 rule is
    written under ``qaware/<rule>/`` (see ``qaware_ref``); ``rules=()`` skips them.
    """
    from qaware_ref import qaware_moe_block, deviation_report, write_qaware

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    case = build_case(m, num_experts, top_k, seed)
    x = case["x"]
    router_w = case["router_weight"]
    shared_gate_w = case["shared_gate_weight"]
    experts_packed = case["experts_packed"]
    shared_packed = case["shared_packed"]

    # ---- float golden (MoE slice) ------------------------------------------
    ref = reference_moe_block(
        x, router_w, shared_gate_w, experts_packed, shared_packed, top_k
    )

    # ---- layer-chain golden (m6#1 -> MoE -> m6#2, float) -------------------
    # m6#2 consumes the *bf16* moe_output tensor (the MoE block's stored output), so
    # the post-MoE goldens are derived from the bf16-rounded value.
    res1, x_norm1 = rmsnorm_ref(case["x_res"], case["gamma1"])
    res2, y_final = rmsnorm_ref(f32_to_bf16(ref["moe_output"]), case["gamma2"],
                                residual=res1)
    shared_gate_logits = (x @ shared_gate_w.T).astype(np.float32).reshape(-1)

    # ---- write inputs --------------------------------------------------------
    files = []

    def put(name: str, arr, dtype_name: str, kind: str, desc: str):
        d = write_bin(outdir / name, arr, dtype_name)
        d.update({"kind": kind, "description": desc})
        # rewrite descriptor with the extra fields
        with open(str(outdir / name) + ".json", "w") as f:
            json.dump(d, f, indent=2)
            f.write("\n")
        files.append(d)

    put("x.bin", x, "bf16", "input",
        "MoE segment input activations [m, hidden], bf16 (chain mode: m6#1 output, "
        "i.e. post-norm — NOT norm(x_res, gamma1))")
    put("router_weight.bin", router_w, "bf16", "input",
        "router gate weight [num_experts, hidden], bf16")
    put("shared_expert_gate_weight.bin", shared_gate_w, "bf16", "input",
        "shared expert gate weight [1, hidden], bf16")
    put("x_res.bin", case["x_res"], "bf16", "input",
        "layer residual input [m, hidden], bf16 — m6#1 x (with a zero residual); "
        "also the fp32 residual the MoE block adds to (layer stacking)")
    put("gamma1.bin", case["gamma1"][None, :], "bf16", "input",
        "m6#1 RMSNorm weight [1, hidden], bf16 (pre-MoE norm)")
    put("gamma2.bin", case["gamma2"][None, :], "bf16", "input",
        "m6#2 RMSNorm weight [1, hidden], bf16 (post-MoE norm)")
    for name in ("experts.gate_up_proj", "experts.down_proj"):
        put(name + ".bin", experts_packed[name], "uint8", "input",
            f"MXFP4 packed weights {list(experts_packed[name].shape)}, lohi nibbles along K")
        put(name + ".weight_scale.bin", experts_packed[name + ".weight_scale"], "uint8",
            "input", "E8M0 group scales [.., N, K//32], bias 127, group 32 along K")
    for name in ("shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj"):
        put(name + ".bin", shared_packed[name], "uint8", "input",
            f"MXFP4 packed weights {list(shared_packed[name].shape)}, lohi nibbles along K")
        put(name + ".weight_scale.bin", shared_packed[name + ".weight_scale"], "uint8",
            "input", "E8M0 group scales [N, K//32], bias 127, group 32 along K")

    # ---- write golden outputs ----------------------------------------------
    put("router_logits.bin", ref["router_logits"], "fp32", "golden",
        "router logits [m, num_experts], fp32 (softmax input, max-shifted)")
    put("shared_gate_logits.bin", shared_gate_logits, "fp32", "golden",
        "shared expert gate raw dot [m], fp32 (sigmoid input, device sgate)")
    put("topk_ids.bin", ref["topk_ids"], "int32", "golden",
        "top-k expert ids [m, top_k] sorted by descending score")
    put("topk_weights.bin", ref["topk_weights"], "fp32", "golden",
        "top-k routing weights [m, top_k], renormalized to sum 1")
    put("perm_src_token.bin", ref["perm_src_token"], "int32", "golden",
        "grouped permute: source token per sorted slot [m*top_k]")
    put("perm_expert.bin", ref["perm_expert"], "int32", "golden",
        "grouped permute: expert id per sorted slot [m*top_k]")
    put("expert_token_counts.bin", ref["expert_token_counts"], "int32", "golden",
        "tokens per expert [num_experts]")
    put("x_sorted.bin", ref["x_sorted"], "bf16", "golden",
        "rearranged activations grouped by expert [m*top_k, hidden], bf16")
    put("routed_output.bin", ref["routed_output"], "bf16", "golden",
        "weighted routed-expert sum before shared expert [m, hidden], bf16")
    put("shared_output.bin", ref["shared_output"], "bf16", "golden",
        "shared expert output after sigmoid gate [m, hidden], bf16")
    put("moe_output.bin", ref["moe_output"], "bf16", "golden",
        "final MoE block output = routed + shared [m, hidden], bf16")
    put("x_norm1.bin", x_norm1, "bf16", "golden",
        "m6#1 output = bf16(rmsnorm(x_res, gamma1)) [m, hidden] — the layer-chain MoE "
        "segment input (differs from x.bin, see README)")
    put("res1.bin", res1, "fp32", "golden",
        "m6#1 residual out (fp32) = f32(x_res) + 0 [m, hidden]; the m6#2 residual input")
    put("y_final.bin", y_final, "bf16", "golden",
        "m6#2 output = bf16(rmsnorm(moe_output + res1, gamma2)) [m, hidden]")
    put("res2.bin", res2, "fp32", "golden",
        "m6#2 residual out (fp32) = f32(moe_output) + res1 [m, hidden] — the residual "
        "stream handed to the next layer")

    # ---- quant-aware golden groups (one per activation e8m0 rule) ----------
    # The deviation report needs the *layer-chain* float reference too, otherwise the
    # y_final/res2 entries are skipped silently (review M26-r1 N1).  Build the reference
    # bundle explicitly so the report's key sets are complete.
    layer_ref = dict(ref)
    layer_ref.update({"x_norm1": x_norm1, "res1": res1, "res2": res2, "y_final": y_final})
    qaware_groups = {}
    for rule in rules:
        q = qaware_moe_block(x, router_w, shared_gate_w, experts_packed, shared_packed,
                             top_k, rule=rule,
                             layer={"x_res": case["x_res"], "gamma1": case["gamma1"],
                                    "gamma2": case["gamma2"]})
        report = deviation_report(layer_ref, q, x, experts_packed, shared_packed, rule=rule,
                                 layer={"gamma2": case["gamma2"]})
        man = write_qaware(outdir / "qaware" / rule, q, report,
                           {"m": m, "num_experts": num_experts, "top_k": top_k, "seed": seed})
        qaware_groups[rule] = {
            "dir": f"qaware/{rule}",
            "tensors": len(man["files"]),
            # scalar summary of the budget: measured deviation of the q-aware chain vs
            # the float golden, and the rigorous worst-case ceiling for the same tensor
            "summary": {
                name: {
                    "measured_max_abs": report["measured"][name]["max_abs"],
                    "ceiling": report["ceiling"][name]["analytic_max"],
                }
                for name in ("routed_output", "shared_output", "moe_output")
                if name in report["measured"] and name in report["ceiling"]
            },
        }

    manifest = {
        "format_version": MANIFEST_VERSION,
        "name": outdir.name,
        "m": m,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden": HIDDEN,
        "moe_intermediate": MOE_INTERMEDIATE,
        "shared_expert_intermediate": SHARED_INTERMEDIATE,
        "group_size": GROUP_SIZE,
        "routing": {
            "scoring": "softmax over all experts",
            "topk_order": "descending score, ties to lower expert id",
            "renormalize_topk": RENORM_TOPK,
            "shared_expert_gate": "sigmoid(x @ shared_expert_gate_weight.T)",
        },
        "quantization": {
            "format": "mxfp4-pack-quantized-e8m0",
            "weight": "e2m1 nibbles, lohi byte order, group 32 along K",
            "weight_scale": "e8m0 bias-127 bytes as given — the checkpoint was packed by the "
                            "model author's CPU RTN tool (cpu_rtn_provenance.tool = "
                            "qwen3.8-flash-next-cpu-rtn-mxfp4); we consume the bytes and "
                            "never re-pack.  This dataset's *synthetic* weights are packed "
                            "with the legacy ceil rule (pack_mxfp4).",
            "activation_scale_rules": {r: SCALE_RULE_DOC[r] for r in SCALE_RULES},
            "activation_data_rounding": {r: SCALE_RULE_ROUNDING[r] for r in SCALE_RULES},
            "activation_default_rule": "floor",
            "activation_rule_status": "authoritative: floor = the official ops-nn MXFP4 "
                                      "sequence (OCP; npu_dynamic_mx_quant(round_mode="
                                      "\"round\", scale_alg=0)) transcribed line by line "
                                      "in moe_block_ref.quantize_ocp — see docs/13 and "
                                      "tools/golden/README.md for the step-by-step table. "
                                      "ceil = legacy (m3 scalar quantizer / synthetic "
                                      "weight packing), kept only for reproducing "
                                      "pre-ruling goldens.",
            "qaware_golden": qaware_groups,
        },
        "layer_chain": {
            "input": "x_res.bin -> m6#1 (Add+RMSNorm with a zero residual, gamma1)",
            "segment_input": "x.bin is the MoE segment input (post-norm); it is NOT "
                             "norm(x_res, gamma1) — RMSNorm is invariant to a per-token "
                             "scale, so the slice and the chain use separate inputs",
            "norm_semantics": "resOut = f32(x) + f32(residual) (fp32, bit-exact); "
                              "rstd = 1/sqrt(mean(xAdd^2) + 1e-6); y = bf16((xAdd*rstd)*gamma)",
            "norm_eps": 1e-6,
            "gammas": "gamma1 (m6#1, pre-MoE), gamma2 (m6#2, post-MoE); bf16 [1, hidden]",
            "residuals": "res1 = f32(x_res); res2 = res1 + moe_output; the MoE block adds "
                         "no residual of its own",
        },
        "seed": seed,
        "files": files,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def load_dataset(outdir: Path) -> dict:
    """Load a generated dataset directory back into memory."""
    outdir = Path(outdir)
    with open(outdir / "manifest.json") as f:
        manifest = json.load(f)
    tensors = {}
    for d in manifest["files"]:
        tensors[d["tensor"]] = read_bin(outdir / d["tensor"], d)
    return {"manifest": manifest, "tensors": tensors}
