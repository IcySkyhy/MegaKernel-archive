"""Quantization-aware (w4a4) reference for the MoE block slice — double precision.

The float golden in :mod:`moe_block_ref` never quantizes activations, so it cannot
grade a w4a4 kernel at strict tolerance (finding
``20260926-agent-moelayer-bug-tools-golden-golden``).  This module recomputes the
device chain in float64 with the two activation quantizations the device performs:

    router -> permute -> A quant (x_sorted / x)
    -> GMM#1 (bf16 out) -> SwiGLU (bf16 out) -> H quant
    -> GMM#2 (bf16 out) -> weighted combine (+ sigmoid-gated shared expert)
    -> m6#2 Add+RMSNorm (layer chain)

Quantization and rounding points mirror the device (docs/12-layer-integration §3,
m13 S1-S10): the A/H quantizers are the MXFP4 group-32 quantizers of
``moe_block_ref.quantize_activations``; GU, H, Y and every output are rounded to bf16
(RNE) exactly where the device rounds (FIXP bf16 output / bf16 UB hand-off); the
unpermute-combine folds the *bf16* top-k weights (``w_tk_packed``) into bf16 Y.

Two activation quantization rules are parameterized (``rule``):

  * ``floor`` — **default and authoritative**: the official ops-nn MXFP4 sequence
    (OCP, ``npu_dynamic_mx_quant(round_mode="round", scale_alg=0)``), transcribed
    statement by statement in ``moe_block_ref.quantize_ocp``; see
    ``docs/13-mx-quant-primitives.md`` §4/§5 and the step table in the README.
    This is also the path our device kernels implement.  The one corner where they used to
    diverge -- the official ``:413`` halfScale override for groups containing ±Inf/NaN
    (``halfScale`` -> bf16 NaN so that ``Mul(±Inf, NaN) = NaN`` and the codes flush to 0)
    -- is implemented on the device side as of 2026-09-26 (M32).  Device side, in each of
    ``m2_mxfp4_quant.asc`` / ``m5_swiglu_quant.asc`` / ``m13_moe_layer.asc``, inside
    ``MxQuantComputeScale``: ``Duplicate(nanRegTensor, NAN_CUSTOMIZATION)`` with
    ``constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;`` -- **lowercase**, so a case-sensitive
    ``grep 0x7F81`` does *not* find it; the 3 uppercase hits per file are **not on the device
    path**.  Reference and device therefore agree on the Inf corner.
  * ``ceil``  — **legacy**: ``scale = 2**ceil(log2(amax/6))`` with nearest/tie-to-even
    codes, the replaced m3 scalar quantizer's rule.  Emitted only so the pre-ruling
    goldens stay reproducible; it is not the reference for grading a device kernel.

:func:`deviation_report` quantifies the gap to the float golden and derives the
deviation budget: a tight analytic bound on the quantizers themselves
(``source_budget``), a conservative worst-case bound on the GEMM/output stages
(``ceiling``), and the per-element oracle deviations a layer chain is graded against
(``measured``, the m13 acceptance criterion).

Pure numpy (2.5.1), no torch. Python 3.12.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import moe_block_ref as ref

QAWARE_RULES = ("floor", "ceil")
DEFAULT_RULE = "floor"  # current device path (m5 VEC / hardware OCP semantics)

# Envelope constants for the deviation budget.
BF16_REL = 2.0 ** -8  # full bf16 relative ulp — safe envelope at every bf16 round point
F32_REL = 2.0 ** -23  # f32 accumulation / add envelope (dominated by the bf16 terms)


# ---------------------------------------------------------------------------
# float64 helpers
# ---------------------------------------------------------------------------


def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """float64 ``a[r, K] @ b[N, K].T -> [r, N]`` (checkpoint weights are ``[N, K]``)."""
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64).T


def _silu64(v: np.ndarray) -> np.ndarray:
    return v / (1.0 + np.exp(-v))


def _sigmoid64(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


E2M1_POS64 = ref.E2M1_POS.astype(np.float64)


def quant_element_bound(abs_values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Analytic bound on ``|q - v|`` for the MXFP4 group quantizers.

    The E2M1 levels are *not* uniform (gaps 0.5/1/2 in level units), so the rounding
    error is bounded by half the gap between the two enclosing levels, not by half a
    step; the floor rule additionally saturates at ±6, contributing ``|v| - 6*s`` for
    elements above ``6*s``.  Both the nearest (ceil rule) and the half-away-from-zero
    (floor rule) data paths stay inside this bound.  ``scale`` broadcasts against
    ``abs_values``; returns float64.
    """
    a = np.asarray(abs_values, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    h = np.where(s > 0.0, a / np.maximum(s, tiny), 0.0)
    idx = np.clip(np.searchsorted(E2M1_POS64, h, side="left"), 0, len(E2M1_POS64) - 1)
    lo = np.clip(idx - 1, 0, len(E2M1_POS64) - 1)
    half_gap = 0.5 * (E2M1_POS64[idx] - E2M1_POS64[lo])
    return half_gap * s + np.maximum(0.0, a - float(ref.E2M1_MAX) * s)


def rule_expand(scales_bytes: np.ndarray, k: int) -> np.ndarray:
    """Expand ``[rows, K // 32]`` E8M0 bytes to per-element float64 scales ``[rows, K]``."""
    s = ref.e8m0_decode(np.asarray(scales_bytes, dtype=np.uint8)).astype(np.float64)
    return np.repeat(s, ref.GROUP_SIZE, axis=1)[:, :k]


def _deq(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize MXFP4 bytes to float64."""
    return ref.unpack_mxfp4(np.asarray(packed), np.asarray(scales)).astype(np.float64)


# ---------------------------------------------------------------------------
# quant-aware MoE block
# ---------------------------------------------------------------------------


def qaware_moe_block(
    x: np.ndarray,
    router_weight: np.ndarray,
    shared_gate_weight: np.ndarray,
    experts_packed: dict,
    shared_packed: dict,
    top_k: int,
    rule: str = DEFAULT_RULE,
    renorm: bool = ref.RENORM_TOPK,
    layer: dict | None = None,
) -> dict:
    """MoE block slice with device-path activation quantization, computed in float64.

    ``x`` / ``router_weight`` / ``shared_gate_weight`` are bf16-rounded float32 (the
    dataset's values); ``experts_packed`` / ``shared_packed`` are the MXFP4 byte
    tensors (identical to the float golden's — weights are never re-quantized here).
    ``layer`` optionally carries ``{"x_res", "gamma1", "gamma2"}`` to continue the
    chain through m6#1 / m6#2 (see :func:`moe_block_ref.rmsnorm_ref`).

    Returns a dict of float32 tensors (on the bf16 grid where the device stores bf16):
      routing: router_logits, topk_ids, topk_weights, perm_src_token, perm_expert,
               expert_token_counts, expert_offsets, x_sorted, inv_slot
      A side : a_qx, a_scale, a_qx_shd, a_scale_shd
      GEMM#1 : gu, gu_shd
      H side : h_swiglu, h_swiglu_shd, h_qx, h_scale, h_qx_shd, h_scale_shd
      GEMM#2 : y_sorted, y_shd
      outputs: routed_output, shared_output, moe_output, shared_gate_logits
      layer  : x_norm1, res1, res2, y_final (None when ``layer`` is not given)
    """
    if rule not in QAWARE_RULES:
        raise ValueError(f"unknown activation e8m0 rule {rule!r}; expected {QAWARE_RULES}")
    x = np.ascontiguousarray(x, dtype=np.float32)
    m = x.shape[0]
    hidden = ref.HIDDEN
    group = ref.GROUP_SIZE

    # ---- routing (identical to the float golden) ---------------------------
    logits, topk_ids, topk_weights = ref.router_topk(x, router_weight, top_k, renorm)
    perm = ref.moe_permute(topk_ids, router_weight.shape[0])
    x_sorted = np.ascontiguousarray(x[perm["perm_src_token"]])
    counts = perm["expert_token_counts"]
    offsets = perm["expert_offsets"]
    total = int(counts.sum())

    gu_packed = experts_packed["experts.gate_up_proj"]
    gu_scales = experts_packed["experts.gate_up_proj.weight_scale"]
    dn_packed = experts_packed["experts.down_proj"]
    dn_scales = experts_packed["experts.down_proj.weight_scale"]
    num_experts = int(gu_packed.shape[0])
    gu_n = int(gu_packed.shape[1])
    inter = gu_n // 2

    a_qx = np.zeros((total, hidden // 2), dtype=np.uint8)
    a_scale = np.zeros((total, hidden // group), dtype=np.uint8)
    gu_out = np.zeros((total, gu_n), dtype=np.float32)
    h_out = np.zeros((total, inter), dtype=np.float32)
    h_qx = np.zeros((total, inter // 2), dtype=np.uint8)
    h_scale = np.zeros((total, inter // group), dtype=np.uint8)
    y_out = np.zeros((total, hidden), dtype=np.float32)

    # ---- routed experts: A quant -> GMM#1 -> SwiGLU -> H quant -> GMM#2 ----
    for e in range(num_experts):
        off, t = int(offsets[e]), int(counts[e])
        if t == 0:
            continue
        sl = slice(off, off + t)
        pq, sq = ref.quantize_activations(np.ascontiguousarray(x_sorted[sl]), rule)
        a_qx[sl] = pq
        a_scale[sl] = sq
        gu_e = ref.f32_to_bf16(
            _mm(_deq(pq, sq), _deq(gu_packed[e], gu_scales[e])).astype(np.float32))
        gu_out[sl] = gu_e
        g = gu_e[:, :inter].astype(np.float64)
        u = gu_e[:, inter:].astype(np.float64)
        h_e = ref.f32_to_bf16((_silu64(g) * u).astype(np.float32))  # swigluOut bf16
        h_out[sl] = h_e
        pq2, sq2 = ref.quantize_activations(np.ascontiguousarray(h_e), rule)
        h_qx[sl] = pq2
        h_scale[sl] = sq2
        y_out[sl] = ref.f32_to_bf16(
            _mm(_deq(pq2, sq2), _deq(dn_packed[e], dn_scales[e])).astype(np.float32))

    # ---- shared expert (1 slot, sigmoid gate) ------------------------------
    a_shd_qx, a_shd_scale = ref.quantize_activations(np.ascontiguousarray(x), rule)
    w_sg = _deq(shared_packed["shared_expert.gate_proj"],
                shared_packed["shared_expert.gate_proj.weight_scale"])
    w_su = _deq(shared_packed["shared_expert.up_proj"],
                shared_packed["shared_expert.up_proj.weight_scale"])
    w_sd = _deq(shared_packed["shared_expert.down_proj"],
                shared_packed["shared_expert.down_proj.weight_scale"])
    gu_shd = ref.f32_to_bf16(_mm(_deq(a_shd_qx, a_shd_scale),
                                 np.concatenate([w_sg, w_su], axis=0)).astype(np.float32))
    h_shd = ref.f32_to_bf16((_silu64(gu_shd[:, :inter].astype(np.float64))
                             * gu_shd[:, inter:].astype(np.float64)).astype(np.float32))
    h_shd_qx, h_shd_scale = ref.quantize_activations(np.ascontiguousarray(h_shd), rule)
    y_shd = ref.f32_to_bf16(_mm(_deq(h_shd_qx, h_shd_scale), w_sd).astype(np.float32))

    # ---- combine (unpermute folds bf16 Y with the bf16 top-k weights) ------
    flat_ids = topk_ids.reshape(-1)
    order = np.argsort(flat_ids, kind="stable")
    inv_slot = np.empty(total, dtype=np.int64)
    inv_slot[order] = np.arange(total, dtype=np.int64)
    w_bf16 = ref.f32_to_bf16(topk_weights).astype(np.float64)  # w_tk_packed is bf16
    y64 = y_out.astype(np.float64)
    routed = np.zeros((m, hidden), dtype=np.float64)
    for t in range(m):
        for k in range(top_k):
            routed[t] += w_bf16[t, k] * y64[inv_slot[t * top_k + k]]
    routed_output = ref.f32_to_bf16(routed.astype(np.float32))

    shared_gate_logits = (x @ shared_gate_weight.T).astype(np.float32).reshape(-1)
    gate_v = _sigmoid64(shared_gate_logits.astype(np.float64))[:, None]
    shared_output = ref.f32_to_bf16((gate_v * y_shd.astype(np.float64)).astype(np.float32))
    moe_output = ref.f32_to_bf16(
        shared_output.astype(np.float32) + routed_output.astype(np.float32))

    out = {
        "rule": rule,
        "router_logits": logits,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "perm_src_token": perm["perm_src_token"],
        "perm_expert": perm["perm_expert"],
        "expert_token_counts": counts,
        "expert_offsets": offsets,
        "inv_slot": inv_slot.astype(np.int32),
        "x_sorted": x_sorted,
        "a_qx": a_qx,
        "a_scale": a_scale,
        "a_qx_shd": a_shd_qx,
        "a_scale_shd": a_shd_scale,
        "gu": gu_out,
        "gu_shd": gu_shd,
        "h_swiglu": h_out,
        "h_swiglu_shd": h_shd,
        "h_qx": h_qx,
        "h_scale": h_scale,
        "h_qx_shd": h_shd_qx,
        "h_scale_shd": h_shd_scale,
        "y_sorted": y_out,
        "y_shd": y_shd,
        "routed_output": routed_output,
        "shared_output": shared_output,
        "moe_output": moe_output,
        "shared_gate_logits": shared_gate_logits,
        "x_norm1": None,
        "res1": None,
        "res2": None,
        "y_final": None,
    }

    # ---- optional layer chain (m6#1 / m6#2, gammas from the dataset) --------
    if layer is not None:
        x_res = np.ascontiguousarray(layer["x_res"], dtype=np.float32)
        res1, x_norm1 = ref.rmsnorm_ref(x_res, layer["gamma1"])
        res2, y_final = ref.rmsnorm_ref(moe_output, layer["gamma2"], residual=res1)
        out.update({"x_norm1": x_norm1, "res1": res1, "res2": res2, "y_final": y_final})
    return out


# ---------------------------------------------------------------------------
# deviation report (quant-aware vs float golden) + analytic budget
# ---------------------------------------------------------------------------


def _stats(ref_arr: np.ndarray, got_arr: np.ndarray, rel_scale: float = 1e-3) -> dict:
    """Absolute/relative deviation statistics of ``got`` vs ``ref`` (float64).

    Relative statistics are taken over the "non-negligible" elements only
    (``|ref| > rel_scale * max|ref|``): near-zero outputs have huge but meaningless
    relative deviations, which would otherwise dominate the report.
    """
    a = np.asarray(ref_arr, dtype=np.float64).reshape(-1)
    b = np.asarray(got_arr, dtype=np.float64).reshape(-1)
    dev = np.abs(a - b)
    ra = np.abs(a)
    thr = max(1e-12, rel_scale * (float(ra.max()) if ra.size else 0.0))
    nz = ra > thr
    rel = dev[nz] / ra[nz]
    return {
        "n": int(dev.size),
        "max_abs": float(dev.max()) if dev.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(dev * dev))) if dev.size else 0.0,
        "n_rel": int(np.count_nonzero(nz)),
        "rel_threshold": float(thr),
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "p99_rel": float(np.percentile(rel, 99.0)) if rel.size else 0.0,
        "frac_rel_gt_10pct": float(np.mean(rel > 0.10)) if rel.size else 0.0,
    }


def _budget(env: np.ndarray, dev: np.ndarray) -> dict:
    """Compare a deviation against its analytic envelope (should hold everywhere)."""
    e = np.asarray(env, dtype=np.float64).reshape(-1)
    d = np.abs(np.asarray(dev, dtype=np.float64).reshape(-1))
    tol = e * (1.0 + 1e-6) + 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(e > 0, d / np.maximum(e, 1e-300), 0.0)
    return {
        "analytic_max": float(e.max()) if e.size else 0.0,
        "violations": int(np.count_nonzero(d > tol)),
        "max_utilization": float(util.max()) if util.size else 0.0,
    }


def _group_ceilings(a_ref, da_rows, gu_q, h_q, h_scale_bytes, y_q, w_gu, w_dn, inter, rule):
    """Ceiling walk for one GEMM#1 -> SwiGLU -> GEMM#2 group.

    The GEMM stages propagate the *actual* quantization error with absolute weights
    (worst case: every error term aligned), so ``ceil_gu`` / ``ceil_y`` are rigorous
    upper bounds on ``|q-aware - float-chain|`` for those stages; the SwiGLU stage uses
    the exact deviation ``|h_q - h_r|`` (its bound would only restate the GEMM ceiling).
    Returns ``(gu_r, h_r, y_r, ceil_gu, ceil_y, src_h)``.
    """
    gu_r = _mm(a_ref, w_gu)
    ceil_gu = (np.abs(da_rows) @ np.abs(w_gu).T
               + BF16_REL * np.abs(gu_q) + F32_REL * np.abs(gu_r))
    g_r = gu_r[:, :inter]
    u_r = gu_r[:, inter:]
    h_r = _silu64(g_r) * u_r
    h_q = np.asarray(h_q, dtype=np.float64)
    src_h = quant_element_bound(np.abs(h_q), rule_expand(h_scale_bytes, inter))
    ceil_h_dq = np.abs(h_q - h_r) + src_h  # exact SwiGLU/bf16 deviation + H quant error
    y_r = h_r @ w_dn.T
    ceil_y = (ceil_h_dq @ np.abs(w_dn).T
              + BF16_REL * np.abs(y_q) + F32_REL * np.abs(y_r))
    return gu_r, h_r, y_r, ceil_gu, ceil_y, src_h


def deviation_report(
    float_ref: dict,
    q: dict,
    x: np.ndarray,
    experts_packed: dict,
    shared_packed: dict,
    rule: str = DEFAULT_RULE,
    layer: dict | None = None,
) -> dict:
    """Deviations of the quant-aware chain vs the float golden + the deviation budget.

    Three blocks (README "偏差预算"):

    ``source_budget``  per-element analytic bound on the A/H quantizers themselves
                       (half the enclosing E2M1 level gap + the ±6 saturation term),
                       against the measured error -> the tight, operative bound on the
                       *input* of every GEMM.
    ``ceiling``        rigorous worst-case bound on the GEMM/output stages: the actual
                       quantization error propagated with absolute weights (all error
                       terms aligned), plus the bf16 rounding points.  Conservative by
                       the sign-cancellation factor of the K-sum.
    ``measured``       the deviations of the q-aware chain vs the float golden (and vs
                       the same-stage float values for the intermediate stages) — the
                       per-element oracle a layer-chain kernel is graded against, with
                       the m13 criterion as ``criterion``.

    Intermediate stages are graded against the same-stage float value (unquantized A,
    no bf16 rounding); the outputs against the float golden.  ``layer`` (with
    ``gamma2``) only extends the ceiling to ``y_final``.
    """
    layer_gamma2 = None if layer is None else layer["gamma2"]
    hidden = ref.HIDDEN
    inter = int(shared_packed["shared_expert.gate_proj"].shape[0])
    counts = np.asarray(q["expert_token_counts"])
    offsets = np.asarray(q["expert_offsets"])
    total = int(counts.sum())
    m = int(x.shape[0])
    top_k = int(q["topk_ids"].shape[1])

    gu_packed = experts_packed["experts.gate_up_proj"]
    gu_scales = experts_packed["experts.gate_up_proj.weight_scale"]
    dn_packed = experts_packed["experts.down_proj"]
    dn_scales = experts_packed["experts.down_proj.weight_scale"]

    x_sorted = np.ascontiguousarray(float_ref["x_sorted"], dtype=np.float64)
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    a_q = _deq(q["a_qx"], q["a_scale"])
    a_q_shd = _deq(q["a_qx_shd"], q["a_scale_shd"])
    da = a_q - x_sorted
    da_shd = a_q_shd - x64
    # source budgets (analytic per-rule bound vs measured quantizer error)
    src_a = quant_element_bound(np.abs(x_sorted), rule_expand(q["a_scale"], hidden))
    src_a_shd = quant_element_bound(np.abs(x64), rule_expand(q["a_scale_shd"], hidden))
    h_q = q["h_swiglu"].astype(np.float64)
    h_q_shd = q["h_swiglu_shd"].astype(np.float64)
    src_h = quant_element_bound(np.abs(h_q), rule_expand(q["h_scale"], inter))
    src_h_shd = quant_element_bound(np.abs(h_q_shd), rule_expand(q["h_scale_shd"], inter))

    gu_ref = np.zeros((total, 2 * inter))
    h_ref = np.zeros((total, inter))
    y_ref = np.zeros((total, hidden))
    ceil_gu = np.zeros((total, 2 * inter))
    ceil_y = np.zeros((total, hidden))
    for e in range(int(counts.shape[0])):
        off, t = int(offsets[e]), int(counts[e])
        if t == 0:
            continue
        sl = slice(off, off + t)
        w_gu = _deq(gu_packed[e], gu_scales[e])
        w_dn = _deq(dn_packed[e], dn_scales[e])
        gu_r, h_r, y_r, c_gu, c_y, _ = _group_ceilings(
            x_sorted[sl], da[sl], q["gu"][sl].astype(np.float64), h_q[sl],
            q["h_scale"][sl], q["y_sorted"][sl].astype(np.float64), w_gu, w_dn, inter, rule)
        gu_ref[sl], h_ref[sl], y_ref[sl] = gu_r, h_r, y_r
        ceil_gu[sl], ceil_y[sl] = c_gu, c_y

    w_gu_shd = np.concatenate([
        _deq(shared_packed["shared_expert.gate_proj"],
             shared_packed["shared_expert.gate_proj.weight_scale"]),
        _deq(shared_packed["shared_expert.up_proj"],
             shared_packed["shared_expert.up_proj.weight_scale"]),
    ], axis=0)
    w_dn_shd = _deq(shared_packed["shared_expert.down_proj"],
                    shared_packed["shared_expert.down_proj.weight_scale"])
    (gu_ref_shd, h_ref_shd, y_ref_shd, ceil_gu_shd, ceil_y_shd,
     _) = _group_ceilings(
        x64, da_shd, q["gu_shd"].astype(np.float64), h_q_shd, q["h_scale_shd"],
        q["y_shd"].astype(np.float64), w_gu_shd, w_dn_shd, inter, rule)

    # ---- combine ceiling ---------------------------------------------------
    inv = np.asarray(q["inv_slot"], dtype=np.int64)
    w_bf16 = ref.f32_to_bf16(q["topk_weights"]).astype(np.float64)
    dw = np.abs(w_bf16 - q["topk_weights"].astype(np.float64))
    y_q = q["y_sorted"].astype(np.float64)
    ceil_routed = np.zeros((m, hidden))
    for t in range(m):
        for k in range(top_k):
            slot = int(inv[t * top_k + k])
            ceil_routed[t] += (w_bf16[t, k] * ceil_y[slot]
                               + dw[t, k] * (np.abs(y_q[slot]) + ceil_y[slot]))
    ceil_routed += BF16_REL * np.abs(float_ref["routed_output"].astype(np.float64))

    gate_v = _sigmoid64(np.asarray(q["shared_gate_logits"], np.float64).reshape(-1))
    ceil_shared = (gate_v[:, None] * ceil_y_shd
                   + BF16_REL * np.abs(q["shared_output"].astype(np.float64)))
    ceil_moe = (ceil_routed + ceil_shared
                + BF16_REL * np.abs(q["moe_output"].astype(np.float64))
                + F32_REL * np.abs(float_ref["moe_output"].astype(np.float64)))

    pairs = {
        "a_dq": (x_sorted, a_q),
        "a_dq_shd": (x64, a_q_shd),
        "h_dq": (h_q, _deq(q["h_qx"], q["h_scale"])),
        "h_dq_shd": (h_q_shd, _deq(q["h_qx_shd"], q["h_scale_shd"])),
        "gu": (gu_ref, q["gu"].astype(np.float64)),
        "gu_shd": (gu_ref_shd, q["gu_shd"].astype(np.float64)),
        "h_swiglu": (h_ref, h_q),
        "h_swiglu_shd": (h_ref_shd, h_q_shd),
        "y_sorted": (y_ref, y_q),
        "y_shd": (y_ref_shd, q["y_shd"].astype(np.float64)),
        "routed_output": (float_ref["routed_output"], q["routed_output"]),
        "shared_output": (float_ref["shared_output"], q["shared_output"]),
        "moe_output": (float_ref["moe_output"], q["moe_output"]),
    }
    ceilings = {
        "gu": ceil_gu,
        "gu_shd": ceil_gu_shd,
        "y_sorted": ceil_y,
        "y_shd": ceil_y_shd,
        "routed_output": ceil_routed,
        "shared_output": ceil_shared,
        "moe_output": ceil_moe,
    }
    sources = {
        "a_routed": (src_a, da),
        "a_shared": (src_a_shd, da_shd),
        "h_routed": (src_h, _deq(q["h_qx"], q["h_scale"]) - h_q),
        "h_shared": (src_h_shd, _deq(q["h_qx_shd"], q["h_scale_shd"]) - h_q_shd),
    }
    if float_ref.get("y_final") is not None and q.get("y_final") is not None:
        res2_f = float_ref["res2"].astype(np.float64)
        rstd = 1.0 / np.sqrt(np.mean(res2_f * res2_f, axis=1) + float(ref.RMSNORM_EPS))
        g2 = np.abs(np.asarray(layer_gamma2, dtype=np.float64))
        pairs["y_final"] = (float_ref["y_final"], q["y_final"])
        ceilings["y_final"] = (ceil_moe * rstd[:, None] * g2[None, :]
                               + BF16_REL * np.abs(np.asarray(float_ref["y_final"], np.float64)))
        pairs["res2"] = (float_ref["res2"], q["res2"])
        ceilings["res2"] = ceil_moe + F32_REL * np.abs(res2_f)

    measured = {name: _stats(r, g) for name, (r, g) in pairs.items()}
    ceiling = {}
    for name, env in ceilings.items():
        r, g = pairs[name]
        ceiling[name] = _budget(env, np.asarray(r, np.float64) - np.asarray(g, np.float64))
    source_budget = {}
    for name, (bound, err) in sources.items():
        e = np.asarray(bound, np.float64)
        d = np.abs(np.asarray(err, np.float64))
        util = np.where(e > 0, d / np.maximum(e, 1e-300), 0.0)
        source_budget[name] = {
            "analytic_max": float(e.max()) if e.size else 0.0,
            "measured_max": float(d.max()) if d.size else 0.0,
            "max_utilization": float(util.max()) if util.size else 0.0,
            "violations": int(np.count_nonzero(d > e * (1.0 + 1e-6) + 1e-12)),
        }

    # ---- f32 float-golden accumulation gap (see the note below) ------------
    # The float golden runs its GEMMs in float32; the ceiling above bounds only the
    # quantization + bf16-rounding terms, so the f32 accumulation error is an extra,
    # unmodelled term.  Measure it against the f64 chain on the same (unquantized)
    # inputs so the "empirical" label is reproducible from this very manifest.
    routed_f64 = np.zeros((m, hidden))
    for t in range(m):
        for k in range(top_k):
            routed_f64[t] += w_bf16[t, k] * y_ref[int(inv[t * top_k + k])]
    shared_f64 = gate_v[:, None] * y_ref_shd
    f64_chain = {
        "routed_output": routed_f64,
        "shared_output": shared_f64,
        "moe_output": routed_f64 + shared_f64,
    }
    f32_chain_gap = {}
    for name, val in f64_chain.items():
        gap = np.abs(np.asarray(float_ref[name], np.float64) - val)
        env = ceilings[name]
        f32_chain_gap[name] = {
            "max_abs": float(gap.max()),
            "max_utilization_vs_ceiling": float(
                (gap / np.maximum(np.asarray(env, np.float64), 1e-300)).max()),
        }
    if "res2" in ceilings and float_ref.get("res2") is not None:
        res2_f64 = np.asarray(float_ref["res1"], np.float64) + f64_chain["moe_output"]
        gap = np.abs(np.asarray(float_ref["res2"], np.float64) - res2_f64)
        f32_chain_gap["res2"] = {
            "max_abs": float(gap.max()),
            "max_utilization_vs_ceiling": float(
                (gap / np.maximum(np.asarray(ceilings["res2"], np.float64), 1e-300)).max()),
        }

    return {
        "rule": rule,
        "source_budget": source_budget,
        "ceiling": ceiling,
        "measured": measured,
        "f32_chain_gap": f32_chain_gap,
        "criterion": (
            "layer-chain acceptance (m13 style): |device - float golden| must be explained by "
            "the quant-aware reference -- ad <= 1.5*|qaware - float| + 0.02*|float| + 5e-3"
        ),
        "note": (
            "Blocks: source_budget = analytic per-element bound on the quantizer itself "
            "(tight, operative); ceiling = worst-case bound propagated with absolute weights "
            "(conservative by the K-sum sign-cancellation factor); measured = the per-element "
            "q-aware deviations a layer-chain kernel is graded against; f32_chain_gap = "
            "|f32 float golden - f64 chain|, i.e. the float golden's own accumulation error. "
            "ceiling is *rigorous* against the f64 chain (gu/gu_shd/y_sorted/y_shd compare two "
            "f64 computations) but only an *empirical* bound against the stored f32 float "
            "golden for the three outputs (and res2): f32 accumulation is not in the envelope "
            "(see f32_chain_gap), so there the criterion's slack -- 1.5x oracle + 0.02*|gold| "
            "+ 5e-3 -- is what carries the margin.  All absolute values; relative stats are vs "
            "the float golden."
        ),
    }


# ---------------------------------------------------------------------------
# writing / loading a quant-aware golden group
# ---------------------------------------------------------------------------

# name -> (key in the q-aware dict, dtype, description)
QAWARE_TENSORS = (
    ("a_qx", "a_qx", "uint8", "routed A = MXFP4(x_sorted), compact rows [m*top_k, hidden/2], lohi along K (row r = perm slot r)"),
    ("a_scale", "a_scale", "uint8", "routed A E8M0 group scales [m*top_k, hidden/32]"),
    ("a_qx_shd", "a_qx_shd", "uint8", "shared-expert A = MXFP4(x) [m, hidden/2]"),
    ("a_scale_shd", "a_scale_shd", "uint8", "shared-expert A E8M0 group scales [m, hidden/32]"),
    ("gu", "gu", "bf16", "GMM#1 output (bf16), compact rows [m*top_k, 2*moe_inter] (gate|up)"),
    ("gu_shd", "gu_shd", "bf16", "shared-expert GMM#1 output (bf16) [m, 2*moe_inter] (gate|up)"),
    ("h_swiglu", "h_swiglu", "bf16", "SwiGLU output before the H quantizer [m*top_k, moe_inter]"),
    ("h_swiglu_shd", "h_swiglu_shd", "bf16", "shared-expert SwiGLU output [m, moe_inter]"),
    ("h_qx", "h_qx", "uint8", "routed H = MXFP4(h_swiglu) [m*top_k, moe_inter/2]"),
    ("h_scale", "h_scale", "uint8", "routed H E8M0 group scales [m*top_k, moe_inter/32]"),
    ("h_qx_shd", "h_qx_shd", "uint8", "shared-expert H [m, moe_inter/2]"),
    ("h_scale_shd", "h_scale_shd", "uint8", "shared-expert H E8M0 group scales [m, moe_inter/32]"),
    ("y_sorted", "y_sorted", "bf16", "GMM#2 output (bf16), compact rows [m*top_k, hidden]"),
    ("y_shd", "y_shd", "bf16", "shared-expert GMM#2 output (bf16) [m, hidden]"),
    ("routed_output", "routed_output", "bf16", "weighted routed-expert combine before the shared expert [m, hidden]"),
    ("shared_output", "shared_output", "bf16", "shared expert output (sigmoid gate) [m, hidden]"),
    ("moe_output", "moe_output", "bf16", "MoE block output = routed + shared [m, hidden]"),
)

QAWARE_LAYER_TENSORS = (
    ("x_norm1", "x_norm1", "bf16", "m6#1 output on the quant-aware chain (not equal to x.bin)"),
    ("res1", "res1", "fp32", "m6#1 residual out (fp32)"),
    ("y_final", "y_final", "bf16", "m6#2 output (bf16) from the quant-aware moe_output"),
    ("res2", "res2", "fp32", "m6#2 residual out (fp32) from the quant-aware moe_output"),
)


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def write_qaware(outdir: Path, q: dict, report: dict, meta: dict) -> dict:
    """Write one quant-aware golden group (bins + descriptors + manifest.json)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    specs = list(QAWARE_TENSORS)
    if q.get("y_final") is not None:
        specs += list(QAWARE_LAYER_TENSORS)
    for name, key, dtype_name, desc in specs:
        d = ref.write_bin(outdir / f"{name}.bin", q[key], dtype_name)
        d.update({"kind": "golden", "description": desc})
        _write_json(Path(str(outdir / f"{name}.bin") + ".json"), d)
        files.append(d)

    manifest = {
        "format_version": 1,
        "name": outdir.name,
        "rule": q["rule"],
        "m": meta["m"],
        "num_experts": meta["num_experts"],
        "top_k": meta["top_k"],
        "hidden": ref.HIDDEN,
        "moe_intermediate": ref.MOE_INTERMEDIATE,
        "shared_expert_intermediate": ref.SHARED_INTERMEDIATE,
        "group_size": ref.GROUP_SIZE,
        "seed": meta["seed"],
        "precision": {
            "accumulate": "float64",
            "weights": "dataset MXFP4 bytes (checkpoint layout, ceil packing) — not re-quantized",
            "round_points": "GU / H / Y / outputs rounded to bf16 (RNE) where the device stores bf16",
            "combine": "bf16 Y folded with bf16 top-k weights (w_tk_packed); gate on the f32 raw dot",
        },
        "activation_scale_rule": ref.SCALE_RULE_DOC[q["rule"]],
        "activation_data_rounding": ref.SCALE_RULE_ROUNDING[q["rule"]],
        "deviation_budget": report,
        "files": files,
    }
    _write_json(outdir / "manifest.json", manifest)
    return manifest


def load_qaware(outdir: Path) -> dict:
    """Load a quant-aware golden group back into memory (+ its manifest)."""
    outdir = Path(outdir)
    with open(outdir / "manifest.json") as f:
        manifest = json.load(f)
    tensors = {d["tensor"]: ref.read_bin(outdir / d["tensor"], d) for d in manifest["files"]}
    return {"manifest": manifest, "tensors": tensors}

