"""Self-checks for the MXFP4 utilities, the activation-scale rules and the datasets.

Checks:
  1. E2M1 exhaustive round trip: decode(encode(v)) == nearest level for a dense value
     sweep; all 16 codes decode and re-encode identically.
  2. E8M0: decode/encode round trip; the weight-packing (ceil) rule
     scale = 2**ceil(log2(amax/6)) keeps max|q| <= 6.
  3. Activation scale rules (M26): the floor rule equals
     clip(bf16 exponent field of amax - 2, 0, 254); floor and ceil agree while
     amax is in [4*2^k, 6*2^k] and differ on (6*2^k, 8*2^k]; the analytic
     per-element bound (half the enclosing E2M1 level gap + the ±6 saturation term)
     is never violated by either rule; the CAST_ROUND data path rounds ties away from
     zero and saturates at ±6.
  4. MXFP4 round trip: pack -> unpack -> pack is bitwise identical at the value level.
  5. bf16 RNE round trip is idempotent.
  6. Independently written quantizer agreement: m13_moe_layer/check_ref.py's
     hardware-rule quantizer (`quant_mxfp4_hw`, byte-verified against the NPU) and its
     golden-rule quantizer (`quant_mxfp4_f32`) must reproduce `quantize_activations`
     byte-for-byte for the floor / ceil rules.  Skipped when that file is absent.
  7. Dataset groups: byte-deterministic regeneration (bins + descriptors + manifests,
     quant-aware groups included), the float golden recomputed from the loaded bins,
     the layer-chain goldens (residuals bit-exact, RMSNorm invariant on x_norm1 /
     y_final), the quant-aware golden recomputed from the inputs, its deviation report
     reproduced, and every analytic bound respected (0 violations).
  8. Cross-tier consistency: every group carries the same tensor set / dtypes /
     shape pattern, and hidden / moe_intermediate / group size match the module
     constants.
  9. Optional: device cross-check against `M13_DUMP=1` dumps of m13_moe_layer in MoE
     slice mode (mode 0) — the A/H quantizer bytes and the device intermediate
     tensors must match the floor-rule quant-aware golden (see README).
  10. Router fp32 FTZ (M46): the hardware flush of subnormal fp32 results is modelled
     in `moe_block_ref.router_topk` (docs/05 §6.1, user ruling: subnormal FTZ is a
     hardware mode, not something to avoid).  Pinned on a designed input that really
     lands in the subnormal band without any real weight — the pre-M46 reference must
     FAIL the device semantics and the modelled one must PASS, so deleting the
     modelling is caught immediately.

Environment identification (M75): before any dataset judgement the run prints the actual
interpreter / numpy / BLAS and whether that is the stack the frozen `.bin` files were
built with (README pins it in "复现与自检").  No criterion is relaxed — the point is
that a failure on another numeric stack exits 3 instead of 1, so "the committed `.bin`
drifted" and "the numeric environment changed" are distinguishable instead of both
reading as a bare `[FAIL]` (README "环境识别（M75）").

Usage:
    python3.12 selfcheck.py [--data DIR] [--groups m1,m33,real] [--real]
                            [--m13-dump DIR] [--no-regen] [--env-ok]
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import moe_block_ref as ref
import qaware_ref as qr
from moe_block_ref import (
    E2M1_POS,
    FTZ_MIN_NORMAL,
    bf16_bits_to_f32,
    e2m1_decode,
    e2m1_encode,
    e8m0_decode,
    e8m0_encode_pow2,
    f32_to_bf16,
    f32_to_bf16_bits,
    ftz_f32,
    generate_dataset,
    group_scale_exp,
    load_dataset,
    pack_mxfp4,
    quantize_activations,
    reference_moe_block,
    rmsnorm_ref,
    router_topk,
    unpack_mxfp4,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
HIDDEN = ref.HIDDEN
INTER = ref.MOE_INTERMEDIATE
GROUP = ref.GROUP_SIZE

# name -> (m, num_experts, top_k, seed)
GROUP_SPECS = {
    "m1": (1, 4, 2, 20260926),
    "m33": (33, 4, 2, 20260927),
    "real": (8, 512, 10, 20260928),
}
COMMITTED_GROUPS = ("m1", "m33")
REAL_WEIGHTS_MIB = 1280.8  # 1343062304 B = 1.2509 GiB for E=512 / topK=10 weights

# expected deviation-report key sets (a silent skip -- e.g. a missing layer-chain entry --
# must fail the self-check rather than shrink the published budget; review M26-r1 N1)
QAWARE_MEASURED_KEYS = {
    "a_dq", "a_dq_shd", "h_dq", "h_dq_shd", "gu", "gu_shd", "h_swiglu", "h_swiglu_shd",
    "y_sorted", "y_shd", "routed_output", "shared_output", "moe_output", "y_final", "res2",
}
QAWARE_CEILING_KEYS = {
    "gu", "gu_shd", "y_sorted", "y_shd", "routed_output", "shared_output", "moe_output",
    "y_final", "res2",
}
QAWARE_SOURCE_KEYS = {"a_routed", "a_shared", "h_routed", "h_shared"}
QAWARE_GAP_KEYS = {"routed_output", "shared_output", "moe_output", "res2"}

# 判定项 / guard 项计数（docs/17 §2.1：恒真与自洽类要单列）
COUNTS = {"check": 0, "guard": 0}


class SelfcheckFailure(SystemExit):
    """A judgement item (check()/guard()) failed.

    Subclasses SystemExit with a string code, so an uncaught one still prints the message
    and exits 1 exactly as before.  ``main`` catches it to map the exit code onto the
    numeric environment; plain SystemExit (a usage error, e.g. an unknown ``--groups``
    value) is left alone and never re-labelled as environment-ambiguous.
    """


def check(cond: bool, msg: str) -> None:
    COUNTS["check"] += 1
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise SelfcheckFailure(f"selfcheck failed: {msg}")


def guard(cond: bool, msg: str) -> None:
    """Harness / metadata invariant (docs/17 §2.1 guard 栏): counted separately."""
    COUNTS["guard"] += 1
    status = "GUARD" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise SelfcheckFailure(f"selfcheck guard failed: {msg}")


def skip(msg: str) -> None:
    print(f"  [SKIP] {msg}")


# ---------------------------------------------------------------------------
# environment identification (M75)
# ---------------------------------------------------------------------------

# The numeric stack the frozen `.bin` files were built with.  Every judgement against a
# committed bin below is *bit-exact*, and the reference chain's fp32 GEMM accumulation
# order is a property of the BLAS numpy links against, so another stack flips a small
# number of bf16 codes and makes those judgement items FAIL with no code change at all
# (measured magnitudes: README "环境识别（M75）").  This block only *identifies* the
# stack and labels the exit code accordingly; it does not touch any criterion.
CANONICAL_ENV = {
    "interpreter": "/usr/local/python3.12.13/bin/python3.12",
    "numpy": "2.5.1",
    "blas": "scipy-openblas",
}
ENV_MISMATCH_RC = 3  # a judgement item failed on a non-canonical numeric stack


def _blas_identity() -> tuple:
    """``(name, version)`` of the BLAS numpy is linked against, ``"unknown"`` if opaque."""
    try:
        blas = np.__config__.CONFIG["Build Dependencies"]["blas"]
        return blas.get("name", "unknown"), blas.get("version", "unknown")
    except Exception:
        return "unknown", "unknown"


def report_environment(assume_ok: bool = False) -> dict:
    """Print the actual numeric environment; say whether it is the pinned one.

    Returns ``{"interpreter", "numpy", "blas", "blas_version", "canonical"[, "why"]}``.
    The two env lines and the ``[ENV]`` notices are deliberately NOT check()/guard() items
    -- counting them would move the frozen `644 判定项 + 6 guard 项` baseline for a
    disclosure that grades nothing.
    """
    blas_name, blas_version = _blas_identity()
    env = {"interpreter": sys.executable, "numpy": np.__version__,
           "blas": blas_name, "blas_version": blas_version}
    print(f"env: interpreter={env['interpreter']}")
    print(f"env: numpy={env['numpy']} blas={blas_name} {blas_version}")
    why = []
    if np.__version__ != CANONICAL_ENV["numpy"]:
        why.append(f"numpy {np.__version__} != {CANONICAL_ENV['numpy']}")
    if blas_name != CANONICAL_ENV["blas"]:
        why.append(f"BLAS {blas_name} != {CANONICAL_ENV['blas']}")
    if assume_ok:
        env["canonical"] = True
        print("  [ENV] --env-ok: this run declares its numeric stack authoritative; "
              "a failing judgement item will exit 1")
        return env
    if not why:
        env["canonical"] = True
        print(f"  [ENV] canonical numeric environment: numpy {np.__version__} / "
              f"{blas_name} {blas_version} (README pins numpy {CANONICAL_ENV['numpy']} / "
              f"{CANONICAL_ENV['blas']} via {CANONICAL_ENV['interpreter']})")
        return env
    env["canonical"] = False
    env["why"] = why
    print(f"  [ENV] non-canonical numeric environment: {'; '.join(why)}")
    print(f"  [ENV] README pins numpy {CANONICAL_ENV['numpy']} / {CANONICAL_ENV['blas']} "
          f"via {CANONICAL_ENV['interpreter']} for the frozen `.bin` comparisons.")
    print("  [ENV] those comparisons are bit-exact, so a FAIL here is environment-ambiguous "
          "rather than a `.bin` drift: measured 2026-09-26 on numpy 1.26.4 / openblas64 "
          "0.3.23.dev, m33 flipped 17/84480 `routed_output` codes (max 2 ulp), 15/84480 "
          "`shared_output` (max 4 ulp), 11/84480 `moe_output` (max 2 ulp).")
    print(f"  [ENV] a failing judgement item will exit {ENV_MISMATCH_RC} instead of 1; "
          f"re-run with {CANONICAL_ENV['interpreter']}, or pass --env-ok to grade this "
          f"stack as authoritative (then a failure exits 1).")
    return env


# ---------------------------------------------------------------------------
# 1-5: numeric fundamentals
# ---------------------------------------------------------------------------


def check_e2m1() -> None:
    print("check 1: e2m1 encode/decode")
    codes = np.arange(16, dtype=np.uint8)
    vals = e2m1_decode(codes)
    check(np.array_equal(e2m1_encode(vals), codes), "all 16 e2m1 codes round-trip")
    check(np.allclose(np.abs(vals), np.concatenate([E2M1_POS, E2M1_POS])),
          "e2m1 levels are 0,.5,1,1.5,2,3,4,6 with sign")
    sweep = np.linspace(-7.0, 7.0, 200001, dtype=np.float32)
    q = e2m1_decode(e2m1_encode(sweep))
    levels = np.concatenate([-E2M1_POS[1:][::-1], E2M1_POS])
    dist_q = np.abs(q[..., None] - levels).min(axis=1)
    dist_exact = np.abs(sweep.astype(np.float64)[..., None] - levels).min(axis=1)
    check(np.all(dist_q <= dist_exact + 1e-6), "encode picks a nearest level everywhere")
    check(e2m1_encode(np.array([-0.0], dtype=np.float32))[0] == 0x8,
          "-0.0 encodes to sign-bit code 0x8")


def check_e8m0() -> None:
    print("check 2: e8m0 scales (weight-packing ceil rule)")
    exps = np.array([-127, -6, 0, 7, 127], dtype=np.int64)
    b = e8m0_encode_pow2(exps)
    check(np.allclose(e8m0_decode(b), np.exp2(exps.astype(np.float32))),
          "e8m0 encode/decode round trip")
    rng = np.random.default_rng(0)
    amax = np.abs(rng.standard_normal(4096)).astype(np.float32) * np.float32(0.1)
    amax[0] = 0.0
    s = np.exp2(group_scale_exp(amax).astype(np.float32))
    qmax = amax / s
    check(bool(np.all(qmax <= 6.0)) and np.isclose(qmax[0], 0.0),
          "scale=2**ceil(log2(amax/6)) keeps max|q|<=6")


def check_scale_rules() -> None:
    """Check 3: official OCP/floor sequence vs the legacy ceil rule, incl. the corners."""
    print("check 3: activation quantization — official OCP/floor sequence vs legacy ceil")
    rng = np.random.default_rng(11)
    amax = np.abs(rng.standard_normal(4000)).astype(np.float32) * 0.05
    amax[:4] = np.array([0.0, 6.0, 7.999, 2.0 ** -20], dtype=np.float32)

    f = ref.e8m0_bytes_floor(amax)
    c = ref.e8m0_bytes_ceil(amax)
    field = ref.exp_field(amax).astype(np.int32)
    check(np.array_equal(f, np.clip(field - 2, 0, 254)),
          "official finite path: byte = clip(bf16 exponent field of amax - 2, 0, 254) "
          "(= 127 + floor(log2 amax) - 2)")
    check(np.array_equal(c, e8m0_encode_pow2(group_scale_exp(amax))),
          "legacy ceil byte = 127 + ceil(log2(amax/6))")
    check(np.array_equal(f, ref.scale_rule_bytes(amax, "floor")) and
          np.array_equal(c, ref.scale_rule_bytes(amax, "ceil")),
          "scale_rule_bytes dispatches both rules")

    # the legacy rule differs from the official one exactly on (6*2^k, 8*2^k]
    base = np.exp2(np.arange(-6, 7, dtype=np.float64))
    lo = (4.0 * base).astype(np.float32)          # amax in [4,6]*2^k -> both give 2^k
    hi = (7.0 * base).astype(np.float32)          # amax in (6,8]*2^k -> official 2^k, legacy 2^(k+1)
    check(np.array_equal(ref.e8m0_bytes_floor(lo), ref.e8m0_bytes_ceil(lo)),
          "official == legacy while amax in [4*2^k, 6*2^k]")
    diff = ref.e8m0_bytes_floor(hi).astype(np.int32) - ref.e8m0_bytes_ceil(hi).astype(np.int32)
    check(bool(np.all(diff == -1)), "official = legacy - 1 on amax in (6*2^k, 8*2^k] "
                                    "(the legacy rule is the deviating one)")

    # the analytic bound must hold for both rules, and be tight (hit) somewhere.
    # Inputs go on the bf16 grid first: that is the dtype the device quantizer loads and
    # the quantization-bound below covers only the MXFP4 step, not the bf16 input rounding.
    for rule in ref.SCALE_RULES:
        x = f32_to_bf16(np.abs(rng.standard_normal((512, 64))).astype(np.float32) *
                        np.float32(0.2))
        packed, scales = quantize_activations(x, rule)
        xq = unpack_mxfp4(packed, scales)
        bound = qr.quant_element_bound(np.abs(x), qr.rule_expand(scales, x.shape[1]))
        dev = np.abs(xq.astype(np.float64) - x.astype(np.float64))
        check(bool(np.all(dev <= bound * (1 + 1e-6) + 1e-12)),
              f"{rule} rule: |q-v| <= analytic per-element bound (0 violations)")
        check(float((dev / np.maximum(bound, 1e-300)).max()) > 0.5,
              f"{rule} rule: bound is tight (utilization "
              f"{float((dev / np.maximum(bound, 1e-300)).max()):.2f} > 0.5)")

    # data-side rounding: ties away from zero, saturation at ±6 (official CAST_ROUND)
    codes = ref.e2m1_encode_away(np.array([[0.75, 1.25, 2.5, 5.0, 7.0, -7.0]], np.float32))
    vals = e2m1_decode(codes)
    check(list(vals[0]) == [1.0, 1.5, 3.0, 6.0, 6.0, -6.0],
          "CAST_ROUND path: ties away from zero, saturating at ±6")
    check(int(ref.e2m1_encode_away(np.array([[7.0]], np.float32))[0, 0]) == 0x7,
          "=6 saturation uses e2m1 code 7 (=6.0)")

    # --- corners of the official sequence (docs/13 §4.1 :402-415, §6) ------
    # (a) all-zero / bf16-denormal group: exponent field clamps to emax -> halfScale 0
    tiny = np.zeros((2, 64), dtype=np.float32)
    tiny[1, :] = np.float32(1e-40)               # bf16 denormal, field < emax
    packed, scales = quantize_activations(tiny, "floor")
    check(bool(np.all(scales == 0)) and bool(np.all(packed == 0)),
          "official corner: a group whose field clamps to emax gets byte 0 + all-zero codes "
          "(halfScale 0, :402-403/:414)")
    # (b) ±Inf / NaN group: scale byte 0xFF, halfScale 0x7F81 -> all codes 0
    inf = np.full((2, GROUP), np.inf, dtype=np.float32)
    inf[0, 1] = -np.inf
    inf[1, :] = np.nan
    packed, scales = quantize_activations(inf, "floor")
    check(bool(np.all(scales == 0xFF)), "official corner: non-finite group -> E8M0 NaN byte "
                                        "0xFF (:406 / docs/13 §4.2)")
    check(bool(np.all(packed == 0)), "official corner: non-finite group -> Mul(±Inf, NaN) "
                                     "= NaN -> cast -> all-zero codes (:413, docs/13 §6)")
    # 这条注记是**运行时证据**（每跑一次就被写进一份新日志）⇒ 必须与同 commit 的事实一致。
    # 旧句「device kernels miss the :413 override … M32 is fixing that」在 M32 落地后即成过期断言，
    # 且会随每次运行自我复制。现改为**当场核对**设备侧 override 是否在（可复算的判据）。
    #
    # ⚠ 上一版把这句写成「`grep 0x7F81` 各 3 处、**都在各自 MxQuantComputeScale 内**」——**假的**
    #   （`0c163b5` 那版实测）：设备侧真正用的是**小写** `NAN_CUSTOMIZATION = 0x7f81`，**不被
    #   `grep 0x7F81` 匹配**；而大写命中每份文件 3 处里**只有 1 处在那个函数内，且是注释**，
    #   另两处是文件头注释与 host 参考 `RefQuantMxfp4`。故这里改成分开核三件事（见下）。
    _kernels = ("m2_mxfp4_quant/m2_mxfp4_quant.asc",
                "m5_swiglu_quant/m5_swiglu_quant.asc",
                "m13_moe_layer/m13_moe_layer.asc")

    def _count(k, pat):
        p = REPO / k
        return p.read_text().count(pat) if p.exists() else -1

    _dev = {k: _count(k, "Duplicate(nanRegTensor, NAN_CUSTOMIZATION)") for k in _kernels}
    _name = {k: _count(k, "NAN_CUSTOMIZATION") for k in _kernels}
    _upper = {k: _count(k, "0x7F81") for k in _kernels}
    _lower = {k: _count(k, "0x7f81") for k in _kernels}
    _missing = [k for k, n in _dev.items() if n <= 0]
    # (c) 的措辞**刻意不做逐文件分解**：静态写出"哪 3 处分别是文件头注释/函数内注释/host 参考"
    # 只在 m2/m5 成立（m13 没有文件头那处、host 参考叫 `H_RefQuantMxfp4`）—— 那正是会自行过期的东西
    # （`96e7e96` 那版曾把它拆成逐文件三条；该句由 `daf3344` 删）这里只陈述**由上面的计数当场支撑**的
    # 两件事：大写命中数、以及设备侧常量是小写。
    print("  [note] device side of the :413 halfScale override (non-finite group -> halfScale "
          "0x7f81, then Mul(±Inf, NaN) = NaN -> cast -> all-zero codes). Counts per kernel, because a "
          "case-sensitive grep is NOT a valid locator here:\n"
          "        (a) grep 'Duplicate(nanRegTensor, NAN_CUSTOMIZATION)' = %s  <- 真正的设备侧覆盖点，"
          "在 MxQuantComputeScale 内\n"
          "        (b) grep 'NAN_CUSTOMIZATION' = %s\n"
          "        (c) grep '0x7F81'（大写）= %s：这些命中都**不在设备路径上**（设备侧的常量是**小写** "
          "`0x7f81`，见下面那栏 —— 大小写敏感的 `0x7F81` 不匹配它，故 (c) 数不到任何设备侧覆盖点）\n"
          "        (c') grep '0x7f81'（小写）= %s：设备侧 `constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;`"
          "（定义处），被 (a) 消费\n"
          "        %s As of 2026-09-26 (M32) reference and device agree on the Inf corner; "
          "see m13_moe_layer/README §5.4 and docs/13 §6."
          % (_dev, _name, _upper, _lower,
             "-1 = 文件不存在。" if not _missing else "MISSING/EMPTY: %s" % _missing))


def check_mxfp4_roundtrip() -> None:
    print("check 4: mxfp4 pack/unpack round trip")
    w = np.array([[0.0, 0.5, 1.0, 1.5] + [0.0] * 28], dtype=np.float32)
    packed, scales = pack_mxfp4(w)
    check(scales[0, 0] == 125, "amax=1.5 group encodes e8m0 byte 125 (2^-2)")
    check(packed[0, 0] == 0x40, "lohi: byte0 = code(w0/s)=0 | code(w1/s)=4 <<4")
    check(packed[0, 1] == 0x76, "lohi: byte1 = code(w2/s)=6 | code(w3/s)=7 <<4")
    check(np.array_equal(unpack_mxfp4(packed, scales), w), "known-answer unpack exact")

    rng = np.random.default_rng(1)
    for n, k in [(640, 2560), (2560, 640), (1280, 2560), (17, 32)]:
        w = (rng.standard_normal((n, k)) * 0.03).astype(np.float32)
        packed, scales = pack_mxfp4(w)
        w1 = unpack_mxfp4(packed, scales)
        w2 = unpack_mxfp4(*pack_mxfp4(w1))
        check(np.array_equal(w2, w1),
              f"dequant fixed point: unpack(pack(unpack(pack(w)))) == unpack(pack(w)) [{n}x{k}]")
        codes = np.empty((n, k), dtype=np.uint8)
        codes[:, 0::2] = packed & 0x0F
        codes[:, 1::2] = packed >> 4
        lv = e2m1_decode(codes).reshape(n, k // 32, 32)
        check(np.allclose(lv * e8m0_decode(scales).astype(np.float32)[:, :, None], w1.reshape(n, k // 32, 32)),
              f"dequant == e2m1(level) * e8m0(scale) [{n}x{k}]")
        err = np.max(np.abs(w1 - w))
        check(err < 0.2, f"dequant error bounded [{n}x{k}] max|err|={err:.5f}")
    # the two activation rules share the byte layout with the weight packer
    x = (rng.standard_normal((8, 64)) * 0.4).astype(np.float32)
    for rule in ref.SCALE_RULES:
        packed, scales = quantize_activations(x, rule)
        check(packed.shape == (8, 32) and scales.shape == (8, 2),
              f"quantize_activations({rule}) layout [rows, K/2] / [rows, K/32]")
        lv = e2m1_decode(_codes_of(packed)).reshape(8, 2, 32)
        s = e8m0_decode(scales).astype(np.float32)[:, :, None]
        check(np.array_equal((lv * s).reshape(8, 64), unpack_mxfp4(packed, scales)),
              f"quantize_activations({rule}) dequant == e2m1(level) * e8m0(scale)")
        v = unpack_mxfp4(packed, scales)
        check(np.array_equal(unpack_mxfp4(*quantize_activations(v, rule)), v),
              f"quantize_activations({rule}) is a value fixed point")


def _codes_of(packed: np.ndarray) -> np.ndarray:
    rows, kh = packed.shape
    codes = np.empty((rows, kh * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return codes


def check_bf16() -> None:
    print("check 5: bf16 RNE conversion")
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(100000) * 10).astype(np.float32)
    once = f32_to_bf16(x)
    twice = f32_to_bf16(once)
    check(np.array_equal(f32_to_bf16_bits(once), f32_to_bf16_bits(x)),
          "f32->bf16 is idempotent (RNE)")
    check(np.array_equal(once, twice), "bf16 round trip stable")
    specials = np.array([0.0, -0.0, np.inf, -np.inf, 1e-40], dtype=np.float32)
    out = bf16_bits_to_f32(f32_to_bf16_bits(specials))
    check(np.isinf(out[2]) and np.isinf(out[3]) and out[0] == 0.0 and out[1] == 0.0,
          "inf/zero preserved through bf16 conversion")


def check_hw_quantizer() -> None:
    """Check 6: agreement with m13_moe_layer's independently written quantizers."""
    print("check 6: agreement with m13_moe_layer/check_ref.py quantizers")
    path = REPO / "m13_moe_layer" / "check_ref.py"
    if not path.exists():
        skip(f"{path} not found — independent quantizer witness unavailable")
        return
    sys.path.insert(0, str(path.parent))
    prev_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # don't drop a __pycache__ into m13_moe_layer/
    try:
        import check_ref as m13
    except Exception as exc:  # pragma: no cover - defensive
        skip(f"cannot import {path}: {exc}")
        return
    finally:
        sys.dont_write_bytecode = prev_bytecode
    rng = np.random.default_rng(7)
    rows = np.vstack([
        (rng.standard_normal((16, 256)) * 0.5).astype(np.float32),          # N(0, .5)
        (rng.standard_normal((16, 256)) * 0.02).astype(np.float32),         # small
        (np.abs(rng.standard_normal((16, 256))) * 7.0).astype(np.float32),  # saturating
        (np.exp2(rng.integers(-4, 5, (16, 256))).astype(np.float32)
         * rng.choice([1.0, 1.5, 1.75], (16, 256)).astype(np.float32)),     # boundary-heavy
    ]).astype(np.float32)
    # the device quantizer loads bf16 data; put the witness inputs on the bf16 grid
    rows = f32_to_bf16(rows)
    pq, sq = quantize_activations(rows, "floor")
    check(np.array_equal(pq, m13.quant_mxfp4_hw(rows)[0]) and
          np.array_equal(sq, m13.quant_mxfp4_hw(rows)[1]),
          f"floor (official OCP sequence) == m13 quant_mxfp4_hw byte-for-byte "
          f"({pq.size}+{sq.size} bytes, bf16-grid activations)")
    pq, sq = quantize_activations(rows, "ceil")
    check(np.array_equal(pq, m13.quant_mxfp4_f32(rows)[0]) and
          np.array_equal(sq, m13.quant_mxfp4_f32(rows)[1]),
          f"legacy ceil rule == m13 quant_mxfp4_f32 byte-for-byte ({pq.size}+{sq.size} bytes)")


# ---------------------------------------------------------------------------
# 7: dataset groups
# ---------------------------------------------------------------------------


def expected_tensors(m: int, num_experts: int, top_k: int) -> dict:
    """tensor name -> (dtype, shape) expected for every dataset group."""
    e = num_experts
    shapes = {
        "x.bin": ("bf16", (m, HIDDEN)),
        "router_weight.bin": ("bf16", (e, HIDDEN)),
        "shared_expert_gate_weight.bin": ("bf16", (1, HIDDEN)),
        "x_res.bin": ("bf16", (m, HIDDEN)),
        "gamma1.bin": ("bf16", (1, HIDDEN)),
        "gamma2.bin": ("bf16", (1, HIDDEN)),
        "experts.gate_up_proj.bin": ("uint8", (e, 2 * INTER, HIDDEN // 2)),
        "experts.gate_up_proj.weight_scale.bin": ("uint8", (e, 2 * INTER, HIDDEN // GROUP)),
        "experts.down_proj.bin": ("uint8", (e, HIDDEN, INTER // 2)),
        "experts.down_proj.weight_scale.bin": ("uint8", (e, HIDDEN, INTER // GROUP)),
        "shared_expert.gate_proj.bin": ("uint8", (INTER, HIDDEN // 2)),
        "shared_expert.gate_proj.weight_scale.bin": ("uint8", (INTER, HIDDEN // GROUP)),
        "shared_expert.up_proj.bin": ("uint8", (INTER, HIDDEN // 2)),
        "shared_expert.up_proj.weight_scale.bin": ("uint8", (INTER, HIDDEN // GROUP)),
        "shared_expert.down_proj.bin": ("uint8", (HIDDEN, INTER // 2)),
        "shared_expert.down_proj.weight_scale.bin": ("uint8", (HIDDEN, INTER // GROUP)),
        "router_logits.bin": ("fp32", (m, e)),
        "shared_gate_logits.bin": ("fp32", (m,)),
        "topk_ids.bin": ("int32", (m, top_k)),
        "topk_weights.bin": ("fp32", (m, top_k)),
        "perm_src_token.bin": ("int32", (m * top_k,)),
        "perm_expert.bin": ("int32", (m * top_k,)),
        "expert_token_counts.bin": ("int32", (e,)),
        "x_sorted.bin": ("bf16", (m * top_k, HIDDEN)),
        "routed_output.bin": ("bf16", (m, HIDDEN)),
        "shared_output.bin": ("bf16", (m, HIDDEN)),
        "moe_output.bin": ("bf16", (m, HIDDEN)),
        "x_norm1.bin": ("bf16", (m, HIDDEN)),
        "res1.bin": ("fp32", (m, HIDDEN)),
        "y_final.bin": ("bf16", (m, HIDDEN)),
        "res2.bin": ("fp32", (m, HIDDEN)),
    }
    return shapes


def expected_qaware_tensors(m: int, top_k: int) -> dict:
    t = m * top_k
    return {
        "a_qx.bin": ("uint8", (t, HIDDEN // 2)),
        "a_scale.bin": ("uint8", (t, HIDDEN // GROUP)),
        "a_qx_shd.bin": ("uint8", (m, HIDDEN // 2)),
        "a_scale_shd.bin": ("uint8", (m, HIDDEN // GROUP)),
        "gu.bin": ("bf16", (t, 2 * INTER)),
        "gu_shd.bin": ("bf16", (m, 2 * INTER)),
        "h_swiglu.bin": ("bf16", (t, INTER)),
        "h_swiglu_shd.bin": ("bf16", (m, INTER)),
        "h_qx.bin": ("uint8", (t, INTER // 2)),
        "h_scale.bin": ("uint8", (t, INTER // GROUP)),
        "h_qx_shd.bin": ("uint8", (m, INTER // 2)),
        "h_scale_shd.bin": ("uint8", (m, INTER // GROUP)),
        "y_sorted.bin": ("bf16", (t, HIDDEN)),
        "y_shd.bin": ("bf16", (m, HIDDEN)),
        "routed_output.bin": ("bf16", (m, HIDDEN)),
        "shared_output.bin": ("bf16", (m, HIDDEN)),
        "moe_output.bin": ("bf16", (m, HIDDEN)),
        "x_norm1.bin": ("bf16", (m, HIDDEN)),
        "res1.bin": ("fp32", (m, HIDDEN)),
        "y_final.bin": ("bf16", (m, HIDDEN)),
        "res2.bin": ("fp32", (m, HIDDEN)),
    }


def _all_files(dirpath: Path) -> list[Path]:
    return sorted(p for p in dirpath.rglob("*") if p.is_file())


def check_qaware_group(name: str, outdir: Path, man: dict, tensors: dict) -> None:
    """Check the quant-aware groups of one dataset: layout, recompute, budget."""
    q_root = outdir / "qaware"
    rules = sorted(man["quantization"]["qaware_golden"].keys())
    on_disk = sorted(p.parent.name for p in q_root.glob("*/manifest.json"))
    check(rules == on_disk, f"{name}: qaware manifest dirs match the manifest "
                            f"({rules} vs {on_disk})")
    guard("floor" in rules, f"{name}: the device-path floor rule is emitted "
                            f"(rules={rules}) -- harness invariant")
    m, e, top_k = man["m"], man["num_experts"], man["top_k"]
    layer = {"x_res": tensors["x_res.bin"], "gamma1": tensors["gamma1.bin"].reshape(-1),
             "gamma2": tensors["gamma2.bin"].reshape(-1)}
    experts_packed = {k: tensors[k + ".bin"] for k in
                      ("experts.gate_up_proj", "experts.gate_up_proj.weight_scale",
                       "experts.down_proj", "experts.down_proj.weight_scale")}
    shared_packed = {k: tensors[k + ".bin"] for k in
                     ("shared_expert.gate_proj", "shared_expert.gate_proj.weight_scale",
                      "shared_expert.up_proj", "shared_expert.up_proj.weight_scale",
                      "shared_expert.down_proj", "shared_expert.down_proj.weight_scale")}
    float_ref = reference_moe_block(tensors["x.bin"], tensors["router_weight.bin"],
                                    tensors["shared_expert_gate_weight.bin"],
                                    experts_packed, shared_packed, top_k)
    float_ref = dict(float_ref)
    float_ref["res1"], float_ref["x_norm1"] = rmsnorm_ref(layer["x_res"], layer["gamma1"])
    # m6#2 consumes the bf16 moe_output tensor (same construction as generate_dataset,
    # README "层链语义"): keep this identical or the res2/y_final budget entries diverge
    float_ref["res2"], float_ref["y_final"] = rmsnorm_ref(
        f32_to_bf16(float_ref["moe_output"]), layer["gamma2"], residual=float_ref["res1"])

    want = expected_qaware_tensors(m, top_k)
    for rule in rules:
        qdir = q_root / rule
        ds = qr.load_qaware(qdir)
        qman = ds["manifest"]
        qt = ds["tensors"]
        got = {d["tensor"]: (d["dtype"], tuple(d["shape"])) for d in qman["files"]}
        check(got == want, f"{name}/qaware/{rule}: tensor set / dtypes / shapes as declared")

        q = qr.qaware_moe_block(tensors["x.bin"], tensors["router_weight.bin"],
                                tensors["shared_expert_gate_weight.bin"],
                                experts_packed, shared_packed, top_k, rule=rule, layer=layer)
        for d in qman["files"]:
            stored = qt[d["tensor"]]
            got_arr = q[d["tensor"][:-4]]
            if d["dtype"] == "bf16":
                ok = np.array_equal(f32_to_bf16_bits(stored), f32_to_bf16_bits(got_arr))
            else:
                ok = np.array_equal(stored, got_arr)
            check(ok, f"{name}/qaware/{rule}: recomputed '{d['tensor']}' matches the "
                      f"stored bin bit-exactly")

        report = qr.deviation_report(float_ref, q, tensors["x.bin"], experts_packed,
                                    shared_packed, rule=rule, layer=layer)
        # key sets must be complete: a silently skipped entry (e.g. the layer-chain
        # y_final/res2 before review M26-r1 N1) would shrink the published budget
        check(set(report["measured"]) == QAWARE_MEASURED_KEYS,
              f"{name}/qaware/{rule}: report measured key set complete "
              f"(missing {sorted(QAWARE_MEASURED_KEYS - set(report['measured']))})")
        check(set(report["ceiling"]) == QAWARE_CEILING_KEYS,
              f"{name}/qaware/{rule}: report ceiling key set complete "
              f"(missing {sorted(QAWARE_CEILING_KEYS - set(report['ceiling']))})")
        check(set(report["source_budget"]) == QAWARE_SOURCE_KEYS,
              f"{name}/qaware/{rule}: report source_budget key set complete")
        check(set(report["f32_chain_gap"]) == QAWARE_GAP_KEYS,
              f"{name}/qaware/{rule}: report f32_chain_gap key set complete")
        stored_rep = qman["deviation_budget"]
        check(set(stored_rep["measured"]) == QAWARE_MEASURED_KEYS,
              f"{name}/qaware/{rule}: stored budget measured key set complete "
              f"(missing {sorted(QAWARE_MEASURED_KEYS - set(stored_rep['measured']))})")
        check(set(stored_rep["ceiling"]) == QAWARE_CEILING_KEYS,
              f"{name}/qaware/{rule}: stored budget ceiling key set complete "
              f"(missing {sorted(QAWARE_CEILING_KEYS - set(stored_rep['ceiling']))})")
        for key, block in (("source_budget", report["source_budget"]),
                           ("ceiling", report["ceiling"])):
            bad = {k: v for k, v in block.items() if v["violations"] != 0}
            check(not bad, f"{name}/qaware/{rule}: {key} respected on every element "
                           f"(violations {bad if bad else 0})")
        util = max(v["max_utilization"] for v in report["source_budget"].values())
        check(util <= 1.0 + 1e-6,
              f"{name}/qaware/{rule}: source bounds hold (worst utilization {util:.3f})")
        # stored report must be reproduced (the manifest is the published budget)
        check(stored_rep["criterion"] == report["criterion"].strip() or
              stored_rep["criterion"] == report["criterion"],
              f"{name}/qaware/{rule}: stored budget criterion matches")
        for tname in ("routed_output", "shared_output", "moe_output",
                      "y_final", "res2"):
            if tname not in stored_rep["measured"]:
                continue
            a = stored_rep["measured"][tname]["max_abs"]
            b = report["measured"][tname]["max_abs"]
            check(abs(a - b) <= 1e-9,
                  f"{name}/qaware/{rule}: stored measured max_abs for {tname} reproduced "
                  f"({a:.6g})")
        # the q-aware deviation must explain the float-golden gap it reports
        for tname in ("routed_output", "shared_output", "moe_output",
                      "y_final", "res2"):
            if tname not in report["measured"]:
                continue
            check(report["measured"][tname]["max_abs"] > 0.0,
                  f"{name}/qaware/{rule}: {tname} deviates from the float golden (>0)")


def check_dataset_group(name: str, spec, data_root: Path, regenerate: bool = True) -> None:
    print(f"check 7: dataset '{name}' (m={spec[0]}, E={spec[1]}, top_k={spec[2]})")
    outdir = data_root / name
    check((outdir / "manifest.json").exists(), f"{outdir} exists")
    ds = load_dataset(outdir)
    man = ds["manifest"]
    t = ds["tensors"]

    # --- schema: tensor set / dtypes / shapes ------------------------------
    want = expected_tensors(spec[0], spec[1], spec[2])
    got = {d["tensor"]: (d["dtype"], tuple(d["shape"])) for d in man["files"]}
    check(got == want, f"{name}: tensor set / dtypes / shapes as declared "
                       f"(extra={sorted(set(got) - set(want))}, missing={sorted(set(want) - set(got))})")
    check((man["hidden"], man["moe_intermediate"], man["shared_expert_intermediate"],
           man["group_size"]) == (HIDDEN, INTER, INTER, GROUP),
          f"{name}: real model K/N kept (hidden={HIDDEN}, inter={INTER}, group={GROUP})")
    for d in man["files"]:
        path = outdir / d["tensor"]
        check(path.stat().st_size == d["size_bytes"], f"{name}/{d['tensor']} size matches")
    guard(man["format_version"] == ref.MANIFEST_VERSION,
          f"{name}: manifest format_version == {ref.MANIFEST_VERSION} (generator/harness invariant)")

    # --- byte determinism (also proves published bins are reproducible) ----
    if regenerate:
        with tempfile.TemporaryDirectory() as td:
            generate_dataset(Path(td) / name, *spec)
            tmproot = Path(td) / name
            for f in _all_files(outdir):
                if not f.name.endswith((".bin", ".json")):
                    continue
                other = tmproot / f.relative_to(outdir)
                check(other.exists() and filecmp.cmp(f, other, shallow=False),
                      f"{name}/{f.relative_to(outdir)} deterministic regeneration")
    else:
        skip(f"{name}: regeneration skipped (--no-regen); sizes + sha256 only")

    # --- float golden recomputed from the loaded bins ----------------------
    experts_packed = {k: t[k + ".bin"] for k in
                      ("experts.gate_up_proj", "experts.gate_up_proj.weight_scale",
                       "experts.down_proj", "experts.down_proj.weight_scale")}
    shared_packed = {k: t[k + ".bin"] for k in
                     ("shared_expert.gate_proj", "shared_expert.gate_proj.weight_scale",
                      "shared_expert.up_proj", "shared_expert.up_proj.weight_scale",
                      "shared_expert.down_proj", "shared_expert.down_proj.weight_scale")}
    m, top_k = man["m"], man["top_k"]
    recomputed = reference_moe_block(
        t["x.bin"], t["router_weight.bin"], t["shared_expert_gate_weight.bin"],
        experts_packed, shared_packed, top_k,
    )
    dtype_by_tensor = {f["tensor"]: f["dtype"] for f in man["files"]}
    for key in ("router_logits", "topk_ids", "topk_weights", "perm_src_token",
                "perm_expert", "expert_token_counts", "x_sorted",
                "routed_output", "shared_output", "moe_output"):
        stored = t[key + ".bin"]
        got_arr = recomputed[key]
        if dtype_by_tensor[key + ".bin"] == "bf16":
            ok = np.array_equal(f32_to_bf16_bits(stored), f32_to_bf16_bits(got_arr))
        else:
            ok = np.array_equal(stored, got_arr)
        check(ok, f"{name}: recomputed '{key}' matches stored bin bit-exactly")
    gate_logits = (t["x.bin"] @ t["shared_expert_gate_weight.bin"].T).astype(np.float32)
    check(np.allclose(t["shared_gate_logits.bin"], gate_logits.reshape(-1), atol=0, rtol=0),
          f"{name}: shared_gate_logits == x @ shared_expert_gate_weight.T (bit-exact)")

    # --- layer-chain goldens ----------------------------------------------
    res1 = t["res1.bin"]
    check(np.array_equal(res1, t["x_res.bin"].astype(np.float32)),
          f"{name}: res1 == f32(x_res) bit-exact (zero residual at m6#1)")
    res1_ref, x_norm1_ref = rmsnorm_ref(t["x_res.bin"], t["gamma1.bin"].reshape(-1))
    check(np.array_equal(f32_to_bf16_bits(t["x_norm1.bin"]), f32_to_bf16_bits(x_norm1_ref)),
          f"{name}: x_norm1 == rmsnorm(x_res, gamma1) bit-exact")
    res2_ref, y_final_ref = rmsnorm_ref(t["moe_output.bin"], t["gamma2.bin"].reshape(-1),
                                        residual=res1)
    check(np.array_equal(t["res2.bin"], res2_ref),
          f"{name}: res2 == f32(moe_output) + res1 bit-exact")
    check(np.array_equal(f32_to_bf16_bits(t["y_final.bin"]), f32_to_bf16_bits(y_final_ref)),
          f"{name}: y_final == rmsnorm(moe_output + res1, gamma2) bit-exact")
    # RMSNorm invariant (independent of the implementation): rms(y / gamma) == 1
    for tag, y, gam in (("x_norm1", t["x_norm1.bin"], t["gamma1.bin"].reshape(-1)),
                        ("y_final", t["y_final.bin"], t["gamma2.bin"].reshape(-1))):
        rms = np.sqrt(np.mean((y / gam[None, :]) ** 2, axis=1))
        check(bool(np.all(np.abs(rms - 1.0) < 5e-3)),
              f"{name}: {tag}/gamma has unit RMS per row (max |rms-1| "
              f"{float(np.abs(rms - 1.0).max()):.2e})")
    check(not np.array_equal(t["x_norm1.bin"], t["x.bin"]),
          f"{name}: x_norm1 != x (slice and chain inputs differ by construction)")

    # --- quant-aware golden ------------------------------------------------
    check_qaware_group(name, outdir, man, t)

    # --- routing sanity ----------------------------------------------------
    check(t["topk_ids.bin"].shape == (m, top_k), f"{name}: topk_ids shape")
    check(np.all(t["topk_ids.bin"] >= 0) and np.all(t["topk_ids.bin"] < man["num_experts"]),
          f"{name}: topk_ids in range")
    wsum = t["topk_weights.bin"].sum(axis=1)
    check(np.allclose(wsum, 1.0, atol=1e-6), f"{name}: topk_weights renormalized to 1")
    check(np.all(np.diff(t["perm_expert.bin"]) >= 0), f"{name}: perm grouped by expert")
    check(int(t["expert_token_counts.bin"].sum()) == m * top_k,
          f"{name}: token counts sum m*top_k")


def check_cross_tier(data_root: Path, groups) -> None:
    print("check 8: cross-tier consistency")
    seen = {}
    for name in groups:
        man_path = data_root / name / "manifest.json"
        if not man_path.exists():
            skip(f"{name}: no manifest, cross-tier check skipped")
            continue
        with open(man_path) as f:
            man = json.load(f)
        seen[name] = {d["tensor"]: d["dtype"] for d in man["files"]}
    ref_name = next(iter(seen), None)
    if ref_name is None:
        skip("no datasets loaded")
        return
    ref_rules = None
    for name, tensors in seen.items():
        check(set(tensors) == set(seen[ref_name]),
              f"{name}: same tensor names as {ref_name}")
        check(tensors == seen[ref_name],
              f"{name}: same dtypes per tensor as {ref_name}")
        with open(data_root / name / "manifest.json") as f:
            rules = sorted(json.load(f)["quantization"]["qaware_golden"].keys())
        if ref_rules is None:
            ref_rules = rules
            guard("floor" in rules, f"{name}: floor (device path) rule emitted "
                                    "(harness invariant)")
        check(rules == ref_rules, f"{name}: same activation rules as {ref_name} ({rules})")


# ---------------------------------------------------------------------------
# 9: optional device cross-check (m13 MoE slice dumps)
# ---------------------------------------------------------------------------


def check_m13_dump(dump_dir: Path, data_root: Path, groups) -> None:
    print("check 9: device cross-check against m13_moe_layer dumps (mode 0)")
    if not dump_dir.exists():
        skip(f"{dump_dir} not found")
        return
    ulp = 2.0 ** -8  # bf16 relative ulp
    for name in groups:
        case_dir = data_root / name
        if not (case_dir / "manifest.json").exists():
            skip(f"{name}: no dataset")
            continue
        if not (dump_dir / f"{name}_mode0_meta.txt").exists():
            skip(f"{name}: no mode0 dump (run M13_DUMP=1 ./m13_moe_layer <data> {name})")
            continue
        ds = load_dataset(case_dir)
        man, t = ds["manifest"], ds["tensors"]
        m, e, top_k = man["m"], man["num_experts"], man["top_k"]
        dsq = qr.load_qaware(case_dir / "qaware" / "floor")
        q = dsq["tensors"]
        counts = t["expert_token_counts.bin"]
        offs = np.concatenate([[0], np.cumsum(counts)])
        total = m * top_k
        mm = 64  # m13 M_MAX (BASE_M) row padding of the per-slot device tensors

        def dump(tag):
            return np.fromfile(dump_dir / f"{name}_mode0_{tag}_device.bin", dtype=np.uint8)

        def dump_bf16(tag):
            return np.fromfile(dump_dir / f"{name}_mode0_{tag}_device.bin", dtype=np.uint16)

        # --- A/H quantizer bytes (per-slot device layout -> compact rows) ---
        for tag, dt, width, comp in (
            ("a_qx", np.uint8, HIDDEN // 2, q["a_qx.bin"]),
            ("a_scale", np.uint8, HIDDEN // GROUP, q["a_scale.bin"]),
            ("h_qx", np.uint8, INTER // 2, q["h_qx.bin"]),
            ("h_scale", np.uint8, INTER // GROUP, q["h_scale.bin"]),
        ):
            dev = dump(tag).reshape(e, mm, -1)[:, :, :width]
            bad = tot = 0
            for ei in range(e):
                tt = int(counts[ei])
                if tt == 0:
                    continue
                sl = slice(int(offs[ei]), int(offs[ei]) + tt)
                bad += int((dev[ei, :tt] != comp[sl]).sum())
                tot += comp[sl].size
            check(bad == 0, f"{name} mode0: {tag} device bytes == floor-rule golden "
                            f"({bad}/{tot} mismatched)")
        for tag, width, comp in (("a_qx_shd", HIDDEN // 2, q["a_qx_shd.bin"]),
                                 ("a_scale_shd", HIDDEN // GROUP, q["a_scale_shd.bin"]),
                                 ("h_qx_shd", INTER // 2, q["h_qx_shd.bin"]),
                                 ("h_scale_shd", INTER // GROUP, q["h_scale_shd.bin"])):
            dev = dump(tag).reshape(mm, -1)[:m, :width]
            check(np.array_equal(dev, comp),
                  f"{name} mode0: {tag} device bytes == floor-rule golden")

        # --- bf16 intermediates: per-slot layout ---------------------------
        for tag, width, comp in (("gu", 2 * INTER, q["gu.bin"]),
                                 ("y_sorted", HIDDEN, q["y_sorted.bin"])):
            dev = dump_bf16(tag).reshape(e, mm, -1)[:, :, :width]
            bad = tot = 0
            for ei in range(e):
                tt = int(counts[ei])
                if tt == 0:
                    continue
                sl = slice(int(offs[ei]), int(offs[ei]) + tt)
                bad += int((dev[ei, :tt] != f32_to_bf16_bits(comp[sl])).sum())
                tot += comp[sl].size
            check(bad == 0, f"{name} mode0: {tag} device bits == q-aware golden "
                            f"({bad}/{tot} mismatched)")
        for tag, width, comp in (("gu_shd", 2 * INTER, q["gu_shd.bin"]),
                                 ("y_shd", HIDDEN, q["y_shd.bin"]),
                                 ("shared_output", HIDDEN, q["shared_output.bin"])):
            dev = dump_bf16(tag).reshape(mm, -1)[:m, :width]
            check(np.array_equal(dev, f32_to_bf16_bits(comp)),
                  f"{name} mode0: {tag} device bits == q-aware golden")

        # --- H_swiglu is stored compactly (perm order) ---------------------
        dev = dump_bf16("h_swiglu")[: total * INTER].reshape(total, INTER)
        check(np.array_equal(dev, f32_to_bf16_bits(q["h_swiglu.bin"])),
              f"{name} mode0: h_swiglu device bits == q-aware golden (compact rows)")
        dev = dump_bf16("h_swiglu_shd").reshape(mm, INTER)[:m]
        check(np.array_equal(dev, f32_to_bf16_bits(q["h_swiglu_shd.bin"])),
              f"{name} mode0: h_swiglu_shd device bits == q-aware golden")

        # --- combine outputs: device folds in f32, the golden in f64 -------
        for tag, comp in (("routed_output", q["routed_output.bin"]),
                          ("moe_output", q["moe_output.bin"])):
            dev = dump_bf16(tag).reshape(mm, HIDDEN)[:m].astype(np.uint16)
            dv = bf16_bits_to_f32(dev).astype(np.float64)
            gv = comp.astype(np.float64)
            ad = np.abs(dv - gv)
            budget = 2.0 * ulp * max(float(np.abs(dv).max()), 1e-30)
            check(float(ad.max()) <= budget,
                  f"{name} mode0: {tag} within 2 bf16 ulp of the golden "
                  f"(max |diff| {float(ad.max()):.3e} <= {budget:.3e})")

        # --- the m13 acceptance criterion is usable with this reference ----
        # |device - float golden| <= 1.5*|qaware - float golden| + 0.02*|float golden| + 5e-3
        for tag in ("routed_output", "shared_output", "moe_output"):
            dev = bf16_bits_to_f32(
                dump_bf16(tag).reshape(mm, HIDDEN)[:m].astype(np.uint16)).astype(np.float64)
            gold = t[f"{tag}.bin"].astype(np.float64)
            qaw = q[f"{tag}.bin"].astype(np.float64)
            tol = 1.5 * np.abs(qaw - gold) + 0.02 * np.abs(gold) + 5e-3
            ratio = float((np.abs(dev - gold) / tol).max())
            bad = int((np.abs(dev - gold) > tol).sum())
            check(bad == 0,
                  f"{name} mode0: {tag} satisfies the m13 layer-chain criterion against the "
                  f"floor-rule golden (violations {bad}/{tol.size}, worst ratio {ratio:.2f})")


# ---------------------------------------------------------------------------
# 10: router fp32 FTZ (hardware subnormal flush)
# ---------------------------------------------------------------------------


def _router_topk_no_ftz(x, w, top_k):
    """The pre-M46 ``router_topk`` body, kept as the regression's negative control.

    Returns ``(topk_ids, topk_weights)``.  If a later change deletes the FTZ modelling
    from ``moe_block_ref.router_topk``, the library call starts matching this replica
    and check 10 fails.
    """
    logits = x @ w.T
    logits = logits - logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    scores = scores / scores.sum(axis=1, keepdims=True)
    order = np.argsort(-scores, axis=1, kind="stable")
    topk_ids = order[:, :top_k].astype(np.int32)
    topk_weights = scores[np.arange(x.shape[0])[:, None], topk_ids]
    return topk_ids, topk_weights / topk_weights.sum(axis=1, keepdims=True)


def _deep_tail_router_case(e: int = 20, m: int = 2):
    """A router input that really lands in the fp32 subnormal band, no real weights.

    ``x[:, 0] = 1`` and ``W[e, 0] = L_e`` (everything else 0) make the logits *exactly*
    the designed per-expert values L_e — a single fp32 product, no accumulation noise,
    so the case is fully controlled.  Each row (max-shifted; experts 0/1 hot, expert 2
    just *above* the flush line, experts 3..E-1 strictly inside the subnormal band and
    *increasing* with the expert id):

      id :  0,   1,   2,     3,     4,   ...,  E-1
      L  :  0,   0,  -85,   -96,  -95.5,  ..., -88
              \\_ exp=1     \\__ exp=1.6e-37  \\___ exp in [6.05e-39, 5.6e-42]

    * ``exp(-85) = 1.6e-37 > 2**-126``: normal, the device keeps it (id 2 is 3rd).
    * ``exp(-88) = 6.05e-39`` down to ``exp(-96) = 5.6e-42``: subnormal, the device
      flushes them all to 0 and then breaks the tie by *lower expert id*.
    """
    hidden = ref.HIDDEN
    x = np.zeros((m, hidden), dtype=np.float32)
    x[:, 0] = 1.0                                   # bf16-exact
    lp = np.zeros((e, hidden), dtype=np.float32)
    lp[0, 0] = lp[1, 0] = np.float32(5.0)           # hot: max-shifted 0
    lp[2, 0] = np.float32(-80.0)                    # max-shifted -85 (normal score)
    for i in range(3, e):                           # max-shifted -96 .. -88, rising
        lp[i, 0] = np.float32(-91.0 + 0.5 * (i - 3))
    return f32_to_bf16(x), f32_to_bf16(lp)


def check_router_ftz() -> None:
    """Check 10: the device fp32 FTZ modelling in ``router_topk``.

    docs/05 §6.1 + user ruling 2026-09-26: "fp32 次正规 FTZ 是硬件模式，不在规避范围" —
    the hardware flushes subnormal fp32 results, so *the device is right and the golden
    was missing a term*.  M42 hit it at real scale (real checkpoint router weight,
    m=4097: 20/4097 rows selected a different top-10 set; with the flush modelled
    0 rows; evidence/router_ftz_device_repro.log).  This check pins that modelling on a
    weight-free, fully designed input, so deleting it fails here immediately.
    """
    print("check 10: router fp32 FTZ (subnormal flush to zero)")
    # --- the helper itself: the boundary is exactly 2**-126, both signs ---------
    tiny = np.nextafter(FTZ_MIN_NORMAL, np.float32(0.0), dtype=np.float32)
    probe = np.array([tiny, -tiny, FTZ_MIN_NORMAL, -FTZ_MIN_NORMAL, np.float32(0.0),
                      np.float32(1e-30)], dtype=np.float32)
    want = np.array([0.0, 0.0, FTZ_MIN_NORMAL, -FTZ_MIN_NORMAL, 0.0, 1e-30],
                    dtype=np.float32)
    check(np.array_equal(ftz_f32(probe), want),
          f"ftz_f32 flushes |v| < 2**-126 (both signs) to +0 and keeps the smallest "
          f"normal and zero (probe {probe.tolist()})")

    # --- a designed input that really reaches the subnormal band ---------------
    e, top_k = 20, 8
    x, w = _deep_tail_router_case(e)
    logits = x @ w.T
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_raw = np.exp(logits).astype(np.float32)
    n_sub = (exp_raw < FTZ_MIN_NORMAL).sum(axis=1)
    check(bool((n_sub == e - 3).all()),
          f"the case really lands in the subnormal band: {e - 3}/{e} experts per row "
          f"have exp(logit) < 2**-126 (reach {int(n_sub.min())})")
    check(bool((exp_raw[:, 2] > FTZ_MIN_NORMAL).all()),
          f"expert 2 sits just *above* the flush line (exp = {float(exp_raw[0, 2]):.3e} "
          f"> 2**-126) — the boundary is pinned from both sides")
    check(bool((exp_raw[:, 3:] > 0.0).all()),
          "the tail scores are non-zero fp32 subnormals (the un-modelled reference "
          "genuinely mis-ranks them, so the case is not vacuous)")

    # --- device semantics = flushed scores, ties to the lower expert id --------
    expected = np.array([[0, 1, 2, 3, 4, 5, 6, 7]] * x.shape[0], dtype=np.int32)
    expected_old = np.array([[0, 1, 2, e - 1, e - 2, e - 3, e - 4, e - 5]] * x.shape[0],
                            dtype=np.int32)
    ids_old, _ = _router_topk_no_ftz(x, w, top_k)
    _, ids_lib, wt_lib = router_topk(x, w, top_k)
    check(np.array_equal(ids_old, expected_old),
          f"the pre-M46 reference ranks the largest subnormal scores instead "
          f"({ids_old[0].tolist()} = the highest expert ids) — it FAILs the device "
          f"semantics")
    check(np.array_equal(ids_lib, expected),
          f"router_topk ids == device semantics (2 hot experts, then the one "
          f"normal-score expert, then the zeroed tail by lower id) = {expected[0].tolist()}")
    check(not np.array_equal(ids_lib, ids_old),
          "REMOVAL GUARD: the modelled ids differ from the pre-M46 replica — deleting "
          "the FTZ modelling makes them identical and fails the check above")
    check(bool((wt_lib[:, 3:] == 0.0).all()) and bool((wt_lib[:, :3] > 0.0).all()),
          "flushed top-k weights are exactly 0 and the surviving ones are > 0")
    rowsum = wt_lib.sum(axis=1)
    check(bool(np.allclose(rowsum, 1.0, atol=1e-6)),
          f"top-k weights still renormalize to 1 (max |sum-1| "
          f"{float(np.abs(rowsum - 1).max()):.2e})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _run_checks(args) -> None:
    check_e2m1()
    check_e8m0()
    check_scale_rules()
    check_mxfp4_roundtrip()
    check_bf16()
    check_hw_quantizer()

    groups = [g for g in args.groups.split(",") if g]
    if args.real and "real" not in groups:
        groups.append("real")
    for name in groups:
        if name not in GROUP_SPECS:
            raise SystemExit(f"unknown group {name!r}; available {list(GROUP_SPECS)}")
        outdir = args.data / name
        if not (outdir / "manifest.json").exists():
            if name == "real" and args.real:
                print(f"check 7: generating '{name}' (E=512, topK=10) — this takes a few "
                      f"minutes and ~{REAL_WEIGHTS_MIB:.0f} MiB")
                generate_dataset(outdir, *GROUP_SPECS[name])
            else:
                skip(f"dataset '{name}' not found under {args.data} "
                     f"(generated on demand: gen_dataset.py --groups {name})")
                continue
        elif name == "real" and not _has_bins(outdir):
            print(f"check 7: '{name}' has a manifest but no bins — regenerating")
            generate_dataset(outdir, *GROUP_SPECS[name])
        check_dataset_group(name, GROUP_SPECS[name], args.data,
                            regenerate=not args.no_regen)
    check_cross_tier(args.data, [g for g in groups if (args.data / g / "manifest.json").exists()])
    if args.m13_dump is not None:
        check_m13_dump(args.m13_dump, args.data, groups)
    check_router_ftz()

    guard(qr.DEFAULT_RULE == "floor",
          "qaware default rule is floor = the official ops-nn sequence (harness invariant)")
    print(f"all selfchecks passed — {COUNTS['check']} 判定项 + {COUNTS['guard']} guard 项")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data",
                    help="dataset root (default: <script_dir>/data)")
    ap.add_argument("--groups", type=str, default=",".join(COMMITTED_GROUPS),
                    help=f"comma-separated dataset groups (available: {','.join(GROUP_SPECS)})")
    ap.add_argument("--real", action="store_true",
                    help=f"also generate + check the real-topology group (E=512, topK=10, "
                         f"~{REAL_WEIGHTS_MIB:.0f} MiB of weights, a few minutes)")
    ap.add_argument("--m13-dump", type=Path, default=None,
                    help="directory with M13_DUMP=1 dumps for the device cross-check")
    ap.add_argument("--no-regen", action="store_true",
                    help="skip the byte-determinism regeneration (sizes/sha256 only)")
    ap.add_argument("--env-ok", action="store_true",
                    help=f"grade the current numeric stack as authoritative: a failing "
                         f"judgement item exits 1 instead of {ENV_MISMATCH_RC}")
    args = ap.parse_args()

    env = report_environment(assume_ok=args.env_ok)

    try:
        _run_checks(args)
    except SelfcheckFailure as exc:
        sys.stdout.flush()
        print(exc.code, file=sys.stderr)
        if env["canonical"]:
            return 1
        print(f"  [ENV] the failure above is environment-ambiguous (rc={ENV_MISMATCH_RC}): "
              f"this stack is not the pinned one ({'; '.join(env['why'])}) and the frozen "
              f"`.bin` comparisons are bit-exact.", file=sys.stderr)
        print(f"  [ENV] re-run with {CANONICAL_ENV['interpreter']}, or pass --env-ok to "
              f"grade this stack as authoritative (then a failure exits 1).", file=sys.stderr)
        return ENV_MISMATCH_RC
    return 0


def _has_bins(outdir: Path) -> bool:
    with open(outdir / "manifest.json") as f:
        man = json.load(f)
    return bool(man["files"]) and (outdir / man["files"][0]["tensor"]).exists()


if __name__ == "__main__":
    sys.exit(main())
