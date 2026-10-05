#!/usr/bin/env python3
"""M17（MoE block 真实权重端到端）的 numpy/double 独立参考链与分段比对。

从**同一份真实 checkpoint**（safetensors，独立于 kernel host 的 manifest+pread 路径）
出发，按真实 MoE block 语义复算整条链，与 `M17_DUMP` 落盘的 device 中间张量**分段抽点**
比对：

    S1  x_norm = RMSNorm(x, gamma1)              （x = 真实 embedding 行，gamma = 真实 hc_norm 切片）
    S2  logits = x_norm @ W_routerᵀ → softmax(max-shift) → top-10 降序 → renorm；sgate = x_norm·g
    S3  计数排序 → perm_src / perm_expert / counts / offsets / inv_slot / w_tk_packed
    S4  x_sorted = x_norm[perm_src]
    S5  A 侧 MXFP4 量化（kernel 采用的 m5 `MxQuant` = 硬件 floor-指数 e8m0 规则）
    S6  grouped gate_up GEMM（MXFP4：A·Wᵀ，W = 真实 checkpoint 的 packed e2m1 + e8m0）
    S7  SwiGLU + MXFP4 量化
    S8  grouped down GEMM
    S9  unpermute 加权折叠 + sigmoid(sgate)·Y_shd → moe
    S10 y_final = RMSNorm(moe, gamma2)（残差 = S1 的 fp32 res1）

**比对口径（`docs/17` §1 分档；权威实现见下方 `tol_bf16` / `tol_l1` / `EPS_*` 与 README §5.3）**：
* 每个分段都报 `n / 越界数(含非有限) / mean / p50 / p90 / p99 / max / 最差占预算`；
  判定 = 越界元素数 0（`NaN/Inf` 一律计入越界）。
* **T1 整数域/位域**：量化器输出字节、perm/counts/inv/w_tk 索引、`x_sorted` gather、
  `res1`/`res2`、e2e 链重算的 `ids`/`counts` 与链上量化字节 → **逐位/逐字节**。
* **T3 长 fp32 链 / 超越函数近似 / mmad 累加**：输出侧 bf16 落盘量与 fp32 GEMV/GEMM 输出 →
  推导界 `tol = nulp·spacing_bf16(ref) + noise`，`noise ∈ {EPS_FAST·|ref|, EPS_·Σ|terms|}`；
  逐元素检查，`docs/17` §1.1 的边界元素分类只作报告。
* **T4 结构性**：非空洞性护栏（`Σt_e == m·topk`，见 `[check_ref][guard]`）、
  换层/换 token 改变 top-10 集合、device 侧 `aclrtMemset(0xCD)` 污染。
* **量化规则的对外 pin 与设备无关见证（M60，`W` 系列，见 README §5.5）**：S5/S7 的 `A_qx`/
  `A_scale`/`H_qx`/`H_scale` 字节判据托在 `quant_hw` 上，因此该镜像的**规则来源**必须钉在
  仓外工件上，并另有一条**不是**本镜像的字节见证。`W1`–`W3` 是规则级自检（定点方向读数、
  角落平价、base 版负向对照），`W4`–`W6` 是 device 侧的交叉见证（`tools/golden::
  quantize_ocp` 逐字节 / argmax 码定点 / 值域往返）与各自的负向对照。四项负向自检在
  判定项里以 `自检-` 前缀单列（计数分开报），任何一条不成立即 FAIL。
* **判定项 / guard / 参考项 / 报告项分栏**：判定项（含 `自检-` 负向对照）计入 PASS 计数；
  `[guard]`、`[参考]`、`[报告]` 一律 print-only。

用法：
    # 1) 在 worktree 根跑 device（落 ws_* 与 meta）
    ./m17_moe_real/build/m17_moe_real m17_moe_real/m17_weight_manifest.txt --m 1 --dumpdir /tmp/m17_dump
    # 2) 独立复算并比对
    /usr/local/python3.12.13/bin/python3.12 m17_moe_real/check_ref.py /tmp/m17_dump layer0_tok1000_m1

    # `自检-W3n` 的 base 版镜像负向对照需要 `git`（`git show <base>:m17_moe_real/check_ref.py`，
    # base 默认 `f0286f6`、可用环境变量 `M17_BASE_REV` 覆盖，见 `_base_quant_hw`）；
    # `自检-W2/W4n/W5n/W6n` 只用本文件里的 `quant_rule_reversed`，不需要任何外部输入。

**退出码（三态，调用方/CI 只看码也能分辨；见 tower 规则「审校脚本必须报 SKIPPED」）**：
    0 = 比过且通过（OK 文案里带实际比较的判定项条数）
    1 = 比过且有差异（FAIL）
    2 = 没得比/输入缺失（dump 目录无 `ws_*.bin`、`ws_meta_<tag>.txt` 或 `<tag>` 不存在）→ 打 SKIPPED
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "weights"))
sys.path.insert(0, os.path.join(ROOT, "tools", "golden"))

from safetensors_reader import ShardReader  # noqa: E402
from moe_block_ref import (  # noqa: E402
    E2M1_POS,
    bf16_bits_to_f32,
    e2m1_decode,
    e8m0_decode,
    f32_to_bf16_bits,
    group_scale_exp,
    quantize_ocp,
)

HIDDEN, INTER, GROUP = 2560, 640, 32
GUN = 2 * INTER          # 1280
E_ALL = 512
MM = 64                  # M_MAX（device 槽位 padding 行数）
TOPK = 10
GU_SCALE_STRIDE = HIDDEN // GROUP      # 80
DN_SCALE_STRIDE = 32                   # 20 有效

MODEL_PREFIX = "model.language_model"

RESULTS: list[tuple[str, bool, str]] = []


# ---------------------------------------------------------------------------
# 判据：分段统计（mean/p50/p90/p99/max + 预算占比），不只报 max
#
# 容差按 docs/17 §1 分档；本 mission 的 T3（长 fp32 链 / 超越函数近似 / mmad-cube 累加）
# 界一律取「输出网格项 + 非舍入项」的显式形式：
#
#     tol = nulp · spacing_bf16(ref)  +  noise
#           └── 网格项（可推导）──┘      └── 非舍入项（逐项 ε 来源见下）──┘
#
# noise 的三种来源（每个调用点必须在注释里写清用了哪一种）：
#   * `EPS_FAST · |ref|`：device 的 fast-math 路径（`Reg::Exp`、Newton-Raphson rsqrt）——
#     相对误差按 **2 个 bf16 网格步长**建模（ε_fast = 2·2^-8 ≈ 7.8e-3）；
#     这是一条**显式的建模假设**（依据：m5/m6 donor 的 Exp / NR-rsqrt 路径以 bf16 域中间量
#     为准，见 README §5.3），需要后续用独立探针标定 Reg::Exp 的实测精度。
#   * `EPS_ACC · Σ|terms|`：fp32 长累加的推导上界，Σ|terms| 逐元素算（不做常量余项）。
#   * 0：整数/位域/逐字节 → 走 check_exact，不在这里。
# ---------------------------------------------------------------------------
def _quantiles(ad: np.ndarray) -> tuple[float, float, float, float, float]:
    f = ad.astype(np.float64).ravel()
    if f.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (float(f.mean()), float(np.percentile(f, 50)), float(np.percentile(f, 90)),
            float(np.percentile(f, 99)), float(f.max()))


def spacing_bf16(x: np.ndarray) -> np.ndarray:
    """bf16 输出网格步长：2^(floor(log2|x|)-8)（尾数 7 位 → [1,2) 上是 2^-8）"""
    e = np.abs(np.asarray(x, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(e > 0, 2.0 ** (np.floor(np.log2(np.where(e > 0, e, 1.0))) - 8.0), 0.0)


EPS_FAST = 2.0 * 2.0 ** -8                 # fast-math（Exp / NR-rsqrt）相对误差模型：2 个 bf16 网格步长
EPS_GEMV = 2.0 * 48 * 2.0 ** -24           # router GEMV：Dot8Row 40 次 chunk 累加 + 64-lane 归约 ~8 级，×2 余量
EPS_FOLD = 2.0 * 11 * 2.0 ** -24           # unpermute 加权折叠：10 项乘加 + 收尾，×2 余量


def eps_cube(k: int, k_base: int = 128) -> float:
    """cube 的 fp32 累加项：K 次累加 + 每 kBlock 归约 ~8 级，×2 余量（mmad 内部顺序未公开）"""
    return 2.0 * (k + (k // k_base + 1) * 8) * 2.0 ** -24


CUR_NOISE: np.ndarray | None = None   # 本段「非舍入噪声」逐元素值（供 §1.1 分类与口径打印）


def tol_bf16(ref: np.ndarray, nulp: float, noise) -> np.ndarray:
    """docs/17 T3 推导界：`tol = nulp·spacing_bf16(ref) + noise`（noise 可为标量或逐元素）"""
    global CUR_NOISE
    refa = np.asarray(ref, dtype=np.float64)
    n = np.broadcast_to(np.asarray(noise, dtype=np.float64), refa.shape).astype(np.float64)
    CUR_NOISE = n
    return float(nulp) * spacing_bf16(refa) + n


def tol_l1(ref: np.ndarray, l1: np.ndarray, eps: float, nulp: float = 0.0) -> np.ndarray:
    """长累加段的推导界：网格项（默认 0：fp32 输出无 bf16 网格）+ ε·Σ|terms|"""
    return tol_bf16(ref, nulp, eps * np.asarray(l1, dtype=np.float64))


def report(tag: str, dev: np.ndarray, ref: np.ndarray, budget: np.ndarray | float,
           kind: str = "abs") -> None:
    """分段统计判据。budget 可为标量或逐元素数组；判定 = 越界元素数 0（**非有限元素一律算越界**）。"""
    global CUR_NOISE
    dev = np.asarray(dev, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert dev.shape == ref.shape, f"{tag}: shape {dev.shape} vs {ref.shape}"
    b = np.broadcast_to(np.asarray(budget, dtype=np.float64), dev.shape)
    # 段内绝对下限：bf16 网格在 0 附近会退化到 0，给「参考恰好为 0」的元素留下零预算。
    # 取段内 max|ref| 的 1e-6（远低于任何有意义的 bf16 步长）——**这条只对 report() 口径生效，
    # 真逐位的判据一律走 check_exact()**
    floor = 1e-6 * float(np.abs(ref).max()) if ref.size else 0.0
    b = np.maximum(b, max(floor, 1e-30))
    ad = np.abs(dev - ref)
    # NaN/Inf：`NaN > b` 为 False，若只看 (ad > b) 会让 NaN 段打印「越界=0 PASS」→ 必须显式计入
    nonfin = ~np.isfinite(dev) | ~np.isfinite(ref)
    nbad = int(nonfin.sum() + np.sum(~nonfin & (ad > b)))
    with np.errstate(invalid="ignore"):
        ratio = float(np.nanmax(np.where(nonfin, np.inf, ad / b)))
    mean, p50, p90, p99, mx = _quantiles(np.where(nonfin, np.inf, ad))
    ok = nbad == 0
    extra = ""
    if CUR_NOISE is not None and ref.size:
        # docs/17 §1.1 的边界元素分类（只作报告）：参考距最近 bf16 网格中点的距离 < δ = noise
        sp = spacing_bf16(ref)
        e = np.abs(ref)
        with np.errstate(divide="ignore", invalid="ignore"):
            mult = np.where(sp > 0, b / np.where(sp > 0, sp, 1.0), 0.0)
        kind = (f"tol = {float(np.median(mult)):.3g}×spacing(x)（中位等效倍数；"
                f"= nulp×spacing + noise，noise 来源见调用点注释与 README §5.3）")
        pos = np.where(sp > 0, e / np.where(sp > 0, sp, 1.0), 0.0)
        d_mid = np.minimum(np.abs(pos - np.round(pos)), 1.0 - np.abs(pos - np.round(pos)))
        nbnd = int(((0.5 - d_mid) * sp < CUR_NOISE).sum())
        if nbnd >= ref.size:
            extra = (f" | δ=noise ≥ spacing/2 ⇒ 全部 {ref.size} 个元素记为「边界元素」"
                     f"（非舍入噪声大于网格步长，分类退化）⇒ 判定用绝对界（网格项 + 非舍入项）")
        else:
            extra = (f" | 常规 {ref.size - nbnd} / 边界 {nbnd}（δ=noise；绝对界已覆盖方向翻转）")
    RESULTS.append((tag, ok,
                    f"n={dev.size} 越界={nbad}（非有限 {int(nonfin.sum())}） | dev-ref| mean={mean:.3e} "
                    f"p50={p50:.3e} p90={p90:.3e} p99={p99:.3e} max={mx:.3e} | 最差占预算={ratio:.3f} "
                    f"({kind}){extra}"))
    CUR_NOISE = None   # 分类标记只对本段有效


def check_exact(tag: str, dev: np.ndarray, ref: np.ndarray, what: str = "元素") -> None:
    dev = np.asarray(dev)
    ref = np.asarray(ref)
    nbad = int((dev != ref).sum())
    RESULTS.append((tag, nbad == 0, f"{nbad}/{dev.size} {what}不符"))


# 兼容别名（旧调用点用）：语义 = `tol_bf16(ref, nulp, EPS_FAST·|ref|)`，即
# 「bf16 网格项 + fast-math 非舍入项」。**只适用于 device 侧经 Exp / NR-rsqrt 的输出段**；
# 长累加段（GEMM / GEMV / 加权折叠）与 fp32 输出段必须直接用 `tol_bf16(..., noise)` /
# `tol_l1(...)` 并写明 noise 来源（逐段对照见 README §5.3）。
def budgets_from_bf16_ulp(ref: np.ndarray, nulp: float) -> np.ndarray:
    return tol_bf16(ref, nulp, EPS_FAST * np.abs(np.asarray(ref, dtype=np.float64)))


# ---------------------------------------------------------------------------
# MXFP4 编解码（与 device/checkpoint 同一规范；见 README「布局转换表」）
# ---------------------------------------------------------------------------
def dequant(packed: np.ndarray, scale: np.ndarray, rows: int, k: int,
            packed_stride: int | None = None, scale_stride: int | None = None) -> np.ndarray:
    """packed [.., k/2] u8（lohi：偶 k 在低半字节）+ e8m0 scale [.., k/32] u8 → float64 [rows, k]"""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    scale = np.asarray(scale, dtype=np.uint8).reshape(-1)
    p_stride = packed_stride if packed_stride is not None else k // 2
    s_stride = scale_stride if scale_stride is not None else k // GROUP
    p = packed.reshape(-1, p_stride)[:rows, : k // 2]
    s = scale.reshape(-1, s_stride)[:rows, : k // GROUP]
    codes = np.empty((rows, k), dtype=np.uint8)
    codes[:, 0::2] = p & 0x0F
    codes[:, 1::2] = p >> 4
    vals = e2m1_decode(codes).astype(np.float64).reshape(rows, k // GROUP, GROUP)
    sc = e8m0_decode(s).astype(np.float64)
    return (vals * sc[:, :, None]).reshape(rows, k)


def quant_hw(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """kernel 激活量化（m5 `MxQuant` 全 VEC 路径）的 numpy 镜像。**规则来源（外部 pin）**：

    * **官方工件**：`add_rms_norm_dynamic_mx_quant_common.h`（ops-nn 源码 `ops-nn/norm/
      add_rms_norm_dynamic_mx_quant/op_kernel/arch35/`，与随包 CANN 的
      `opp/.../ops_nn/ascendc/dynamic_mx_quant/arch35/add_rms_norm_dynamic_mx_quant_common.h`
      同构；**行号只是查阅提示、以符号名/常量名为定位依据**——两份副本的行号并不相同
      （下面括号里给的是 **ops-nn 副本现版本**的行号，已逐行核过；行号随上游版本与本仓注释
      变动。注意 `docs/13-mx-quant-primitives.md` §4 记的是 M28 当时那次阅读的区间，其中
      `MxQuantComputeMaxExpOCP` 那处与本机现版本对不上，已由 M60 的 reviewer 另开 finding，
      不在本 mission scope）：
        - `MxQuantComputeMaxExpOCP`（:308-356）：组内 **指数域** max（`And 0x7F80` ×2 →
          `Max` → `ReduceDataBlock<MAX>`）——对应下面的 `field`（**指数域 max 不是 abs 值 max**
          在 bf16 网格上等价，因为输入已在 bf16 网格且指数域随量值单调）；
        - `MxQuantComputeScaleOCP`（函数自 :358 起；下面引的细粒度行号都在它的**缩放循环体**
          :397-418 内）：`clamp 下界 0x0100`（:402-403）→ `Sub`（:404）
          → `ShiftRights 7`（:405）⇒ e8m0 字节 = `max(exp(amax),2) − 2`；非有限 → `0xFF`
          （:406）；`halfScale = bf16(0x7F00 − shared)`（:412）⇒ **除**（不是乘）`2^(byte−127)`，
          非有限 → `0x7F81`（:413）、`shared == 0` → `0`（:414）；
        - `MxQuantComputeDataFP4`（函数自 :661 起）的 **bf16 分支**（:751-758：`} else {` +
          `// for bf16` → 两条 `Mul` + `Interleave` + 两条 `Cast`）→ `Cast` 到 fp4
          （`castTraitRM<round_mode="round">` = `CAST_ROUND`：平局远离零、饱和到 ±6）；
        - 常量在 :71-98：`expMask 0x7F80` / `emax 0x0100` / `SHR 7` / `BF16_EXP_BIAS 0x7F00` /
          scale NaN `0x00FF` / `NAN_CUSTOMIZATION 0x7F81`。
    * **仓内第二见证**：`tools/golden/moe_block_ref.py::quantize_ocp`（:293-348，逐句带官方行号）
      —— `W3`/`W4` 就是拿它与本镜像逐字节对拍。
    * **device 侧符号（对照 pin，不是规则来源）**：`m17_moe_real/m17_moe_layer.asc` 的
      `MxQuantComputeScale` / `MxQuantComputeDataFP4`（`0x7f81` 的 `Select` 覆盖见 :1631/:1660）。

    **pin 的强度（自评，按 M59 §1.2 第 2 条）**：`quant_hw` 与 `quantize_ocp` 是**同一工作区
    里的两个 agent 对同一份官方头文件的两次转写**（同源转录 / correlated transcription），
    共享同一份读法与同一组常量。因此两者逐字节一致**只能**咬住「转录笔误 + 角落分支实现
    差异」，**咬不住**「对规范的共同误读」（m3 型方向/规则错若发生在两次转写里就仍然自洽）。
    真正的外部权威是官方头文件本身（代码里写出 `文件:符号` 只是可追溯，不等于已验证）；
    `W1`（定点方向读数）与 `W6`（值域往返）就是为这条缺口补的**规则不敏感**判据
    （它们只引用「MXFP4 group32 的数学」；两份转写只并列出现、不拿其一当基准），见 README §5.5。

    **角落分支（与官方 :406/:412-414 逐句对齐；M60 修复）**：此前本镜像缺 `subnormal/极大
    尺度` 与 `非有限` 两条覆盖，实测在合成角落上与官方分叉（±Inf 组给 scale 253 / code 7，
    官方给 255 / code 0；`amax = 2⁻¹²⁶` 组给 code 4，官方给 code 0）—— 真实数据不触发
    （5 份归档 dump 的 122000 个组里 0 个非有限组、0 个指数域 < 2 的组），故 M29 以来一直
    未被既有判据发现；`W3` 现在把这条钉住，并以 base 版镜像作负向对照。

    x: bf16 网格的 float32 [rows, K] → (packed u8 [rows,K/2], scale u8 [rows,K/32])"""
    x = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
    rows, k = x.shape
    assert k % GROUP == 0
    g = x.reshape(rows, k // GROUP, GROUP)
    amax = np.max(np.abs(g), axis=2).astype(np.float32)
    field = ((f32_to_bf16_bits(amax) >> 7) & 0xFF).astype(np.int32)
    nonfinite = field == 0xFF                 # 官方 :406/:413（组内含 ±Inf/NaN）
    degenerate = field < 2                    # 官方 :402-403 + :414（shared 夹到 0 → halfScale = 0）
    byte = np.clip(field - 2, 0, 254).astype(np.uint8)          # 官方 :404-405（指数域受 0..254）
    scale_v = np.exp2((byte.astype(np.int32) - 127).astype(np.float32))   # 只有正常组用得到
    with np.errstate(invalid="ignore", over="ignore"):
        # 非有限输入（NaN / Inf）在这里必然产 NaN：官方序列同样让 `Mul` 出 NaN 再由 `Cast` 收成 0
        # （`quantize_ocp` 用 errstate 静默同一条），故此处显式静默
        h = np.abs(g / scale_v[:, :, None])
    idx = np.clip(np.searchsorted(E2M1_POS * 2.0, h * 2.0, side="left"), 0, 7)
    lo = np.clip(idx - 1, 0, 7)
    d_hi = np.abs(E2M1_POS[idx] - h)
    d_lo = np.abs(h - E2M1_POS[lo])
    code = np.where(d_hi <= d_lo, idx, lo).astype(np.uint8)
    code |= np.where(np.signbit(g), 8, 0).astype(np.uint8)
    # 退化组：halfScale = 0 ⇒ Mul 得 ±0 ⇒ 只剩符号位（官方 :414 / 设备 `Cast(-0)`）
    code = np.where(degenerate[:, :, None], np.where(np.signbit(g), np.uint8(8), np.uint8(0)), code)
    # 非有限组：Mul(±Inf, NaN) = NaN ⇒ Cast 得 0（官方 :406/:413，**含符号位一起变 0**）
    code = np.where(nonfinite[:, :, None], np.uint8(0), code).astype(np.uint8)
    byte = np.where(nonfinite, np.uint8(0xFF), byte).astype(np.uint8)    # 官方 :406（E8M0 NaN）
    code = code.reshape(rows, k)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8), byte


def quant_rule_reversed(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """**负向对照专用**（不是任何判据的参考实现）：把官方 :412/:753 的 `halfScale` 用法
    反成「乘 `2^(byte−127)` 的倒数」—— 即 m3 当年 `QuantRowH` / `M3VfQuantTwoGroups` 的
    那份错读法（`ScaleByteToFloatH(sByte) = (uint32)b << 23 = 2^(byte−127)` 后 `Mul`）。

    `W2`/`W4n`/`W5n`/`W6n` 用它证明「判据咬得住方向错」：如果这四处负向对照不 FAIL，
    说明见证对方向级错误失明（M59 §5 的 m3 形态）。e8m0 字节的算法与 `quant_hw` 相同
    （m3 错的只是乘/除方向，scale 字节本身没错 —— 这正是它几个月全 PASS 的原因）。"""
    x = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
    rows, k = x.shape
    g = x.reshape(rows, k // GROUP, GROUP)
    amax = np.max(np.abs(g), axis=2).astype(np.float32)
    field = ((f32_to_bf16_bits(amax) >> 7) & 0xFF).astype(np.int32)
    byte = np.clip(field - 2, 0, 254).astype(np.uint8)
    scale_v = np.exp2((byte.astype(np.int32) - 127).astype(np.float32))
    h = np.abs(g) * scale_v[:, :, None]                     # ← 方向反了（应为 / scale）
    idx = np.clip(np.searchsorted(E2M1_POS * 2.0, h * 2.0, side="left"), 0, 7)
    lo = np.clip(idx - 1, 0, 7)
    code = np.where(np.abs(E2M1_POS[idx] - h) <= np.abs(h - E2M1_POS[lo]), idx, lo).astype(np.uint8)
    code |= np.where(np.signbit(g), 8, 0).astype(np.uint8)
    code = code.reshape(rows, k)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8), byte


def quant_ceil(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """「golden 权重打包规范」（scale = 2^ceil(log2(amax/6))，平局取偶）——仅作参考项打印，
    不参与判定（kernel 激活侧跟的是硬件 floor-指数规范，见 m17 README「已知限制」）。"""
    x = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
    rows, k = x.shape
    g = x.reshape(rows, k // GROUP, GROUP)
    exps = group_scale_exp(np.max(np.abs(g), axis=2))
    scale_bytes = np.clip(exps + 127, 0, 254).astype(np.uint8)
    scale_f = np.exp2(exps.astype(np.float32))[:, :, None]
    v = np.abs(g / scale_f)
    idx = np.clip(np.searchsorted(E2M1_POS, v, side="left"), 0, 7)
    lo = np.clip(idx - 1, 0, 7)
    d_hi = np.abs(E2M1_POS[idx] - v)
    d_lo = np.abs(v - E2M1_POS[lo])
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & ((idx & 1) == 0))
    code = np.where(pick_hi, idx, lo).astype(np.uint8)
    code |= np.where(np.signbit(g), 8, 0).astype(np.uint8)
    code = code.reshape(rows, k)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8), scale_bytes


# ---------------------------------------------------------------------------
# M60 见证工具：nibble 解包 / base 版镜像（负向对照用）
# ---------------------------------------------------------------------------
def _codes(packed: np.ndarray, k: int) -> np.ndarray:
    """packed u8 [rows, k/2]（lo-hi：偶 k 在低半字节）→ nibble 码 [rows, k]（0..15）"""
    p = np.asarray(packed, dtype=np.uint8)
    c = np.empty((p.shape[0], k), dtype=np.uint8)
    c[:, 0::2] = p & 0x0F
    c[:, 1::2] = p >> 4
    return c


def _base_quant_hw():
    """取出 **M60 修复前**的 `quant_hw`（`git show <base>:m17_moe_real/check_ref.py`，不写盘），
    作为 W3 的负向对照：修复前的镜像缺官方 :406/:413 与 :414 两条覆盖，**必须**与
    `quantize_ocp` 在这些角落上分叉，否则说明角落判据失明。base 版本号可用环境变量
    `M17_BASE_REV` 覆盖（默认 = M29/M60 的 base 提交 `f0286f6`）。"""
    import subprocess
    import types
    rev = os.environ.get("M17_BASE_REV", "f0286f6")
    try:
        src = subprocess.check_output(
            ["git", "-C", ROOT, "show", f"{rev}:m17_moe_real/check_ref.py"],
            stderr=subprocess.DEVNULL).decode()
    except Exception as exc:                                   # noqa: BLE001
        return None, f"git show {rev}:m17_moe_real/check_ref.py 失败（{exc}）"
    mod = types.ModuleType("m17_check_ref_base")
    mod.__dict__.update({"__name__": "m17_check_ref_base", "__file__": os.path.join(HERE, "check_ref.py")})
    exec(compile(src, f"<{rev}:m17_moe_real/check_ref.py>", "exec"), mod.__dict__)   # noqa: S102
    return mod.quant_hw, f"{rev}:m17_moe_real/check_ref.py::quant_hw"


# ---------------------------------------------------------------------------
# device dump 访问（ws_<tag>.bin + ws_meta_<tag>.txt）
# ---------------------------------------------------------------------------
class Dump:
    def __init__(self, dumpdir: str, tag: str):
        meta_path = os.path.join(dumpdir, f"ws_meta_{tag}.txt")
        self.seg: dict[str, tuple[int, int]] = {}
        for line in open(meta_path):
            t = line.split()
            if t and t[0] == "seg":
                kv = dict(x.split("=", 1) for x in t[1:] if "=" in x)
                self.seg[t[1]] = (int(kv["off"]), int(kv["size"]))
        self.raw = np.fromfile(os.path.join(dumpdir, f"ws_{tag}.bin"), dtype=np.uint8)

    def u8(self, name: str) -> np.ndarray:
        off, size = self.seg[name]
        return self.raw[off:off + size]

    def view(self, name: str, dtype: str, count: int) -> np.ndarray:
        off, _ = self.seg[name]
        return np.frombuffer(self.raw, dtype=dtype, count=count, offset=off)


# ---------------------------------------------------------------------------
# 真实权重装载（独立于 kernel host 的 manifest/pread 路径）
# ---------------------------------------------------------------------------
def load_real(model_dir: str, layer: int, token: int, m: int) -> dict:
    rdr = ShardReader(model_dir)
    p = f"{MODEL_PREFIX}.layers.{layer}.mlp."

    def u8(name):
        return np.asarray(rdr.load(name), dtype=np.uint8)

    def bf16(name, sl=None):
        a = rdr.load_f32(name, sl) if sl is not None else rdr.load_f32(name)
        return np.asarray(a, dtype=np.float32)

    w = {
        "router_w": bf16(p + "gate.weight"),
        "sgate_w": bf16(p + "shared_expert_gate.weight").reshape(HIDDEN),
        "gu_packed": u8(p + "experts.gate_up_proj").reshape(E_ALL, GUN, HIDDEN // 2),
        "gu_scale": u8(p + "experts.gate_up_proj.weight_scale").reshape(E_ALL, GUN, GU_SCALE_STRIDE),
        "dn_packed": u8(p + "experts.down_proj").reshape(E_ALL, HIDDEN, INTER // 2),
        "dn_scale": u8(p + "experts.down_proj.weight_scale").reshape(E_ALL, HIDDEN, INTER // GROUP),
        "shd_gate": u8(p + "shared_expert.gate_proj.weight").reshape(INTER, HIDDEN // 2),
        "shd_gate_scale": u8(p + "shared_expert.gate_proj.weight_scale").reshape(INTER, GU_SCALE_STRIDE),
        "shd_up": u8(p + "shared_expert.up_proj.weight").reshape(INTER, HIDDEN // 2),
        "shd_up_scale": u8(p + "shared_expert.up_proj.weight_scale").reshape(INTER, GU_SCALE_STRIDE),
        "shd_down": u8(p + "shared_expert.down_proj.weight").reshape(HIDDEN, INTER // 2),
        "shd_down_scale": u8(p + "shared_expert.down_proj.weight_scale").reshape(HIDDEN, INTER // GROUP),
        "gamma1": bf16(f"{MODEL_PREFIX}.layers.{layer}.mlp_hyper_connection.hc_norm.weight",
                       slice(0, HIDDEN)),
        "gamma2": bf16(f"{MODEL_PREFIX}.layers.{layer}.mlp_hyper_connection.hc_norm.weight",
                       slice(HIDDEN, 2 * HIDDEN)),
        "x": bf16(f"{MODEL_PREFIX}.embed_tokens.weight", slice(token, token + m)).reshape(m, HIDDEN),
    }
    # 注意：`rdr.load()` 返回的是**分片 mmap 上的 view**（不复制，838MB 的 gate_up 全靠它）。
    # 因此 reader 必须活到参考链算完 —— 若在此 close()，后续对 W[...] 的第一次取数就是
    # 已 unmap 的地址（实测：直接段错误）。挂在返回值上把生命周期带到调用方。
    w["_reader"] = rdr
    return w


# ---------------------------------------------------------------------------
# 单 case 的完整参考链
# ---------------------------------------------------------------------------
def rms_norm_rows(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """double 参考：y = x / sqrt(mean(x²)+eps) * gamma"""
    x = x.astype(np.float64)
    ms = np.mean(x * x, axis=1, keepdims=True)
    return x / np.sqrt(ms + eps) * gamma.astype(np.float64)


def bf16_round(x: np.ndarray) -> np.ndarray:
    """RNE 落到 bf16 网格（与 device 的 fixpipe/StoreAlign 落盘一致），返回 float32 值"""
    return bf16_bits_to_f32(f32_to_bf16_bits(np.asarray(x, dtype=np.float32))).astype(np.float64)


def e2e_chain(W: dict, x: np.ndarray, m: int, xnorm_override: np.ndarray | None = None) -> dict:
    """**不打断的端到端参考链**：从真实层输入 x 起步，按 kernel 的段序与数据流一路复算到
    `moe_output` / `y_final` —— 全程**不取任何 device 中间张量当输入**（docs/17 L0 的要求）。

    每个"落盘"点都按 device 的数据流做一次 bf16 RNE（x_norm / GU / H / Y / routed / shared /
    moe / y_final 在 device 上都是 bf16），量化点用与 kernel 同规范的 `quant_hw`。

    `xnorm_override`：传入 device 的 bf16 x_norm 时，得到的是"除 S1 外全链"的变体 —
    两条链的差即 S1 段残差在链上的**实际放大**，用于给端到端判据做误差预算合成
    （|A−dev| ≤ |A−B| + |B−dev|）。
    """
    g1, g2 = W["gamma1"], W["gamma2"]
    # ---- S1 ----
    if xnorm_override is None:
        xnorm = bf16_round(rms_norm_rows(x, g1))
    else:
        xnorm = np.asarray(xnorm_override, dtype=np.float64)
    # ---- S2 router ----
    lg = xnorm @ W["router_w"].astype(np.float64).T
    lg = lg - lg.max(axis=1, keepdims=True)
    e = np.exp(lg)
    ids = np.argsort(-e, axis=1, kind="stable")[:, :TOPK]
    ew = np.take_along_axis(e, ids, axis=1)
    wts = ew / ew.sum(axis=1, keepdims=True)
    sgate = xnorm @ W["sgate_w"].astype(np.float64)
    # ---- S3 计数排序（stable，按 expert 分组）----
    psrc, pexp = [], []
    for ex in range(E_ALL):
        for t in range(m):
            for k in range(TOPK):
                if ids[t, k] == ex:
                    psrc.append(t)
                    pexp.append(ex)
    psrc = np.array(psrc, dtype=np.int64)
    pexp = np.array(pexp, dtype=np.int64)
    counts = np.bincount(ids.ravel(), minlength=E_ALL).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(counts)])
    total = int(counts.sum())
    inv = np.zeros((m, TOPK), dtype=np.int64)
    for pos in range(total):
        inv[int(psrc[pos]), int(np.where(ids[int(psrc[pos])] == pexp[pos])[0][0])] = int(pexp[pos]) * MM + (
            pos - int(off[pexp[pos]]))
    # ---- S4 permute（bf16 逐位搬）----
    xsort = xnorm[psrc]
    # ---- S5 A 侧量化（自己的字节）----
    aq = np.zeros((E_ALL, MM, HIDDEN // 2), dtype=np.uint8)
    asc = np.zeros((E_ALL, MM, GU_SCALE_STRIDE), dtype=np.uint8)
    for ex in range(E_ALL):
        t = int(counts[ex])
        if t == 0:
            continue
        pq, sq = quant_hw(np.ascontiguousarray(xsort[off[ex]:off[ex] + t].astype(np.float32)))
        aq[ex, :t] = pq
        asc[ex, :t, :HIDDEN // GROUP] = sq
    # ---- S6/S7/S8（routed 槽位）----
    gu = np.zeros((E_ALL, MM, GUN))
    h = np.zeros((E_ALL, MM, INTER))
    y = np.zeros((E_ALL, MM, HIDDEN))
    hq = np.zeros((E_ALL, MM, INTER // 2), dtype=np.uint8)
    hs = np.zeros((E_ALL, MM, DN_SCALE_STRIDE), dtype=np.uint8)
    for ex in range(E_ALL):
        t = int(counts[ex])
        if t == 0:
            continue
        adq = dequant(aq[ex], asc[ex], t, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
        wgu = dequant(W["gu_packed"][ex], W["gu_scale"][ex], GUN, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
        g_b = bf16_round(adq @ wgu.T)                       # S6 落 bf16
        gu[ex, :t] = g_b
        gd = g_b[:, :INTER]
        ud = g_b[:, INTER:]
        h_b = bf16_round((gd / (1.0 + np.exp(-gd))) * ud)   # S7 落 bf16
        h[ex, :t] = h_b
        pq, sq = quant_hw(np.ascontiguousarray(h_b.astype(np.float32)))   # S7 量化
        hq[ex, :t] = pq
        hs[ex, :t, :INTER // GROUP] = sq
        hd = dequant(hq[ex], hs[ex], t, INTER, INTER // 2, DN_SCALE_STRIDE)
        wd = dequant(W["dn_packed"][ex], W["dn_scale"][ex], HIDDEN, INTER, INTER // 2, INTER // GROUP)
        y[ex, :t] = bf16_round(hd @ wd.T)                   # S8 落 bf16
    # ---- 共享专家（紧凑布局）----
    pq, sq = quant_hw(np.ascontiguousarray(xnorm.astype(np.float32)))
    aq_s = np.zeros((MM, HIDDEN // 2), dtype=np.uint8)
    as_s = np.zeros((MM, GU_SCALE_STRIDE), dtype=np.uint8)
    aq_s[:m] = pq
    as_s[:m, :HIDDEN // GROUP] = sq
    adq_s = dequant(aq_s, as_s, m, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
    wg = dequant(W["shd_gate"], W["shd_gate_scale"], INTER, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
    wu = dequant(W["shd_up"], W["shd_up_scale"], INTER, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
    gu_s = bf16_round(adq_s @ np.concatenate([wg, wu], axis=0).T)
    gsg = gu_s[:, :INTER]
    usu = gu_s[:, INTER:]
    h_s = bf16_round((gsg / (1.0 + np.exp(-gsg))) * usu)
    pq_s, sq_s = quant_hw(np.ascontiguousarray(h_s.astype(np.float32)))
    hq_s = np.zeros((MM, INTER // 2), dtype=np.uint8)
    hs_s = np.zeros((MM, DN_SCALE_STRIDE), dtype=np.uint8)
    hq_s[:m] = pq_s
    hs_s[:m, :INTER // GROUP] = sq_s
    hd_s = dequant(hq_s, hs_s, m, INTER, INTER // 2, DN_SCALE_STRIDE)
    wd_s = dequant(W["shd_down"], W["shd_down_scale"], HIDDEN, INTER, INTER // 2, INTER // GROUP)
    y_s = bf16_round(hd_s @ wd_s.T)
    # ---- S9 unpermute + combine ----
    y_flat = y.reshape(E_ALL * MM, HIDDEN)
    routed = np.zeros((m, HIDDEN))
    wb = bf16_bits_to_f32(f32_to_bf16_bits(wts.astype(np.float32)))   # device 落盘的 bf16 权重
    for t in range(m):
        acc = np.zeros(HIDDEN)
        for k in range(TOPK):
            acc += float(wb[t, k]) * y_flat[int(inv[t, k])]
        routed[t] = bf16_round(acc)
    gv = 1.0 / (1.0 + np.exp(-sgate))
    gated = gv[:, None] * y_s
    shared = bf16_round(gated)
    moe = bf16_round(routed + gated)          # kernel：Add 用未取整的 gated
    # ---- S10 ----
    yfin = bf16_round(rms_norm_rows(x.astype(np.float64) + moe, g2))
    return {"xnorm": xnorm, "ids": ids, "wts": wts, "aq": aq, "asc": asc, "xsort": xsort,
            "gu": gu, "h": h, "hq": hq, "hs": hs, "y": y, "aq_s": aq_s, "as_s": as_s,
            "gu_s": gu_s, "h_s": h_s, "hq_s": hq_s, "hs_s": hs_s, "y_s": y_s,
            "routed": routed, "shared": shared, "moe": moe, "yfin": yfin,
            "counts": counts, "off": off, "inv": inv, "psrc": psrc, "pexp": pexp}


def main() -> int:
    dumpdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/m17_dump"
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    if not os.path.isdir(dumpdir):
        print(f"[check_ref] RESULT: SKIPPED (dump 目录不存在: {dumpdir}) —— 退出码 2")
        return 2
    if tag is None:
        cands = [f[4:-4] for f in sorted(os.listdir(dumpdir)) if f.startswith("ws_") and f.endswith(".bin")]
        if not cands:
            print(f"[check_ref] RESULT: SKIPPED（{dumpdir} 下没有 ws_*.bin，先用 device 跑一次）"
                  f" —— 退出码 2；未比较任何判据")
            return 2
        tag = cands[0]
    # 输入缺失（meta/bin 不在）→ SKIPPED(2)，不许发合格证
    need = [os.path.join(dumpdir, f"ws_meta_{tag}.txt"), os.path.join(dumpdir, f"ws_{tag}.bin")]
    miss = [p for p in need if not os.path.exists(p)]
    if miss:
        print(f"[check_ref] RESULT: SKIPPED（缺输入 {miss}） —— 退出码 2；未比较任何判据")
        return 2
    print(f"[check_ref] dump={dumpdir} tag={tag}")

    d = Dump(dumpdir, tag)
    # tag = layer{L}_tok{T}_m{M}
    import re
    mm_ = re.fullmatch(r"layer(\d+)_tok(\d+)_m(\d+)", tag)
    if mm_ is None:
        print(f"[check_ref] tag 格式应为 layer{{L}}_tok{{T}}_m{{M}}，得到 {tag}")
        return 1
    layer, token, m = (int(mm_.group(1)), int(mm_.group(2)), int(mm_.group(3)))
    print(f"[check_ref] layer={layer} token={token} m={m}")

    # ---- dump 身份 pin（M60）：证明本轮的 device 字节就是**归档 dump** 的字节 ----
    # M60 的量化规则见证（W4/W5/W6）声明「输入 = 已归档 dump」；这条 pin 把它变成可核对的事实：
    # `evidence/dump_manifest.md` 列了该 tag 的整段 sha256 时逐字节核对（不等 → 判定 FAIL，
    # 因为此时 W4/W5/W6 判的不是归档字节）；未列该 tag（现场新 dump）→ 打 SKIPPED，不计数。
    import hashlib
    pin_path = os.path.join(HERE, "evidence", "dump_manifest.md")
    pin_sha = None
    if os.path.exists(pin_path):
        for line in open(pin_path):
            if f"`ws_{tag}.bin`" in line:
                mt = re.search(r"sha256\s*=\s*`([0-9a-f]{64})`", line)
                if mt:
                    pin_sha = mt.group(1)
    if pin_sha is None:
        print(f"[check_ref][pin] SKIPPED（evidence/dump_manifest.md 未列 tag={tag} 的整段 sha256；"
              f"本轮 dump 不是归档 dump，W4/W5/W6 的「归档字节」声明不适用，该 pin 不计数）")
    else:
        hh = hashlib.sha256()
        with open(os.path.join(dumpdir, f"ws_{tag}.bin"), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                hh.update(chunk)
        got = hh.hexdigest()
        RESULTS.append(("W0 dump 身份 pin：本轮的 ws_*.bin 整段 sha256 == evidence/dump_manifest.md",
                        got == pin_sha, f"{got} vs {pin_sha}"
                        f"{'（= 归档字节）' if got == pin_sha else '（≠ 归档字节！W4/W5/W6 判的不是归档 dump）'}"))

    mf = {}
    for line in open(os.path.join(HERE, "m17_weight_manifest.txt")):
        if line.startswith("model_dir="):
            mf["model_dir"] = line.split("=", 1)[1].strip()
    W = load_real(mf["model_dir"], layer, token, m)
    print(f"[check_ref] 真实权重装载完成（独立 safetensors 路径）：E={E_ALL} 专家，"
          f"gate_up packed {W['gu_packed'].nbytes / 2**30:.3f} GiB")

    x = W["x"]                                   # [m, HIDDEN] bf16→f32
    ids_dev = d.view("topk_ids", np.int32, MM * TOPK).reshape(MM, TOPK)[:m]
    counts_dev = d.view("counts", np.int32, E_ALL)
    off_dev = np.concatenate([[0], np.cumsum(counts_dev)])
    total = int(counts_dev.sum())
    # 非空洞性护栏（docs/17 §4）：counts 全 0 的 dump 会让后面所有 perm/量化类判据比较**空数组**
    # 并真空 PASS。这里显式断言 Σt_e == m·topk 且 perm 数组非空，不满足就记 FAIL 并直接返回。
    if total != m * TOPK:
        RESULTS.append((f"非空洞性：Σt_e == m·topk（m={m} topk={TOPK}）", False,
                        f"Σt_e={total} != {m * TOPK} —— 后续 perm/量化判据会比较空数组，已中止"))
        nbad = sum(1 for _, ok, _ in RESULTS if not ok)
        for tag_, ok, info in RESULTS:
            print(f"[check_ref] {'PASS' if ok else 'FAIL'} {tag_}: {info}")
        print(f"[check_ref] ===== 判定项 {len(RESULTS)} 条，{len(RESULTS) - nbad} PASS / {nbad} FAIL =====")
        return 1
    # 非空洞性护栏按 docs/17 §2.1 属**报告项（guard）**，与判定项分栏打印（不计入 PASS 计数）；
    # 上面 total != m·TOPK 的分支保留为硬中止（那是"判据会比较空数组"的失效条件，不是 guard）。
    # 对照 docs/17 §6 的 m15 口径（"761 判定项 + 109 guard"）。
    print(f"[check_ref][guard] 非空洞性：Σt_e={total} == m·topk={m * TOPK}；perm/counts/inv 数组非空")
    diag = int(d.view("offsets", np.int32, E_ALL + 16)[E_ALL + 8])   # 诊断槽（IG 越界计数）

    # ================= S1: RMSNorm #1 =================
    xnorm_dev = bf16_bits_to_f32(d.view("x_norm", np.uint16, MM * HIDDEN).reshape(MM, HIDDEN)[:m])
    xnorm_ref = rms_norm_rows(x, W["gamma1"])
    # noise = EPS_FAST·|ref|（device 的 NR-rsqrt 走 bf16 域快速路径 ⇒ 见 helper 头注释）
    report("S1 x_norm (RMSNorm#1, gamma=真实 hc_norm)", xnorm_dev, xnorm_ref,
           tol_bf16(xnorm_ref, 1.0, EPS_FAST * np.abs(xnorm_ref)), "fast-math noise")
    # res1 = bf16 层输入（精确可表）→ 真逐位（不用 report()，后者会抬 1e-6·max|ref| 下限）
    res1_dev = d.view("res1", np.float32, MM * HIDDEN).reshape(MM, HIDDEN)[:m]
    check_exact("S1 res1 (残差 = bf16 层输入, fp32 逐位)", res1_dev, x.astype(np.float32), "元素")

    # ================= S2: router =================
    # **逐 op 口径**：router 的段输入取 device 自己的 bf16 x_norm（S1 判据已见证 x_norm 的正确性），
    # 这样 S2 的残差只含 router 本段（GEMV 累加 + softmax/topk），不会把 S1 的 bf16 舍入
    # （相对 2^-9）放大成 router 的"误差"——实测若用 double 段输入，logits 残差会达 1e-2 量级。
    # 端到端（不打断）的那条链见 §e2e 段（`e2e_chain`），它从真实 x 起步。
    # 代价：这几段的**输入**取自 device 上一段落盘值（docs/17 L0「不许拿设备自己的中间输出
    # 当参考」的边界情形）——本 mission 以「段内算术仍独立 double 复算」+「每段输入由上游段
    # 判据见证」+「另有一条不打断的端到端链」三点共同兜住，见 README §5.2。
    xin = xnorm_dev.astype(np.float64)
    logits_dev = d.view("logits", np.float32, MM * E_ALL).reshape(MM, E_ALL)[:m]
    lg_ref = xin @ W["router_w"].astype(np.float64).T
    lg_ref = lg_ref - lg_ref.max(axis=1, keepdims=True)
    # noise = EPS_GEMV·Σ|x_norm·w_e|：GEMV 的 fp32 累加上界（Dot8Row 40 次 chunk 累加 +
    # 64-lane 归约树），逐元素按该元素的 L1 量级缩放（**不按 |logit| 缩放**：相消元素上后者会失效）。
    # logits 是 fp32 输出 ⇒ 网格项 nulp=0，无常量余项。
    l1 = np.abs(xin) @ np.abs(W["router_w"].astype(np.float64)).T
    report("S2 logits (x_norm·Wᵀ, max-shift)", logits_dev, lg_ref,
           tol_l1(lg_ref, l1, EPS_GEMV), "EPS_GEMV·Σ|a·w|")
    e_ref = np.exp(lg_ref)
    ids_ref = np.argsort(-e_ref, axis=1, kind="stable")[:, :TOPK]
    check_exact("S2 topk_ids (降序/并列取小 id)", ids_dev, ids_ref, "元素")
    w_dev = d.view("topk_weights", np.float32, MM * TOPK).reshape(MM, TOPK)[:m]
    ew = np.take_along_axis(e_ref, ids_ref, axis=1)
    w_ref = ew / ew.sum(axis=1, keepdims=True)
    report("S2 topk_weights (renorm)", w_dev, w_ref,
           tol_bf16(w_ref, 1.0, EPS_FAST * np.abs(w_ref)), "fast-math noise")
    sg_dev = d.view("sgate", np.float32, MM)[:m]
    sg_ref = xin @ W["sgate_w"].astype(np.float64)
    l1sg = np.abs(xin) @ np.abs(W["sgate_w"].astype(np.float64))
    report("S2 sgate (共享门裸点积)", sg_dev, sg_ref, tol_l1(sg_ref, l1sg, EPS_GEMV), "EPS_GEMV·Σ|a·w|")

    # ================= S3: 索引生成 =================
    psrc_dev = d.view("perm_src", np.int32, MM * TOPK)[:total]
    pexp_dev = d.view("perm_expert", np.int32, MM * TOPK)[:total]
    psrc_ref, pexp_ref = [], []
    for e in range(E_ALL):
        for t in range(m):
            for k in range(TOPK):
                if ids_ref[t, k] == e:
                    psrc_ref.append(t)
                    pexp_ref.append(e)
    psrc_ref = np.array(psrc_ref, dtype=np.int32)
    pexp_ref = np.array(pexp_ref, dtype=np.int32)
    check_exact("S3 perm_src (计数排序, 稳定)", psrc_dev, psrc_ref, "元素")
    check_exact("S3 perm_expert", pexp_dev, pexp_ref, "元素")
    check_exact("S3 counts", counts_dev, np.bincount(ids_ref.ravel(), minlength=E_ALL).astype(np.int32), "元素")
    inv_dev = d.view("inv_slot", np.int32, MM * TOPK).reshape(MM, TOPK)[:m]
    inv_ref = np.zeros((m, TOPK), dtype=np.int32)
    for pos in range(total):
        t, e = int(psrc_ref[pos]), int(pexp_ref[pos])
        loc = pos - int(off_dev[e])                    # 该专家槽位内的行号
        k = int(np.where(ids_ref[t] == e)[0][0])
        inv_ref[t, k] = e * MM + loc
    check_exact("S3 inv_slot (=e*M_MAX+槽位行号)", inv_dev, inv_ref, "元素")
    wtk_dev = d.view("w_tk_packed", np.int32, MM * 16).reshape(MM, 16)[:m, :TOPK]
    wtk_ref = f32_to_bf16_bits(w_dev.astype(np.float32)).astype(np.int32)
    check_exact("S3 w_tk_packed (低16位=bf16(权重))", wtk_dev & 0xFFFF, wtk_ref, "元素")

    # ================= S4: permute =================
    xsort_dev = d.view("x_sorted", np.uint16, MM * TOPK * HIDDEN).reshape(MM * TOPK, HIDDEN)[:total]
    xnorm_bits = f32_to_bf16_bits(xnorm_dev.astype(np.float32))
    check_exact("S4 x_sorted = gather(x_norm, perm_src)", xsort_dev, xnorm_bits[psrc_ref], "元素")

    # ================= S5: A 侧量化（逐字节） =================
    aq = d.u8("aq").reshape(E_ALL, MM, HIDDEN // 2)
    asc = d.u8("as").reshape(E_ALL, MM, GU_SCALE_STRIDE)
    xs_f = bf16_bits_to_f32(xsort_dev).astype(np.float32)
    nq = ns = 0
    badq = bads = 0
    refstat = {"aq": [0, 0], "as": [0, 0]}
    for e in range(E_ALL):
        t = int(counts_dev[e])
        if t == 0:
            continue
        rows = np.ascontiguousarray(xs_f[off_dev[e]:off_dev[e] + t])
        pq, sq = quant_hw(rows)
        badq += int((pq != aq[e, :t]).sum())
        bads += int((sq != asc[e, :t, :HIDDEN // GROUP]).sum())
        nq += pq.size
        ns += sq.size
        pq_g, sq_g = quant_ceil(rows)
        refstat["aq"][0] += int((aq[e, :t] != pq_g).sum())
        refstat["aq"][1] += aq[e, :t].size
        refstat["as"][0] += int((asc[e, :t, :HIDDEN // GROUP] != sq_g).sum())
        refstat["as"][1] += asc[e, :t, :HIDDEN // GROUP].size
    RESULTS.append(("S5 A_qx 逐字节 vs numpy(MxQuant 硬件规范)", badq == 0, f"{badq}/{nq} 字节不符"))
    RESULTS.append(("S5 A_scale 逐字节 vs numpy(MxQuant 硬件规范)", bads == 0, f"{bads}/{ns} 字节不符"))
    aq_s = d.u8("aq_shd").reshape(MM, HIDDEN // 2)
    as_s = d.u8("as_shd").reshape(MM, GU_SCALE_STRIDE)
    pq, sq = quant_hw(np.ascontiguousarray(xnorm_dev.astype(np.float32)))
    RESULTS.append(("S5 A_shd_qx 逐字节 vs numpy", bool(np.array_equal(pq, aq_s[:m])),
                    f"{int((pq != aq_s[:m]).sum())}/{pq.size}"))
    RESULTS.append(("S5 A_shd_scale 逐字节 vs numpy", bool(np.array_equal(sq, as_s[:m, :HIDDEN // GROUP])),
                    f"{int((sq != as_s[:m, :HIDDEN // GROUP]).sum())}/{sq.size}"))

    # ================= S6: grouped gate_up GEMM =================
    gu_dev = bf16_bits_to_f32(d.view("gu", np.uint16, E_ALL * MM * GUN).reshape(E_ALL, MM, GUN))
    gu_shd_dev = bf16_bits_to_f32(d.view("gu_shd", np.uint16, MM * GUN).reshape(MM, GUN)[:m])
    h_dev = bf16_bits_to_f32(d.view("h_swiglu", np.uint16, MM * TOPK * INTER).reshape(MM * TOPK, INTER)[:total])
    hq = d.u8("hq").reshape(E_ALL, MM, INTER // 2)
    hs = d.u8("hs").reshape(E_ALL, MM, DN_SCALE_STRIDE)
    y_dev = bf16_bits_to_f32(d.view("y", np.uint16, E_ALL * MM * HIDDEN).reshape(E_ALL, MM, HIDDEN))
    y_shd_dev = bf16_bits_to_f32(d.view("y_shd", np.uint16, MM * HIDDEN).reshape(MM, HIDDEN)[:m])
    badhq = badhs = nhq = nhs = 0
    for e in range(E_ALL):
        t = int(counts_dev[e])
        if t == 0:
            continue
        adq = dequant(aq[e], asc[e], t, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
        wgu_e = dequant(W["gu_packed"][e], W["gu_scale"][e], GUN, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
        gu_ref = adq @ wgu_e.T
        # noise = eps_cube(2560)·Σ|a·w|（cube 的 fp32 累加上界，逐元素算 L1，不用相对代理）
        l1gu = np.abs(adq) @ np.abs(wgu_e).T
        report(f"S6 GU slot{e} (t={t})", gu_dev[e, :t], gu_ref,
               tol_bf16(gu_ref, 1.0, eps_cube(HIDDEN) * l1gu), "eps_cube(2560)·Σ|a·w|")
        gd = gu_dev[e, :t, :INTER].astype(np.float64)
        ud = gu_dev[e, :t, INTER:].astype(np.float64)
        h_ref = (gd / (1.0 + np.exp(-gd))) * ud
        report(f"S7 H_swiglu slot{e}", h_dev[off_dev[e]:off_dev[e] + t], h_ref,
               tol_bf16(h_ref, 1.0, EPS_FAST * np.abs(h_ref)), "fast-math noise（输入取 device bf16 GU）")
        rows = np.ascontiguousarray(h_dev[off_dev[e]:off_dev[e] + t].astype(np.float32))
        pq, sq = quant_hw(rows)
        badhq += int((pq != hq[e, :t]).sum())
        badhs += int((sq != hs[e, :t, :INTER // GROUP]).sum())
        nhq += pq.size
        nhs += sq.size
        hd = dequant(hq[e], hs[e], t, INTER, INTER // 2, DN_SCALE_STRIDE)
        wd = dequant(W["dn_packed"][e], W["dn_scale"][e], HIDDEN, INTER, INTER // 2, INTER // GROUP)
        y_ref = hd @ wd.T
        l1y = np.abs(hd) @ np.abs(wd).T
        report(f"S8 Y slot{e}", y_dev[e, :t], y_ref,
               tol_bf16(y_ref, 1.0, eps_cube(INTER) * l1y), "eps_cube(640)·Σ|a·w|")
    RESULTS.append(("S7 H_qx 逐字节 vs numpy(MxQuant 硬件规范)", badhq == 0, f"{badhq}/{nhq} 字节不符"))
    RESULTS.append(("S7 H_scale 逐字节 vs numpy(MxQuant 硬件规范)", badhs == 0, f"{badhs}/{nhs} 字节不符"))

    # 共享专家（紧凑布局，行 = m）
    adq = dequant(aq_s, as_s, m, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
    wg = dequant(W["shd_gate"], W["shd_gate_scale"], INTER, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
    wu = dequant(W["shd_up"], W["shd_up_scale"], INTER, HIDDEN, HIDDEN // 2, GU_SCALE_STRIDE)
    gu_s_ref = adq @ np.concatenate([wg, wu], axis=0).T
    l1gs = np.abs(adq) @ np.abs(np.concatenate([wg, wu], axis=0)).T
    report("S6 GU_shd (真实共享专家)", gu_shd_dev, gu_s_ref,
           tol_bf16(gu_s_ref, 1.0, eps_cube(HIDDEN) * l1gs), "eps_cube(2560)·Σ|a·w|")
    g = gu_shd_dev[:, :INTER].astype(np.float64)
    u = gu_shd_dev[:, INTER:].astype(np.float64)
    h_s_ref = (g / (1.0 + np.exp(-g))) * u
    h_shd_dev = bf16_bits_to_f32(d.view("h_swiglu_shd", np.uint16, MM * INTER).reshape(MM, INTER)[:m])
    report("S7 H_shd", h_shd_dev, h_s_ref,
           tol_bf16(h_s_ref, 1.0, EPS_FAST * np.abs(h_s_ref)), "fast-math noise")
    hq_s = d.u8("hq_shd").reshape(MM, INTER // 2)
    hs_s = d.u8("hs_shd").reshape(MM, DN_SCALE_STRIDE)
    pq, sq = quant_hw(np.ascontiguousarray(h_shd_dev.astype(np.float32)))
    RESULTS.append(("S7 H_shd_qx 逐字节 vs numpy", bool(np.array_equal(pq, hq_s[:m])),
                    f"{int((pq != hq_s[:m]).sum())}/{pq.size}"))
    RESULTS.append(("S7 H_shd_scale 逐字节 vs numpy", bool(np.array_equal(sq, hs_s[:m, :INTER // GROUP])),
                    f"{int((sq != hs_s[:m, :INTER // GROUP]).sum())}/{sq.size}"))
    hd = dequant(hq_s, hs_s, m, INTER, INTER // 2, DN_SCALE_STRIDE)
    wd = dequant(W["shd_down"], W["shd_down_scale"], HIDDEN, INTER, INTER // 2, INTER // GROUP)
    y_shd_ref = hd @ wd.T
    l1ys = np.abs(hd) @ np.abs(wd).T
    report("S8 Y_shd", y_shd_dev, y_shd_ref,
           tol_bf16(y_shd_ref, 1.0, eps_cube(INTER) * l1ys), "eps_cube(640)·Σ|a·w|")

    # ================= S9: unpermute + combine =================
    routed_dev = bf16_bits_to_f32(d.view("routed", np.uint16, MM * HIDDEN).reshape(MM, HIDDEN)[:m])
    shared_dev = bf16_bits_to_f32(d.view("shared", np.uint16, MM * HIDDEN).reshape(MM, HIDDEN)[:m])
    moe_dev = bf16_bits_to_f32(d.view("moe", np.uint16, MM * HIDDEN).reshape(MM, HIDDEN)[:m])
    y_flat = y_dev.reshape(E_ALL * MM, HIDDEN)
    wtk_bits = (wtk_dev & 0xFFFF).astype(np.uint16)
    wr = bf16_bits_to_f32(wtk_bits)
    routed_ref = np.zeros((m, HIDDEN), dtype=np.float64)
    l1r = np.zeros((m, HIDDEN), dtype=np.float64)
    for t in range(m):
        for k in range(TOPK):
            yy = y_flat[int(inv_ref[t, k])]
            routed_ref[t] += float(wr[t, k]) * yy
            l1r[t] += abs(float(wr[t, k])) * np.abs(yy)
    # noise = EPS_FOLD·Σ|w·Y|（10 项乘加 + 收尾的 fp32 累加上界；w/Y 都是精确 bf16 值）
    report("S9a routed_output (加权折叠)", routed_dev, routed_ref,
           tol_bf16(routed_ref, 1.0, EPS_FOLD * l1r), "EPS_FOLD·Σ|w·Y|")
    gv = 1.0 / (1.0 + np.exp(-sg_dev.astype(np.float64)))
    shared_ref = gv[:, None] * y_shd_dev.astype(np.float64)
    report("S9b shared_output (sigmoid(g)·Y_shd)", shared_dev, shared_ref,
           tol_bf16(shared_ref, 1.0, EPS_FAST * np.abs(shared_ref)), "fast-math noise")
    # kernel 的 combine（m13 CombineStage）：shared = bf16(fp32(g)·Y_shd)；moe = bf16(fp32(routed) +
    # **未取整的** fp32(g·Y_shd))（Add 用的是 Mul 的 fp32 结果，不是 shared 的 bf16 值）→
    # 参考按同一序列复算（g 取 device 的 sgate 裸点积，Y_shd 取 device 的 bf16 落盘值）
    gated = gv[:, None] * y_shd_dev.astype(np.float64)
    moe_ref = routed_dev.astype(np.float64) + gated
    # noise：加法本身 ~2·2^-24·Σ|加数|（可忽略），门控的 exp 误差按 EPS_FAST·|gated| 计
    report("S9b moe_output (routed + 未取整 gated shared)", moe_dev, moe_ref,
           tol_bf16(moe_ref, 1.0, 2.0 * 2.0 ** -24 * (np.abs(routed_dev) + np.abs(gated)) +
                    EPS_FAST * np.abs(gated)),
           "2·2^-24·Σ|加数| + EPS_FAST·|gated|")

    # ================= S10: RMSNorm #2 =================
    # 与 kernel 一致：norm 的输入是 **fp32 残差 res2 = res1 + bf16(moe)**，不是 moe 本身
    yfin_dev = bf16_bits_to_f32(d.view("y_final", np.uint16, MM * HIDDEN).reshape(MM, HIDDEN)[:m])
    res2_dev = d.view("res2", np.float32, MM * HIDDEN).reshape(MM, HIDDEN)[:m]
    yfin_ref = rms_norm_rows(res1_dev.astype(np.float64) + moe_dev.astype(np.float64), W["gamma2"])
    report("S10 y_final (RMSNorm#2 over res2, gamma=真实 hc_norm)", yfin_dev, yfin_ref,
           tol_bf16(yfin_ref, 1.0, EPS_FAST * np.abs(yfin_ref)), "fast-math noise")
    # res2 = fp32(res1 + fp32(moe))：device 的加法是 fp32 RNE；参考也必须在 **fp32** 精度上比较
    # （直接把精确 double 和与 device 的 fp32 值比会在「恰好落在两个 fp32 值中点」时假报 1 ulp —— 
    #  实测 m=64 有 1/163840 个元素正是这种平局，device 的 RNE 结果 = 偶数尾数那个值，是正确的）
    res2_ref = np.float32(res1_dev + moe_dev)
    RESULTS.append(("S10 res2 (fp32(res1+bf16(moe)) 逐位)", bool((res2_dev == res2_ref).all()),
                    f"{int((res2_dev != res2_ref).sum())}/{res2_dev.size} 不符"))

    # ================= 端到端（不打断）链：从真实 x 起步，全程不取 device 中间量当输入 =================
    # 定位：§5.2 的逐段判据把每段残差隔离了，但 x→moe 的整链从未在单次独立复算里走过一遍
    # （docs/17 L0 的要求）。这里补上：
    #   A 链 = 从**真实 x** 起步（S1 用 double RMSNorm → 落 bf16）；
    #   B 链 = 同一个 chain 但 S1 的输入换成 device 的 bf16 x_norm（只差 S1 这一段）。
    # 于是 |A − dev| ≤ |A − B| + |B − dev|：第一项是 S1 段残差在链上的**实际放大**
    # （逐元素实测，不是拟合），第二项是链自身的舍入/累加预算（与 §5.2 同口径）。
    # 量化点另做**字节级**比对（我们自己的量化字节 vs device 的字节），把差异源钉死。
    e2eA = e2e_chain(W, x, m)
    e2eB = e2e_chain(W, x, m, xnorm_override=xnorm_dev)
    # 我们链上算出的路由必须与 device 一致（否则整链比对无从谈起）
    check_exact("e2e router ids == device topk_ids（链上重算）", e2eA["ids"], ids_dev, "元素")
    check_exact("e2e counts == device counts", e2eA["counts"].astype(np.int32), counts_dev, "元素")
    bs_bad = 0
    bs_tot = 0
    for ex in range(E_ALL):
        t = int(counts_dev[ex])
        if t == 0:
            continue
        bs_bad += int((e2eA["aq"][ex, :t] != aq[ex, :t]).sum()) + int(
            (e2eA["asc"][ex, :t, :HIDDEN // GROUP] != asc[ex, :t, :HIDDEN // GROUP]).sum())
        bs_tot += e2eA["aq"][ex, :t].size + e2eA["asc"][ex, :t, :HIDDEN // GROUP].size
        bs_bad += int((e2eA["hq"][ex, :t] != hq[ex, :t]).sum()) + int(
            (e2eA["hs"][ex, :t, :INTER // GROUP] != hs[ex, :t, :INTER // GROUP]).sum())
        bs_tot += e2eA["hq"][ex, :t].size + e2eA["hs"][ex, :t, :INTER // GROUP].size
    RESULTS.append(("e2e 链上量化字节 == device 字节（A_qx/A_scale/H_qx/H_scale）", bs_bad == 0,
                    f"{bs_bad}/{bs_tot} 字节不符（A 侧应 0：x_sorted 逐位一致；H 侧差异来自 GU 的舍入）"))
    tolB = tol_bf16(e2eB["moe"], 1.0, EPS_FAST * np.abs(e2eB["moe"]))
    report("e2e moe_output（A 链，输入=真实 x）", e2eA["moe"], moe_dev,
           np.abs(e2eA["moe"] - e2eB["moe"]) + tolB,
           "|A−B|（S1 残差的链上放大）+ |B−dev| 预算")
    tolB2 = tol_bf16(e2eB["yfin"], 1.0, EPS_FAST * np.abs(e2eB["yfin"]))
    report("e2e y_final（A 链，输入=真实 x）", e2eA["yfin"], yfin_dev,
           np.abs(e2eA["yfin"] - e2eB["yfin"]) + tolB2, "|A−B| + |B−dev| 预算")
    print(f"[check_ref][报告] e2e A/B 链差（S1 残差放大）：moe max={np.abs(e2eA['moe'] - e2eB['moe']).max():.3e} "
          f"y_final max={np.abs(e2eA['yfin'] - e2eB['yfin']).max():.3e}")

    # ============ 量化规则的对外 pin 与设备无关见证（M60；README §5.5）============
    # 定位（M59 §1 的机理代号）：S5/S7 的 A/H 量化字节判据托在 `quant_hw` 上，此前只有
    # 「device vs 同一工作区镜像」这一层 ⇒ **S2 残余**（无外部权威 pin：m3 型的规则/方向错
    # 会两侧一起错、逐字节 0 失配而全 PASS）。这一段补三层，每层都配负向对照（`自检-` 前缀，
    # 负向对照不 FAIL 即判 FAIL）：
    #   W1/W3 规则级定点与角落（**设备无关**：只用合成输入 + 两份规则实现，不需要任何 dump）；
    #   W4    设备侧**第二来源**字节见证（`tools/golden::quantize_ocp` 逐字节）；
    #   W5/W6 两条**对规则来源不敏感**的 device 侧判据（方向定点 / 值域往返）——它们只引用
    #         「MXFP4 group32 的数学」，故能咬住「两次转写共同误读」（m3 形态）。
    # 输入 = **已归档 dump**（其整段 sha256 与 `evidence/dump_manifest.md` 逐份核对，见
    # `evidence/m60_quant_rule_witness.log` §0）；本轮**不新跑 device**。
    # 自指形态标注：W4 的输入取 device 的 `x_sorted`/`bf16 GU`/`xnorm` ⇒ **S1（隔离判据，不得
    # 计入「独立端到端」）**；W4 的规则来自 `quantize_ocp` ⇒ **同源转录**（correlated
    # transcription：同一工作区两个 agent 对同一份官方头文件的两次转写，不是两处独立）；
    # W5/W6 吃 device 字节 + host 参考值 ⇒ S1 + 规则不敏感。真 S1-free 的变体见 `[报告] W6b`。
    # 口径之外：本段不引入 device 代码（`m17_moe_layer.asc` / `m17_resources.h` 零改动，见 README §5.5.4），
    # 故「禁 memory-based API」不适用；判定量只有字节相等 / 整数计数 / 推导界下的分位，不含时间量
    # （当场核见 `evidence/m60_quant_rule_witness.log` §5⑥ 打印的「计时调用 0 行」）。
    fp_x = np.zeros((3, GROUP), dtype=np.float32)
    fp_x[:, 0] = (0.5, 1.0, 2.0)
    fp_bytes, fp_code = (124, 125, 126), 6
    w1_lines, w1_ok = [], True
    for fn_name, fn in (("quant_hw", quant_hw), ("quantize_ocp", quantize_ocp)):
        pk, sc = fn(fp_x)
        cd = _codes(pk, GROUP)[:, 0]
        val = tuple(float(E2M1_POS[int(c)] * 2.0 ** (int(b) - 127)) for b, c in zip(sc[:, 0], cd))
        ok = (tuple(int(b) for b in sc[:, 0]) == fp_bytes
              and tuple(int(c) for c in cd) == (fp_code,) * 3 and val == (0.5, 1.0, 2.0))
        w1_ok &= ok
        w1_lines.append(f"{fn_name}: byte={tuple(int(b) for b in sc[:, 0])} "
                        f"code={tuple(int(c) for c in cd)} 反量化={val}")
    RESULTS.append(("W1 量化规则方向定点（amax∈{0.5,1,2} ⇒ scale 字节 124/125/126、code 6、"
                    "反量化回 amax；设备无关、规则不敏感）", w1_ok, " | ".join(w1_lines)))
    pk_r, sc_r = quant_rule_reversed(fp_x)
    cd_r = tuple(int(c) for c in _codes(pk_r, GROUP)[:, 0])
    RESULTS.append((f"自检-W2 负向对照：方向反的 legacy 规则必须在定点上分叉（code ≠ {fp_code}）",
                    all(c != fp_code for c in cd_r),
                    f"方向反（x × 2^(byte−127)）：code={cd_r}，而 scale 字节与正确规则**逐字节相同** "
                    f"= {tuple(int(b) for b in sc_r[:, 0])} ⇒ 只比 scale 字节的 vf-probe 对 data 侧翻转"
                    f"完全无感（M59 §6 第 3 步）"))
    c_inf = np.full((2, GROUP), 1.0, dtype=np.float32)
    c_inf[0, 3] = np.inf
    c_inf[1, 7] = np.nan
    c_sub = np.zeros((2, GROUP), dtype=np.float32)
    c_sub[0, 0] = 2.0 ** -126
    c_sub[1, 0] = -(2.0 ** -130)
    corners = (("±Inf/NaN 组", c_inf), ("次正规/最小正规组", c_sub),
               ("全零组", np.zeros((1, GROUP), dtype=np.float32)))
    w3_lines, w3_ok = [], True
    for cname, cx in corners:
        pk_c, sc_c = quant_hw(cx)
        pk_o, sc_o = quantize_ocp(cx)
        same = bool(np.array_equal(pk_c, pk_o) and np.array_equal(sc_c, sc_o))
        w3_ok &= same
        w3_lines.append(f"{cname}: byte={tuple(int(b) for b in sc_c[:, 0])} "
                        f"code={tuple(tuple(int(x) for x in row[:4]) for row in _codes(pk_c, GROUP))} "
                        f"{'== quantize_ocp' if same else '≠ quantize_ocp'}")
    RESULTS.append(("W3 角落平价：quant_hw == tools/golden::quantize_ocp（±Inf/NaN / 次正规 / 全零，"
                    "逐字节；设备无关）", w3_ok, " | ".join(w3_lines)))
    base_fn, base_desc = _base_quant_hw()
    if base_fn is None:
        RESULTS.append(("自检-W3n 负向对照：M60 修复前的镜像必须在角落上与官方分叉", False,
                        f"负向对照无法构建：{base_desc} ⇒ 本判据不发合格证"
                        f"（tower 规则「无输入可比不得发合格证」）"))
    else:
        w3n_bad = 0
        for _, cx in corners:
            pk_b, sc_b = base_fn(cx)
            pk_o, sc_o = quantize_ocp(cx)
            w3n_bad += int((pk_b != pk_o).sum()) + int((sc_b != sc_o).sum())
        w3n_tot = sum(cx.size // 2 + cx.shape[0] for _, cx in corners)
        RESULTS.append(("自检-W3n 负向对照：M60 修复前的镜像必须在角落上与官方分叉",
                        w3n_bad > 0,
                        f"{base_desc}：与 quantize_ocp 失配 {w3n_bad}/{w3n_tot} 字节"
                        f"（+Inf 组给 scale 253/code 7，官方 255/code 0；`amax=2⁻¹²⁶` 组给 code 4，"
                        f"官方 code 0）⇒ M29 以来既有判据从未覆盖这两个角落（真实数据 0 个这样的组）"))
    w4_ocp = 0
    w4_rev = 0
    w4_tot = 0
    w5_dev = 0
    w5_rev = 0
    w5_n = 0

    def _byte_witness(rows, p_dev, s_dev, K):
        """device 可见输入 rows [t,K] → `quantize_ocp` / 方向反规则，与 device 字节逐字节"""
        nonlocal w4_ocp, w4_rev, w4_tot, w5_dev, w5_rev, w5_n
        r32 = np.ascontiguousarray(rows.astype(np.float32))
        pk_o, sc_o = quantize_ocp(r32)
        pk_r, sc_r = quant_rule_reversed(r32)
        gs = K // GROUP
        w4_ocp += int((pk_o != p_dev[:, :K // 2]).sum()) + int((sc_o != s_dev[:, :gs]).sum())
        w4_rev += int((pk_r != p_dev[:, :K // 2]).sum()) + int((sc_r != s_dev[:, :gs]).sum())
        w4_tot += int(pk_o.size + sc_o.size)
        # W5：组内 |amax| 那个元素在 device 上的码必须是 4.0/6.0（即码 6 或 7）——
        # 方向反的实现会把该元素压到 0/1（小尺度组）或顶到码 7（大尺度组）
        t = r32.shape[0]
        arg = np.argmax(np.abs(r32).reshape(t, gs, GROUP), axis=2)
        cd_d = _codes(p_dev[:, :K // 2], K).reshape(t, gs, GROUP)
        cd_r = _codes(pk_r, K).reshape(t, gs, GROUP)
        am_d = np.take_along_axis(cd_d, arg[:, :, None], axis=2)[:, :, 0] & 0x7
        am_r = np.take_along_axis(cd_r, arg[:, :, None], axis=2)[:, :, 0] & 0x7
        w5_dev += int((~np.isin(am_d, (6, 7))).sum())
        w5_rev += int((~np.isin(am_r, (6, 7))).sum())
        w5_n += int(am_d.size)

    for e in range(E_ALL):
        t = int(counts_dev[e])
        if t == 0:
            continue
        o = int(off_dev[e])
        _byte_witness(xs_f[o:o + t], aq[e, :t], asc[e, :t], HIDDEN)
        _byte_witness(h_dev[o:o + t].astype(np.float32), hq[e, :t], hs[e, :t], INTER)
    _byte_witness(xnorm_dev, aq_s[:m], as_s[:m], HIDDEN)
    _byte_witness(h_shd_dev, hq_s[:m], hs_s[:m], INTER)
    RESULTS.append(("W4 device 量化字节 == tools/golden::quantize_ocp（A/H routed + 共享，含 scale 与 "
                    "nibble，逐字节；S1 输入 + 同源转录规则）", w4_ocp == 0, f"{w4_ocp}/{w4_tot} 字节不符"))
    RESULTS.append(("自检-W4n 负向对照：方向反的 legacy 规则 vs device 字节必须失配",
                    w4_rev > 0,
                    f"{w4_rev}/{w4_tot} 字节不符 = {100.0 * w4_rev / max(w4_tot, 1):.1f}%"
                    f"（M59 在 m3 上的归档同类读数 94.8%、在 m1 golden 上 44.2%）"))
    RESULTS.append(("W5 设备侧方向定点：每个活动组的 |amax| 元素码必须 ∈ {6,7}（= 值 4.0/6.0）",
                    w5_dev == 0, f"违反 {w5_dev}/{w5_n} 组（A/H routed + 共享；分母 = 全部活动组）"))
    RESULTS.append(("自检-W5n 负向对照：方向反的规则在同一统计上必须违反", w5_rev > 0,
                    f"违反 {w5_rev}/{w5_n} 组 ⇒ 该统计对方向翻转是活的"))
    rt = {"d": [], "b": [], "rel": [], "badd": 0, "n": 0}
    rt_rev = {"d": [], "b": [], "rel": [], "badd": 0, "n": 0}
    rt_chain = {"d": [], "b": [], "rel": [], "badd": 0, "n": 0}

    def _roundtrip(acc, packed, scale, h_ref, K, s_stride):
        """值域往返：|dequant(device 字节) − h_ref| vs 预算 2·scale + (1×spacing + EPS_FAST·|h_ref|)"""
        t = int(h_ref.shape[0])
        deq = dequant(packed, scale, t, K, K // 2, s_stride)
        sc = e8m0_decode(np.asarray(scale)[:, :K // GROUP]).astype(np.float64)
        sc = np.broadcast_to(sc[:, :, None], (t, K // GROUP, GROUP)).reshape(t, K)
        b = 2.0 * sc + tol_bf16(h_ref, 1.0, EPS_FAST * np.abs(h_ref))
        d = np.abs(deq - h_ref)
        acc["d"].append(d.ravel())
        acc["b"].append(np.broadcast_to(b, d.shape).ravel())
        nz = h_ref != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            acc["rel"].append((d[nz] / np.abs(h_ref[nz])).ravel())
        acc["badd"] += int((d > b).sum())
        acc["n"] += int(d.size)

    for e in range(E_ALL):
        t = int(counts_dev[e])
        if t == 0:
            continue
        o = int(off_dev[e])
        gd = gu_dev[e, :t, :INTER].astype(np.float64)
        ud = gu_dev[e, :t, INTER:].astype(np.float64)
        h_ref = (gd / (1.0 + np.exp(-gd))) * ud
        _roundtrip(rt, hq[e, :t], hs[e, :t], h_ref, INTER, DN_SCALE_STRIDE)
        if e2eA is not None:
            _roundtrip(rt_chain, hq[e, :t], hs[e, :t], e2eA["h"][e, :t].astype(np.float64),
                       INTER, DN_SCALE_STRIDE)
        pk_rc, sc_rc = quant_rule_reversed(np.ascontiguousarray(h_ref.astype(np.float32)))
        _roundtrip(rt_rev, pk_rc, sc_rc, h_ref, INTER, INTER // GROUP)   # 量化器输出：紧凑行宽 20
    gd_s = gu_shd_dev[:, :INTER].astype(np.float64)
    ud_s = gu_shd_dev[:, INTER:].astype(np.float64)
    h_s_ref = (gd_s / (1.0 + np.exp(-gd_s))) * ud_s
    _roundtrip(rt, hq_s[:m], hs_s[:m], h_s_ref, INTER, DN_SCALE_STRIDE)
    if e2eA is not None:
        _roundtrip(rt_chain, hq_s[:m], hs_s[:m], e2eA["h_s"][:m].astype(np.float64),
                   INTER, DN_SCALE_STRIDE)
    pk_rshd, sc_rshd = quant_rule_reversed(np.ascontiguousarray(h_s_ref.astype(np.float32)))
    _roundtrip(rt_rev, pk_rshd, sc_rshd, h_s_ref, INTER, INTER // GROUP)
    d6 = np.concatenate(rt["d"])
    b6 = np.concatenate(rt["b"])
    r6 = np.concatenate(rt["rel"])
    RESULTS.append(("W6 值域往返：|dequant(device H 字节) − h_ref| ≤ 2·scale + (1×spacing + EPS_FAST·|h_ref|)"
                    "（h_ref = host double SwiGLU，**不由 device 字节反推**）",
                    rt["badd"] == 0,
                    f"n={rt['n']} 越界={rt['badd']} | |Δ| p50={np.percentile(d6, 50):.3e} "
                    f"p99={np.percentile(d6, 99):.3e} max={d6.max():.3e} | 预算 max={b6.max():.3e} "
                    f"最差占预算={float((d6 / b6).max()):.3f} | 相对误差（h_ref ≠ 0，p50/p99/max）="
                    f"{np.percentile(r6, 50):.3e}/{np.percentile(r6, 99):.3e}/{r6.max():.3e}"))
    dr6 = np.concatenate(rt_rev["d"])
    rr6 = np.concatenate(rt_rev["rel"])
    RESULTS.append(("自检-W6n 负向对照：方向反的编码在同一判据上必须越界", rt_rev["badd"] > 0,
                    f"越界={rt_rev['badd']}/{rt_rev['n']} | 相对误差 p50/max="
                    f"{np.percentile(rr6, 50):.3e}/{rr6.max():.3e}"
                    f"（同口径下 M59 在 m3 上的归档读数 = p50 142 / max 5.79e76；本数据组的尺度更小，"
                    f"方向反后码被压到 0，相对误差饱和在 1.0 —— 这也是本判据用**绝对界 + 预算比**"
                    f"而不是相对误差作判定的原因）"))
    dc6 = np.concatenate(rt_chain["d"])
    bc6 = np.concatenate(rt_chain["b"])
    rc6 = np.concatenate(rt_chain["rel"])
    print(f"[check_ref][报告] W6b 值域往返的 **S1-free 变体**：h_ref 取 §5.4 的 A 链（输入 = 真实 x，"
          f"全程不取 device 中间量）⇒ n={rt_chain['n']} | |Δ| p50={np.percentile(dc6, 50):.3e} "
          f"p99={np.percentile(dc6, 99):.3e} max={dc6.max():.3e} | 同预算下最差占预算="
          f"{float((dc6 / bc6).max()):.3f} 越界={rt_chain['badd']}/{rt_chain['n']} | 相对误差 p50/max="
          f"{np.percentile(rc6, 50):.3e}/{rc6.max():.3e}")
    print(f"[check_ref][报告] W6/W4/W5 的自指形态：W4/W5/W6 的**输入**都含 device 中间量"
          f"（`x_sorted` / `bf16 GU` / `xnorm`）⇒ 按 M59 §1 记为 **S1（隔离判据）**，"
          f"**不得计入「独立端到端」**；W4 的规则来自 `quantize_ocp` ⇒ **同源转录**（非两处独立）。"
          f"规则不敏感的那一半是 W1/W5/W6 的「方向 + 值域」口径 —— 它们只引用 MXFP4 group32 的数学"
          f"（量化实现只以「两份转写并列」的形式出现，不拿其一当基准）。")

    # ================= 参考项（非判定） =================
    for kk, name in (("aq", "A_qx(routed)"), ("as", "A_scale(routed)")):
        bad, tot = refstat[kk]
        print(f"[check_ref][参考] {name}: device 字节 vs「golden 权重 ceil 规范」量化器 "
              f"{bad}/{tot} 不同 = {100.0 * bad / max(tot, 1):.1f}%")
    print(f"[check_ref][参考] IG 诊断槽(offsets[E+8]) = {diag}（0 = 无越界 ids）")

    nbad = 0
    print()
    for tag_, ok, info in RESULTS:
        if not ok:
            nbad += 1
        print(f"[check_ref] {'PASS' if ok else 'FAIL'} {tag_}: {info}")
    njudged = len(RESULTS)
    n_e2e = sum(1 for t_, _, _ in RESULTS if t_.startswith("e2e"))
    n_self = sum(1 for t_, _, _ in RESULTS if t_.startswith("自检-"))
    # 量化规则见证（M60）= `W*` 判定项 + 它们的 `自检-*` 负向对照（两者都计入判定项总数）
    n_wit = sum(1 for t_, _, _ in RESULTS
                if t_.startswith(("W0", "W1", "W3", "W4", "W5", "W6")) or t_.startswith("自检-"))
    print(f"[check_ref] ===== 判定项 {njudged} 条，{njudged - nbad} PASS / {nbad} FAIL "
          f"（分段 {njudged - n_e2e - n_wit} + e2e {n_e2e} + 量化规则见证 {n_wit}，"
          f"其中负向自检 {n_self}）=====")
    # ---- 覆盖范围交代（tower 规则「统计/校验工具必须交代自己真正的覆盖范围」）----
    # 计数与匹配器同源：都来自本次运行的同一份 counts_dev/循环，不另写数字。
    n_active = int((counts_dev > 0).sum())
    n_skipped = int(E_ALL - n_active)
    n_rows = int(counts_dev.sum())
    print(f"[check_ref][coverage] 本次实际比较：判定项 {njudged}（分段 {njudged - n_e2e - n_wit} + "
          f"e2e {n_e2e} + 量化规则见证 {n_wit}，其中负向自检 {n_self}）；guard 1（单列）")
    print(f"[check_ref][coverage]   全量比较：x_norm/res1/logits(全部 {E_ALL} 个专家)/topk_ids/topk_weights/"
          f"perm_src/perm_expert/inv_slot/w_tk_packed/sgate/x_sorted(前 {n_rows} 行)/"
          f"routed/shared/moe/y_final/res2（m={m} 行全部元素）")
    print(f"[check_ref][coverage]   部分比较：A/H 量化字节与 GU/h_swiglu/Y/共享四段只比**非空槽位的前 t_e 行**"
          f"（活动槽 {n_active}/{E_ALL}，跳过 {n_skipped} 个空槽）")
    print(f"[check_ref][coverage]   不在本脚本范围（各由 host 不变量 / guard / device 日志见证）："
          f"counts 与 offsets 的前缀和一致性、t_e∈[0,64]、ids 范围与行内唯一、perm 分组有序、"
          f"inv_slot 自洽、w_tk 低 16 位、全体有限、topk 行和≈1、offsets[E+1] 之后的诊断槽")
    print(f"[check_ref][coverage]   量化规则见证（M60）的三态分栏："
          f"**W1/W2/W3/W3n 设备无关**（只吃合成输入，不读 dump；可与装置无关地复跑）；"
          f"**W0/W4/W4n/W5/W5n/W6/W6n 吃已归档 dump**（`W0` 读整段 `ws_*.bin` 的 sha256，"
          f"其余读段内字节；身份与 `evidence/dump_manifest.md` 逐份核对），"
          f"覆盖 = 全部 {n_active} 个活动槽（A/H routed + 共享）的**前 t_e 行**，"
          f"即 {w4_tot} 个 A/H 量化字节与 {w5_n} 个活动组 —— 计数与匹配器同源（同一循环，不另写数字）；"
          f"**不在覆盖内**：空槽（{n_skipped} 个）与各行 padding、scale 行距尾部字节（"
          f"`hs[:, :, {INTER // GROUP}:]` 等 device 不写或不用的槽位）、`W6` 的 h_ref 自身误差"
          f"（由 S7 判据见证，本判据只把它的界并入预算）")
    print(f"[check_ref][coverage]   ⇒ **本脚本的 OK 不等于「整个 workspace 都比过了」**："
          f"空槽位/各段 padding 行/上述 host 侧项不在**内容**判据的比较范围"
          f"（负向对照见 README §5.3.1：**在归档 dump 上篡改空槽位或行距尾部的字节 ⇒ 内容判据仍全 OK，"
          f"但 `W0` 会因整段 sha 变化而 FAIL** —— 此时要按「**除 W0 外全 PASS**」读；"
          f"内容是覆盖边界、不是漏判。历史口径：M29 时这条篡改只报「内容判据 OK」，M60 新增的 "
          f"`W0`（整段身份 pin）把它变成「内容 OK + 身份 FAIL」，逐条对照见 README §5.3.1 的"
          f"「M29 → M60 的一处行为变化」注）")
    # 三态退出码（tower 规则「审校脚本必须报 SKIPPED」）：0 = 比过且通过 / 1 = 比过有差异 /
    # 2 = 没得比。OK 文案必须带**实际比较计数**，且计数为 0 时不得报 OK。
    if njudged == 0:
        print("[check_ref] RESULT: SKIPPED (0 条判定项被比较) —— 退出码 2")
        return 2
    if nbad == 0:
        print(f"[check_ref] RESULT: OK ({njudged} 条判定项全部比较通过：分段判据 "
              f"{njudged - n_e2e - n_wit} + e2e 链 {n_e2e} + 量化规则见证 {n_wit}（含负向自检 {n_self}）；"
              f"另有 1 条非空洞性 guard 单列、未计入)")
        return 0
    print(f"[check_ref] RESULT: FAIL ({nbad}/{njudged} 条判定项越界) —— 退出码 1")
    return 1


if __name__ == "__main__":
    sys.exit(main())
