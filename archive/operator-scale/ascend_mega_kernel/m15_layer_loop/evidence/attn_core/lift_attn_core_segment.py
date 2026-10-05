#!/usr/bin/env python3
"""M101：从只读 donor `m10_attn_decode/m10_attn_decode.asc` **机械抽取** attention 核心 device 段，
生成 `m15_layer_loop/m15_attn_core.h`。

为什么用脚本而不是手工复制（先例：`m15_layer_loop/lift_moe_segment.py`、
`evidence/ple_wire/lift_ple_device_segment.py`）：把抽取规则写成可执行文件后，`--check` 能证明
「`m15_attn_core.h` 的 device 代码**除了下表列的 8 类机械替换 + 第 9 类整块替换之外**与 donor 逐字相同」，
差异表因此可复核而不是靠人眼。

抽取范围（**内容锚点**，不钉行号）：
    起点 = donor 首个顶格 `namespace m10 {`（device 段起点；其后是 host 侧 `main()` 与
           数据集装载，不抽取 —— 融合后 attention 核心不再自成 kernel，入口由调用方给）
    终点 = donor 里顶格 `}  // namespace m10`（含）
⇒ 抽出来的是 donor 的**全部 device 侧代码**：形状常量 / UB+L1+L0 静态布局 / BufferID /
CrossCore flagId / 三个装载与两个 FIXP 原语 / mmad 封装 / 6 个 `__simd_vf__` / `M10Aic` / `M10Aiv`。

保留逐字不变的替换（**全部是机械替换，无算法改动**；每一类的命中次数都由生成器断言）：
  1. `namespace m10 {`                 → `namespace M15AC {`
     原因：与 GDN/hc/MoE/AP 段的常量空间分离；融合 TU 里那些段各自有具名 namespace。
  2. `}  // namespace m10`             → `}  // namespace M15AC`
  3. `M10Aic`                          → `AttnCoreAic`（类名；`M10` 前缀在 M15 里没有意义）
  4. `M10Aiv`                          → `AttnCoreAiv`
  5. `M10_DBG_STAGE0`                  → `M15AC_DBG_STAGE0`（调试开关，默认不开）
  6. `M10_DEBUG_SKIP_AIC_FD`           → `M15AC_DEBUG_SKIP_AIC_FD`（同上）
  7. `M10_DEBUG_SKIP_COMBINE`          → `M15AC_DEBUG_SKIP_COMBINE`（同上）
  8. printf 标签 `[M10K]`               → `[M15AC]`（**纯文案**：不改任何数值/判据/同步）
  9. **整块替换（不是重命名式）**：FD combine 的归并内循环里两处**经典 memory-based 向量 API**
     （`Muls(tmpR, accR, w, S2T)` / `Add(numR, numR, tmpR, S2T)`，donor `m10_attn_decode.asc:977-978` 同款）
     换成 RegBase VF（`LoadAlign + Muls + Add + StoreAlign`，在 `__VEC_SCOPE__` 内）。
     依据 = 人类逐字规则「…不应该使用 memory base 的 API」；塔 2026-10-04 裁决方案 B。
     ⇒ **抽取物不再是"逐字 + 纯重命名"**：对这两行而言"与 donor 逐行相同"**不成立**，
     其等价性由「与 `m10_attn_decode/data` 归档 dump 逐字节对拍 + donor `check_ref.py` 三档」见证
     （先例：`lift_moe_segment.py` 的第 8 类对 m13 的 router 整段重写）。

生成物形态（塔的硬约束）：**include guard + 只含 inline body + 无 `main()` + 无 `#include`**。
donor 的 device 段里没有 `#include`；只有 3 对 `#if/#else/#endif` 调试开关（生成器会断言
"没有 `#include`，且 `#if` 与 `#endif` 计数相等"）。

用法：
  python3 m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py           # 生成
  python3 m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py --check   # 校验（rc=1 表示漂移）
"""
import argparse
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M15 = os.path.abspath(os.path.join(HERE, "..", ".."))          # m15_layer_loop/
REPO = os.path.abspath(os.path.join(M15, ".."))                # 仓库根
SRC = os.path.join(REPO, "m10_attn_decode", "m10_attn_decode.asc")
DST = os.path.join(M15, "m15_attn_core.h")

BEGIN = "namespace m10 {"
END = "}  // namespace m10"

# 机械替换表： (旧, 新, 期望命中次数) —— 次数写死是为了"上游一改就红"，不是为了好看
REPLACEMENTS = [
    ("namespace m10 {", "namespace M15AC {", 1),
    ("}  // namespace m10", "}  // namespace M15AC", 1),
    ("M10Aic", "AttnCoreAic", 2),
    ("M10Aiv", "AttnCoreAiv", 2),
    ("M10_DBG_STAGE0", "M15AC_DBG_STAGE0", 2),
    ("M10_DEBUG_SKIP_AIC_FD", "M15AC_DEBUG_SKIP_AIC_FD", 1),
    ("M10_DEBUG_SKIP_COMBINE", "M15AC_DEBUG_SKIP_COMBINE", 1),
    ("[M10K]", "[M15AC]", 17),
]


def _extract():
    """返回 (donor 全文, 抽取片段正文, 起止行号(1-based，含端点), donor sha256)。"""
    raw = open(SRC, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode()
    i = text.index("\n" + BEGIN) + 1              # 顶格 `namespace m10 {` 的行首
    j = text.index("\n" + END + "\n") + 1 + len(END)
    body = text[i:j]
    line0 = text[:i].count("\n") + 1
    line1 = text[:j].count("\n") + 1 - (0 if text[j - 1] == "\n" else 0)
    if not body.startswith(BEGIN) or not body.endswith(END):
        raise SystemExit("[FAIL] 内容锚点抽取结果不合预期")
    if "#include" in body:
        raise SystemExit("[FAIL] 抽取片段里出现了 `#include` —— 生成物就不能只含 inline body 了")
    if body.count("#if") != body.count("#endif"):
        raise SystemExit("[FAIL] 抽取片段里的条件编译不成对")
    return text, body, (line0, text.count("\n", 0, j) + 1), sha


def _replace(body):
    for old, new, want in REPLACEMENTS:
        got = body.count(old)
        if got != want:
            raise SystemExit(f"[FAIL] 机械替换 {old!r} -> {new!r}：期望 {want} 处，实得 {got} 处"
                             f"（donor 变了 ⇒ 必须重核替换表）")
        body = body.replace(old, new)
    return body


# ============================================================
# 第 9 类：**整块替换**（不是重命名式机械替换）
# ============================================================
# 依据（人类逐字规则，塔 2026-10-04 裁决方案 B）：
#   「……都应该都用 VEC_SCOPE aka simd vector function，不应该使用 memory base 的 API」
# 被替换的原文（donor `m10_attn_decode.asc:977-978` 与本抽取件的 r1 逐字相同，位于
# `AttnCoreAiv::Combine()` 的 FD 归并内循环）：
#       Muls(tmpR, accR, w, S2T);
#       Add(numR, numR, tmpR, S2T);
#   —— 这两行是经典 memory-based 向量 API（`LocalTensor` + 元素个数），且落在**数值路径**上
#      （加权 + 跨 split 累加），不属于"搬运/常量填充"豁免面。
#
# 为什么可以期望"数值逐位不变"：官方 Reg 矢量精度规格表
#   （`asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md`）
#   里 `Mul` / `Add` / `Muls` 都是 **0 ulp**（同一张表：Exp 1ulp、Div 1ulp）⇒ 换成 RegBase 的
#   `Muls`+`Add` 不引入新误差；且**累加次序一字未动**（`nsplit` 外层循环、逐 i 先乘后加）。
#   **这一步的等价性不能靠论证，只能靠见证** ⇒ 用"与 `m10_attn_decode/data` 归档 dump 逐字节对拍"
#   + donor `check_ref.py` 三档（见 evidence/attn_core/README.md §3.2/§3.4）。
#   **若对拍不成立（任何字节不同），按塔的要求立刻停手报塔**，不许改判据、不许把差异解释掉。
#
# 替换后：`Muls`/`Add` 走 `__VEC_SCOPE__` 内的 `LoadAlign + Muls + Add + StoreAlign`
#   （形态与本段原有的 `AccUpdateTileVf`、以及 `m15_attn_oproj.h` 的 `GateMulVf` 同款）。
#   ⇒ **抽取物不再是"逐字 + 纯重命名"**：本文件是"8 类机械替换 + 1 类整块替换"。
VF_ACC_FMA = '''// M101 r2 · **第 9 类整块替换**新增：FD combine 的归并累加走 RegBase VF。
//   替换前（donor / 本抽取件 r1 逐字形态）：`Muls(tmpR, accR, w, S2T); Add(numR, numR, tmpR, S2T);`
//   —— 经典 memory-based API，人类逐字禁：「…不应该使用 memory base 的 API」。
//   替换后：LoadAlign + Muls + Add + StoreAlign（全在 __VEC_SCOPE__ 内），与 `AccUpdateTileVf`
//   同款。**累加次序（nsplit 循环、逐 i 先乘后加）一字未动**；官方 Reg 规格里 Mul/Add/Muls 都是
//   0 ulp ⇒ 预期与替换前逐位相同，由"与 m10_attn_decode/data 归档 dump 逐字节对拍"见证。
__simd_vf__ inline void AccFmaRowVf(__ubuf__ float* numUb, __ubuf__ float* accUb, float w)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> acc, num;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t g = 0; g < (uint16_t)ROW_VL; ++g) {
            LoadAlign(acc, accUb + g * VL_F32);
            LoadAlign(num, numUb + g * VL_F32);
            Muls(acc, acc, w, fullM);
            Add(num, num, acc, fullM);
            StoreAlign(numUb + g * VL_F32, num, fullM);
        }
    }
}

'''

OLD9_A = ("// ============================================================\n"
          "// AIV\n"
          "// ============================================================\n"
          "class AttnCoreAiv {")
NEW9_A = VF_ACC_FMA + OLD9_A

OLD9_B = ("                Muls(tmpR, accR, w, S2T);\n"
          "                Add(numR, numR, tmpR, S2T);\n")
NEW9_B = ("                // [第 9 类整块替换] 原为经典 memory-based 的\n"
          "                //   `Muls(tmpR, accR, w, S2T); Add(numR, numR, tmpR, S2T);`\n"
          "                // 依人类逐字规则（「…不应该使用 memory base 的 API」）改走 RegBase VF；\n"
          "                // nsplit 累加次序未动。tmpR（UB_TMPR 槽）在替换后不再被引用 —— 声明保留，\n"
          "                // 以免动段内那张编译期静态 UB 地址表（不占运行期资源）。\n"
          "                (void)tmpR;\n"
          "                AccFmaRowVf(reinterpret_cast<__ubuf__ float*>(numR.GetPhyAddr()),\n"
          "                            reinterpret_cast<__ubuf__ float*>(accR.GetPhyAddr()), w);\n")

# 第 9 类只做**整块替换**：每一处都要求"在 donor 里出现且只出现一次"
BLOCK_REPLACEMENTS = [
    (OLD9_A, NEW9_A, 1, "插入 RegBase VF 归并函数（AIV 段 banner 之前）"),
    (OLD9_B, NEW9_B, 1, "把 FD 归并的 2 处经典 memory-based 调用换成 VF 调用"),
]


def _block_replace(body):
    for old, new, want, what in BLOCK_REPLACEMENTS:
        got = body.count(old)
        if got != want:
            raise SystemExit(f"[FAIL] 第 9 类整块替换（{what}）：期望 {want} 处，实得 {got} 处"
                             f"（donor 或上游抽取形态变了 ⇒ 必须重核）")
        body = body.replace(old, new)
    return body


def _cc_table(body):
    """从抽出的片段里**实测** CrossCore 常量与调用分布（清单因此不可能与代码漂移）。"""
    consts = re.findall(r"constexpr uint16_t (CC_\w+)\s*=\s*(\d+);", body)
    calls = re.findall(r"CrossCore(Set|Wait)Flag<0x([0-9A-Fa-f]+),\s*(PIPE_\w+)>\(([^)]*)\)", body)
    per = {}
    for kind, mode, pipe, arg in calls:
        key = (kind, "0x" + mode, pipe)
        per[key] = per.get(key, 0) + 1
    return consts, len(calls), per


def _mutex_table(body):
    consts = re.findall(r"constexpr MutexID (B_\w+)\s*=\s*(\d+);", body)
    acq = re.findall(r"Acq<PIPE_(\w+)>\((B_\w+)\)", body)
    rls = re.findall(r"Rls<PIPE_(\w+)>\((B_\w+)\)", body)
    per = {}
    for kind, pairs in (("Acq", acq), ("Rls", rls)):
        for pipe, buf in pairs:
            key = (kind, pipe, buf)
            per[key] = per.get(key, 0) + 1
    return consts, per


def _head(body, lines, sha):
    cc_consts, cc_n, cc_per = _cc_table(body)
    mx_consts, mx_per = _mutex_table(body)
    cc_rows = "\n".join(f"//   · {n:<11} = {v}" for n, v in cc_consts)
    mx_rows = "\n".join(f"//   · {n:<7} = {v}" for n, v in mx_consts)
    cc_calls = "\n".join(f"//   · {k[0]:<4} mode {k[1]:<3} {k[2]:<9} × {v}"
                         for k, v in sorted(cc_per.items()))
    mx_calls = "\n".join(f"//   · {k[0]:<3} {k[1]:<9} {k[2]:<7} × {v}"
                         for k, v in sorted(mx_per.items()))
    n_vec = body.count("__VEC_SCOPE__")
    n_dup = len(re.findall(r"(?<![\w:])Duplicate\(", body))
    return f'''// ============================================================
// m15_attn_core.h —— **机械生成物，勿手改**（M101）
// ============================================================
// 由 `m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py` 从**只读 donor**
// `m10_attn_decode/m10_attn_decode.asc` 的 device 段抽出，做下表列的 8 类机械替换 **+ 第 9 类整块替换**。
// `--check` 重新抽取并与本文件逐字节比 ⇒ donor 一改，`--check` 立刻红（不存在静默漂移）。
//
//   · 内容锚点：donor asc 行 {lines[0]} .. {lines[1]}（顶格 `namespace m10 {{` .. 顶格 `}}  // namespace m10`）
//   · donor 文件 sha256 = {sha}
//   · 抽取形态：**include guard + 只含 inline body + 无 `main()` + 无 `#include`**
//     （donor 的 device 段本来就没有 `#include`；3 对 `#if/#else/#endif` 是调试开关）
//
// ---- 8 类机械替换（生成器逐类计数断言；重命名式，无算法改动）----
//   1. `namespace m10 {{`           → `namespace M15AC {{`
//   2. `}}  // namespace m10`       → `}}  // namespace M15AC`
//   3. `M10Aic`                    → `AttnCoreAic`
//   4. `M10Aiv`                    → `AttnCoreAiv`
//   5. `M10_DBG_STAGE0`            → `M15AC_DBG_STAGE0`
//   6. `M10_DEBUG_SKIP_AIC_FD`     → `M15AC_DEBUG_SKIP_AIC_FD`
//   7. `M10_DEBUG_SKIP_COMBINE`    → `M15AC_DEBUG_SKIP_COMBINE`
//   8. printf 标签 `[M10K]`         → `[M15AC]`（纯文案，不动数值/判据/同步）
//
// ---- 第 9 类：**整块替换**（**抽取物不再是"逐字 + 纯重命名"**，塔 2026-10-04 裁决）----
//   把 `AttnCoreAiv::Combine()` 的 FD 归并内循环里两处**经典 memory-based 向量 API** 换成 RegBase VF：
//     改前：`Muls(tmpR, accR, w, S2T); Add(numR, numR, tmpR, S2T);`（donor asc:977-978 同款）
//     改后：`AccFmaRowVf(numUb, accUb, w)` = `__VEC_SCOPE__` 内的 `LoadAlign + Muls + Add + StoreAlign`
//     依据：人类逐字规则「……都应该都用 VEC_SCOPE aka simd vector function，**不应该使用 memory base 的 API**」
//     **等价性**：`Mul`/`Add`/`Muls` 在官方 Reg 精度表里都是 **0 ulp**，且 `nsplit` 累加次序一字未动
//     ⇒ 预期逐位不变；由「与 `m10_attn_decode/data` 归档 dump **逐字节**对拍 + donor `check_ref.py`
//     三档」见证（读数见 `evidence/attn_core/README.md` §3.2/§3.4）。**若对拍不成立即停手报塔。**
//
// ============================================================
// 段接口（供 M102 在顶层接线与**打平分配**；人类要求"暴露依赖哪些 pipeline、输出哪些 pipeline、
// 需要哪些 buffer id / cross-core id"）
// ============================================================
// 入口：`M15AC::AttnCoreBody(q, k, v, out, seq, wsAcc, wsM, wsS, maskCol, dbgS, dbgP, dbgC, cfg, dbgC2, gP)`
//   调度形态 `__mix__(1, 2)`；**blockDim = AIC 数**，本段内部假定 28 个 unit（14 split × 2 KV 头，
//   `n2 = bid & 1`、`split = bid >> 1`）。AIV 数 = 2 × AIC 数。
//   形状（全 bf16）：Q `[2][16][256]`（行 0..11 为 12 个 q 头，12..15 pad）、
//   K/V `[2][seqPad][256]`、out `[2][12][256]`；`seq` 是 GM 上的运行期标量（`seqGM.GetValue(0)`）。
//
// **依赖（消费）的管道** —— 语义 = "谁必须先跑完":
//   · 跨段: **无**。段起点是自包含的：调用方只需把 q/k/v/seq/maskCol/workspace/dbg 备好并对全体
//     核可见（前一个相位边界已给）。本段不 wait 任何段外 flag。
//   · 段内: AIC ↔ AIV 经 cross-core（下表），核内经 BufferID（下表）。
// **输出（生产）的管道**:
//   · GM `out`（`[2][12][256]` bf16，由 `Combine()` 按行写出 —— FD 归并的最终结果）。
//   · GM workspace `wsAcc/wsM/wsS`（未归一 partial；`WritePartialsAndSignal()` 写）。
//   · GM 调试面 `dbgS/dbgP/dbgC/dbgC2`（**首 tile 无条件写**，见下"调试面"）。
//   · 跨段: **无 set**。段尾由调用方补 `PipeBarrier<PIPE_ALL>`（与 `m15_attn_passthrough_body` 同款）。
//
// **cross-core flagId：{cc_n} 次调用，本段段内独占；融合后必须由顶层按 `m15_layer_resources.h` §4 重号。**
//   两条必须重号的理由（M102 逐条核）：
//   ① **mode-2 的 9/10/11 与 hc 段 mode-2 的 8/9/10/11 撞号**：同一 (核型, mode) 子空间内不许有
//      同一 id 的两个同步点。本段这三个号在顶层应取"已 drain 的空洞"或另分节。
//   ② **mode-4 不是"新子空间"**（复审 r1-F2 更正；官方口径 = 本机 CANN 文档
//      `asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md`
//      的「flagId取值范围说明」）：模式 0/1/2 每核 16 个（0-15）；**模式 4 时 AIV 仍是同一批 0-15
//      （池没有变宽），只有 AIC 侧放宽到 0-31**，且「AIC 发起 flagId 16-31 ↔ AIV1 的 wait 0-15」。
//      同一核上同一 flagId 跨模式复用是**合法**的，前提是"模式切换前前一个 mode 的所有
//      set/wait 都已执行完（drain）"。⇒ 本段的 AIV mode-4 号 {{0,1,5,6,7}} 与顶层 AIV 上
//      mode0+mode2 已占满的 0-15 **是同一个物理池**，打平表必须按"drain 后复用"登记；
//      本段的 AIC mode-4 高号（`ccMM + AIV_CH` = 16/17）落在 AIC 放宽后的 0-31 里 ——
//      **顶层登记表必须为 `AIC ∧ mode==4` 单独放行到 32**，否则 16/17 会被 `FLAG_PER_CORE = 16`
//      判成越界。截至本提交，主线 `m15_layer_resources.h` 已落 `FlagIdLimit(core, mode)` 与
//      `FLAG_PER_CORE_AIC_MODE4 = 32`（以及 `ATTN_CORE_M4_AIV_MAX` / `ATTN_CORE_M4_AIC_HI_LO`
//      两个见证量）—— 以该文件为准，本行只记"本段需要什么"。
//
// **资源足迹（人类口径"资源分配要全部打平考虑"；复审 r1-F5 要求写进本头）**：
//   · L1：段内 `L1_OFF_P0/P1/P2`(3×8KB) + `L1_OFF_Q`(8KB) + `L1_OFF_KV0/KV1`(2×128KB)
//     ⇒ 末端 = `L1_OFF_KV1 + L1_KV_BYTES` = 32768 + 131072 + 131072 = **294,912 B = 288 KB / 512 KB**。
//   · L0A 64KB（Q 8KB 常驻 + P 8KB 轮转）、L0B 64KB 单 buffer 轮转、L0C 4×16KB 环 = 64KB / 256KB。
//   · AIV UB：段内自报的两个峰值常数 `UB_TOTAL` = 50,208 B、`UB_TOTAL2` = 64,512 B（combine 路径）
//     ⇒ 峰值按 **64,512 B ≈ 63 KB / 248 KB** 记账。
//   · GM：`out` 12,288 B、workspace `wsAcc/wsM/wsS` = 458,752 + 1,792 + 1,792 B、
//     调试面 `dbgC/dbgC2/dbgS/dbgP` = 458,752+458,752+458,752+229,376 B、P 中转 `gP` 688,128 B。
//   ⇒ 以上都是**段内自管理**的编译期常量（没有用 AscendC 的资源管理函数）；融合后与其它段
//     同址叠放时要按相位互斥重核（L1/L0C/UB 的峰值取 max 不是求和）。
//
// **prefill 边界（复审 r1-F5 要求写进本头）**：本段是 **decode 形状** —— `blockDim` = AIC 数、
//   段内假定 28 个 unit（`SPLITS=14` × 2 KV 头）、`n2 = bid & 1`、`split = bid >> 1`、
//   KV tile = `S2T=256` token。**它不能直接吃 prefill `m=4097`**：AIC 的 N-块分派与 AIV 的按头
//   分派都要 per-row 重排（与 M97 §5 第 7 项同一件事，本 mission 未做，见
//   `evidence/attn_core/README.md` §5 第 3 项）。
//   段内常量（由生成器实测 donor 得出）：
{cc_rows}
//   调用分布（生成器实测）：
{cc_calls}
//
// **MutexID（核内局部编号）**：融合后必须与同核其它段打平重编；本段不假定全局编号。
//   段内常量：
{mx_rows}
//   调用分布（生成器实测）：
{mx_calls}
//
// 量测面（生成器实测）：`__VEC_SCOPE__` × {n_vec}、`Duplicate(` × {n_dup}
//   （`Duplicate` 是 UB 常量填充，不在计算链上 —— donor README §7 的存量说明）。
//
// **调试面（M102 接线注意）**：`dbgC`（`[28][16][256]` fp32 = 458,752 B）、`dbgC2`（同尺寸）、
//   `dbgS`（`[56][8][256]` fp32 = 458,752 B）、`dbgP`（`[56][8][256]` bf16 = 229,376 B）
//   是**首 tile 无条件写**的（donor 的定位遗留）。融合形态若要省掉这 4 次 dump 与
//   `printf`，需要改段内代码 —— 本 mission 按"机械抽取"口径**原样保留**，逐条记在
//   `evidence/attn_core/README.md` §5「未完成项」。
//
// ---- 包含前提 ----
// 本文件**不自带任何 include**（不引入系统头/AscendC 头）；依赖 include 它的 TU 已经包含
// `kernel_operator.h`（AscendC 基础 API + `AscendC::Reg`）。donor 的 device 段亦如此。
// ============================================================
#ifndef M15_ATTN_CORE_H
#define M15_ATTN_CORE_H

'''


TAIL_HEAD = '''

// ============================================================
// 段入口（M101 生成）：融合 kernel 与本 mission 的独立验证路共用同一个 body。
// 参数表 = donor `__global__` 入口的参数表（AIC/AIV 两侧 `Init()` 的并集），
// 只是把"分核调度"搬进来 —— donor 原本写在 `__global__` 入口里，那属于 host 侧不抽取的部分。
// ============================================================
namespace M15AC {
__aicore__ inline void AttnCoreBody(__gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* v,
                                    __gm__ uint8_t* out, __gm__ uint32_t* seq, __gm__ uint8_t* wsAcc,
                                    __gm__ uint8_t* wsM, __gm__ uint8_t* wsS, __gm__ float* maskCol,
                                    __gm__ uint8_t* dbgS, __gm__ uint8_t* dbgP, __gm__ uint8_t* dbgC,
                                    __gm__ uint32_t* cfg, __gm__ uint8_t* dbgC2, __gm__ uint8_t* gP)
{
    if ASCEND_IS_AIV {
        AttnCoreAiv op;
        op.Init(q, k, v, out, seq, wsAcc, wsM, wsS, maskCol, dbgS, dbgP, gP);
        op.Run();
    }
    if ASCEND_IS_AIC {
        AttnCoreAic op;
        op.Init(q, k, v, out, seq, wsAcc, wsM, wsS, dbgC, cfg, dbgC2, gP);
        op.Run();
    }
}

// ---- 段间契约的编译期钉子：id 一旦重编号，这里先红（提示同步顶层打平表）----
static_assert(CC_MM0 == 0 && CC_MM1 == 1 && CC_P0 == 5 && CC_P1 == 6 && CC_P2 == 7 && CC_BAR == 8 &&
              CC_AIVDONE == 9 && CC_RDY == 10 && CC_ALLDONE == 11 && AIV_CH == 16,
              "cross-core flagId 清单变了 —— m15_layer_resources.h §4 的顶层打平表必须同步");
static_assert(B_Q == 0 && B_KV0 == 1 && B_KV1 == 2 && B_L0 == 3 && B_C0 == 4 && B_C1 == 5 && B_C2 == 6 &&
              B_C3 == 7 && B_PL1 == 8, "AIC 侧 MutexID 清单变了");
static_assert(B_PC == 0 && B_ACC == 1 && B_MSK == 2 && B_CIN == 3 && B_COUT == 4 && B_ACCIN == 5 &&
              B_DBG == 6, "AIV 侧 MutexID 清单变了");
}  // namespace M15AC

#endif  // M15_ATTN_CORE_H
'''


def build():
    _text, body, lines, sha = _extract()
    body = _replace(body)          # 第 1-8 类：重命名式机械替换
    body = _block_replace(body)    # 第 9 类：整块替换（FD 归并 → RegBase VF）
    return _head(body, lines, sha) + body + TAIL_HEAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验磁盘上的 m15_attn_core.h 与重新抽取一致")
    args = ap.parse_args()
    want = build()
    _text, _body, lines, sha = _extract()
    if args.check:
        try:
            have = open(DST).read()
        except OSError as e:
            raise SystemExit(f"[FAIL] 读不到 {DST}: {e}")
        if have != want:
            raise SystemExit(f"[FAIL] {DST} 与 donor 重新抽取的结果不一致（漂移）—— 重新生成")
        print(f"[ok] {DST} == 从 {SRC} 重新抽取的结果（{len(want)} 字节，逐字节；"
              f"锚点 asc 行 {lines[0]}..{lines[1]}，donor sha256 {sha[:16]}…）")
        return
    with open(DST, "w") as f:
        f.write(want)
    print(f"[ok] 写出 {DST}（{len(want)} 字节；锚点 asc 行 {lines[0]}..{lines[1]}，"
          f"donor sha256 {sha[:16]}…）")


if __name__ == "__main__":
    sys.exit(main())
