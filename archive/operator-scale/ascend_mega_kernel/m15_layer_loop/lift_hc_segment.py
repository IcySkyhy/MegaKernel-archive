#!/usr/bin/env python3
"""从 m20_hyperconn/m20_hyperconn.asc 机械抽取 **hc（hyper-connection）段 device 代码**，
生成 m15_hc_layer.h（融合进 m15 per-layer kernel 的版本）。

为什么用脚本而不是手工复制：docs/17 §3 反例纪律 4 要求「复制改造的代码必须给出与上游的逐文件
差异表」。把抽取规则写成可执行文件后，`--check` 可以证明「入库文件里的算件代码与 m20 逐行相同，
只做了下面列出的 **6 类**替换」，差异表因此可复核而不是靠人眼（与 `lift_moe_segment.py` 同款）。

抽取范围：**内容锚点**（`_src_anchors()`）——从 m20_hyperconn.asc 首个顶格 `namespace {`
（device 段起点）到入口 kernel `m20_hyperconn_kernel` 之前**最后一个** `}  // namespace`。
入口本身与其后的 host 段（数据生成 / double 参考 / 判据 / dump）**不抽取**：融合后 hc 段不再
自成 kernel，入口由 `m15_layer_kernel.h` 给出。

保留逐字不变的替换（**6 类**，全部机械，无算法改动）：
  1. 行首 `namespace {` → `namespace M15H {`（两处：helper + 段序实现）。
     原因：m15 是**单 TU**，`m15_gdn_layer.h`（M15G）与 `m15_moe_layer.h`（M15M）的同名工具
     （BufAcquire/Block1/NormStage/…）在同一 TU 里；匿名 namespace 装同一批名字会重定义。
  2. 删除 `using namespace M20;`（段内常量已由 `m15_hc_resources.h` 定义在 M15H 内）。
  3. 资源表 include 名与注释里的 `m20_resources.h` → `m15_hc_resources.h`；
     标识符/注释里的 `M20` → `M15H`。
  4. **IJ 源的行距可变**（`HcPtrs` 新增 `ijStride`）：独立 `[m,16]` 平面为 `IJ_STRIDE(16)`，
     层内 handoff（IJ 取自前一个边界的 `OH[:, 320:324)`，行距 `OH_W=336`）为 `OH_W`。
     实现 = 把原来的一次整块 `DataCopy` 改成**按行 32B 搬运**（两种行距都是 32B 倍数）。
     原因：vLLM 的层序里，层内第二次 hc 边界的 `inj_logits` **就是**第一次边界的
     `OH[:,320:324]`；m20 是独立 kernel，中间那次提取由 host 用 `[m,16]` 平面物化，融合后必须
     在 kernel 内直接消费。
  5. **新增第 4 档 mode `MODE_COMBINE_ONLY`**（`ProcessAiv` 段首 + `ProcessAic` 段首各一个分支）：
     combine-only = 只跑 W0 + S1（`H' = bf16(H + BO·injW)`），不做 GemmaRMSNorm、不做 mix、
     AIC 不跑任何 GEMM。用途见 `m15_hc_resources.h` §0 与 README（层 0→1 被 PLE 打断的边界，
     m20 的三档 mode 都不提供；其 README §8 已显式披露该缺口）。
  6. 文件头 PROLOGUE 换成 m15 版的来源/差异登记（本文件顶部那段）。

**不做的替换**：段序、同步表、tile 常量、UB/L1/L0 偏移**一字未动**（flagId 的**取值**随
`m15_hc_resources.h` 重编，但代码里全部走 `FLAG_*` 符号，正文无字面量 ⇒ 无文本替换）。

上游不变量（`assert_upstream()`：**只断言、不替换**）：
  · **计算路径合规**（tower M44 裁决 ① + docs/05 §6.1 计算路径规则 ⓒ）：段内所有向量 API 调用
    要么**词法上在 `__VEC_SCOPE__` 块内**，要么在**只被 `__VEC_SCOPE__` 块调用的 helper 函数**里
    （判据落在被调函数上）。两个集合都由本脚本**自动推导**（不写死函数名）。
  · **无 memory-based 等价物之争**：hc 段的计算路径里**没有** `Sort32` / `MrgSort` / 经典
    `Extract(` —— 因此**没有需要援引白名单例外的调用点**；若将来重抽带进来，脚本会失败，
    必须先在 docs/05 §6.1 登记例外与依据注释（四段式：例外声明 / docs/05 §6.1 / 无 `Reg::`
    等价物 / 官方 donor 位置）才能放行。
  · **两处 bring-up 披露必须随代码走**：`StoreDist::DIST_FIRST_ELEMENT_B32` 的「误用」裁定
    （tower 裁定，非平台差异）与 §4.1(b) 的「尚未隔离」声明 —— 它们是 m20 README §4.1 的
    device 侧载体，删掉就等于把「已定性」与「尚未隔离」两条结论从代码里抹掉。
  · **上游未变**：device 段的 sha256 必须等于 `SRC_DEV_SHA256`（重抽需显式更新本脚本）。
    host 段的变化不触发失败（抽取范围不含它），但会在 `--check` 里报告。

用法：
    python3 lift_hc_segment.py            # 生成 m15_hc_layer.h
    python3 lift_hc_segment.py --check    # 只校验已入库的 m15_hc_layer.h 与规则一致
    python3 lift_hc_segment.py --check --verbose
"""

import argparse
import hashlib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "m20_hyperconn" / "m20_hyperconn.asc"
DST = HERE / "m15_hc_layer.h"

# ---- 上游身份（可复核：device 段的 sha256 是硬断言，其余是记录事实）----
SRC_PATH_STR = "m20_hyperconn/m20_hyperconn.asc"
SRC_COMMIT = "a0996ebab356defb743055f7e0178f9361d104c7"   # 最后改动该文件的 commit（M181 donor 注释订正）
SRC_FILE_SHA256 = "b707507d4bc8dd7ec625a346660816b1d1fe9ef5f6b55d85f606fdfbe0cdebe5"
SRC_DEV_SHA256 = "12fe5ca26d6f76e3e138a98405953925a5b912287d285e3ee64af635d80fa9a9"

ENTRY_MARK = 'extern "C" __global__ __mix__(1, 2) void m20_hyperconn_kernel('


def _src_anchors():
    lines = SRC.read_text().splitlines()
    begin = next(i for i, ln in enumerate(lines) if ln == "namespace {")
    entry = next(i for i, ln in enumerate(lines) if ln.startswith(ENTRY_MARK))
    end = max(i for i, ln in enumerate(lines[:entry]) if ln == "}  // namespace")
    return begin, end, entry


def _dev_section() -> str:
    begin, end, _ = _src_anchors()
    lines = SRC.read_text().splitlines()
    return "\n".join(lines[begin:end + 1]) + "\n"


# ============================================================
# 结构解析：括号匹配（用于 __VEC_SCOPE__ 覆盖范围与函数边界）
# ============================================================

def _brace_ranges(lines, start_indices):
    """对每个 start 行，返回 (start_line, end_line)：从该行起第一个 '{' 到其匹配 '}'。"""
    out = []
    for s in start_indices:
        depth = 0
        opened = False
        for i in range(s, len(lines)):
            for ch in lines[i]:
                if ch == '{':
                    depth += 1
                    opened = True
                elif ch == '}':
                    depth -= 1
                    if opened and depth == 0:
                        out.append((s, i))
                        break
            else:
                continue
            break
        else:
            raise AssertionError("括号不匹配：从第 %d 行起的块没有闭合" % (s + 1))
    return out


def _strip_line_comment(s: str) -> str:
    i = s.find("//")
    return s if i < 0 else s[:i]


VECTOR_OPS = (
    "Add", "Sub", "Mul", "Div", "Max", "Min", "Adds", "Muls", "Exp", "Ln", "Sqrt", "Cast",
    "Duplicate", "Select", "Compares", "Compare", "Reduce", "Arange", "LoadAlign", "StoreAlign",
    "Gather", "GatherMask", "Interleave", "DeInterleave", "Transpose", "Concat", "Extract",
    "Sort32", "MrgSort", "Sort", "Copy", "BlockReduce", "ReduceSum", "Brcb", "Load",
)
VECTOR_RE = re.compile(r"\b(" + "|".join(VECTOR_OPS) + r")\s*[<(]")
# 只做跨核/搬运/矩阵的调用（memory-based 但按标准允许）——不算「向量 API 调用点」
ALLOWED_PREFIX = ("AscendC::", "Mutex", "Buf", "Barrier", "CrossCore", "Pipe", "Trap",
                  "GlobalTensor", "LocalTensor", "InitSocState", "GetPhyAddr", "SetGlobalBuffer")


def _vector_call_lines(lines):
    """返回 [(lineno0, line)]：剥掉行尾注释后仍匹配向量 API 的行。"""
    out = []
    for i, raw in enumerate(lines):
        code = _strip_line_comment(raw)
        if VECTOR_RE.search(code):
            out.append((i, raw))
    return out


def _func_of_line(lines, line_idx):
    """若 line_idx 落在某个 `__aicore__ ... <name>(` 函数体内，返回该函数名（否则 None）。"""
    heads = [i for i, ln in enumerate(lines) if re.search(r"__aicore__.*\b(\w+)\s*\(", ln)
             and not ln.lstrip().startswith("//") and ";" not in ln]
    for h in heads:
        m = re.search(r"__aicore__.*?\b(\w+)\s*\(", lines[h])
        if m is None:
            continue
        # 函数体起点 = 参数表闭合后的第一个 '{'（可能在同一行或多行后）
        depth = 0
        started = False
        body_start = None
        for i in range(h, len(lines)):
            for ch in lines[i]:
                if ch == '(':
                    depth += 1
                    started = True
                elif ch == ')':
                    depth -= 1
                elif ch == '{' and started and depth == 0:
                    body_start = i
                    break
            if body_start is not None:
                break
        if body_start is None:
            continue
        (_, body_end), = _brace_ranges(lines, [body_start])
        if body_start <= line_idx <= body_end:
            return m.group(1)
    return None


def assert_upstream(text: str, verbose: bool = False):
    """抽取段的上游不变量断言（全部可复核；任一不满足即 AssertionError）。"""
    lines = text.splitlines()
    problems = []

    # ---- (A) device 段 sha256 未变（重抽必须显式更新本脚本）----
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != SRC_DEV_SHA256:
        problems.append("上游 device 段已变：sha256=%s（本脚本记录 %s）⇒ 需显式重抽并复核差异表"
                        % (got, SRC_DEV_SHA256))

    # ---- (B) 计算路径：向量 API 只在 __VEC_SCOPE__ 内，或在「只被 VEC_SCOPE 调用」的 helper 里 ----
    vec_scope_starts = [i for i, ln in enumerate(lines) if "__VEC_SCOPE__" in _strip_line_comment(ln)]
    if not vec_scope_starts:
        problems.append("device 段里没有 __VEC_SCOPE__（计算路径全 memory-based？）")
    scope_ranges = [_brace_ranges(lines, [s])[0] for s in vec_scope_starts]

    def in_scope(i):
        return any(a <= i <= b for (a, b) in scope_ranges)

    helpers = set()
    outside = []
    for i, _raw in _vector_call_lines(lines):
        if in_scope(i):
            continue
        fn = _func_of_line(lines, i)
        if fn is None:
            outside.append(i)
        else:
            helpers.add(fn)
    if outside:
        problems.append("以下行用了向量 API，却既不在 __VEC_SCOPE__ 内、也不在任何 __aicore__ "
                        "函数体内（memory-based 计算路径？）：行 %s"
                        % ", ".join(str(i + 1) for i in outside))
    for fn in sorted(helpers):
        for i, raw in enumerate(lines):
            if not re.search(r"\b%s\s*\(" % re.escape(fn), _strip_line_comment(raw)):
                continue
            if re.search(r"__aicore__.*\b%s\s*\(" % re.escape(fn), raw):
                continue   # 定义行本身
            if not in_scope(i):
                problems.append("helper `%s` 的调用点在第 %d 行，**不在 __VEC_SCOPE__ 内**"
                                "（tower M44 裁决 ①：判据落在被调函数上 ⇒ 调用点必须在 VEC_SCOPE 内）"
                                % (fn, i + 1))
    if verbose:
        print("[lift] 计算路径判据：%d 个 __VEC_SCOPE__ 块；跨块调用的 helper = %s"
              % (len(scope_ranges), sorted(helpers) if helpers else "（无）"))

    # ---- (C) 例外声明：hc 段不得出现白名单例外调用点（Sort32/MrgSort/经典 Extract）----
    forbidden = []
    for i, raw in enumerate(lines):
        code = _strip_line_comment(raw)
        for op in ("Sort32", "MrgSort", "Extract"):
            if re.search(r"\b%s\s*\(" % op, code):
                forbidden.append("第 %d 行：%s" % (i + 1, op))
    if forbidden:
        problems.append("hc 段出现了白名单例外类调用点（%s）——必须先按 docs/05 §6.1 登记例外声明"
                        "（四段式：例外声明 / docs/05 §6.1 / 无 Reg:: 等价物 / 官方 donor 位置）"
                        % "; ".join(forbidden))

    # ---- (D) 两处 bring-up 披露必须随代码走（m20 README §4.1 的 device 侧载体）----
    for mark, why in (
        ("属**误用**（tower 裁定，非平台差异）",
         "`StoreDist::DIST_FIRST_ELEMENT_B32` 只写元素 0 的**误用**裁定"),
        ("**不作为平台行为引用**（详见 README §4.1）",
         "原 S1「从 GM 取 injW 再 BRC」非确定性的**尚未隔离**声明"),
        ("docs/05 §6.2 第 1 条",
         "同 pipe 复用 UB 行缓冲的 PipeBarrier 依据（docs/05 §6.2）"),
    ):
        if mark not in text:
            problems.append("上游披露标记缺失：%s —— %s 必须随代码一起搬运" % (why, mark))

    # ---- (E) hc 段的段序/同步结构存在（防止抽取范围被改窄成空壳）----
    for mark in ("InjwStage", "CombineStage", "NormStage", "SiluStage", "GateMixStage",
                 "ProcessAiv", "ProcessAic", "class HyperConnOp"):
        if mark not in text:
            problems.append("device 段缺少 `%s`（抽取范围被改窄？）" % mark)

    if problems:
        raise AssertionError("上游不变量断言失败：\n  - " + "\n  - ".join(problems))


# ============================================================
# 6 类替换
# ============================================================

def _sub_once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        raise AssertionError("替换「%s」期望 1 处，实际 %d 处（上游已变）" % (what, n))
    return text.replace(old, new)


T4_PTR_OLD = """    uint32_t m;              // token 数（1..M_MAX）
    uint32_t mode;           // MODE_MIX / MODE_COMBINE_MIX / MODE_FINAL_MIX
    uint32_t stageLimit;     // 段序截断（bring-up 定位；7 = 全开）"""

T4_PTR_NEW = """    uint32_t m;              // token 数（1..M_MAX）
    // ---- M58 新增（替换类 4）：IJ 源的行距（元素）----
    //   独立 [m,16] 平面（m20 的 host 形态）= IJ_STRIDE = 16；
    //   层内 handoff（IJ 直接取自上一个边界的 OH[:,320:324)）= OH_W = 336。
    //   两种行距都是 32B 倍数 ⇒ 每行 32B 搬运天然对齐（见 InjwStage 的按行搬运用法）。
    uint32_t ijStride;
    uint32_t mode;           // MODE_MIX / MODE_COMBINE_MIX / MODE_FINAL_MIX / MODE_COMBINE_ONLY
    uint32_t stageLimit;     // 段序截断（bring-up 定位；7 = 全开）"""

T4_COPY_OLD = "        AscendC::DataCopy(ijL, ijGm[0], Block1(M_MAX * IJ_STRIDE * 2));"

T4_COPY_NEW = """        // M58 替换类 4：IJ 源按**行距 p.ijStride** 逐行 32B 搬运（不再是一次整块拷贝）。
        // 两种行距（独立平面的 16、OH 列 [320,324) 的 336）都是 32B 倍数 ⇒ 每行 32B 天然
        // 对齐，不引入 DataCopyPad（docs/05 §6.1 的 32B 对齐要求）。
        for (uint16_t ijRow = 0; ijRow < static_cast<uint16_t>(M_MAX); ++ijRow) {
            AscendC::DataCopy(ijL[static_cast<uint32_t>(ijRow) * IJ_STRIDE],
                              ijGm[static_cast<uint32_t>(ijRow) * p.ijStride], Block1(IJ_STRIDE * 2));
        }"""

T5_AIV_OLD = """        const bool useCombine = (p.mode != MODE_MIX);
"""

T5_AIV_NEW = """        const bool useCombine = (p.mode != MODE_MIX);

        // M58 替换类 5：**第 4 档 mode = combine-only**（`MODE_COMBINE_ONLY`）。
        // 语义 = 只跑 W0（injW = 2·sigmoid(IJ/HC)）+ S1（H' = bf16(H + BO·injW[s])），
        // 既不做 GemmaRMSNorm、也不做 mix（S2/S3-S6 全部跳过），AIC 也不跑任何 GEMM。
        // 用途：层 0→1 被 PLE 打断的边界（docs/14 §4.1）——那个边界上 combine 与 mix 之间插入了
        // PLE，二者必须分开执行。m20 的三档 mode 都不提供这一档（其 README §8 已披露该缺口）。
        // 段序 = InjwStage → CombineStage → 到齐 barrier（`FLAG_AV0`，与 combine 档同位置），
        // 然后返回；不需要 mode-2 交接（本档没有 AIC 参与）。
        if (p.mode == MODE_COMBINE_ONLY) {
            InjwStage(bid, nAiv);
            CombineStage(bid, nAiv);
            BarrierAiv<PIPE_MTE2, FLAG_AV0>();
            return;
        }
"""

T5_AIC_OLD = """        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nAic = AscendC::GetBlockNum();
"""

T5_AIC_NEW = """        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nAic = AscendC::GetBlockNum();

        // M58 替换类 5：combine-only 不含任何 GEMM（S3 down 与 S5 up 都跳过）⇒ AIC 直接返回。
        // AIV 侧本档也不等待任何 mode-2 flag，故此早退不会造成对侧挂死。
        if (p.mode == MODE_COMBINE_ONLY) {
            return;
        }
"""


PROLOGUE = '''/**
 * m15_hc_layer.h —— **hc（hyper-connection）边界段 device 代码**，融合进 m15 per-layer kernel
 *
 * **本文件由 lift_hc_segment.py 机械生成，不要手改**（改上游 m20 后重跑脚本，再核对 --check
 * 输出的差异表）。抽取范围 = `%(src)s` 的 **device 段**（内容锚点：首个顶格
 * `namespace {` 到入口 kernel `m20_hyperconn_kernel` 前最后一个 `}  // namespace`，**不写行号**）
 * ——入口与其后的 host 段（数据生成 / double 参考 / 判据 / dump）不抽取：融合后 hc 段不再自成
 * kernel，入口由 `m15_layer_kernel.h` 给出。
 *
 * 上游身份（可复核）：文件 sha256 = `%(filesha)s`
 * （last touched by `%(commit)s` = 最后改动 m20 的 commit）；
 * **device 段 sha256 = `%(devsha)s`**（`--check` 的硬断言：device 段一变即失败）。
 *
 * 与上游的差异 = **6 类机械替换**（逐条见 `lift_hc_segment.py` 的模块 docstring，`--check`
 * 会逐条核对）：
 *   1. 两处 `namespace {` → `namespace M15H {`（单 TU 里与 M15G/M15M 的符号分离）；
 *   2. 删除 `using namespace M20;`；
 *   3. `m20_resources.h` → `m15_hc_resources.h`、标识符/注释 `M20` → `M15H`；
 *   4. `HcPtrs.ijStride` + `InjwStage` 的 IJ 源改为**按行 32B 搬运**（层内 handoff：IJ 取自
 *      上一个边界的 `OH[:,320:324)`，行距 `OH_W=336`）；
 *   5. `MODE_COMBINE_ONLY`（第 4 档 mode：只跑 W0+S1；PLE 打断点用）在 `ProcessAiv`/
 *      `ProcessAic` 段首各加一个分支；
 *   6. 本 PROLOGUE。
 *
 * **一字未动**的部分：段序、同步表、tile 常量、UB/L1/L0 偏移、全部向量/矩阵/搬运语句
 * （flagId 的**取值**由 `m15_hc_resources.h` 重编，但正文全走 `FLAG_*` 符号 ⇒ 无文本替换）。
 *
 * 计算路径合规（docs/05 §6.1 计算路径规则 ⓒ + tower M44 裁决 ①）：段内全部向量 API 要么词法上
 * 在 `__VEC_SCOPE__` 内，要么在**只被 `__VEC_SCOPE__` 块调用**的 helper
 * （`NormDonor::LoadRegForDtype` / `ComputeRstdNewtonRaphsonReg` / `SigmoidReg`）里；两个集合
 * 由 `lift_hc_segment.py` **自动推导并断言**（不写死函数名），且**断言段内 0 处**
 * `Sort32`/`MrgSort`/经典 `Extract(` —— 即 hc 段**没有需要援引白名单例外的调用点**。
 * 搬运/矩阵类（`DataCopy`/`Nd2Nz`/`LoadData`/`Mmad`/`Fixpipe`）按标准保留 memory-based。
 *
 * 两处 bring-up 披露随代码搬运（删掉等于抹掉结论，故由脚本断言其存在）：
 *   · `StoreDist::DIST_FIRST_ELEMENT_B32` 只写元素 0 = **误用**（tower 裁定，非平台差异）；
 *   · 原 S1「从 GM 取 injW 再 BRC」非确定性 = **尚未隔离**，不作为平台行为引用（README §4.1）。
 */

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "reg_compute/kernel_reg_compute_intf.h"

#include <cstdint>

#include "m15_hc_resources.h"

'''

DEFINE_START = 'namespace M15H {'


def build() -> str:
    begin, end, entry = _src_anchors()
    lines = SRC.read_text().splitlines()
    text = "\n".join(lines[begin:end + 1]) + "\n"
    assert_upstream(text)

    # 类 1：两处匿名 namespace → namespace M15H {
    n_ns = sum(1 for ln in text.splitlines() if ln == "namespace {")
    if n_ns != 2:
        raise AssertionError("替换类 1：期望 2 处顶格 `namespace {`，实际 %d" % n_ns)
    text = "\n".join("namespace M15H {" if ln == "namespace {" else ln for ln in text.splitlines()) + "\n"

    # 类 3：资源表名 / M20 → M15H（含 `using namespace M20;` → `using namespace M15H;`）
    n_res = text.count("m20_resources.h")
    if n_res == 0:
        raise AssertionError("替换类 3：device 段里没有 `m20_resources.h` 引用（上游已变？）")
    text = text.replace("m20_resources.h", "m15_hc_resources.h")
    n_m20 = len(re.findall(r"\bM20\b", text))
    if n_m20 == 0:
        raise AssertionError("替换类 3：device 段里没有 `M20` 标识符（上游已变？）")
    text = re.sub(r"\bM20\b", "M15H", text)

    # 类 2：删除 using namespace M15H;（段内常量已由 m15_hc_resources.h 定义在 M15H 内）
    text = _sub_once(text, "\nusing namespace M15H;\n", "\n",
                     "类 2：删除 using namespace M15H;（原 M20）")

    # 类 4：IJ 源行距可变
    text = _sub_once(text, T4_PTR_OLD, T4_PTR_NEW, "类 4a：HcPtrs.ijStride 字段")
    text = _sub_once(text, T4_COPY_OLD, T4_COPY_NEW, "类 4b：InjwStage 的 IJ 按行搬运")

    # 类 5：combine-only 档
    text = _sub_once(text, T5_AIV_OLD, T5_AIV_NEW, "类 5a：ProcessAiv 的 combine-only 分支")
    text = _sub_once(text, T5_AIC_OLD, T5_AIC_NEW, "类 5b：ProcessAic 的 combine-only 分支")

    # 类 6：PROLOGUE（## 6 类替换）
    head = PROLOGUE % {"src": SRC_PATH_STR, "commit": SRC_COMMIT, "filesha": SRC_FILE_SHA256,
                       "devsha": SRC_DEV_SHA256}
    if text.startswith(DEFINE_START):
        text = text[len(DEFINE_START):]
        text = head + DEFINE_START + text

    # 生成物自检：改名后不得再出现裸 M20 标识符 / 匿名 namespace
    if re.search(r"\bM20\s*::", text):
        raise AssertionError("生成物里仍有 `M20::` 限定（替换类 3 不完整）")
    body = text.split("namespace M15H {", 1)[1]
    if re.search(r"^namespace \{", body, re.M):
        raise AssertionError("生成物里仍有顶层匿名 namespace（替换类 1 不完整）")
    if "MODE_COMBINE_ONLY" not in text:
        raise AssertionError("生成物里没有 MODE_COMBINE_ONLY（替换类 5 不完整）")
    tail = text.splitlines()[-1]
    if tail != "}  // namespace":
        raise AssertionError("生成物尾部不是 `}  // namespace`（当前 %r）—— 抽取范围被截断？" % tail)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验已入库文件与规则一致")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("[lift] 源 = %s（device 段 %d 行）" % (SRC, SRC_DEV_SHA256 and
          len(_dev_section().splitlines())))
    got = build()

    # 上游 host 段变化（不触发失败，但必须报告 —— 说明「文件变了、device 段没变」）
    file_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    if file_sha != SRC_FILE_SHA256:
        print("[lift][note] 上游**文件** sha256 = %s ≠ 本脚本记录 %s（device 段未变 ⇒ 抽取不受影响；"
              "若 host 段的参考语义变了，check_hc_ref.py 侧需另行复核）" % (file_sha, SRC_FILE_SHA256))
    assert_upstream(_dev_section(), verbose=args.verbose)

    if args.check:
        if not DST.exists():
            print("[lift][FAIL] %s 不存在" % DST)
            return 1
        cur = DST.read_text()
        if cur != got:
            print("[lift][FAIL] %s 与抽取规则不一致（--check 失败）" % DST.name)
            import difflib
            d = list(difflib.unified_diff(cur.splitlines(), got.splitlines(),
                                          "checked-in/" + DST.name, "regenerated/" + DST.name,
                                          lineterm="", n=2))
            print("\n".join(d[:80]))
            return 1
        print("[lift] OK：%s 与抽取规则逐字节一致（6 类替换；device 段 sha256 = %s）"
              % (DST.name, SRC_DEV_SHA256))
        return 0

    DST.write_text(got)
    print("[lift] 写出 %s（%d 行）" % (DST, len(got.splitlines())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
