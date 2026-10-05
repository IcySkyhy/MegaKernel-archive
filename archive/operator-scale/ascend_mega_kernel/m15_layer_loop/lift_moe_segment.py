#!/usr/bin/env python3
"""从 m13_moe_layer/m13_moe_layer.asc 机械抽取 MoE 段 device 代码，生成 m15_moe_layer.h。

为什么用脚本而不是手工复制：docs/17-predeed §3 反例纪律 4 要求「复制改造的代码必须给出与
上游的逐文件差异表」。把抽取规则写成可执行文件后，`--check` 可以证明「m15_moe_layer.h 里
的算件代码**除了下面列出的 8 类替换之外**与 m13 相同」，差异表因此可复核而不是靠人眼。
**注意口径**：第 1–7 类是**重命名式/插入式机械替换**（逐行相同仍是它们的性质）；**第 8 类
（M91-#3）是整块替换** —— 它把 S2 router 的权重加载与点积两段**整段重写**（生成物里约 200 行），
所以对 S2 router 而言「与 m13 逐行相同」**不成立**，等价性由 `moe_relift/m91_README.md` 的
算术论证 + E=4 的 A/B dump 逐字节对拍见证（见该文件 §#3）。

抽取范围：**内容锚点**（见 `_src_anchors()`）——从 m13_moe_layer.asc 首个顶格
`namespace {`（device 段起点）到层 kernel 入口 `m13_moe_layer_kernel` 之前最后一个
`}  // namespace`，即**全部 device 侧代码**；入口本身与其后的 host 段（数据集装载 /
判据 / dump）**不抽取**——融合后 MoE 段不再自成 kernel，入口由 m15_layer_kernel.h 给出。

保留逐字不变的替换（全部是机械替换，无算法改动）：
  1. 行首 `namespace {` → `namespace M15M {`
     原因：m15_gdn_layer.h 的同名工具（BufAcquire/Block1/NormStage/NormDonor/…）也在**同一个 TU**
     的匿名 namespace 里；两个匿名 namespace 装同一批名字会重定义。改用具名 namespace M15M
     后，(a) 与 GDN 段的名字空间分离，(b) 融合入口可以用 `M15M::MoeLayerChain` 显式引用。
  2. 删除 `using namespace M13;`（段内所有常量已由 m15_moe_resources.h 定义在 M15M 内）。
  3. 资源表 include 改名。
  4. 行尾/注释里的 `M13` → `M15M`；
  5. **出口 y 独立成 GM 缓冲**（与 m15 对 m14 的同一处改造）：
     `MoeLayerPtrs` 增加 `yLayer` 字段，S10 的 m6#2 出口由 `ws + WS_YFINAL` 改为 `args.yLayer`。
     原因：融合后层输出必须落在层间的残差流缓冲（`yLayer`）上，而 m13 原版把 S10 出口写在层内
     ws 的 WS_YFINAL 里（m13 的 host 判据直接从 ws 读它）。ws 内的 WS_YFINAL 区间在融合形态下
     不再使用（地址仍保留，dump 判据改为从 yLayer 读）。
  6. **紧凑专家槽 + (expert, mTile, nTile) 工作项**（tower 对 M40 的硬约束，见 README §5.7）：
     把「每专家固定 M_MAX 行」的槽位寻址改成 **Σt_e 紧凑槽**（槽起点 = S3 已产出的
     `expert_offsets[]` 前缀和），并把两组 grouped GEMM 的工作项从
     `(slot, nBlock)` 改成 `(expert, mTile, nBlock)`、循环上界由 `counts` / `active_num` 驱动。
     改动是**纯寻址**：所有 ws 张量的**尺寸一字未改**（在本实例 `NUM_EXPERTS*M_MAX = 256`
     恰好等于紧凑上界 `TOTAL_MAX = M_MAX*TOPK_MAX`，故缓冲区天然够用；真实规模下紧凑布局还能
     把这块 ws 显著缩小 —— 见 README §5.7）。涉及 6 个位置：
       (a) S3 的 `inv_slot` 编码：`e*M_MAX + 局部行号` → 紧凑行号 `pos`；
       (b) S5 的 A 量化目标从 padded 改 compact（`PAD_DST=false`）；
       (c) S7 的 H 量化源与目标都改 compact（`PAD_SRC=false, PAD_DST=false`）；
       (d)(e) S6/S8 两组 GEMM 的 A/H/Y 槽位基址从 `slot*M_MAX*...` 改成 `off[slot]*...`；
       (f) S6/S8 的工作项枚举改成 (expert, mTile, nBlock) 三重循环，空专家由 `counts` 跳过。
  7. **规模无关前置修复（M84，= M77 §2 的 #5/#6/#7/#8）**：这四处**在 E=4/topk=2 下是 no-op**，
     但真实规模 E=512/topk=10 下会失效。只改「与 E 相关的形态」，不动规模常量、不动紧凑 Σt_e
     寻址（M40 的硬约束）：
       (a) **#5** S3 的 `cnt[E]/off[E+1]/cursor[E]` 由 AIV **标量栈数组**改为 **UB 静态槽位**
           （E=512 时 3×2KB = 6KB 压垮标量栈）；槽位尺寸按 NUM_EXPERTS 定尺、随 E 自动增长，
           并对 counts/offsets 各做 32B 对齐（DataCopyPad 源要求）；
       (b) **#6** 诊断槽的 GM 目的从写死的 `offs[8]` 改为 `IG_DIAG_GM_SLOT`
           （= `SZ_OFFSETS` 尾部 32B 槽的起始 i32 下标，恒在真 offsets[0..E] 之后；
           E 为 8 的倍数时 = `E+8`；E=4 时 = 8，与改前同一字节）；
       (c) **#7** unpermute 的 K 链（`FmaChunk`）与模板实例由 `1..3`/`<1..4>` 展开到
           `1..9` + `UnpermuteStage<10>`（topk=10 需要）；
       (d) **#8** unpermute 的 Y 视图上界显式写成 `TOTAL_MAX`（紧凑 Σt_e 上界）；
           **不是** padded 的 `NUM_EXPERTS*M_MAX` —— 两者在 E=4 相等只是巧合；
       (e) **M105** S3 诊断槽的 UB **标量写**（`bUb[32] = oobCountUb;`）从
           `BufAcquire<PIPE_MTE3>(BUF_AIV_IDX)` **之后**提前到写侧 acquire（`MutexLock<PIPE_S>`）
           **之前**，并把写侧释放从 `MutexUnlock<PIPE_S>`（= `RlsBufInternal<PIPE_S,0>`，mode 0）
           换成 **`BufRelease<PIPE_S>`（= `RlsBufInternal<PIPE_S,false>`，mode=false = CANN
           `ASC_LOCK_BLOCK` 默认「阻塞」；`true` = `NON_BLOCK`；两种模式都等本 pipe 已发射指令落地，
           `true` 额外等此前同 id 的释放）**。改前这次
           标量写与末尾诊断槽的 `DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], …)` 抢跑（可能落到该
           UB 位置 `UB_IG_CNT+128` —— 落在 router x 行 0 窗内 —— 的残值）。成对形态 = `docs/05`
           §6.1 ⓔ「写 → 写侧 acquire → 写侧 release → 搬侧 acquire → 搬」（本次的写落在写侧
           acquire **之前**；ⓔ 的覆盖面前提只要求写在该 release 之前），同款已受审站点
           见 `m15_ple.asc` / `m15_hc_layer.h::CombineStage`；m17 的 `IndexGenStage::Run` 是同一诊断的
           先例。（不动 UB 地址、不动 GM 目的；**不新增** set/wait flag 类同步 —— 人类逐字禁用。）
  8. **router 权重流式（M91-#3）**：**整块替换（不是重命名式机械替换）** —— 把 S2 的
     `PrecastWeights`（`[NUM_EXPERTS+1][HIDDEN]` 全量预转 fp32 常驻 UB）换成 m17_moe_real 的
     流式形态，并重组 `ComputeBlock` / `GemvRow`：
       (a) `PrecastWeights` 段 → `LoadWRow` / `PrecastWRow`（bf16 行 ping/pong + `RT_EGRP` 行
           fp32 窗）/ `PrecastSgateW`（共享门 1 行独立槽）；
       (b) `ComputeBlock` 段 → 先 `PadLogitsRow`，再共享门 `SgateRows`，再按 `RT_EGRP` 个专家
           一组流式 `GemvGroupRow`，最后 `SoftmaxTopkRow`；
       (c) `GemvRow` 段 → `PadLogitsRow` / `SgateRows` / `GemvGroupRow`（8 个累加器 + 8 路
           Interleave 树 + `ng = min(RT_EGRP, NUM_EXPERTS - e0)` 掩码）；
       (d) 删除：类内 `static constexpr uint32_t RT_RB = 16;` 的 shadow（改用资源表 §4 的
           `RT_RB` = 8）、`RT_NROWS` 常量、`Run()` 里的 `PrecastWeights()` 调用。
     **算术不变**（同一 `CHUNKS_H`、同一 `MulAddDst` chunk 升序累加、同一 `Reduce SUM`、同一
     Interleave 树；8 路树的 lane 0..3 == 旧 4 路树的 `[l0,l1,l2,l3]`）⇒ E=4 下逐字节相同，
     由 A/B dump 见证；论证与设备读数见 `moe_relift/m91_README.md` 的 §#3。

上游不变量（`assert_upstream_vf()`：**只断言、不替换**——从 M52 起上游 m13 自己就是对的）：
  · m13 的 S2 router（`SoftmaxTopkRow`）已按 tower M44 裁决把经典 memory-based
    `Extract(ovT, oiT, pairT, 1)` 换成厂商 VF 形态：`Sort32` 之后一次
    `LoadAlign<float, LoadDist::DIST_DINTLV_B32>` 解交织 + 两条掩码 `StoreAlign`
    （抄 dav_3510 的 `ExtractVf<float>`；M50 在 m13 落地）；
  · `Sort32` 保留为已批白名单例外，且调用点带 docs/05 §6.1 规则 ⓒ 的依据注释
    （无 `Reg::` 等价物 + 官方 donor 位置）。
  背景：M40 曾在**本脚本里**手写一份等价的 Extract 替换（旧规则 7，落在生成物里）。M52 重抽时
  发现该规则的目标文本已随 M50 失效（脚本自身报 AssertionError），且它的产物与上游形态分叉
  （多算了一处「不落 UB 再读回」的差异）。改为断言后，**副本恒等于上游的当前形态**；上游若
  退回 memory-based `Extract` 或缺 `Sort32` 依据注释，重抽会立刻失败而不是静默落后。

用法：
    python3 lift_moe_segment.py            # 生成 m15_moe_layer.h
    python3 lift_moe_segment.py --check    # 只校验已入库的 m15_moe_layer.h 与规则一致
"""

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "m13_moe_layer" / "m13_moe_layer.asc"
DST = HERE / "m15_moe_layer.h"

# ---- 内容锚点（**不要再用裸行号**：M32 在 main 上给 m13 加了 +454 行，行号锚点会静默错位、
#      写出截断的文件）----
#   SRC_BEGIN  = 第一个顶层匿名 namespace（`namespace {` 且顶格）
#   ENTRY_MARK = 层 kernel 入口（device 段结束于它之前最后一个 `}  // namespace`）
def _src_anchors() -> tuple[int, int, int]:
    lines = SRC.read_text().splitlines()
    begin = next(i for i, ln in enumerate(lines) if ln == "namespace {")
    entry = next(i for i, ln in enumerate(lines) if ln.startswith("__global__ __mix__(1, 2) void m13_moe_layer_kernel("))
    end = max(i for i, ln in enumerate(lines[:entry]) if ln == "}  // namespace")
    return begin, end, entry

PROLOGUE = '''/**
 * m15_moe_layer.h —— MoE 段（S1-S10）device 代码，融合进 m15 per-layer kernel 的版本
 *
 * **本文件由 lift_moe_segment.py 机械生成，不要手改**（改上游 m13 后重跑脚本，再核对
 * --check 输出的差异表）。抽取范围 = m13_moe_layer/m13_moe_layer.asc 的 **device 段**
 * （**内容锚点**：首个顶格 `namespace {` 到入口前最后一个 `}  // namespace`），
 * **除下面列出的 8 类替换之外，算件代码逐行相同**（其中第 1–7 类是重命名式/插入式机械替换；
 * **第 8 类是整块替换，非机械替换**）：
 *   namespace { → namespace M15M { / 删 using namespace M13; / 资源表 include 改名 / M13→M15M /
 *   出口 y 独立成 GM 缓冲 / 紧凑专家槽 + (expert,mTile,nBlock) 工作项 /
 *   **规模无关前置修复（M84）**：S3 标量数组 → UB 静态槽位 + 诊断槽 GM 下标（#5/#6）、
 *   unpermute K 链与实例展开到 10（#7）、unpermute Y 视图上界写成 TOTAL_MAX（#8）/
 *   **S3 诊断槽次序修复（M105）**：`IndexGenStage::Run` 的 UB 标量写
 *   `bUb[32] = oobCountUb;` 从 `BufAcquire<PIPE_MTE3>` **之后**提前到写侧 acquire **之前**，
 *   并把写侧释放 `MutexUnlock<PIPE_S>`（mode 0）换成 **`BufRelease<PIPE_S>`（mode=false = CANN `ASC_LOCK_BLOCK` 默认，与 acquire 同模式）** ——
 *   成对形态见 `docs/05` §6.1 ⓔ（不动地址、不加 set/wait flag）/
 *   **router 权重流式（M91-#3，整块替换）**：`PrecastWeights`（`[NUM_EXPERTS+1][HIDDEN]` 全量
 *   预转 fp32 常驻 UB）→ `LoadWRow`/`PrecastWRow`/`PrecastSgateW`（bf16 行 ping/pong + `RT_EGRP`
 *   行 fp32 窗）；`GemvRow` → `PadLogitsRow`/`SgateRows`/`GemvGroupRow`；`ComputeBlock` 重组。
 *   **⇒ 对 S2 router 而言「与 m13 逐行相同」不成立**（该段是整块重写）；它的算术不变性（同一
 *   chunk 序 / 同一 `Reduce SUM` / 同一 Interleave 树，8 路树 lane 0..3 == 旧 4 路树）由
 *   `moe_relift/m91_README.md` §#3 的论证 + E=4 的 A/B dump 逐字节对拍见证。
 * 另有 **1 条上游不变量断言**（不做替换）：S2 的 `Extract` 已由上游 VF 化、`Sort32` 带依据注释
 * —— 见 lift_moe_segment.py 的 `assert_upstream_vf()` 与 README「本次重抽的位级影响面」。
 *
 * 相对「m13 原版自成 kernel」的**接口差异**（不在本文件里，在 m15_layer_kernel.h）：
 *   1. 不再有 `__global__ m13_moe_layer_kernel` 入口；由 m15_layer_kernel_gdn/_attn 在同一
 *      kernel 内先跑 GDN/attention 段、再做相位边界同步、再调用 `M15M::MoeLayerChain`；
 *   2. m / topk 仍是**运行期参数**（M33/discs15 §5.3 ★5 明确要求不要退化成常量）；
 *      sliceMode / stageLimit / subLimit 三个**bring-up 截断开关退化为编译期常量**
 *      （sliceMode=1 层链模式、stageLimit=8/subLimit=9 = 全链），与 m15_gdn_layer.h 对
 *      m14 的处理同款 —— 段序一字未动，只是把死分支折掉；
 *   3. 相位边界（GDN 段 → MoE 段）的全体 AIV mode-0 barrier 写在融合入口里，不在本文件。
 *
 * 段序（与 m13 README §1 完全一致，逐段同步表见 m13_moe_layer/README.md §3）：
 *   S1 m6#1 Add+RMSNorm → S2 router(GEMV→softmax→top-k→renorm) + 共享门 →
 *   S3 计数排序索引生成 → S4 permute → S5 A 侧 MXFP4 量化（routed+shared）→
 *   S6 grouped gate_up GEMM（5 槽位）→ S7 SwiGLU+量化 → S8 grouped down GEMM →
 *   S9a unpermute 加权折叠 → S9b combine（+sigmoid 门）→ S10 m6#2 Add+RMSNorm。
 */

#ifndef M15_MOE_LAYER_H
#define M15_MOE_LAYER_H

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "reg_compute/kernel_reg_compute_intf.h"

#include <cstdint>

#include "m15_moe_resources.h"

'''


def transform(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if line == "namespace {":
            out_lines.append("namespace M15M {")
            continue
        if line.strip() == "using namespace M13;":
            continue
        line = line.replace('#include "m13_resources.h"', '#include "m15_moe_resources.h"')
        # 6g：模板加 ROUTED 形参（走紧凑槽 + active_num 驱动的行数）
        line = line.replace(
            "template <uint32_t K, uint32_t SCAL_STRIDE, bool SWIGLU, bool PAD_SRC, bool PAD_DST>",
            "template <uint32_t K, uint32_t SCAL_STRIDE, bool SWIGLU, bool PAD_SRC, bool PAD_DST,"
            " bool ROUTED = false>   // [M40-6g] ROUTED = 紧凑槽 + counts/active_num 驱动的行数")
        # 规则 5：出口 y 独立成 GM 缓冲（与 m15 对 m14 的同一处改造）
        line = line.replace("    __gm__ uint8_t* xLayer;     // 层输入激活 [M_MAX, HIDDEN] bf16",
                            "    __gm__ uint8_t* xLayer;     // 层输入激活 [M_MAX, HIDDEN] bf16\n"
                            "    __gm__ uint8_t* yLayer;     // S10 出口（M40 改造：写独立 GM 缓冲 = 层间残差流）")
        line = line.replace("norm2.Init(ws + WS_MOE, ws + WS_RES1, ws + WS_YFINAL, ws + WS_RES2, UB_GAMMA2_F32, args.m);",
                            "norm2.Init(ws + WS_MOE, ws + WS_RES1, args.yLayer, ws + WS_RES2, UB_GAMMA2_F32, args.m);")
        # ---- 规则 6：紧凑专家槽 + (expert, mTile, nTile) 工作项 ----
        # (a) inv_slot 编码：padded 行号 → 紧凑行号（= 计数排序里的产出位置 pos）
        line = line.replace("invUb[s] = static_cast<int32_t>(e * M_MAX + (pos - off[e]));",
                            "invUb[s] = static_cast<int32_t>(pos);   // [M40-6a] 紧凑行号（= 计数排序产出位置）")
        # (b)(c) 量化器的 padding 模板实参：routed 路径改 compact
        line = line.replace("VecQuantStage<HIDDEN, GU_SCALE_STRIDE, false, false, true> quantA;",
                            "VecQuantStage<HIDDEN, GU_SCALE_STRIDE, false, false, false, true> quantA;   // [M40-6b] 紧凑目标+active_num 驱动")
        line = line.replace("VecQuantStage<INTER, DN_SCALE_STRIDE, true, true, true> quantH;",
                            "VecQuantStage<INTER, DN_SCALE_STRIDE, true, false, false, true> quantH;   // [M40-6c] 紧凑源+目标")
        # (d)(e) GEMM 槽位基址：固定 M_MAX 行距 → 紧凑前缀和 off[slot]
        line = line.replace("shd ? (ws + WS_AQ_SHD) : (ws + WS_AQ + slot * M_MAX * (HIDDEN / 2)),",
                            "shd ? (ws + WS_AQ_SHD) : (ws + WS_AQ + offGm.GetValue(slot) * (HIDDEN / 2)),   // [M40-6d]")
        line = line.replace("shd ? (ws + WS_AS_SHD) : (ws + WS_AS + slot * M_MAX * GU_SCALE_STRIDE),",
                            "shd ? (ws + WS_AS_SHD) : (ws + WS_AS + offGm.GetValue(slot) * GU_SCALE_STRIDE),")
        line = line.replace("shd ? (ws + WS_GU_SHD) : (ws + WS_GU + slot * M_MAX * GU_N * 2));",
                            "shd ? (ws + WS_GU_SHD) : (ws + WS_GU + offGm.GetValue(slot) * GU_N * 2));")
        line = line.replace("shd ? (ws + WS_HQ_SHD) : (ws + WS_HQ + slot * M_MAX * (INTER / 2)),",
                            "shd ? (ws + WS_HQ_SHD) : (ws + WS_HQ + offGm.GetValue(slot) * (INTER / 2)),   // [M40-6e]")
        line = line.replace("shd ? (ws + WS_HS_SHD) : (ws + WS_HS + slot * M_MAX * DN_SCALE_STRIDE),",
                            "shd ? (ws + WS_HS_SHD) : (ws + WS_HS + offGm.GetValue(slot) * DN_SCALE_STRIDE),")
        line = line.replace("shd ? (ws + WS_Y_SHD) : (ws + WS_Y + slot * M_MAX * HIDDEN * 2));",
                            "shd ? (ws + WS_Y_SHD) : (ws + WS_Y + offGm.GetValue(slot) * HIDDEN * 2));")
        # ---- 规则 7（M84-5/#6）：S3 的标量槽位常量紧贴其使用点（class IndexGenStage）插入 ----
        if line == "class IndexGenStage {":
            out_lines.extend(SCALAR_SLOT_CONSTS.splitlines())
        line = re.sub(r"\bM13\b", "M15M", line)
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


OFF_DECL_OLD = """        AscendC::GlobalTensor<uint32_t> countsGm;
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ws + WS_COUNTS), NUM_EXPERTS);
"""
OFF_DECL_NEW = """        AscendC::GlobalTensor<uint32_t> countsGm;
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ws + WS_COUNTS), NUM_EXPERTS);
        // [M40-6f] 紧凑槽起点（= S3 产出的 expert_offsets 前缀和）；absent 时由 counts 推算
        AscendC::GlobalTensor<uint32_t> offGm;
        offGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ws + WS_OFFSETS), NUM_EXPERTS + 1);
"""

# (f) 工作项 = (expert, mTile, nBlock)，上界由 counts/active_num 驱动
# (f) 工作项 = (expert, mTile, nBlock)：用正则整体替换两组 GEMM 的 item 枚举循环
GEMM_LOOP_RE = re.compile(
    r"        for \(uint32_t item = bid; item < (?P<items>[A-Z]+_ITEMS); item \+= numBlocks\) \{"
    r".*?\n        \}\n",
    re.S,
)


RUN_HEAD = (
    "        uint32_t startOff[NUM_EXPERTS + 1];\n"
    "        startOff[0] = 0;\n"
    "        uint32_t rows = M;\n"
)
RUN_TAIL = "            const uint32_t dstRow = PAD_DST ? (slot * M_MAX + lr) : r;\n"
RUN_NEW = (
    "        uint32_t startOff[NUM_EXPERTS + 1];\n"
    "        startOff[0] = 0;\n"
    "        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {\n"
    "            startOff[e + 1] = startOff[e] + static_cast<uint32_t>(countsGm.GetValue(e));\n"
    "        }\n"
    "        // [M40-6g] 行数由 counts / active_num 驱动：紧凑槽下 routed 路的行数 = Σt_e\n"
    "        //（等于 m*topk，但**不写成常量**，也不依赖 NUM_EXPERTS 的取值）；共享路仍是 m 行。\n"
    "        uint32_t rows = M;\n"
    "        if constexpr (ROUTED) {\n"
    "            rows = startOff[NUM_EXPERTS];\n"
    "        }\n"
    "        for (uint32_t r = bid; r < rows; r += nAiv) {\n"
    "            uint32_t slot = 0;\n"
    "            uint32_t lr = r;\n"
    "            if constexpr (ROUTED || PAD_DST) {\n"
    "                while (slot < NUM_EXPERTS - 1 && r >= startOff[slot + 1]) {\n"
    "                    ++slot;\n"
    "                }\n"
    "                lr = r - startOff[slot];\n"
    "            }\n"
    "            const uint32_t srcRow = PAD_SRC ? (slot * M_MAX + lr) : r;\n"
    "            const uint32_t dstRow = ROUTED ? r : (PAD_DST ? (slot * M_MAX + lr) : r);\n"
)


def rewrite_quant_run(text: str) -> str:
    """[M40-6g] 量化器的行数与槽位解码改为 counts/active_num 驱动（多行替换）。"""
    i = text.index(RUN_HEAD)
    j = text.index(RUN_TAIL, i) + len(RUN_TAIL)
    return text[:i] + RUN_NEW + text[j:]


# ---- 规则 7 的替代：断言上游已合规，**不做替换** ----
#   M40 曾在**本脚本里**手写一份 Extract→VF 的等价替换（旧规则 7）。它的目标文本随 M50 在 m13
#   落地而失效（重抽直接 AssertionError），且它的产物与上游形态分叉（多了一处「不落 UB 再读回」
#   的差异）。现在改为「断言上游已经合规」：副本恒等于上游的当前形态；上游若退回 memory-based
#   `Extract` 或缺 `Sort32` 依据注释，重抽立刻失败，而不是静默落后于 donor。
CLASSIC_EXTRACT_RE = re.compile(r"\bExtract\s*\(")
DINTLV_VF_MARK = "LoadAlign<float, LoadDist::DIST_DINTLV_B32>(vr, vi, pairUb);"
SORT32_BASIS_MARKS = (
    "Sort32 属**裁定例外**",                        # 白名单例外的声明（docs/05 §6.1 规则 ⓒ）
    "docs/05 §6.1",                                 # 依据出处（章节标签，不写行号）
    "reg_compute/**` 穷举无 Sort32",                # 「Reg 侧无等价物」这一条
    "moe_gating_top_k_softmax_v2_perf_arch35.h",    # 官方 donor（同为 memory-based）
)


def code_only(text: str) -> str:
    """剥掉行尾注释后只留代码。

    必须这么做：m13 的注释里就有「经典 `Extract(..., 1)`」这种**提法**（讲改造来历时写的），
    直接对全文匹配会把注释误判成调用 —— 报出假失败。
    """
    return "\n".join(ln.split("//", 1)[0] for ln in text.splitlines())


def assert_upstream_vf(text: str) -> None:
    """断言上游 m13 的 S2 已 VF 化、且 Sort32 带依据注释（旧规则 7 的替代）。"""
    assert not CLASSIC_EXTRACT_RE.search(code_only(text)), (
        "上游 m13 的 device 段出现经典 memory-based `Extract(` 调用 —— tower M44 裁决认定 "
        "`Extract` 不入例外，副本会落后于 donor")
    assert DINTLV_VF_MARK in text, (
        "上游 m13 的 S2 未按厂商 VF 形态内联 Extract（缺 DINTLV_B32 解交织 + 掩码 StoreAlign）")
    missing = [m for m in SORT32_BASIS_MARKS if m not in text]
    assert not missing, f"上游 m13 的 Sort32 调用点缺依据注释片段：{missing}"


def rewrite_gemm_loop(text: str, items: str, nblk: str) -> str:
    """把 `for (item...) { ... }` 的扁平工作项替换为 (expert, mTile, nBlock) 三重循环。

    原循环体里除 `SetItem/Run` 外的部分（slot/nb 解码、空槽跳过）都被新的循环结构取代；
    `SetItem` 的五个实参原样保留（槽位基址已在单行替换里改成紧凑前缀和 `rowBase`），
    `Run(nb, t)` → `Run(nb, rows)`（curM = 本 mTile 的行数）。
    """
    m = GEMM_LOOP_RE.search(text)
    assert m and m.group("items") == items, f"{items} 循环未找到"
    body = m.group(0)
    mm = re.search(
        r"(\s*)(gemm[A-Za-z]+\.SetItem\(.*?\);)\s*\n\s*(gemm[A-Za-z]+\.Run\(nb, t\);)", body, re.S
    )
    assert mm, f"{items} 的 SetItem/Run 未找到"
    call = re.sub(r"\n\s+", "\n                        ", (mm.group(2) + "\n" + mm.group(3)).strip())
    call = call.replace("Run(nb, t)", "Run(nb, rows)")
    gemm = re.search(r"gemm[A-Za-z]+", mm.group(2)).group(0)
    call = call.replace(gemm + ".Run(nb, t)", gemm + ".Run(nb, rows)")
    call = call.replace("offGm.GetValue(slot) * ", "rowBase * ")
    call = call.replace("(ws + WS_AQ_SHD)", "(ws + WS_AQ_SHD + rowBase * (HIDDEN / 2))")
    call = call.replace("(ws + WS_AS_SHD)", "(ws + WS_AS_SHD + rowBase * GU_SCALE_STRIDE)")
    call = call.replace("(ws + WS_GU_SHD)", "(ws + WS_GU_SHD + rowBase * GU_N * 2)")
    call = call.replace("(ws + WS_HQ_SHD)", "(ws + WS_HQ_SHD + rowBase * (INTER / 2))")
    call = call.replace("(ws + WS_HS_SHD)", "(ws + WS_HS_SHD + rowBase * DN_SCALE_STRIDE)")
    call = call.replace("(ws + WS_Y_SHD)", "(ws + WS_Y_SHD + rowBase * HIDDEN * 2)")
    call = call.replace("(shd ? p.wGuShd :", "(shd ? p.wGuShd :")
    call = call.replace("Run(nb, t)", "Run(nb, rows)")
    new_body = (
        "        // [M40-6f] 工作项 = (expert, mTile, nBlock)：空专家由 counts 跳过（count 驱动）；\n"
        "        // mTile 数由该专家的 t_e 推出（本实例 t_e <= BASE_M ⇒ mTiles 恒 1；prefill 自动分段）。\n"
        "        for (uint32_t slot = 0; slot < NUM_SLOTS; ++slot) {\n"
        "            const bool shd = (slot == NUM_EXPERTS);\n"
        "            const uint32_t t = shd ? p.m : countsGm.GetValue(slot);\n"
        "            if (t == 0) {\n"
        "                continue;   // 空专家：A/H 区未被写，无需计算\n"
        "            }\n"
        "            const uint32_t mTiles = (t + BASE_M - 1) / BASE_M;\n"
        "            for (uint32_t mt = 0; mt < mTiles; ++mt) {\n"
        "                const uint32_t rows = ((t - mt * BASE_M) < BASE_M) ? (t - mt * BASE_M) : BASE_M;\n"
        "                const uint32_t rowBase = shd ? (mt * BASE_M) : (offGm.GetValue(slot) + mt * BASE_M);\n"
        "                for (uint32_t nb = bid; nb < " + nblk + "; nb += numBlocks) {\n"
        "                    " + call + "\n"
        "                }\n"
        "            }\n"
        "        }\n"
    )
    return text.replace(body, new_body)


# ============================================================
# 规则 7（M84）：规模无关前置修复（= M77 §2 的 #5/#6/#7/#8）
#   只修「在 E=4/topk=2 下是 no-op、在真实规模 E=512/topk=10 下失效」的形态；
#   **不动**规模常量、**不动**紧凑 Σt_e 寻址（M40 的硬约束）。
# ============================================================

# ---- #5/#6：S3 标量数组的 UB 静态槽位 + 诊断槽 GM 下标（插在 class IndexGenStage 之前）----
SCALAR_SLOT_CONSTS = """\
// ============================================================
// [M84-5] S3 索引生成的标量数组：AIV 标量栈 → UB 静态槽位（规模无关前置修复）
//   原实现 `uint32_t cnt[E]; uint32_t off[E+1]; uint32_t cursor[E];` 放在 AIV **标量栈**上：
//   E=4 时 3×16B 无感；E=512 时 3×2KB = 6KB 压垮 AIV 标量栈 ⇒ 改 UB 静态槽位。
//   槽位从 S3 自己的 router-x-块窗尾部（UB_IG_END）起算，**尺寸按 NUM_EXPERTS 定尺**、
//   随 E 自动增长；counts 与 offsets 起始各按 32B 对齐（DataCopyPad 源要求，M9 quirk），
//   E=4 时 offsets 起始 = +32B（与改前的 cntUb[8] 同一字节）。
// ============================================================
constexpr uint32_t UB_IG_SCAL = UB_IG_END;                                       // i32 counts[NUM_EXPERTS]
constexpr uint32_t UB_IG_OFF = UB_IG_SCAL + AlignUp(NUM_EXPERTS * 4, 32);        // i32 offsets[E+1]（32B 对齐）
constexpr uint32_t UB_IG_CUR = UB_IG_OFF + AlignUp((NUM_EXPERTS + 1) * 4, 32);   // i32 cursor[NUM_EXPERTS]
constexpr uint32_t UB_IG_SCAL_END = UB_IG_CUR + AlignUp(NUM_EXPERTS * 4, 32);
static_assert(UB_IG_SCAL_END <= UB_RT_XB + RT_RB * HIDDEN * 2, "S3 标量槽超出 router x 块区");

// ============================================================
// [M84-6] S3 诊断槽的 GM 目的下标：必须是 SZ_OFFSETS 尾部 32B 槽的**起始 i32 下标**
//   （即恒在真 offsets[0..E] 之后）。m13 写死 `offs[8]`：E=4 时它恰是槽起点，
//   但 E=512 时 offs[8] 是**真偏移**、会被诊断槽覆盖 ⇒ 由 SZ_OFFSETS 的布局反推：
//   E 为 8 的倍数时 = E+8（真实 E=512 ⇒ 520）；E=4 时 = 8（与改前同一字节）。
// ============================================================
constexpr uint32_t IG_DIAG_GM_SLOT = AlignUp((NUM_EXPERTS + 1) * 4 + 4, WS_ALIGN) / 4;
// offsGm 视图长度（额外 int32 数）：**由 SZ_OFFSETS 反推**，视图恒等于底层 expert_offsets 区。
//   donor m17 用的是写死的 `IG_DIAG_SLOT_END = 16`（`m17_resources.h:190`）——E=4 时
//   `NUM_EXPERTS+16 = 20` 个 int32 = 80 B 会**超出**底层 `SZ_OFFSETS = 64 B`（今天只写
//   index ≤ 8 故无实害，形态上是惰性；E=512 时两者恰好相等 528×4 = 2112）。这里改成派生量：
//   E=4 → 12（视图 16×4 = 64 B = SZ_OFFSETS）、E=512 → 16（视图 528×4 = 2112）。
constexpr uint32_t IG_DIAG_SLOT_END = SZ_OFFSETS / 4 - NUM_EXPERTS;
static_assert(IG_DIAG_GM_SLOT >= NUM_EXPERTS + 1, "诊断槽必须落在真 offsets 之后");
static_assert((IG_DIAG_GM_SLOT + 1) * 4 <= SZ_OFFSETS, "诊断槽超出 expert_offsets GM 区");
static_assert(IG_DIAG_GM_SLOT + 1 <= NUM_EXPERTS + IG_DIAG_SLOT_END, "诊断槽超出 offsGm 视图");

"""

IDX_HEAD_OLD = """\
        uint32_t cnt[NUM_EXPERTS];
        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
            cnt[e] = 0;
        }
        for (uint32_t s = 0; s < S; ++s) {
            const int32_t e = idsGm.GetValue(s);
            if (e < 0 || e >= static_cast<int32_t>(NUM_EXPERTS)) {
                ++badIds;   // 防御：脏 ids 时不做越界写（会踩栈/UB）
                continue;
            }
            cnt[e] += 1;
        }
        oobCountUb = badIds;   // 诊断槽（badIds != 0 时下面的落位循环整体短路）
        uint32_t off[NUM_EXPERTS + 1];
        off[0] = 0;
        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
            off[e + 1] = off[e] + cnt[e];
        }
"""

IDX_HEAD_NEW = """\
        // [M84-5] cnt/off/cursor 由标量栈数组改为 UB 静态槽位（E=512 时 3×2KB 压垮标量栈）
        __ubuf__ int32_t* cntUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_SCAL);
        __ubuf__ int32_t* offUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_OFF);
        __ubuf__ int32_t* curUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_CUR);
        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
            cntUb[e] = 0;
        }
        for (uint32_t s = 0; s < S; ++s) {
            const int32_t e = idsGm.GetValue(s);
            if (e < 0 || e >= static_cast<int32_t>(NUM_EXPERTS)) {
                ++badIds;   // 防御：脏 ids 时不做越界写
                continue;
            }
            cntUb[e] = cntUb[e] + 1;
        }
        oobCountUb = badIds;   // 诊断槽（badIds != 0 时下面的落位循环整体短路）
        // [M105] 诊断槽的 UB 标量写必须落在 MTE3 那条读的**次序 token 之内**。改前的形态是：
        //   「紧邻最后那次 `DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], ExtBlock1(4))`」，却排在
        //   它的 `BufAcquire<PIPE_MTE3>(BUF_AIV_IDX)` **之后** ⇒ 落在 token 之外、与那条读抢跑
        //   （可能落到该 UB 位置 `UB_IG_CNT+128` —— 落在 router x 行 0 窗内 —— 的残值；实测两种
        //   落点都出现过，见 `evidence/moe_race/slot_bytes.txt`）。本处把它提前到写侧 acquire 之前。
        //   次序由**写侧的 release** 建立：`BufRelease<PIPE_S>(BUF_AIV_IDX)`
        //   （= `RlsBufInternal<PIPE_S,false>`，mode=false = CANN `ASC_LOCK_BLOCK` 默认「阻塞」；
        //   两种模式都等本 pipe 已发射指令落地，`true`（`NON_BLOCK`）额外等此前同 id 的释放 —— 更保守），
        //   成对形态 = 本仓 docs/05 §6.1 ⓔ「写 → 写侧 acquire → 写侧 release
        //   → 搬侧 acquire → 搬」（本次的写落在写侧 acquire **之前**；ⓔ 的覆盖面前提只要求写在该
        //   release 之前）。同款的已受审站点（在 `main` 上核过）：`m15_ple.asc` 的 IDS/BODY；
        //   m17 的 `IndexGenStage::Run` 是同一诊断的先例。
        //   见 `evidence/moe_race/README.md` §2①/§2④ 与 `mechanism_note.md`。
        __ubuf__ int32_t* diagUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_CNT);
        diagUb[32] = static_cast<int32_t>(oobCountUb);
        offUb[0] = 0;
        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
            offUb[e + 1] = offUb[e] + cntUb[e];
        }
"""

IDX_MID_OLD = """\
        __ubuf__ int32_t* cntUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_CNT);

        MutexLock<PIPE_S>(BUF_AIV_IDX);
        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
            cntUb[e] = static_cast<int32_t>(cnt[e]);
        }
        for (uint32_t e = 0; e <= NUM_EXPERTS; ++e) {
            cntUb[8 + e] = static_cast<int32_t>(off[e]);   // +32B 对齐槽（DataCopyPad 源需 32B 对齐）
        }
        uint32_t cursor[NUM_EXPERTS];
        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
            cursor[e] = off[e];
        }
"""

IDX_MID_NEW = ("\n"
               "        MutexLock<PIPE_S>(BUF_AIV_IDX);\n"
               "        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {\n"
               "            curUb[e] = offUb[e];\n"
               "        }\n")

IDX_CURSOR_OLD = """\
                const uint32_t pos = cursor[e];
                cursor[e] = pos + 1;
"""

IDX_CURSOR_NEW = """\
                const uint32_t pos = static_cast<uint32_t>(curUb[e]);
                curUb[e] = static_cast<int32_t>(pos + 1);
"""

IDX_TENSOR_OLD = "        LocalTensor<int32_t> cntL(TPosition::VECCALC, UB_IG_CNT, NUM_EXPERTS + 1);\n"
IDX_TENSOR_NEW = ("        LocalTensor<int32_t> cntL(TPosition::VECCALC, UB_IG_SCAL, NUM_EXPERTS);\n"
                  "        LocalTensor<int32_t> offL(TPosition::VECCALC, UB_IG_OFF, NUM_EXPERTS + 1);\n")

IDX_OFFCOPY_OLD = "        DataCopyPad(offsGm[0], cntL[8], ExtBlock1((NUM_EXPERTS + 1) * 4));\n"
IDX_OFFCOPY_NEW = "        DataCopyPad(offsGm[0], offL, ExtBlock1((NUM_EXPERTS + 1) * 4));\n"

IDX_DIAG_OLD = """\
            // 诊断槽：UB 源必须 32B 对齐（+128B），GM 目的也用 32B 对齐槽（offs[8]）
            __ubuf__ int32_t* bUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_CNT);
            bUb[32] = static_cast<int32_t>(oobCountUb);
            LocalTensor<int32_t> bL(TPosition::VECCALC, UB_IG_CNT, 64);
            DataCopyPad(offsGm[8], bL[32], ExtBlock1(4));
"""

IDX_DIAG_NEW = """\
            // [M84-6] 诊断槽：UB 源 32B 对齐（+128B）；GM 目的放 IG_DIAG_GM_SLOT
            //   （= SZ_OFFSETS 尾部 32B 槽起点，恒在真 offsets[0..E] 之后）
            // [M105] 这一读的 UB 源那 4 B 的**标量写**已提前到写侧 acquire 之前
            //   （见本函数开头 `diagUb[32] = …`）—— 此处只留 MTE3 那一读；改前写在这一读紧邻处、
            //   却排在 `BufAcquire<PIPE_MTE3>` **之后**（次序 token 之外），与它无次序保证。
            //   次序现由写侧 release（mode=false）`BufRelease<PIPE_S>(BUF_AIV_IDX)` 建立（见该行注释）。
            LocalTensor<int32_t> bL(TPosition::VECCALC, UB_IG_CNT, 64);
            DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], ExtBlock1(4));
"""

# ---- [M105] 写侧释放：`MutexUnlock<PIPE_S>`（mode 0）→ `BufRelease<PIPE_S>`（mode=false，与 acquire 同模式）----
IDX_RELEASE_OLD = "        MutexUnlock<PIPE_S>(BUF_AIV_IDX);\n"
IDX_RELEASE_NEW = """\
        // [M105] 写侧释放：`BufRelease<PIPE_S>` = `RlsBufInternal<PIPE_S,false>`
        //   —— mode=false = CANN `ASC_LOCK_BLOCK` 默认「阻塞」；`true` 为 `NON_BLOCK`。两种模式都等本
        //   pipe 已发射指令落地，`true` 额外等此前同 id 的释放（更保守）。此处与 acquire 侧同模式。
        //   改前这里是 `MutexUnlock<PIPE_S>`（= `RlsBufInternal<PIPE_S,0>`，同为 mode 0；
        //   见 docs/05 §6 与 CANN kernel_common.h:138-160）。
        //   写侧 acquire 仍是 `MutexLock<PIPE_S>`（= `GetBufInternal<PIPE_S,0>`，与 `BufAcquire<PIPE_S>`
        //   是同一 token 获取）⇒ 成对不破（只放 release 而缺写侧 acquire 会挂死，见 `m15_ple.asc`
        //   文件头 §BUF 的教训）。
        BufRelease<PIPE_S>(BUF_AIV_IDX);
"""

IDX_OFFS_VIEW_OLD = "        offsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets), NUM_EXPERTS + 1);\n"
IDX_OFFS_VIEW_NEW = ("        offsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets),"
                     " NUM_EXPERTS + IG_DIAG_SLOT_END);\n")

# ---- #7：unpermute 的 K 链与模板实例展开到 10 ----
FMA_OLD = "                if constexpr (TK > 3) FmaChunk<3>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);\n"
FMA_NEW = FMA_OLD + "".join(
    f"                if constexpr (TK > {k}) FmaChunk<{k}>(acc, yF, yB16, wRaw, wF, yUb, wRowUb,"
    f" maskAll, off);\n" for k in range(4, 10))

SWITCH_OLD = """\
            case 3: unperm3.Run(bid, nAiv); break;
            default: unperm4.Run(bid, nAiv); break;
"""
SWITCH_NEW = """\
            case 3: unperm3.Run(bid, nAiv); break;
            case 4: unperm4.Run(bid, nAiv); break;
            case 5: unperm5.Run(bid, nAiv); break;
            case 6: unperm6.Run(bid, nAiv); break;
            case 7: unperm7.Run(bid, nAiv); break;
            case 8: unperm8.Run(bid, nAiv); break;
            case 9: unperm9.Run(bid, nAiv); break;
            default: unperm10.Run(bid, nAiv); break;
"""

INIT_OLD = "        unperm4.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);\n"
INIT_NEW = "".join(
    f"        unperm{k}.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);\n"
    for k in range(4, 11))

MEMBER_OLD = "    UnpermuteStage<4> unperm4;\n"
MEMBER_NEW = "".join(f"    UnpermuteStage<{k}> unperm{k};\n" for k in range(4, 11))

UNPERM_COMMENT_OLD = "    // topK 运行时 → 模板分发（m8#2 同款：k 链全展开，下标皆编译期）\n"
UNPERM_COMMENT_NEW = (
    "    // topK 运行时 → 模板分发（m8#2 同款：k 链全展开，下标皆编译期）\n"
    "    // [M84-7] 真实 topk=10 ⇒ 展开到 1..9 + <10>（m13 只到 1..3 / <1..4>）\n")

# ---- #8：unpermute 的 Y 视图上界写成紧凑 Σt_e 上界 ----
YVIEW_OLD = ("        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ySorted),"
             " static_cast<uint64_t>(M_MAX * TOPK_MAX) * HIDDEN);\n")
YVIEW_NEW = ("        // [M84-8] Y 视图上界 = 紧凑 Σt_e 上界 TOTAL_MAX 行（**不是** padded 的\n"
             "        //   NUM_EXPERTS*M_MAX：两者在 E=4 相等只是巧合，E=512 时不等）\n"
             "        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ySorted),"
             " static_cast<uint64_t>(TOTAL_MAX) * HIDDEN);\n")


def _must_replace(text: str, old: str, new: str, tag: str) -> str:
    """规则 7 的单点替换：目标必须**存在且唯一**，否则上游文本已漂移、直接失败。"""
    n = text.count(old)
    assert n == 1, f"[M84] 规则 7 的目标不唯一/未找到（{tag}）：{n} 处"
    return text.replace(old, new)


# ============================================================
# 规则 8（M91-#3）：router 权重「[E+1][HIDDEN] 全量预转 UB 常驻」→ m17 式**流式**
#   形态来源 = `m17_moe_real/m17_moe_layer.asc` 的 `RouterStage`（x 行块常驻 + 权重行 ping/pong
#   + RT_EGRP 个专家一组的 fp32 预转窗），**改抄自 m15 自己的同名函数**（`PrecastWeights` /
#   `GemvRow`），只改「槽位从哪里来、何时搬」：
#     · `PrecastWeights`（E+1 行一次预转常驻）→ `LoadWRow`/`PrecastWRow`/`PrecastSgateW`（流式）
#     · `GemvRow`（5 个累加器、x 行主序）→ `PadLogitsRow` + `SgateRows` + `GemvGroupRow`
#   算术**一字未改**（同一 chunk 数 CHUNKS_H、同一 `MulAddDst` 累加序、同一 `Reduce SUM`、
#   同一 `Interleave` 打包树的前 4 个 lane ⇒ E=4 时逐位相同 —— 见 moe_relift/m91_README.md）。
#   UB 常驻量从 `(NUM_EXPERTS+1)` 行降到 `RT_EGRP` 行 + 1 行 + 2 个 bf16 行（与 E 解耦）。
# ============================================================

RT_RB_MEMBER_OLD = "    static constexpr uint32_t RT_RB = 16;   // row-block（m ≤ 64 → 最多 4 块；m7 同款结构）\n"
RT_NROWS_OLD = "constexpr uint32_t RT_NROWS = NUM_EXPERTS + 1;    // 路由专家 + 共享门\n"
PRECAST_CALL_OLD = """        if (subLimit >= 1) {
            PrecastWeights();
        }
"""

PRECAST_OLD_BEGIN = "    // 权重行 bf16 → fp32 预转（RT_NROWS 行；ping-pong UB）\n"
INDEX_TMPL_BEGIN = "    // 索引模板 0..31（Arange 只填 64 lane → 取低 32 lane）\n"
COMPUTEBLK_OLD_BEGIN = (
    "    // 整块 V 段：x 行全读 + staging 全写 都在同一 span 内，块尾一次性 drain 释放\n")
GEMVROW_OLD_BEGIN = "    // 行内 GEMV（读 UB_RT_XB 的第 r 行）→ logits 行写入 staging\n"
SOFTMAX_BEGIN = "    // 行内 softmax(max-shift) + Sort32 + 前 32 对拆分（Extract 的 VF 内联）+ renorm → staging\n"

_PRECAST_HEAD = '''    // ---- [M91-#3] 权重流式：只保留 RT_EGRP 行 fp32 窗常驻（取用协议抄 m17_moe_real 的 Gemv）----
    // 改前把 [NUM_EXPERTS+1][HIDDEN] 全部预转 fp32 常驻 UB（E=4 → 51,200 B；E=512 → 5.25 MB 装不下）。
    // 现按「RT_EGRP 个专家一组」流式：ping/pong 两个 bf16 行缓冲 + 一个 RT_EGRP 行 fp32 窗，
    // 组内 RT_EGRP 个专家的点积复用同一份 x 载入 ⇒ UB 常驻量与 NUM_EXPERTS **解耦**。
    // 协议：进第一组前 W0=w[0]、W1=w[1]；槽 j 预转 w[e0+j]（缓冲 j%2），预转后立刻把
    // w[e0+j+RT_EGRP] 装进该缓冲（最后一槽即下一组的预取）。
    // 越界槽（e0+j >= NUM_EXPERTS）用 `e % NUM_EXPERTS` 的真实行填充：必须是**真实搬运** ——
    // 本实现按「Acquire/Release 一律成对」使用（与 m17 `LoadW` 的早退形态不同：两种写法都合法，
    // 这里取更保守的那一侧；`GetBufInternal<pipe,false>` 展开到 `get_buf(pipe,bufId,mode)`，那个
    // mode 是 ping/pong 选择位，从 CANN 头文件无法断定其是否阻塞 —— 复审核过这一点），
    // 且不得读越权重区。
    // 这些槽算出的值不落盘（StoreAlign 的 ng 掩码只写本组有效 lane）。
    __aicore__ inline void LoadWRow(uint32_t e, uint32_t bufSel)
    {
        const uint32_t row = (e < NUM_EXPERTS) ? e : (e % NUM_EXPERTS);
        const uint32_t wbOff = bufSel ? UB_RT_WB1 : UB_RT_WB0;
        LocalTensor<bfloat16_t> wbL(TPosition::VECCALC, wbOff, HIDDEN);
        BufAcquire<PIPE_MTE2>(bufSel ? BUF_AIV_WST1 : BUF_AIV_WST0);
        PipeBarrier<PIPE_MTE2>();
        DataCopy(wbL, rwGm[static_cast<uint64_t>(row) * HIDDEN], Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE2>(bufSel ? BUF_AIV_WST1 : BUF_AIV_WST0);
    }

    // 单个槽：bf16 → fp32 预转（写 RT_WF 的 slot 槽）；V 取用/归还该缓冲的 token
    __aicore__ inline void PrecastWRow(uint32_t bufSel, uint32_t slot)
    {
        const uint32_t wbOff = bufSel ? UB_RT_WB1 : UB_RT_WB0;
        __ubuf__ bfloat16_t* wSrc = reinterpret_cast<__ubuf__ bfloat16_t*>(wbOff);
        __ubuf__ float* wDst = reinterpret_cast<__ubuf__ float*>(UB_RT_WF + slot * HIDDEN * 4);
        BufAcquire<PIPE_V>(bufSel ? BUF_AIV_WST1 : BUF_AIV_WST0);
        __VEC_SCOPE__
        {
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t c = 0; c < CHUNKS_H; ++c) {
                RegTensor<bfloat16_t> b0;
                RegTensor<float> f0;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(b0, wSrc + c * VL_F32);
                Cast<float, bfloat16_t, castTraitB162B32>(f0, b0, maskAll);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(wDst + c * VL_F32, f0, maskAll);
            }
        }
        BufRelease<PIPE_V>(bufSel ? BUF_AIV_WST1 : BUF_AIV_WST0);
    }

    // 共享门权重（1 行）：独立 UB 槽（RT_SGB）预转一次常驻 RT_SGWF；token 用 BUF_AIV_STG
    // （与 router 权重的 ping/pong 分开，原因同 m17：复用会让 MTE2 写与 V 读落同一片 UB 而无 token）
    __aicore__ inline void PrecastSgateW()
    {
        LocalTensor<bfloat16_t> wbL(TPosition::VECCALC, UB_RT_SGB, HIDDEN);
        BufAcquire<PIPE_MTE2>(BUF_AIV_STG);
        PipeBarrier<PIPE_MTE2>();
        DataCopy(wbL, sgGm, Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE2>(BUF_AIV_STG);
        __ubuf__ bfloat16_t* wSrc = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_RT_SGB);
        __ubuf__ float* wDst = reinterpret_cast<__ubuf__ float*>(UB_RT_SGWF);
        BufAcquire<PIPE_V>(BUF_AIV_STG);
        __VEC_SCOPE__
        {
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t c = 0; c < CHUNKS_H; ++c) {
                RegTensor<bfloat16_t> b0;
                RegTensor<float> f0;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(b0, wSrc + c * VL_F32);
                Cast<float, bfloat16_t, castTraitB162B32>(f0, b0, maskAll);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(wDst + c * VL_F32, f0, maskAll);
            }
        }
        BufRelease<PIPE_V>(BUF_AIV_STG);
    }

'''

_COMPUTEBLK_NEW = '''    // 整块 V 段：x 行全读 + staging 全写 都在同一 span 内，块尾一次性 release（mode=false）
    // [M91-#3] 顺序：① 对数行先整行 pad 写 RT_NEG_BIG（组循环只写本组有效 lane ⇒ pad 必须
    //   先铺满整行）② 共享门权重预转 + 逐行裸点积（值在 lane0 → UB_RT_SG）③ 权重流式：
    //   RT_EGRP 个专家一组，组内逐行点积写对数行 ④ 逐行 softmax + top-k。
    __aicore__ inline void ComputeBlock(uint32_t b0, uint32_t rows)
    {
        BufAcquire<PIPE_V>(BUF_AIV_ROW0);   // 等整块 x 行就位
        BufAcquire<PIPE_V>(BUF_AIV_ROW1);   // staging（等上一块的 MTE3 拷走）
        for (uint32_t r = 0; r < rows; ++r) {
            PadLogitsRow(r);
        }
        PrecastSgateW();
        SgateRows(rows);
        LoadWRow(0, 0);
        LoadWRow(1, 1);
        for (uint32_t e0 = 0; e0 < NUM_EXPERTS; e0 += RT_EGRP) {
            for (uint32_t j = 0; j < RT_EGRP; ++j) {
                PrecastWRow(j & 1, j);
                LoadWRow(e0 + j + 2, j & 1);
            }
            for (uint32_t r = 0; r < rows; ++r) {
                GemvGroupRow(r, e0);
            }
        }
        for (uint32_t r = 0; r < rows; ++r) {
            SoftmaxTopkRow(r, b0 + r);
        }
        BufRelease<PIPE_V>(BUF_AIV_ROW0);   // x 块全部消费完 → MTE2 可复用
        BufRelease<PIPE_V>(BUF_AIV_ROW1);   // staging 写完 → MTE3 可读
    }

'''

_GEMVROW_NEW = '''    // [M91-#3] 对数行 pad：整行 64 lane 写 RT_NEG_BIG。改前的同值由 GemvRow 打包后的
    //   `Select(rowReg, v4, negInf, mE)` 写（lane >= NUM_EXPERTS 全覆盖）——同一常量、同一范围，
    //   只是搬到组循环之前（组循环只写 [e0, e0+ng)）。
    __aicore__ inline void PadLogitsRow(uint32_t r)
    {
        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG + r * 64 * 4);
        __VEC_SCOPE__
        {
            RegTensor<float> negInf;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            Duplicate(negInf, RT_NEG_BIG, maskAll);
            StoreAlign(rowUb, negInf, maskAll);
        }
    }

    // 共享门裸点积（1 个 fp32 累加器；权重在 RT_SGWF 常驻）→ UB_RT_SG 的 lane0
    //   改前它作为第 NUM_EXPERTS 个累加器 a4 与专家共用一份 x 载入；这里改成独立一圈，
    //   但 **chunk 升序的 MulAddDst 累加序与 Reduce SUM 一字未改** ⇒ 逐位相同。
    __aicore__ inline void SgateRows(uint32_t rows)
    {
        for (uint32_t r = 0; r < rows; ++r) {
            __ubuf__ bfloat16_t* xSrc = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_RT_XB + r * HIDDEN * 2);
            __ubuf__ float* dst = reinterpret_cast<__ubuf__ float*>(UB_RT_SG) + r;
            __VEC_SCOPE__
            {
                RegTensor<float> acc;
                RegTensor<float> xf;
                RegTensor<float> w;
                RegTensor<bfloat16_t> xb;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                Duplicate(acc, 0.0f);
                for (uint16_t c = 0; c < CHUNKS_H; ++c) {
                    const uint32_t off = static_cast<uint32_t>(c) * VL_F32;
                    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xb, xSrc + off);
                    Cast<float, bfloat16_t, castTraitB162B32>(xf, xb, maskAll);
                    LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_SGWF) + off);
                    MulAddDst(acc, xf, w, maskAll);
                }
                Reduce<ReduceType::SUM, float>(acc, acc, maskAll);
                uint32_t n1 = 1;
                MaskReg m1 = UpdateMask<float>(n1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dst, acc, m1);
            }
        }
    }

    // 一组 RT_EGRP 个专家的点积（共享 1 次 x 载入）→ 写对数行的 lane [e0, e0+ng)
    //   RT_EGRP 个 fp32 累加器；Reduce SUM 后按 Interleave 树拼成 RT_EGRP 个**连续且有序**的
    //   lane（8 路树的 lane 0..3 == 改前 4 路树的 [l0,l1,l2,l3]，已逐 lane 推过）。
    //   写盘掩码 ng = min(RT_EGRP, NUM_EXPERTS - e0)：E=4 时 = 4 ⇒ lane 4..63 保持 PadLogitsRow
    //   写的 RT_NEG_BIG ⇒ 与改前 `Select(...)` 的产物逐位相同。
    __aicore__ inline void GemvGroupRow(uint32_t r, uint32_t e0)
    {
        __ubuf__ bfloat16_t* xRowUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_RT_XB + r * HIDDEN * 2);
        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG + r * 64 * 4) + e0;
        __VEC_SCOPE__
        {
            RegTensor<float> a0, a1, a2, a3, a4, a5, a6, a7;
            RegTensor<float> xf;
            RegTensor<float> w;
            RegTensor<bfloat16_t> xb;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            Duplicate(a0, 0.0f);
            Duplicate(a1, 0.0f);
            Duplicate(a2, 0.0f);
            Duplicate(a3, 0.0f);
            Duplicate(a4, 0.0f);
            Duplicate(a5, 0.0f);
            Duplicate(a6, 0.0f);
            Duplicate(a7, 0.0f);
            for (uint16_t c = 0; c < CHUNKS_H; ++c) {
                const uint32_t off = static_cast<uint32_t>(c) * VL_F32;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xb, xRowUb + off);
                Cast<float, bfloat16_t, castTraitB162B32>(xf, xb, maskAll);
                LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_WF + 0 * HIDDEN * 4) + off);
                MulAddDst(a0, xf, w, maskAll);
                LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_WF + 1 * HIDDEN * 4) + off);
                MulAddDst(a1, xf, w, maskAll);
                LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_WF + 2 * HIDDEN * 4) + off);
                MulAddDst(a2, xf, w, maskAll);
                LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_WF + 3 * HIDDEN * 4) + off);
                MulAddDst(a3, xf, w, maskAll);
                LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_WF + 4 * HIDDEN * 4) + off);
                MulAddDst(a4, xf, w, maskAll);
                LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_WF + 5 * HIDDEN * 4) + off);
                MulAddDst(a5, xf, w, maskAll);
                LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_WF + 6 * HIDDEN * 4) + off);
                MulAddDst(a6, xf, w, maskAll);
                LoadAlign(w, reinterpret_cast<__ubuf__ float*>(UB_RT_WF + 7 * HIDDEN * 4) + off);
                MulAddDst(a7, xf, w, maskAll);
            }
            Reduce<ReduceType::SUM, float>(a0, a0, maskAll);
            Reduce<ReduceType::SUM, float>(a1, a1, maskAll);
            Reduce<ReduceType::SUM, float>(a2, a2, maskAll);
            Reduce<ReduceType::SUM, float>(a3, a3, maskAll);
            Reduce<ReduceType::SUM, float>(a4, a4, maskAll);
            Reduce<ReduceType::SUM, float>(a5, a5, maskAll);
            Reduce<ReduceType::SUM, float>(a6, a6, maskAll);
            Reduce<ReduceType::SUM, float>(a7, a7, maskAll);
            RegTensor<float> i01, i23, i45, i67, j0, j1, j2, j3, v, vv;
            Interleave(i01, vv, a0, a4);
            Interleave(i23, vv, a2, a6);
            Interleave(i45, vv, a1, a5);
            Interleave(i67, vv, a3, a7);
            Interleave(j0, j1, i01, i23);
            Interleave(j2, j3, i45, i67);
            Interleave(v, vv, j0, j2);
            uint32_t ng = (NUM_EXPERTS - e0 < RT_EGRP) ? (NUM_EXPERTS - e0) : RT_EGRP;
            MaskReg mg = UpdateMask<float>(ng);
            StoreAlign(rowUb, v, mg);
        }
    }

'''


def _cut(text: str, begin: str, end: str, tag: str) -> tuple[str, str]:
    """把 [begin, end) 之间的整段切出来（两端的锚点各必须**恰好出现一次**）。"""
    assert text.count(begin) == 1, f"[M91] 规则 8 的起点锚点不唯一/未找到（{tag}）：{text.count(begin)} 处"
    assert text.count(end) == 1, f"[M91] 规则 8 的终点锚点不唯一/未找到（{tag}）：{text.count(end)} 处"
    i = text.index(begin)
    j = text.index(end, i)
    return text[:i], text[j:]


def rewrite_router_streaming(text: str) -> str:
    """[M91-#3] 把 router 的「权重全量预转常驻」改成 m17 式流式（3 段整块替换 + 2 处删除）。"""
    text = _must_replace(text, RT_RB_MEMBER_OLD,
                         "    // [M91-#3] row-block 行数由 m15_moe_resources.h §4 的 RT_RB 给出"
                         "（曾在此处 shadow 成 16；删除以免与资源表的 UB 定尺不一致）\n",
                         "M91-8 class RT_RB 成员（删 shadow）")
    text = _must_replace(text, RT_NROWS_OLD, "", "M91-8 RT_NROWS（流式后不再有“全部行”概念）")
    text = _must_replace(text, PRECAST_CALL_OLD, "", "M91-8 Run() 里的 PrecastWeights 调用")
    pre, rest = _cut(text, PRECAST_OLD_BEGIN, INDEX_TMPL_BEGIN, "PrecastWeights 段")
    text = pre + _PRECAST_HEAD + rest
    pre, rest = _cut(text, COMPUTEBLK_OLD_BEGIN, GEMVROW_OLD_BEGIN, "ComputeBlock 段")
    text = pre + _COMPUTEBLK_NEW + rest
    pre, rest = _cut(text, GEMVROW_OLD_BEGIN, SOFTMAX_BEGIN, "GemvRow 段")
    text = pre + _GEMVROW_NEW + rest
    # 产物断言：流式形态到位、且旧的「全量预转」形态不再出现
    for gone in ("PrecastWeights", "UB_RT_WF + i * HIDDEN * 4", "RT_NROWS"):
        assert gone not in text, f"[M91] 规则 8 后仍未消失：{gone}"
    # 窗口读满 RT_EGRP 个槽（0..7），且只在这一处
    assert text.count("UB_RT_WF + 7 * HIDDEN * 4") == 1, "[M91] 权重流式窗的第 8 个槽读点不唯一"
    for need in ("LoadWRow", "PrecastWRow", "PrecastSgateW", "PadLogitsRow", "SgateRows",
                 "GemvGroupRow", "RT_EGRP", "UB_RT_SGWF", "UB_RT_SGB",
                 "BUF_AIV_WST0", "BUF_AIV_WST1"):
        assert need in text, f"[M91] 规则 8 的产物缺 {need}"
    # 正对照：旧形态的两个特征串必须真的在**上游 asc** 里出现过（否则规则 8 是空转）
    assert PRECAST_OLD_BEGIN in SRC.read_text(), "[M91] 规则 8 的起点锚点在上游 asc 里不存在（空转）"
    assert "PrecastWeights()" in SRC.read_text(), "[M91] 规则 8 要删的调用在上游 asc 里不存在（空转）"
    return text


def rewrite_index_gen(text: str) -> str:
    """[M84-5/#6] S3 标量数组 → UB 静态槽位；诊断槽 GM 下标改 IG_DIAG_GM_SLOT；
    [M105] 诊断槽的 UB 标量写提前到写侧 acquire 之前 + 写侧释放改 **mode=false**
    （`MutexUnlock<PIPE_S>` → `BufRelease<PIPE_S>`，与 acquire 同模式）⇒ 那次写落在写侧 release 之前。"""
    text = _must_replace(text, IDX_HEAD_OLD, IDX_HEAD_NEW, "#5 计数/前缀和")
    text = _must_replace(text, IDX_MID_OLD, IDX_MID_NEW, "#5 槽位搬运")
    text = _must_replace(text, IDX_CURSOR_OLD, IDX_CURSOR_NEW, "#5 cursor 落位")
    text = _must_replace(text, IDX_TENSOR_OLD, IDX_TENSOR_NEW, "#5 落盘视图")
    text = _must_replace(text, IDX_OFFCOPY_OLD, IDX_OFFCOPY_NEW, "#5 offsets 落盘源")
    text = _must_replace(text, IDX_DIAG_OLD, IDX_DIAG_NEW, "#6 诊断槽")
    text = _must_replace(text, IDX_OFFS_VIEW_OLD, IDX_OFFS_VIEW_NEW, "#6 offsGm 视图长度")
    text = _must_replace(text, IDX_RELEASE_OLD, IDX_RELEASE_NEW, "[M105] 写侧释放改 mode=false")
    for gone in ("uint32_t cnt[NUM_EXPERTS]", "uint32_t off[NUM_EXPERTS + 1]",
                 "uint32_t cursor[NUM_EXPERTS]", "cntUb[8 + e]", "offsGm[8]"):
        assert gone not in text, f"[M84] #5/#6 改造不完整：仍出现 {gone}"
    # ---- [M105] 诊断槽次序：写唯一 + 该写落在**写侧 release 之前** ----
    # 内容锚点（不用行号）：`Run()` 里这四句是本段 S→MTE3 的次序 token 全链
    #   `MutexLock<PIPE_S>`（写侧 acquire）→ 标量写 → `BufRelease<PIPE_S>`（写侧 release，mode=false）
    #   → `BufAcquire<PIPE_MTE3>`（搬侧 acquire）→ `DataCopyPad`（搬）。
    diag_store = "diagUb[32] = static_cast<int32_t>(oobCountUb);"
    diag_lock = "MutexLock<PIPE_S>(BUF_AIV_IDX);"
    diag_rls = "BufRelease<PIPE_S>(BUF_AIV_IDX);"
    diag_acq = "BufAcquire<PIPE_MTE3>(BUF_AIV_IDX);"
    up = SRC.read_text()
    assert diag_lock in up and diag_acq in up, "(M105 自检：次序 token 锚点串不在上游)"
    assert text.count(diag_store) == 1, f"[M105] 诊断槽标量写不唯一：{text.count(diag_store)} 处"
    # ① 写侧**必须是 mode=false 的释放**（`BufRelease` = `RlsBufInternal<pipe,false>`）——
    #    人类 M181 裁定「整个项目所有场景都是 false」；`true` 与 mode-0 `MutexUnlock` 都不得出现。
    assert "RlsBufInternal<pipe, false>(id);" in text, (
        "[M181] 段内缺 mode=false 的 `BufRelease` 封装（上游 m13 回退到 true？）")
    assert "RlsBufInternal<pipe, true>" not in text, (
        "[M181] 段内仍出现 `RlsBufInternal<... ,true>`（应为 mode=false = ASC_LOCK_BLOCK 默认）")
    #    改前的 `MutexUnlock`（mode 0）不得残留 —— 这是本 mission 第 2 轮复审的 P1 落点。
    assert text.count(diag_rls) == 1, f"[M105] 写侧 release 不唯一：{text.count(diag_rls)} 处"
    assert "MutexUnlock<PIPE_S>(BUF_AIV_IDX);" not in text, (
        "[M105] S 侧释放仍是 `MutexUnlock<PIPE_S>`（本段应用与 acquire 同模式的 `BufRelease`）")
    # ② 次序：写 → 写侧 acquire → 写侧 release → 搬侧 acquire
    tail = text[text.index(diag_store):]
    for need in (diag_lock, diag_rls, diag_acq):
        assert need in tail, f"[M105] 诊断槽标量写之后缺 {need}"
    assert tail.index(diag_lock) < tail.index(diag_rls) < tail.index(diag_acq), (
        "[M105] 次序不对：必须是 写 → MutexLock<PIPE_S> → BufRelease<PIPE_S>(mode=false) → BufAcquire<PIPE_MTE3>")
    # ③ 负向形状不得残留；④ 上游确实是抢跑形态（防「规则空转」）
    assert "bUb[32] = static_cast<int32_t>(oobCountUb);" not in text, "[M105] 改前的诊断槽标量写仍在"
    assert "MutexUnlock<PIPE_S>(BUF_AIV_IDX);" in up, (
        "[M105] 上游 m13 的 S 侧释放已不是 `MutexUnlock<PIPE_S>` ⇒ 本规则空转，须重核")
    assert up.index("bUb[32] = static_cast<int32_t>(oobCountUb);") > up.index(diag_acq), (
        "[M105] 上游 m13 已不是「写落在 BufAcquire 之后」的形态 ⇒ 本规则空转，须重核")
    return text


def rewrite_unpermute(text: str) -> str:
    """[M84-7] unpermute 的 K 链与模板实例展开到 10（topk=10 需要）。"""
    text = _must_replace(text, FMA_OLD, FMA_NEW, "#7 FmaChunk K 链")
    text = _must_replace(text, SWITCH_OLD, SWITCH_NEW, "#7 topk 分发 switch")
    text = _must_replace(text, INIT_OLD, INIT_NEW, "#7 PrepareUnpermute")
    text = _must_replace(text, MEMBER_OLD, MEMBER_NEW, "#7 UnpermuteStage 成员")
    text = _must_replace(text, UNPERM_COMMENT_OLD, UNPERM_COMMENT_NEW, "#7 注释")
    assert "default: unperm4.Run" not in text, "[M84] #7 k 链仍截断在 4"
    return text


def rewrite_y_view(text: str) -> str:
    """[M84-8] unpermute 的 Y 视图上界显式写成紧凑 Σt_e 上界 TOTAL_MAX。"""
    return _must_replace(text, YVIEW_OLD, YVIEW_NEW, "#8 Y 视图")


# ============================================================
# 规则 9（M95-#4）：top-k 归并树（16 块 Sort32 + 4 级二路 MrgSort + Extract 64 对 → 前 TOPK）
#   形状来源 = `m17_moe_real/m17_moe_layer.asc` 的 `SoftmaxTopkRenormRow` / `MergeTree` /
#   `Merge2`（E=512/top-10 的已验收形态；`m17_moe_real/README.md:33`）。改动的四件：
#     (a) 索引模板从 0..31 铺到 0..RT_ROWL-1（16 个 32 块的全局 expert id）；
#     (b) 对数行行距 64 → RT_ROWL（`PadLogitsRow` / `GemvGroupRow` / `SoftmaxTopkRow` /
#         `CopyOutBlock` 的 4 处 `r * 64 * 4`）；top-k **结果**的 IDS/WS 行距固定 RT_WROW（= 64
#         lane，与 E 无关 —— 只放 ≤ 64 个 top-k 结果，m17 亦如此），它们不随行宽重排；
#     (c) `Sort32(..., 1)` → `Sort32(..., RT_SORT_NBLK)` + `MergeTree()`/`Merge2()`（MrgSort 二路树）；
#     (d) 行内 max/exp 从「单寄存器 32 lane」改成「RT_NCHUNK_W 个 64-lane chunk + Reduce」。
#   **E=4 的退化档**：`RT_SORTLN = 512 ≥ NUM_EXPERTS` ⇒ 行内有效 logit 后面的 lane 全是
#   `RT_NEG_BIG`（exp → 0），排序/归并落在尾部，top-TOPK 与改前单块 Sort32 同集合同次序；
#   且排序、归并、pair 拆分**无算术** ⇒ 逐字节相同的论证成立（由 A/B dump 见证，见 m95_README）。
# ============================================================

TOPK_IDX_TMPL_OLD = """\
    // 索引模板 0..31（Arange 只填 64 lane → 取低 32 lane）
    __aicore__ inline void BuildIndexTemplate()
    {
        __ubuf__ int32_t* idxUb = reinterpret_cast<__ubuf__ int32_t*>(UB_RT_IDX);
        __VEC_SCOPE__
        {
            RegTensor<int32_t> rg;
            uint32_t n32 = RT_LANES;
            MaskReg m32 = UpdateMask<int32_t>(n32);
            Arange(rg, static_cast<int32_t>(0));
            StoreAlign(idxUb, rg, m32);
        }
    }
"""

TOPK_IDX_TMPL_NEW = """\
    // 索引模板 0..RT_ROWL-1（`Arange` 一次只填 64 lane ⇒ 按 RT_NCHUNK_W 个 chunk 拼）
    // [M95-#4] 改前只铺 0..31（单块 Sort32 的模板宽度）。归并树有 RT_SORT_NBLK = 16 个 32 块，
    //   每块内的索引模板必须是**全局 expert id 且随块递增** ⇒ 按行宽铺满 0..RT_ROWL-1 再交给
    //   `Sort32(..., RT_SORT_NBLK)`（它以 32 lane 为单位连续取块）。
    __aicore__ inline void BuildIndexTemplate()
    {
        __ubuf__ int32_t* idxUb = reinterpret_cast<__ubuf__ int32_t*>(UB_RT_IDX);
        __VEC_SCOPE__
        {
            uint32_t n64 = 64;               // 显式 64 lane：不依赖 int32 ALL mask 的宽度
            MaskReg m64i = UpdateMask<int32_t>(n64);
            for (uint16_t c = 0; c < static_cast<uint16_t>(RT_NCHUNK_W); ++c) {
                RegTensor<int32_t> rg;
                Arange(rg, static_cast<int32_t>(static_cast<uint32_t>(c) * 64));
                StoreAlign(idxUb + static_cast<uint32_t>(c) * 64, rg, m64i);
            }
        }
    }
"""

TOPK_PADLOGITS_OLD = """\
    __aicore__ inline void PadLogitsRow(uint32_t r)
    {
        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG + r * 64 * 4);
        __VEC_SCOPE__
        {
            RegTensor<float> negInf;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            Duplicate(negInf, RT_NEG_BIG, maskAll);
            StoreAlign(rowUb, negInf, maskAll);
        }
    }
"""

TOPK_PADLOGITS_NEW = """\
    __aicore__ inline void PadLogitsRow(uint32_t r)
    {
        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG + r * RT_ROWL * 4);
        __VEC_SCOPE__
        {
            RegTensor<float> negInf;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            Duplicate(negInf, RT_NEG_BIG, maskAll);
            for (uint16_t c = 0; c < static_cast<uint16_t>(RT_NCHUNK_W); ++c) {
                StoreAlign(rowUb + static_cast<uint32_t>(c) * VL_F32, negInf, maskAll);
            }
        }
    }
"""

TOPK_GEMVROW_OLD = "        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG + r * 64 * 4) + e0;\n"
TOPK_GEMVROW_NEW = ("        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG"
                    " + r * RT_ROWL * 4) + e0;\n")

TOPK_COPYOUT_OLD = """\
            LocalTensor<float> logitsL(TPosition::VECCALC, UB_RT_LOG + r * 64 * 4, 64);
            LocalTensor<int32_t> idsL(TPosition::VECCALC, UB_RT_IDS + r * 64 * 4, 64);
            LocalTensor<float> wsL(TPosition::VECCALC, UB_RT_WS + r * 64 * 4, 64);
"""

TOPK_COPYOUT_NEW = """\
            // [M95-#4] 对数行 = RT_ROWL lane；top-k 结果 staging（IDS/WS）行距 = RT_WROW（固定 64）
            LocalTensor<float> logitsL(TPosition::VECCALC, UB_RT_LOG + r * RT_ROWL * 4, RT_ROWL);
            LocalTensor<int32_t> idsL(TPosition::VECCALC, UB_RT_IDS + r * RT_WROW * 4, RT_WROW);
            LocalTensor<float> wsL(TPosition::VECCALC, UB_RT_WS + r * RT_WROW * 4, RT_WROW);
"""

TOPK_SOFTMAX_OLD = """\
    // 行内 softmax(max-shift) + Sort32 + 前 32 对拆分（Extract 的 VF 内联）+ renorm → staging
    __aicore__ inline void SoftmaxTopkRow(uint32_t r, uint32_t grow)
    {
        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG + r * 64 * 4);
        __ubuf__ float* valUb = reinterpret_cast<__ubuf__ float*>(UB_RT_VAL);
        __ubuf__ float* ovUb = reinterpret_cast<__ubuf__ float*>(UB_RT_OV);
        __ubuf__ int32_t* oiUb = reinterpret_cast<__ubuf__ int32_t*>(UB_RT_OI);
        __ubuf__ float* pairUb = reinterpret_cast<__ubuf__ float*>(UB_RT_PAIR);
        __ubuf__ float* wsUb = reinterpret_cast<__ubuf__ float*>(UB_RT_WS + r * 64 * 4);
        __ubuf__ int32_t* idsUb = reinterpret_cast<__ubuf__ int32_t*>(UB_RT_IDS + r * 64 * 4);

        __VEC_SCOPE__
        {
            RegTensor<float> rowReg, mx, dm;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            LoadAlign(rowReg, rowUb);
            uint32_t n32 = RT_LANES;
            MaskReg m32 = UpdateMask<float>(n32);
            Reduce<ReduceType::MAX, float>(mx, rowReg, m32);
            Duplicate(dm, mx, maskAll);
            Sub(rowReg, rowReg, dm, maskAll);
            StoreAlign(rowUb, rowReg, maskAll);   // logits -= rowmax（对齐 golden 返回形式）
            Exp(rowReg, rowReg, maskAll);         // padding lane = exp(-3e38-max) = 0
            StoreAlign(valUb, rowReg, m32);
        }
        {
            // Sort32 属**裁定例外**（docs/05 §6.1 计算路径规则 ⓒ + §2 指令级原语白名单）：
            //   ① Reg 侧**无**等价物（CANN 9.1.0 `reg_compute/**` 穷举无 Sort32/MrgSort）；
            //   ② 官方 donor 同为 memory-based：ops-transformer/moe/moe_gating_top_k_softmax_v2/
            //      op_kernel/arch35/moe_gating_top_k_softmax_v2_perf_arch35.h:187（Sort32）、
            //      :225/:243（MrgSort）。仅限本 topk 排序段，不得扩散。
            //   `MrgSort4` 是 dav-3510 上的 deprecated 空函数体（静默 no-op），本文件 0 处使用。
            LocalTensor<float> pairT(TPosition::VECCALC, UB_RT_PAIR, 2 * RT_LANES);
            LocalTensor<float> valT(TPosition::VECCALC, UB_RT_VAL, RT_LANES);
            LocalTensor<uint32_t> idxT(TPosition::VECCALC, UB_RT_IDX, RT_LANES);
            Sort32(pairT, valT, idxT, 1);
        }
        // 前 32 对 (value, idx) 拆分 = 经典 `Extract(..., 1)` 的 VF 内联。抄厂商 dav_3510
        //   ExtractVf（asc/impl/basic_api/dav_3510/kernel_operator_vec_gather_mask_impl.h:426-481）
        //   的 float 分支：repeatTime=1 ⇒ loopTimes=0/tail=1 的「单寄存器载入 + DeInterleave +
        //   半寄存器存储」路径；与官方 topk donor 落盘同形（…perf_arch35.h:309-343：
        //   LoadAlign<DIST_DINTLV_B32> 载入 (value, idx) 交错对 + 两条掩码 StoreAlign）。
        //   纯 pair 拆分、无算术 ⇒ 与经典 Extract 逐位等价（判据 T1：topk_ids/weights 逐位）。
        __VEC_SCOPE__
        {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            RegTensor<float> vr, vi;
            uint32_t n32 = RT_LANES;
            MaskReg m32 = UpdateMask<float>(n32);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(vr, vi, pairUb);
            StoreAlign(ovUb, vr, m32);
            StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(UB_RT_OI),
                       reinterpret_cast<RegTensor<uint32_t>&>(vi), m32);
        }
        __VEC_SCOPE__
        {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            RegTensor<float> vals;
            RegTensor<int32_t> idxs;
            RegTensor<float> s, ds;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskAllI = CreateMask<int32_t, MaskPattern::ALL>();
            LoadAlign(vals, ovUb);
            LoadAlign(idxs, oiUb);
            uint32_t nk = TOPK;
            MaskReg mk = UpdateMask<float>(nk);
            Reduce<ReduceType::SUM, float>(s, vals, mk);
            Duplicate(ds, s, maskAll);
            Div(vals, vals, ds, mk);
            StoreAlign(wsUb, vals, maskAll);
            StoreAlign(idsUb, idxs, maskAllI);
        }
    }
"""

TOPK_SOFTMAX_NEW = """\
    // 行内 softmax(max-shift) + **16 块 Sort32 + 4 级二路 MrgSort 归并树** → 根列表 →
    //   Extract 前 64 对（VF 内联）→ 前 TOPK 对 renorm → staging
    //
    // 形状来源：`m17_moe_real/m17_moe_layer.asc` 的 `SoftmaxTopkRenormRow` / `MergeTree` /
    //   `Merge2`（E=512/top-10 的已验收形态，`m17_moe_real/README.md:33`）；`Extract` 用
    //   **本段已有的 VF 内联形态**（M50 落地：`LoadAlign<DIST_DINTLV_B32>` 解交织 + 两条掩码
    //   `StoreAlign`）重复 RT_EXTRACT_REP 次 = 经典 `Extract(..., 2)` 的两遍 32 对；
    //   **不引入**经典 memory-based `Extract`（不在 M44 白名单内）。
    //
    // `Sort32` / `MrgSort` 属**裁定例外**（docs/05 §6.1 计算路径规则 ⓒ + §2 指令级原语白名单）：
    //   ① Reg 侧**无**等价物（CANN 9.1.0 `reg_compute/**` 穷举无 Sort32/MrgSort）；
    //   ② 官方 donor 同为 memory-based：ops-transformer/moe/moe_gating_top_k_softmax_v2/
    //      op_kernel/arch35/moe_gating_top_k_softmax_v2_perf_arch35.h:187（Sort32）、
    //      :225/:243（MrgSort）。仅限本 topk 排序段，不得扩散。
    //   `MrgSort4` 是 dav-3510 上的 deprecated 空函数体（静默 no-op），本文件 0 处使用。
    //
    // **取前 TOPK 仍严格正确的证明**：每级归并各取两条输入列表的**前 32 对**、输出 64 对。
    //   设 P(k)：第 k 级每条列表的前 32 对 = 该子树元素的 top-32。k=1 时输入是两个 32 块的
    //   完整内容、输出 64 对 = 完整归并 ⇒ P(1) 成立。若 x ∈ top-32(S)，S = S_A ∪ S_B，则比 x
    //   大的元素在 S_A 内至多 31 个 ⇒ x 在 S_A 内的名次 ≤ 32 ⇒ x ∈ top-32(S_A) ⊆ A 的前 32 对
    //   （由 P(k)）⇒ merge 后 x 落在输出前 32 ⇒ P(k+1)。归纳到根：**根的前 32 对 = 全体候选的
    //   top-32（精确）**；`Extract` 取根的前 64 对 ⊇ top-32 ⇒ top-TOPK（TOPK ≤ 32）精确。
    //   **口径警告**：根的 64 对**不是**全局 top-64（第 2 级起每路只取前 32 对，名次 33..64 不再
    //   上行）。m17 注释里的「各级 top-64」指的是「该级输出的 64 对」；本实现的正确性声明只到
    //   top-32（= TOPK_MAX 的上界，由 §4 的 static_assert 守住）。
    //
    // 退化档（RT_ROWL ≥ NUM_EXPERTS 时行内 pad）：lane ≥ NUM_EXPERTS 全是 `RT_NEG_BIG` ⇒ `Exp`
    //   得 0、排序/归并落在尾部（索引模板更大，并列时排在真实专家之后）⇒ top-TOPK 与改前单块
    //   Sort32 同集合同次序；排序 / 归并 / pair 拆分**无算术** ⇒ 逐位相同（A/B dump 见证）。
    __aicore__ inline void SoftmaxTopkRow(uint32_t r, uint32_t grow)
    {
        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG + r * RT_ROWL * 4);
        __ubuf__ float* valUb = reinterpret_cast<__ubuf__ float*>(UB_RT_VAL);
        __ubuf__ float* ovUb = reinterpret_cast<__ubuf__ float*>(UB_RT_OV);
        __ubuf__ int32_t* oiUb = reinterpret_cast<__ubuf__ int32_t*>(UB_RT_OI);
        __ubuf__ float* pairUb = reinterpret_cast<__ubuf__ float*>(UB_RT_PAIR);
        __ubuf__ float* wsUb = reinterpret_cast<__ubuf__ float*>(UB_RT_WS + r * RT_WROW * 4);
        __ubuf__ int32_t* idsUb = reinterpret_cast<__ubuf__ int32_t*>(UB_RT_IDS + r * RT_WROW * 4);

        // ① 行内 max（覆盖整行 RT_ROWL lane：pad lane 是 RT_NEG_BIG，不影响 max）+ max-shift + exp。
        //    改前是「单寄存器 32 lane 的 Reduce」；现在按 64-lane chunk 做 Max 树再一次 Reduce。
        //    pad lane 的 exp(-3e38 - max) = 0（与改前同值：改前也对 32 lane 里的 pad 求 exp）。
        __VEC_SCOPE__
        {
            RegTensor<float> v, m, mx, dm;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            Duplicate(m, RT_NEG_BIG, maskAll);
            for (uint16_t c = 0; c < static_cast<uint16_t>(RT_NCHUNK_W); ++c) {
                LoadAlign(v, rowUb + static_cast<uint32_t>(c) * VL_F32);
                Max(m, m, v, maskAll);
            }
            Reduce<ReduceType::MAX, float>(mx, m, maskAll);
            Duplicate(dm, mx, maskAll);
            for (uint16_t c = 0; c < static_cast<uint16_t>(RT_NCHUNK_W); ++c) {
                LoadAlign(v, rowUb + static_cast<uint32_t>(c) * VL_F32);
                Sub(v, v, dm, maskAll);
                StoreAlign(rowUb + static_cast<uint32_t>(c) * VL_F32, v, maskAll);   // logits -= rowmax（对齐 golden 返回形式）
                Exp(v, v, maskAll);
                StoreAlign(valUb + static_cast<uint32_t>(c) * VL_F32, v, maskAll);
            }
        }
        // ② Sort32：RT_SORT_NBLK 个 32 块内降序（值, 全局 expert id）
        {
            LocalTensor<float> pairT(TPosition::VECCALC, UB_RT_PAIR, 2 * RT_ROWL);
            LocalTensor<float> valT(TPosition::VECCALC, UB_RT_VAL, RT_SORTLN);
            LocalTensor<uint32_t> idxT(TPosition::VECCALC, UB_RT_IDX, RT_SORTLN);
            Sort32(pairT, valT, idxT, RT_SORT_NBLK);
        }
        // ③ 二路归并树：16 个 32 块 → 4 级 → 根列表落在 UB_RT_MB（前 32 对 = 全局 top-32）
        MergeTree();
        // ④ Extract 根的前 64 对 = 经典 `Extract(..., 2)` 的 VF 内联。抄厂商 dav_3510
        //   ExtractVf（asc/impl/basic_api/dav_3510/kernel_operator_vec_gather_mask_impl.h:426-481）
        //   的 float 分支：repeatTime=1 ⇒ loopTimes=0/tail=1 的「单寄存器载入 + DeInterleave +
        //   半寄存器存储」路径；与官方 topk donor 落盘同形（…perf_arch35.h:309-343：
        //   LoadAlign<DIST_DINTLV_B32> 载入 (value, idx) 交错对 + 两条掩码 StoreAlign）。
        //   纯 pair 拆分、无算术 ⇒ 与经典 Extract 逐位等价（判据 T1：topk_ids/weights 逐位）。
        for (uint32_t rep = 0; rep < RT_EXTRACT_REP; ++rep) {
            __VEC_SCOPE__
            {
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                RegTensor<float> vr, vi;
                uint32_t n32 = RT_LANES;
                MaskReg m32 = UpdateMask<float>(n32);
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    vr, vi, reinterpret_cast<__ubuf__ float*>(UB_RT_MB) + rep * 2 * RT_LANES);
                StoreAlign(ovUb + rep * RT_LANES, vr, m32);
                StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(UB_RT_OI) + rep * RT_LANES,
                           reinterpret_cast<RegTensor<uint32_t>&>(vi), m32);
            }
        }
        // ⑤ renorm（前 TOPK 对除以它们的和）→ staging。改前的 ids store 用 `int32_t ALL` mask
        //   （其宽度在本架构上未定，M94 在裁）；这里改成**显式 64 lane**（= RT_WROW 行距，
        //   恰好一行、不跨行）。
        __VEC_SCOPE__
        {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            RegTensor<float> vals;
            RegTensor<int32_t> idxs;
            RegTensor<float> s, ds;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            uint32_t nw = RT_WROW;
            MaskReg mwi = UpdateMask<int32_t>(nw);
            LoadAlign(vals, ovUb);
            LoadAlign(idxs, oiUb);
            uint32_t nk = TOPK;
            MaskReg mk = UpdateMask<float>(nk);
            Reduce<ReduceType::SUM, float>(s, vals, mk);
            Duplicate(ds, s, maskAll);
            Div(vals, vals, ds, mk);
            StoreAlign(wsUb, vals, maskAll);
            StoreAlign(idsUb, idxs, mwi);
        }
    }

    // ---- 2 路归并树（UB_RT_MA / UB_RT_MB 乒乓）：16 个 32 块 → 4 级 → 根 = UB_RT_MB[0] ----
    //      每级 `elementLengths = [32, 32, 0, 0]` ⇒ 每级输出 64 对（正确性见 SoftmaxTopkRow 头注）。
    //      `LocalMemBar` 必须在 `__VEC_SCOPE__` 内（3510 quirk，m17/m22 同款）。
    __aicore__ inline void MergeTree()
    {
        LocalTensor<float> srcT(TPosition::VECCALC, UB_RT_PAIR, 2 * RT_ROWL);
        LocalTensor<float> aT(TPosition::VECCALC, UB_RT_MA, 2 * RT_ROWL);
        LocalTensor<float> bT(TPosition::VECCALC, UB_RT_MB, 2 * RT_ROWL);
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        for (uint32_t g = 0; g < 8; ++g) {   // level1: 16 块 → 8 组（PAIR → MA）
            Merge2(aT[g * 128], srcT[(2 * g) * 64], srcT[(2 * g + 1) * 64], 32);
        }
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        for (uint32_t g = 0; g < 4; ++g) {   // level2: 8 组 → 4 组（MA → MB）
            Merge2(bT[g * 128], aT[(2 * g) * 128], aT[(2 * g + 1) * 128], 32);
        }
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        for (uint32_t g = 0; g < 2; ++g) {   // level3: 4 组 → 2 组（MB → MA）
            Merge2(aT[g * 128], bT[(2 * g) * 128], bT[(2 * g + 1) * 128], 32);
        }
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        Merge2(bT[0], aT[0], aT[128], 32);   // level4: 2 组 → 1（MA → MB，根 = bT[0]）
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
    }

    // 2 路归并：走 `MrgSort(dst, srcList, MrgSort4Info)`（validBit = 0b0011），**不用** `MrgSort4`
    //   （后者在 dav-3510 上是 deprecated 空函数体，见 probe_sync_quirks A02/A15/A20 与
    //   docs/05 §6 #19；同参数的 4 路 `MrgSort` 真机正确 ⇒ 这里选 2 路是够用，不是硬件限制）。
    //   src3/src4 在 validBit = 0b0011 下不参与，填 src2（m17_moe_real / m22_router512 同款写法）。
    __aicore__ inline void Merge2(const LocalTensor<float>& dst, const LocalTensor<float>& s1,
                                  const LocalTensor<float>& s2, uint32_t pairLen)
    {
        MrgSort4Info params;
        params.elementLengths[0] = static_cast<uint16_t>(pairLen);
        params.elementLengths[1] = static_cast<uint16_t>(pairLen);
        params.elementLengths[2] = 0;
        params.elementLengths[3] = 0;
        params.validBit = 0b0011;
        params.repeatTimes = 1;
        params.ifExhaustedSuspension = false;
        MrgSortSrcList<float> sl;
        sl.src1 = s1;
        sl.src2 = s2;
        sl.src3 = s2;
        sl.src4 = s2;
        MrgSort(dst, sl, params);
    }
"""


def rewrite_topk_tree(text: str) -> str:
    """[M95-#4] 单块 Sort32 → 16 块 Sort32 + 4 级二路 MrgSort 归并树（5 处定点替换）。"""
    text = _must_replace(text, TOPK_SOFTMAX_OLD, TOPK_SOFTMAX_NEW, "M95-9 SoftmaxTopkRow 段")
    text = _must_replace(text, TOPK_IDX_TMPL_OLD, TOPK_IDX_TMPL_NEW, "M95-9 索引模板宽度")
    text = _must_replace(text, TOPK_PADLOGITS_OLD, TOPK_PADLOGITS_NEW, "M95-9 PadLogitsRow 行距")
    text = _must_replace(text, TOPK_GEMVROW_OLD, TOPK_GEMVROW_NEW, "M95-9 GemvGroupRow 行距")
    text = _must_replace(text, TOPK_COPYOUT_OLD, TOPK_COPYOUT_NEW, "M95-9 CopyOutBlock 行距/视图")
    # 产物：归并树到位、单块形态（`Sort32(..., 1)`）消失、行距换名
    assert "Sort32(pairT, valT, idxT, RT_SORT_NBLK)" in text, "[M95] 归并树的 Sort32 块数未接上"
    assert "Sort32(pairT, valT, idxT, 1)" not in text, "[M95] 单块 Sort32 仍在"
    for need in ("MergeTree()", "Merge2(", "MrgSort(dst, sl, params)", "UB_RT_MA", "UB_RT_MB",
                 "RT_ROWL", "RT_WROW", "RT_SORTLN", "RT_NCHUNK_W", "RT_EXTRACT_REP"):
        assert need in text, f"[M95] 规则 9 的产物缺 {need}"
    assert "MrgSort4(" not in text, "[M95] 引入了 dav-3510 上静默 no-op 的 MrgSort4"
    # 行距：对数行的 4 处必须用 RT_ROWL（`r * 64 * 4` 不许再落在对数行上）
    assert text.count("UB_RT_LOG + r * 64 * 4") == 0, "[M95] 仍有写死 64 lane 的对数行取址"
    assert text.count("UB_RT_LOG + r * RT_ROWL * 4") == 4, (
        f"[M95] 对数行取址应为 4 处（PadLogitsRow / GemvGroupRow / SoftmaxTopkRow / CopyOutBlock），"
        f"实为 {text.count('UB_RT_LOG + r * RT_ROWL * 4')}")
    assert text.count("LocalTensor<float> logitsL(TPosition::VECCALC, UB_RT_LOG + r * RT_ROWL * 4") == 1
    # 正对照：改前的形态必须真的在上游 asc 里（否则规则 9 是空转）
    up = SRC.read_text()
    for probe in ("Sort32(pairT, valT, idxT, 1)", "// 索引模板 0..31",
                  "UB_RT_LOG + r * 64 * 4"):
        assert probe in up, f"[M95] 规则 9 的改前特征串在上游 asc 里不存在（空转）：{probe}"
    return text


def build() -> str:
    b, e, _ = _src_anchors()
    lines = SRC.read_text().splitlines()
    body = "\n".join(lines[b:e + 1])
    text = transform(body)
    assert OFF_DECL_OLD in text, "ProcessAic 的 countsGm 声明未找到"
    text = text.replace(OFF_DECL_OLD, OFF_DECL_NEW)
    assert_upstream_vf(text)
    text = rewrite_quant_run(text)
    text = rewrite_gemm_loop(text, "GU_ITEMS", "GU_NBLK")
    text = rewrite_gemm_loop(text, "DN_ITEMS", "DN_NBLK")
    text = rewrite_index_gen(text)
    text = rewrite_unpermute(text)
    text = rewrite_y_view(text)
    text = rewrite_router_streaming(text)
    text = rewrite_topk_tree(text)
    # 规则 6 的产物核对：不能再出现固定槽行距（slot * M_MAX）
    leftover = [ln.strip() for ln in text.splitlines() if "slot * M_MAX" in ln]
    assert all(ln.startswith("const uint32_t srcRow = PAD_SRC") or ln.startswith("const uint32_t dstRow = ROUTED")
               for ln in leftover), f"紧凑槽改造不完整：{leftover}"
    return PROLOGUE + text + "\n#endif  // M15_MOE_LAYER_H\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    want = build()
    if args.check:
        have = DST.read_text() if DST.exists() else ""
        b, e, entry = _src_anchors()
        if have == want:
            print(f"[lift] {DST.name} 与抽取规则一致（内容锚点：asc 行 {b + 1}..{e + 1}，入口在第 {entry + 1} 行）")
            return 0
        print(f"[lift][FAIL] {DST.name} 与抽取规则不一致（被手改或上游已变）")
        import difflib
        diff = difflib.unified_diff(have.splitlines(), want.splitlines(), "on-disk", "lifted", lineterm="")
        for i, d in enumerate(diff):
            if i > 60:
                print("  ...（截断）")
                break
            print("  " + d)
        return 1
    DST.write_text(want)
    b, e, entry = _src_anchors()
    print(f"[lift] 写出 {DST.name}（{want.count(chr(10))} 行；内容锚点 asc 行 {b + 1}..{e + 1}，"
          f"入口第 {entry + 1} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
