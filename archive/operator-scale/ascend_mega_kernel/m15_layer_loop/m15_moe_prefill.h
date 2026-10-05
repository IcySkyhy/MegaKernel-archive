/**
 * m15_moe_prefill.h —— MoE **prefill** 段（B4）设备代码，可被融合 TU 直接 include
 *
 * 交付形态（`docs/15` §M103-2.7 的三条硬要求）：
 *   ① include guard + **无 `main()`**（`main()` 只在 `m26_moe_prefill/m26_moe_prefill.asc`）；
 *   ② **用具名 `namespace M15MP`**（不用匿名 namespace：融合 TU 里已有多个同名工具段）；
 *   ③ 设备代码只依赖设备头 + 自家的 `_res.h`（不反向依赖 host 头）；
 *      资源（UB 窗 / L1 / L0 / L0C / BufferID / flagId）**全部编译期静态**，峰值带 `static_assert`。
 *
 * 段序（与 decode 档 `m15_moe_layer.h` 同一套 S1–S10 语义，但**按 `MT=64` 行分块**）：
 *   S1  m6#1 Add+RMSNorm（res=0）                         [AIV, 行切分]
 *   S2  **router 打分 = AIC 上的 bf16 `Mmad`**（`x[MT,2560] @ W[E_PAD,2560]^T` → fp32 logits）
 *       —— 人类裁决：凡涉及矩阵乘法必须 mmad，不管 M=1 或者是多大；
 *       `docs/20` §4 WO-A2 把 AIV 逐列 GEMV 的 router 打分列为 **P1 违规**，本段是第一次 mmad 化。
 *   S2b AIV 行内 softmax(max-shift) + 16 块 Sort32 + 4 级二路 MrgSort 归并树 + renorm
 *       （`Sort32`/`MrgSort` 属 `docs/05` §6.1 ⓒ 的裁定例外，仍留在 AIV）
 *   S3  计数排序索引生成（counts/offsets/perm_src/perm_expert/inv_slot）      [AIV0]
 *   S4  permute（四级流水 gather）                        [AIV, 行切分]
 *   S5  A 侧 MXFP4 全 VEC 量化（routed 紧凑槽 + shared）   [AIV, 行切分]
 *   S6  grouped gate_up GEMM（`MXFP4GemmItem`，AIC；槽 = 512 专家 + 1 共享）  [AIC, N 条带]
 *   S7  SwiGLU + MXFP4 量化                              [AIV, 行切分]
 *   S8  grouped down GEMM                                 [AIC, N 条带]
 *   S9a unpermute 加权折叠 / S9b combine（+sigmoid 门）    [AIV, 行切分]
 *   S10 m6#2 Add+RMSNorm（res = S1 的 fp32 resOut）        [AIV, 行切分]
 *
 * 算件来源（**复制改造，不改动原文件**；本 mission 的 scope 只允许新建这三个头 + `m26_moe_prefill/` 目录）：
 *   · `MXFP4GemmItem` / `NormStage` / `NormDonor`：**直接复用** `m15_moe_layer.h` 的 M15M 版本
 *     （`MXFP4GemmItem` 是形状模板、与 E 解耦；`NormStage` 的 GM/UB 定尺按 `M_MAX=64` = 本段的 `MT`，恰好可用）。
 *   · `PermuteStage` / `UnpermuteStage` / `CombineStage`：**真复制**一份为 `PermuteP` /
 *     `UnpermuteP<TK>` / `CombineP`（§4b），把 GM 视图从 decode 档的 `TOTAL_MAX`(256) /
 *     `M_MAX*TOPK_MAX`(256) 改成 prefill 档的 `TP_MAX`(640) / `MT*TOPK`(640)
 *     —— decode 档那三份的 size 实参在 topk=10 下**装不下**（r1 复审 P2-1）。
 *     `UnpermuteP` 另按新规把 top-k 权重的 bf16 取整**从 scalar 位打包改成 VF 的 Cast**
 *     （矩阵乘法⇒cube、其余⇒VF、scalar 只做控制流）。
 *   · `VecQuantStage` / `IndexGenStage`：**复制 + 改 E**（decode 档那份把 `NUM_EXPERTS=4`
 *     写进了 counts 扫描与 UB 静态槽定尺，E=512 下直接装不下）。
 *   · router 的 AIV top-k（softmax / Sort32 / 归并树 / Extract / renorm）：**复制**自
 *     `m15_moe_layer.h` 的 `SoftmaxTopkRow`/`MergeTree`/`Merge2`（同一 `RT_ROWL=512` 形状，
 *     E=512 恰好铺满 ⇒ 无 pad lane）。
 *   · **新写**：`RouterMmadStage`（AIC 的 bf16 mmad router）与 `MoePrefillChain`（按 m 分块的编排）。
 *
 * 同步（用户原话 + `docs/15` Wave B 共同契约 2）：核内 pipeline 用 **BufferID**；核间用
 * `CrossCoreSetFlag/WaitFlag`；**不用** set flag/wait flag 系列；除值依赖外**不用 `PIPE_S`**。
 * 落 GM 一律 DMA（规则 ⓔ）：本文件的每一处 GM 写都是 `DataCopy`/`DataCopyPad`/`Fixpipe`。
 */

#ifndef M15_MOE_PREFILL_H
#define M15_MOE_PREFILL_H

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "reg_compute/kernel_reg_compute_intf.h"

#include <cstdint>

// ---- include 前缀 ----
// **调用方必须先引入 `m15_gdn_layer.h` 与 `m15_hc_layer.h`**：这两个头**没有 include guard**
// （它们各自开一个匿名 namespace 提供全局作用域的 `BufAcquire`/`BufRelease`/`Block1`/`ExtBlock1`
// 等 pipeline 帮手，且有各自的段体），因此**只能被引入一次**，本头不重复引入它们。
// 融合 TU（`m15_layer_loop.asc:60-61`）与独立工程（`m26_moe_prefill.asc`）都满足这一前提。
// 其余设备头都带 guard，本头按正确次序列全（在融合 TU 里全是 no-op）。
#include "m15_attn_layer.h"
#include "m15_loop_layout.h"
#include "m15_attn_kv.h"          // 必须先于 m15_attn_prolog.h（见 m15_attn_kv.h 末尾注释）
#include "m15_attn_prolog.h"
#include "m15_layer_resources.h"  // MWP_* / MOE_W_PREFILL_STRIDE / MOE_E_PREFILL / MOE_TOPK_PREFILL（唯一 owner）
#include "m15_moe_layer.h"        // 复用 M15M::MXFP4GemmItem / NormStage（不改动它）
#include "m15_moe_prefill_res.h"

namespace M15MP {

using namespace AscendC;
using namespace AscendC::Reg;
using namespace M15PFR;

// ---- 消歧（实跑踩到过）：全局作用域已带 `using namespace M15G/M15H/M15AP`（各段匿名 namespace 的
//      做法），与 `using namespace M15PFR` 带进来的**同名**常量在同一次查找里会被判 ambiguous。
//      这里在 M15MP 里显式取一份（内层声明遮蔽外层），名字与语义都不变。
constexpr uint32_t HIDDEN = M15PFR::HIDDEN;
constexpr uint32_t INTER = M15PFR::INTER;
constexpr uint32_t GU_N = M15PFR::GU_N;
constexpr uint32_t GROUP = M15PFR::GROUP;
constexpr uint32_t GU_SCALE_STRIDE = M15PFR::GU_SCALE_STRIDE;
constexpr uint32_t DN_SCALE_STRIDE = M15PFR::DN_SCALE_STRIDE;
constexpr uint32_t E = M15PFR::E;
constexpr uint32_t TOPK = M15PFR::TOPK;
constexpr uint32_t MT = M15PFR::MT;
constexpr uint32_t E_PAD = M15PFR::E_PAD;
constexpr uint32_t TP_MAX = M15PFR::TP_MAX;
constexpr uint32_t RB_K = M15PFR::RB_K;
constexpr uint32_t RB_N = M15PFR::RB_N;
constexpr uint32_t RB_KBLK = M15PFR::RB_KBLK;
constexpr uint32_t RB_NBLK = M15PFR::RB_NBLK;

// 复用 decode 档的算件（**只读**，不改 `m15_moe_layer.h`）：
using M15M::MXFP4GemmItem;
using M15M::NormStage;
// DataCopy 的 `{1, len, 0, 0}` 参数助手（decode 档 MoE 段按元素计数写 GM 的那两个重载）
using M15M::Block1;
using M15M::ExtBlock1;

constexpr uint32_t VL_F32 = 64;                                   // 向量寄存器 fp32 lane 数
constexpr uint32_t CHUNKS_H = HIDDEN / VL_F32;                    // 每行 64-lane chunk 数 = 40
constexpr uint32_t RT_LANES = 32;                                 // Sort32 的 lane 数
constexpr float RT_NEG_BIG = -3.0e38f;
constexpr uint32_t RT_SORT_NBLK = M15M::RT_SORT_NBLK;             // 16 块（固定）
constexpr uint32_t RT_SORTLN = RT_SORT_NBLK * RT_LANES;           // 512
constexpr uint32_t RT_NCHUNK_W = RT_ROWL / VL_F32;                // 8
constexpr uint32_t RT_EXTRACT_REP = 2;
static_assert(RT_SORTLN == E, "E 必须恰好等于 16×32 = 512（无 pad lane 的前提）");
static_assert(RT_ROWL == E, "logits 行宽必须恰好覆盖 E 个专家");

// ============================================================
// 1. S2（AIC）：router 打分 = bf16 `Mmad`
//    `logits[MT, n] = Σ_k x[MT, k] · W[n, k]`（W 是模型原始 [N, K] 行主序 bf16 = 真权重矩阵）
//    ⇒ 这就是 `docs/20` WO-A2 要求的 `x[m,2560] @ W[E,2560]^T`，**M 小也算**（M=1 也走 mmad）。
//    流水（与 `m15_gdn_layer.h` 的 `Cube::Bf16Gemm` 同款，参数换成 router 的形状）：
//      MTE2 Nd2Nz(GM→L1) ∥ MTE1 LoadData2D(L1→L0) ∥ M(Mmad 累加) ∥ FIXP(L0C→GM, fp32)
//    N 方向按 `RB_N=128` 分块，各 AIC 按 `nb = bid, bid+numBlocks, …` 条带划分（AIC 打平②）。
// ============================================================

class RouterMmadStage {
public:
    __aicore__ inline RouterMmadStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* w, __gm__ uint8_t* logits, uint32_t m)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w));
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logits));
        mTotal = m;
    }

    // 一个 N tile：nBlock ∈ [0, RB_NBLK)；写 `logits[0:curM, nBlock*RB_N : …]`
    __aicore__ inline void RunTile(uint32_t nBlock)
    {
        const uint32_t curM = (mTotal < MT) ? mTotal : MT;
        // 3510 quirk（m1 契约，实测）：Nd2Nz 行数 = 1 时不做 NZ 切分（退化成 1D 拷贝）⇒ 数据错位。
        // 计算侧统一提升为 >= 2 行；多读的行由 host 保证可读（x/w 的分配都按满 tile 算），
        // 结果只按 curM 行写出。
        const uint32_t calcM = (curM < 2u) ? 2u : curM;
        const uint32_t calcMAlign = ((calcM + M15M::CUBE_BLOCK - 1u) / M15M::CUBE_BLOCK) * M15M::CUBE_BLOCK;

        AscendC::LocalTensor<bfloat16_t> a1Ping(AscendC::TPosition::A1, L1R_A0, MT * RB_K);
        AscendC::LocalTensor<bfloat16_t> a1Pong(AscendC::TPosition::A1, L1R_A1, MT * RB_K);
        AscendC::LocalTensor<bfloat16_t> b1Ping(AscendC::TPosition::B1, L1R_B0, RB_K * RB_N);
        AscendC::LocalTensor<bfloat16_t> b1Pong(AscendC::TPosition::B1, L1R_B1, RB_K * RB_N);
        AscendC::LocalTensor<bfloat16_t> a2Ping(AscendC::TPosition::A2, L0R_OFF0, MT * RB_K);
        AscendC::LocalTensor<bfloat16_t> a2Pong(AscendC::TPosition::A2, L0R_OFF1, MT * RB_K);
        AscendC::LocalTensor<bfloat16_t> b2Ping(AscendC::TPosition::B2, L0R_OFF0, RB_K * RB_N);
        AscendC::LocalTensor<bfloat16_t> b2Pong(AscendC::TPosition::B2, L0R_OFF1, RB_K * RB_N);
        AscendC::LocalTensor<float> cL0C(AscendC::TPosition::CO1, L0R_L0C, MT * RB_N);

        // M 先取 L0C 所有权（挡上一 tile 的 Fixpipe 读、允许本 tile 累加）
        AscendC::Mutex::Lock<PIPE_M>(PFR_BUF_L0C);
        for (uint32_t kBlock = 0; kBlock < RB_KBLK; ++kBlock) {
            const uint32_t p = kBlock & 1u;
            const AscendC::MutexID bufA = p ? PFR_BUF_A1 : PFR_BUF_A0;
            const AscendC::MutexID bufB = p ? PFR_BUF_B1 : PFR_BUF_B0;
            const AscendC::MutexID bufL0 = p ? PFR_BUF_L0_1 : PFR_BUF_L0_0;
            AscendC::LocalTensor<bfloat16_t> a1 = p ? a1Pong : a1Ping;
            AscendC::LocalTensor<bfloat16_t> b1 = p ? b1Pong : b1Ping;
            AscendC::LocalTensor<bfloat16_t> a2 = p ? a2Pong : a2Ping;
            AscendC::LocalTensor<bfloat16_t> b2 = p ? b2Pong : b2Ping;

            AscendC::Mutex::Lock<PIPE_MTE2>(bufA);
            CopyInA(a1, kBlock, calcM);
            AscendC::Mutex::Unlock<PIPE_MTE2>(bufA);
            AscendC::Mutex::Lock<PIPE_MTE2>(bufB);
            CopyInB(b1, kBlock, nBlock);
            AscendC::Mutex::Unlock<PIPE_MTE2>(bufB);

            AscendC::Mutex::Lock<PIPE_MTE1>(bufL0);
            AscendC::Mutex::Lock<PIPE_MTE1>(bufA);
            AscendC::Mutex::Lock<PIPE_MTE1>(bufB);
            LoadA(a1, a2, calcMAlign);
            LoadB(b1, b2);
            AscendC::Mutex::Unlock<PIPE_MTE1>(bufA);
            AscendC::Mutex::Unlock<PIPE_MTE1>(bufB);
            AscendC::Mutex::Unlock<PIPE_MTE1>(bufL0);

            AscendC::Mutex::Lock<PIPE_M>(bufL0);
            AscendC::MmadParams mmadParams = {};
            mmadParams.m = calcM;
            mmadParams.n = RB_N;
            mmadParams.k = RB_K;
            mmadParams.cmatrixInitVal = (kBlock == 0u);
            AscendC::Mmad(cL0C, a2, b2, mmadParams);
            AscendC::Mutex::Unlock<PIPE_M>(bufL0);
        }
        AscendC::Mutex::Unlock<PIPE_M>(PFR_BUF_L0C);
        AscendC::Mutex::Lock<PIPE_FIX>(PFR_BUF_L0C);
        CopyOut(cL0C, nBlock, curM, calcMAlign);
        AscendC::Mutex::Unlock<PIPE_FIX>(PFR_BUF_L0C);
    }

private:
    // GM → L1：A 的一个 RB_K 大包（calcM 行 × RB_K bf16），Nd2Nz 落 L1
    __aicore__ inline void CopyInA(AscendC::LocalTensor<bfloat16_t>& a1, uint32_t kBlock, uint32_t curM)
    {
        AscendC::Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = curM;              // 行数（调用侧保证 >= 2：3510 行数=1 退化 quirk）
        par.dValue = RB_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = HIDDEN;
        par.dstNzC0Stride = MT;         // L1 NZ 布局按满 tile 行距（与 LoadA 的 srcStride 配套）
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1, xGm[kBlock * RB_K], par);
    }

    // GM → L1：B 按模型原始 [N, K] layout 搬 RB_N 行 × RB_K bf16
    __aicore__ inline void CopyInB(AscendC::LocalTensor<bfloat16_t>& b1, uint32_t kBlock, uint32_t nBlock)
    {
        AscendC::Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = RB_N;
        par.dValue = RB_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = HIDDEN;
        par.dstNzC0Stride = RB_N;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1, wGm[kBlock * RB_K + nBlock * RB_N * HIDDEN], par);
    }

    __aicore__ inline void LoadA(AscendC::LocalTensor<bfloat16_t>& a1, AscendC::LocalTensor<bfloat16_t>& a2,
                                 uint32_t calcMAlign)
    {
        AscendC::LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = calcMAlign / M15M::CUBE_BLOCK;
        lp.kStep = RB_K / M15M::CUBE_BLOCK;
        lp.srcStride = MT / M15M::CUBE_BLOCK;
        lp.dstStride = calcMAlign / M15M::CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        AscendC::LoadData(a2, a1[0], lp);
    }

    __aicore__ inline void LoadB(AscendC::LocalTensor<bfloat16_t>& b1, AscendC::LocalTensor<bfloat16_t>& b2)
    {
        AscendC::LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = RB_N / M15M::CUBE_BLOCK;
        lp.kStep = RB_K / M15M::CUBE_BLOCK;
        lp.srcStride = RB_N / M15M::CUBE_BLOCK;
        lp.dstStride = RB_N / M15M::CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        AscendC::LoadData(b2, b1[0], lp);
    }

    // L0C → GM：**fp32 直出**（`NoQuant` = 不量化 ⇒ L0C 的 fp32 累加值按 fp32 落 GM）。
    // 为什么不用 F322BF16：router 的 logits 要进 softmax / top-k，bf16 的 8 位尾数会把
    // 512 个专家的近邻名次搅乱（top-k 的选择是**比较**运算，误差直接变成 id 差异）。
    __aicore__ inline void CopyOut(AscendC::LocalTensor<float>& cL0C, uint32_t nBlock, uint32_t curM,
                                   uint32_t calcMAlign)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp = {};
        fp.nSize = RB_N;
        fp.mSize = curM;
        fp.srcStride = calcMAlign;
        fp.dstStride = E_PAD;
        fp.quantPre = QuantMode_t::NoQuant;
        fp.reluScalar = 0;
        fp.vectorRelu = 0;
        fp.deqScalar = 0;
        AscendC::Fixpipe(cGm[nBlock * RB_N], cL0C, fp);
    }

private:
    AscendC::GlobalTensor<bfloat16_t> xGm;
    AscendC::GlobalTensor<bfloat16_t> wGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t mTotal = 0;
};

// ============================================================
// 2. S2b（AIV）：行内 softmax(max-shift) + top-k（`docs/05` §6.1 ⓒ 的裁定例外路径）
//    输入 = **S2 的 mmad 产物**（GM 上的 fp32 logits，行距 `E_PAD`）；输出 = ids / weights / 共享门裸点积。
//    与 decode 档 `RouterStage` 的差别**只有输入来源与多核行分派**：排序/归并/Extract/renorm 的
//    算件代码逐字相同（同一 `RT_ROWL=512` 形状、同一 ub 窗语义）。
// ============================================================

class RouterTopkStage {
public:
    __aicore__ inline RouterTopkStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* logits, __gm__ uint8_t* ids, __gm__ uint8_t* weights,
                                __gm__ uint8_t* sgate, uint32_t m)
    {
        M = m;
        logitsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logits));
        idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
        sgateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sgate));
    }

    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        BuildIndexTemplate();
        for (uint32_t r = bid; r < M; r += nAiv) {
            CopyInRow(r);
            ComputeRow();
            CopyOutRow(r);
        }
    }

private:
    // 索引模板 0..RT_ROWL-1（`Arange` 一次只填 64 lane ⇒ 按 RT_NCHUNK_W 个 chunk 拼）
    // [逐字复制 m15_moe_layer.h 的 BuildIndexTemplate；E=512 恰好铺满 512 lane]
    __aicore__ inline void BuildIndexTemplate()
    {
        __ubuf__ int32_t* idxUb = reinterpret_cast<__ubuf__ int32_t*>(UB_PF_IDX);
        __VEC_SCOPE__
        {
            uint32_t n64 = 64;
            MaskReg m64i = UpdateMask<int32_t>(n64);
            for (uint16_t c = 0; c < static_cast<uint16_t>(RT_NCHUNK_W); ++c) {
                RegTensor<int32_t> rg;
                Arange(rg, static_cast<int32_t>(static_cast<uint32_t>(c) * 64));
                StoreAlign(idxUb + static_cast<uint32_t>(c) * 64, rg, m64i);
            }
        }
    }

    __aicore__ inline void CopyInRow(uint32_t r)
    {
        LocalTensor<float> logL(TPosition::VECCALC, UB_PF_LOG, RT_ROWL);
        LocalTensor<float> sgL(TPosition::VECCALC, UB_PF_SG, 1);
        BufAcquire<PIPE_MTE2>(PFR_BUF_ROW0);
        PipeBarrier<PIPE_MTE2>();   // 逐行复用同一 UB 窗 ⇒ 补 MTE2 排空（小改 B）
        DataCopyPad(logL, logitsGm[static_cast<uint64_t>(r) * E_PAD], ExtBlock1(E * 4),
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        // 共享门裸点积：mmad 输出的第 E 列（**未被本段改动**，是原始 logit）
        DataCopyPad(sgL, logitsGm[static_cast<uint64_t>(r) * E_PAD + E], ExtBlock1(4),
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        BufRelease<PIPE_MTE2>(PFR_BUF_ROW0);
    }

    __aicore__ inline void ComputeRow()
    {
        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_PF_LOG);
        __ubuf__ float* valUb = reinterpret_cast<__ubuf__ float*>(UB_PF_VAL);
        __ubuf__ float* ovUb = reinterpret_cast<__ubuf__ float*>(UB_PF_OV);
        __ubuf__ int32_t* oiUb = reinterpret_cast<__ubuf__ int32_t*>(UB_PF_OI);
        __ubuf__ float* wsUb = reinterpret_cast<__ubuf__ float*>(UB_PF_WS);
        __ubuf__ int32_t* idsUb = reinterpret_cast<__ubuf__ int32_t*>(UB_PF_IDS);

        BufAcquire<PIPE_V>(PFR_BUF_ROW0);   // 等 MTE2 写就位
        BufAcquire<PIPE_V>(PFR_BUF_ROW1);   // staging（等上一行的 MTE3 读走）

        // ① 行内 max + max-shift + exp（整行 512 lane，无 pad lane）
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
                StoreAlign(rowUb + static_cast<uint32_t>(c) * VL_F32, v, maskAll);   // logits -= rowmax
                Exp(v, v, maskAll);
                StoreAlign(valUb + static_cast<uint32_t>(c) * VL_F32, v, maskAll);
            }
        }
        // ② Sort32：16 个 32 块内降序（值, 全局 expert id）
        {
            LocalTensor<float> pairT(TPosition::VECCALC, UB_PF_PAIR, 2 * RT_ROWL);
            LocalTensor<float> valT(TPosition::VECCALC, UB_PF_VAL, RT_SORTLN);
            LocalTensor<uint32_t> idxT(TPosition::VECCALC, UB_PF_IDX, RT_SORTLN);
            Sort32(pairT, valT, idxT, RT_SORT_NBLK);
        }
        // ③ 二路归并树 → 根列表落在 UB_PF_MB（前 32 对 = 全局 top-32）
        MergeTree();
        // ④ Extract 根的前 64 对（VF 内联；纯 pair 拆分、无算术）
        for (uint32_t rep = 0; rep < RT_EXTRACT_REP; ++rep) {
            __VEC_SCOPE__
            {
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                RegTensor<float> vr, vi;
                uint32_t n32 = RT_LANES;
                MaskReg m32 = UpdateMask<float>(n32);
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    vr, vi, reinterpret_cast<__ubuf__ float*>(UB_PF_MB) + rep * 2 * RT_LANES);
                StoreAlign(ovUb + rep * RT_LANES, vr, m32);
                StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(UB_PF_OI) + rep * RT_LANES,
                           reinterpret_cast<RegTensor<uint32_t>&>(vi), m32);
            }
        }
        // ⑤ renorm（前 TOPK 对除以它们的和）→ staging
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
        BufRelease<PIPE_V>(PFR_BUF_ROW0);
        BufRelease<PIPE_V>(PFR_BUF_ROW1);
    }

    // 2 路归并树（UB_PF_MA / UB_PF_MB 乒乓）：16 个 32 块 → 4 级 → 根 = MB[0]
    // [逐字复制 m15_moe_layer.h 的 MergeTree，仅换 UB 偏移]
    __aicore__ inline void MergeTree()
    {
        LocalTensor<float> srcT(TPosition::VECCALC, UB_PF_PAIR, 2 * RT_ROWL);
        LocalTensor<float> aT(TPosition::VECCALC, UB_PF_MA, 2 * RT_ROWL);
        LocalTensor<float> bT(TPosition::VECCALC, UB_PF_MB, 2 * RT_ROWL);
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        for (uint32_t g = 0; g < 8; ++g) {
            Merge2(aT[g * 128], srcT[(2 * g) * 64], srcT[(2 * g + 1) * 64], 32);
        }
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        for (uint32_t g = 0; g < 4; ++g) {
            Merge2(bT[g * 128], aT[(2 * g) * 128], aT[(2 * g + 1) * 128], 32);
        }
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        for (uint32_t g = 0; g < 2; ++g) {
            Merge2(aT[g * 128], bT[(2 * g) * 128], bT[(2 * g + 1) * 128], 32);
        }
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        Merge2(bT[0], aT[0], aT[128], 32);
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
    }

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

    __aicore__ inline void CopyOutRow(uint32_t r)
    {
        LocalTensor<int32_t> idsL(TPosition::VECCALC, UB_PF_IDS, TOPK);
        LocalTensor<float> wsL(TPosition::VECCALC, UB_PF_WS, TOPK);
        LocalTensor<float> sgL(TPosition::VECCALC, UB_PF_SG, 1);
        BufAcquire<PIPE_MTE3>(PFR_BUF_ROW1);
        DataCopyPad(idsGm[static_cast<uint64_t>(r) * TOPK], idsL, ExtBlock1(TOPK * 4));
        DataCopyPad(wGm[static_cast<uint64_t>(r) * TOPK], wsL, ExtBlock1(TOPK * 4));
        DataCopyPad(sgateGm[r], sgL, ExtBlock1(4));
        BufRelease<PIPE_MTE3>(PFR_BUF_ROW1);
    }

private:
    uint32_t M = 0;
    AscendC::GlobalTensor<float> logitsGm;
    AscendC::GlobalTensor<int32_t> idsGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> sgateGm;
};

// ============================================================
// 3. S3（AIV0）：routing 索引生成（计数排序 → perm_src/perm_expert/counts/offsets/inv_slot）
//    **复制改自** `m15_moe_layer.h::IndexGenStage`，唯一差别 = `NUM_EXPERTS(4)` → `E(512)`：
//    decode 档那份把 E=4 写进了 counts 扫描、offsets 定尺与 GM 诊断槽下标，E=512 下装不下。
//    仍是 AIV0 单核（多核化 = S3 的**未完成项**，见 m26_moe_prefill/README.md「未完成项」）。
// ============================================================
constexpr uint32_t IGP_DIAG_SLOT = ((E + 1u) * 4u + 4u + PWS_ALIGN - 1u) / PWS_ALIGN * PWS_ALIGN / 4u;

class IndexGenP {
public:
    __aicore__ inline IndexGenP() {}

    // **只读 ids**（新规）：权重**不再**在这里做 scalar 的 bf16 位打包（那是"用标量算数据"）——
    // 改由 `UnpermuteP` 在 VF 里 Cast 取整（见 §4b）。故本函数不再需要 weights/wtk 两个平面的入口。
    __aicore__ inline void Init(__gm__ uint8_t* ids, __gm__ uint8_t* counts, __gm__ uint8_t* offsets,
                                __gm__ uint8_t* permSrc, __gm__ uint8_t* permExp, __gm__ uint8_t* inv,
                                uint32_t m, uint32_t topk)
    {
        M = m;
        TOPK = topk;
        idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts));
        offsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
        permSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(permSrc));
        permExpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(permExp));
        invGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(inv));
    }

    __aicore__ inline uint32_t OobCount() const { return oobCountUb; }

    __aicore__ inline void Run()
    {
        // **落 GM 的可见性**：ids/weights 是 router 段（MTE3）写的，本段用**标量**读它们的 GM 值
        // ⇒ 必须有 MTE3→S 的次序。规则 ⓔ 禁 `SetFlag/WaitFlag` 系列（人类逐字禁令 + docs/05 §2），
        // 故这里用**同一 buffer token** 的阻塞释放 / 读侧 acquire 表达（调用方在进本函数前
        // 已 `BufRelease<PIPE_MTE3>(PFR_BUF_ROW1)`，本函数 `BufAcquire<PIPE_S>(PFR_BUF_ROW1)` 配对）。
        BufAcquire<PIPE_S>(PFR_BUF_ROW1);

        const uint32_t S = M * TOPK;
        uint32_t badIds = 0;
        __ubuf__ int32_t* cntUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_SCAL);
        __ubuf__ int32_t* offUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_OFF);
        __ubuf__ int32_t* curUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_CUR);
        for (uint32_t e = 0; e < E; ++e) {
            cntUb[e] = 0;
        }
        for (uint32_t s = 0; s < S; ++s) {
            const int32_t e = idsGm.GetValue(s);
            if (e < 0 || e >= static_cast<int32_t>(E)) {
                ++badIds;
                continue;
            }
            cntUb[e] = cntUb[e] + 1;
        }
        oobCountUb = badIds;
        offUb[0] = 0;
        for (uint32_t e = 0; e < E; ++e) {
            offUb[e + 1] = offUb[e] + cntUb[e];
        }

        __ubuf__ int32_t* srcUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_SRC);
        __ubuf__ int32_t* expUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_EXP);
        __ubuf__ int32_t* invUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_INV);

        MutexLock<PIPE_S>(PFR_BUF_IDX);
        for (uint32_t e = 0; e < E; ++e) {
            curUb[e] = offUb[e];
        }
        for (uint32_t t = 0; t < M; ++t) {
            for (uint32_t k = 0; k < TOPK; ++k) {
                const uint32_t s = t * TOPK + k;
                const int32_t e = idsGm.GetValue(s);
                if (e < 0 || e >= static_cast<int32_t>(E) || badIds != 0) {
                    ++badIds;
                    continue;
                }
                const uint32_t pos = static_cast<uint32_t>(curUb[e]);
                curUb[e] = static_cast<int32_t>(pos + 1);
                srcUb[pos] = static_cast<int32_t>(t);
                expUb[pos] = e;
                invUb[s] = static_cast<int32_t>(pos);   // 紧凑行号
            }
        }
        MutexUnlock<PIPE_S>(PFR_BUF_IDX);
        BufRelease<PIPE_S>(PFR_BUF_ROW1);

        // UB → GM（S 写 → MTE3 读 的 BufferID 握手）
        LocalTensor<int32_t> srcL(TPosition::VECCALC, UB_IG_SRC, S);
        LocalTensor<int32_t> expL(TPosition::VECCALC, UB_IG_EXP, S);
        LocalTensor<int32_t> invL(TPosition::VECCALC, UB_IG_INV, S);
        LocalTensor<int32_t> cntL(TPosition::VECCALC, UB_IG_SCAL, E);
        LocalTensor<int32_t> offL(TPosition::VECCALC, UB_IG_OFF, E + 1);
        BufAcquire<PIPE_MTE3>(PFR_BUF_IDX);
        DataCopyPad(permSrcGm[0], srcL, ExtBlock1(S * 4));
        DataCopyPad(permExpGm[0], expL, ExtBlock1(S * 4));
        DataCopyPad(invGm[0], invL, ExtBlock1(S * 4));
        DataCopyPad(countsGm[0], cntL, ExtBlock1(E * 4));
        DataCopyPad(offsGm[0], offL, ExtBlock1((E + 1) * 4));
        // 诊断槽（IG 越界计数）：UB 源 32B 对齐（+128B）；GM 目的落在 offsets 区尾部 32B 槽
        {
            __ubuf__ int32_t* bUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_CUR);
            bUb[32] = static_cast<int32_t>(oobCountUb);
            LocalTensor<int32_t> bL(TPosition::VECCALC, UB_IG_CUR, 64);
            DataCopyPad(offsGm[IGP_DIAG_SLOT], bL[32], ExtBlock1(4));
        }
        BufRelease<PIPE_MTE3>(PFR_BUF_IDX);
    }

private:
    uint32_t M = 0;
    uint32_t TOPK = 1;
    uint32_t oobCountUb = 0;
    AscendC::GlobalTensor<int32_t> idsGm;
    AscendC::GlobalTensor<int32_t> countsGm;
    AscendC::GlobalTensor<int32_t> offsGm;
    AscendC::GlobalTensor<int32_t> permSrcGm;
    AscendC::GlobalTensor<int32_t> permExpGm;
    AscendC::GlobalTensor<int32_t> invGm;
};

// ============================================================
// 4. S5'/S7'（AIV）：MXFP4 行级量化 / SwiGLU+量化
//    **复制改自** `m15_moe_layer.h::VecQuantStage`：`NUM_EXPERTS` → `E`；去掉 prefill 不需要的
//    `PAD_SRC/PAD_DST`（本段的槽一律**紧凑**：源与目的行号都就是 `r`，不需要 slot 映射）。
// ============================================================

template <uint32_t K, uint32_t SCAL_STRIDE, bool SWIGLU, bool ROUTED>
class VecQuantP {
public:
    static constexpr uint32_t TILE = 256;
    static constexpr uint32_t FULL_TILES = K / TILE;
    static constexpr uint32_t NTILES = (K % TILE == 0) ? FULL_TILES : FULL_TILES + 1;
    static constexpr uint32_t PACKED_K = K / 2;
    static constexpr uint32_t SRC_ROW = SWIGLU ? (2 * K) : K;
    static constexpr uint32_t GP = K / GROUP;

    __aicore__ inline VecQuantP() {}

    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* swigluOut, __gm__ uint8_t* qx,
                                __gm__ uint8_t* scale, __gm__ uint8_t* counts, uint32_t m, uint32_t topk)
    {
        M = m;
        TOPK = topk;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(src));
        swigluGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(swigluOut));
        qxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(qx));
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scale));
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts));
    }

    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        // 行数：routed 路 = Σt_e（**不写成常量**，也不依赖 E 的取值），shared 路 = m 行
        uint32_t rows = M;
        if constexpr (ROUTED) {
            rows = 0;
            for (uint32_t e = 0; e < E; ++e) {
                rows += static_cast<uint32_t>(countsGm.GetValue(e));
            }
        }
        for (uint32_t r = bid; r < rows; r += nAiv) {
            for (uint32_t t = 0; t < NTILES; ++t) {
                const uint32_t t0 = (t < FULL_TILES) ? t * TILE : (K - TILE);
                ProcessTile(r, r, r, t0);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t packedRow, uint32_t srcRow, uint32_t dstRow, uint32_t t0)
    {
        const uint64_t xOff = static_cast<uint64_t>(srcRow) * SRC_ROW + t0;
        const uint64_t hOff = static_cast<uint64_t>(packedRow) * K + t0;
        const uint64_t qxOff = static_cast<uint64_t>(dstRow) * PACKED_K + t0 / 2;
        const uint64_t sOff = static_cast<uint64_t>(dstRow) * (SCAL_STRIDE / 2) + t0 / GROUP / 2;

        LocalTensor<bfloat16_t> gateL(TPosition::VECCALC, UB_QT_GATE, TILE);
        LocalTensor<bfloat16_t> upL(TPosition::VECCALC, UB_QT_GATE + TILE * 2, TILE);
        LocalTensor<bfloat16_t> swigluL(TPosition::VECCALC, UB_QT_SWIGLU, TILE);
        LocalTensor<int8_t> qxL(TPosition::VECCALC, UB_QT_QX, TILE / 2);
        LocalTensor<uint16_t> scaleL(TPosition::VECCALC, UB_QT_SCALE, TILE / GROUP / 2);
        LocalTensor<uint16_t> halfL(TPosition::VECCALC, UB_QT_HALF, TILE / GROUP);

        __ubuf__ bfloat16_t* gateUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_QT_GATE);
        __ubuf__ bfloat16_t* upUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_QT_GATE + TILE * 2);
        __ubuf__ bfloat16_t* swigluUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_QT_SWIGLU);
        __ubuf__ int8_t* qxUb = reinterpret_cast<__ubuf__ int8_t*>(UB_QT_QX);
        __ubuf__ uint16_t* scaleUb = reinterpret_cast<__ubuf__ uint16_t*>(UB_QT_SCALE);
        __ubuf__ uint16_t* halfUb = reinterpret_cast<__ubuf__ uint16_t*>(UB_QT_HALF);

        BufAcquire<PIPE_MTE2>(PFR_BUF_ROW0);
        PipeBarrier<PIPE_MTE2>();   // 小改 B：tile 循环同 pipe 复用同一 UB tile
        DataCopy(gateL, srcGm[xOff], Block1(TILE * 2));
        if constexpr (SWIGLU) {
            DataCopy(upL, srcGm[xOff + K], Block1(TILE * 2));
        }
        BufRelease<PIPE_MTE2>(PFR_BUF_ROW0);

        BufAcquire<PIPE_V>(PFR_BUF_ROW0);
        BufAcquire<PIPE_V>(PFR_BUF_OUT);
        if constexpr (SWIGLU) {
            SwigluComputeTile(gateUb, upUb, swigluUb);
            MxQuantComputeScale(swigluUb, scaleUb, halfUb);
            MxQuantComputeDataFP4(swigluUb, halfUb, qxUb);
        } else {
            MxQuantComputeScale(gateUb, scaleUb, halfUb);
            MxQuantComputeDataFP4(gateUb, halfUb, qxUb);
        }
        BufRelease<PIPE_V>(PFR_BUF_ROW0);
        BufRelease<PIPE_V>(PFR_BUF_OUT);

        BufAcquire<PIPE_MTE3>(PFR_BUF_OUT);
        if constexpr (SWIGLU) {
            DataCopy(swigluGm[hOff], swigluL, Block1(TILE * 2));
        }
        DataCopy(qxGm[qxOff], qxL, Block1(TILE / 2));
        DataCopyPad<uint16_t, PaddingMode::Compact>(scaleGm[sOff], scaleL, ExtBlock1(TILE / GROUP));
        BufRelease<PIPE_MTE3>(PFR_BUF_OUT);
    }

    // SwiGLU 五元组（**逐字复制** m15_moe_layer.h::VecQuantStage::SwigluComputeTile）
    __aicore__ inline void SwigluComputeTile(__ubuf__ bfloat16_t* gateUb, __ubuf__ bfloat16_t* upUb,
                                             __ubuf__ bfloat16_t* swigluUb)
    {
        constexpr uint32_t VL = 64;
        constexpr uint16_t LOOP = TILE / VL;
        __VEC_SCOPE__
        {
            RegTensor<bfloat16_t> tg;
            RegTensor<bfloat16_t> tu;
            RegTensor<bfloat16_t> ty;
            RegTensor<float> g;
            RegTensor<float> u;
            RegTensor<float> y;
            RegTensor<float> v;
            uint32_t vl = VL;
            MaskReg dataMask = UpdateMask<float>(vl);
            for (uint16_t j = 0; j < LOOP; ++j) {
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tg, gateUb + j * VL);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tu, upUb + j * VL);
                Cast<float, bfloat16_t, M15M::NormDonor::castTraitB162B32>(g, tg, dataMask);
                Cast<float, bfloat16_t, M15M::NormDonor::castTraitB162B32>(u, tu, dataMask);
                Muls(v, g, -1.0f, dataMask);
                Exp(v, v, dataMask);
                Adds(v, v, 1.0f, dataMask);
                Div(v, g, v, dataMask);
                Mul(y, v, u, dataMask);
                Cast<bfloat16_t, float, M15M::NormDonor::castTraitB322B16>(ty, y, dataMask);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(swigluUb + j * VL, ty, dataMask);
            }
        }
    }

    // 以下两段**逐字复制** m15_moe_layer.h::VecQuantStage 的同名函数（MxQuant 三段式）
    __aicore__ inline void MxQuantComputeScale(__ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                               __ubuf__ uint16_t* halfScaleLocalAddr)
    {
        constexpr uint32_t groups = TILE / GROUP;
        constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
        constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
        constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
        constexpr int16_t SHR_NUM_FOR_BF16 = 7;
        constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
        constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
        constexpr uint32_t VL_B16 = 128;
        constexpr int64_t DIGIT_TWO = 2;
        __VEC_SCOPE__
        {
            RegTensor<bfloat16_t> vdExp0;
            RegTensor<bfloat16_t> vdExp1;
            RegTensor<uint16_t> vdExpExtract0;
            RegTensor<uint16_t> vdExpExtract1;
            RegTensor<uint16_t> expMaskBF16;
            Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
            RegTensor<uint16_t> vdMaxExp;
            MaskReg Mask = CreateMask<uint16_t, MaskPattern::ALL>();
            MaskReg cmpResult;
            MaskReg zeroMask;
            MaskReg preMaskScale;
            RegTensor<uint16_t> maxExpValue;
            Duplicate(maxExpValue, FP4_E2M1_BF16_MAX_EXP);
            RegTensor<uint16_t> sharedExp;
            RegTensor<uint16_t> scaleValue;
            RegTensor<uint16_t> scaleBias;
            Duplicate(scaleBias, BF16_EXP_BIAS);
            RegTensor<uint16_t> halfScale;
            RegTensor<uint16_t> fp8NanRegTensor;
            Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
            RegTensor<uint16_t> nanRegTensor;
            Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
            RegTensor<uint16_t> zeroRegTensor;
            Duplicate(zeroRegTensor, 0);

            uint32_t nGroups = groups;
            preMaskScale = UpdateMask<uint16_t>(nGroups);
            LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            And(vdExpExtract0, (RegTensor<uint16_t>&)vdExp0, expMaskBF16, Mask);
            And(vdExpExtract1, (RegTensor<uint16_t>&)vdExp1, expMaskBF16, Mask);
            Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
            ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(vdMaxExp, vdMaxExp, Mask);

            Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMaskBF16, preMaskScale);
            Compare<uint16_t, CMPMODE::LE>(zeroMask, vdMaxExp, maxExpValue, preMaskScale);
            Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, zeroMask);
            Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, nGroups / 2, preMaskScale);
            Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroRegTensor, preMaskScale);
            Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr, halfScale, nGroups, preMaskScale);
        }
    }

    __aicore__ inline void MxQuantComputeDataFP4(__ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                                 __ubuf__ int8_t* outLocalAddr)
    {
        constexpr uint32_t VL_B16 = 128;
        constexpr int64_t DIGIT_TWO = 2;
        __VEC_SCOPE__
        {
            MaskReg dataMask1;
            RegTensor<uint16_t> halfScaleForMul;
            RegTensor<bfloat16_t> vdExp0;
            RegTensor<bfloat16_t> vdExp1;
            RegTensor<fp4x2_e2m1_t> vdExp0FP4;
            RegTensor<fp4x2_e2m1_t> vdExp1FP4;
            uint32_t totalCount = TILE;
            dataMask1 = UpdateMask<bfloat16_t>(totalCount);
            LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, VL_B16 * DIGIT_TWO);
            LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, TILE / GROUP);
            Mul(vdExp0, vdExp0, (RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
            Mul(vdExp1, vdExp1, (RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
            Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            Cast<fp4x2_e2m1_t, bfloat16_t, M15M::castTraitRM_Round>(vdExp0FP4, vdExp0, dataMask1);
            Cast<fp4x2_e2m1_t, bfloat16_t, M15M::castTraitRM_Round>(vdExp1FP4, vdExp1, dataMask1);
            StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (RegTensor<int8_t>&)vdExp0FP4, 64, dataMask1);
            StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (RegTensor<int8_t>&)vdExp1FP4, 64, dataMask1);
        }
    }

private:
    uint32_t M = 0;
    uint32_t TOPK = 1;
    AscendC::GlobalTensor<bfloat16_t> srcGm;
    AscendC::GlobalTensor<bfloat16_t> swigluGm;
    AscendC::GlobalTensor<int8_t> qxGm;
    AscendC::GlobalTensor<uint16_t> scaleGm;
    AscendC::GlobalTensor<int32_t> countsGm;
};

// ============================================================
// 4b. S4' permute / S9a' unpermute / S9b' combine
//     **复制改自** `m15_moe_layer.h` 的同名类（r1 复审 P2-1）：decode 档那份把 GM 视图上界写死成
//     `TOTAL_MAX = M_MAX*TOPK_MAX = 256` 行 / 条，而 prefill 的 `Σt_e` 上界是 `TP_MAX = MT*TOPK = 640`
//     ⇒ 真 shape 下**视图说小了**（设备侧不做越界检查，但形态上不可用）。这里按 prefill 定尺真复制。
//     另按新规（矩阵乘法⇒cube；其余用 VF；**scalar 只做控制流**）：unpermute 的 top-k 权重
//     **不再走 scalar 的 bf16 RNE 位打包**（那是"用标量算数据"），改成读 fp32 权重 + **VF 里
//     Cast 成 bf16 再展回 fp32**（与 donor 的舍入点逐位相同：bf16(RNE) 再乘）。
// ============================================================

constexpr uint32_t PM_STAGES = 4;

class PermuteP {
public:
    __aicore__ inline PermuteP() {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* permSrc, __gm__ uint8_t* counts,
                                __gm__ uint8_t* xSorted, uint32_t numExperts, uint32_t m, uint32_t topk)
    {
        M = m;
        TOPK = topk;
        E_ = numExperts;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), static_cast<uint64_t>(MT) * HIDDEN);
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(permSrc), TP_MAX);
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts), numExperts);
        xSortedGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xSorted),
                                  static_cast<uint64_t>(TP_MAX) * HIDDEN);
    }

    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        uint32_t total = 0;
        for (uint32_t e = 0; e < E_; ++e) {
            total += static_cast<uint32_t>(countsGm.GetValue(e));
        }
        TOTAL = total;
        const uint32_t myRows = (total > bid) ? (total - bid - 1) / nAiv + 1 : 0;
        for (uint32_t s = 0; s < PM_STAGES && s < myRows; ++s) {
            CopyIn(bid + s * nAiv, s);
        }
        for (uint32_t r = 0; r < myRows; ++r) {
            const uint32_t stage = r % PM_STAGES;
            CopyOut(bid + r * nAiv, stage);
            if (r + PM_STAGES < myRows) {
                CopyIn(bid + (r + PM_STAGES) * nAiv, stage);
            }
        }
    }

    __aicore__ inline uint32_t Total() const { return TOTAL; }

private:
    __aicore__ inline void CopyIn(uint32_t row, uint32_t stage)
    {
        LocalTensor<bfloat16_t> bufL(TPosition::VECCALC, UB_PM_ROW + stage * HIDDEN * 2, HIDDEN);
        const int32_t src = srcGm.GetValue(row);   // 标量 GM 读 = 地址推导（合法）
        BufAcquire<PIPE_MTE2>(PFR_BUF_PERM0 + stage);
        PipeBarrier<PIPE_MTE2>();
        DataCopy(bufL, xGm[static_cast<uint64_t>(src) * HIDDEN], Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE2>(PFR_BUF_PERM0 + stage);
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t stage)
    {
        LocalTensor<bfloat16_t> bufL(TPosition::VECCALC, UB_PM_ROW + stage * HIDDEN * 2, HIDDEN);
        BufAcquire<PIPE_MTE3>(PFR_BUF_PERM0 + stage);
        DataCopy(xSortedGm[static_cast<uint64_t>(row) * HIDDEN], bufL, Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE3>(PFR_BUF_PERM0 + stage);
    }

private:
    uint32_t M = 0;
    uint32_t TOPK = 1;
    uint32_t E_ = 1;
    uint32_t TOTAL = 0;
    AscendC::GlobalTensor<bfloat16_t> xGm;
    AscendC::GlobalTensor<int32_t> srcGm;
    AscendC::GlobalTensor<int32_t> countsGm;
    AscendC::GlobalTensor<bfloat16_t> xSortedGm;
};

// 每个 stage 的 UB 子布局（字节；全部 32B 对齐）：y = TOPK*HIDDEN*2，out = HIDDEN*2，w = TOPK*4
constexpr uint32_t UB_U_Y = 0;
constexpr uint32_t UB_U_OUT = UB_U_Y + TOPK * HIDDEN * 2;                 // 51200
constexpr uint32_t UB_U_W = UB_U_OUT + HIDDEN * 2;                       // 56320（+40B 权重，含 64B 余量）
static_assert(UB_U_W + ((TOPK * 4u + 31u) / 32u) * 32u <= U_STAGE_B, "UnpermuteP 的每 stage 子布局超出 U_STAGE_B");

template <uint32_t TK>
__aicore__ inline void FmaChunkP(RegTensor<float>& acc, RegTensor<float>& yF, RegTensor<bfloat16_t>& yB16,
                                 RegTensor<float>& wF, __ubuf__ bfloat16_t* yUb, __ubuf__ float* wRowUb,
                                 MaskReg& maskAll, uint32_t off)
{
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(yB16, yUb + TK * HIDDEN + off);
    Cast<float, bfloat16_t, M15M::NormDonor::castTraitB162B32>(yF, yB16, maskAll);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(wF, wRowUb + TK);   // 广播**已 VF 取整**的 w_k
    Mul(yF, yF, wF, maskAll);
    Add(acc, acc, yF, maskAll);
}

template <uint32_t TK>
class UnpermuteP {
public:
    __aicore__ inline UnpermuteP() {}

    // wFp32 = **fp32** 权重平面（`PWS_W`，`MT*TOPK` 条）；bf16 取整在 VF 里做（scalar 只做下标）
    __aicore__ inline void Init(__gm__ uint8_t* ySorted, __gm__ uint8_t* invSlot, __gm__ uint8_t* wFp32,
                                __gm__ uint8_t* out, uint32_t m)
    {
        M = m;
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ySorted), static_cast<uint64_t>(TP_MAX) * HIDDEN);
        slotGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(invSlot), static_cast<uint64_t>(MT) * TOPK);
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wFp32), static_cast<uint64_t>(MT) * TOPK);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(MT) * HIDDEN);
    }

    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        const uint32_t myRows = (M > bid) ? (M - bid - 1) / nAiv + 1 : 0;
        for (uint32_t r = 0; r < myRows; ++r) {
            const uint32_t t = bid + r * nAiv;
            const uint32_t stage = r % 2u;
            CopyIn(t, stage);
            ComputeRow(stage);
            CopyOut(t, stage);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t t, uint32_t stage)
    {
        LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_Y, TK * HIDDEN);
        LocalTensor<float> wL(TPosition::VECCALC, UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_W, TK);
        BufAcquire<PIPE_MTE2>(PFR_BUF_ROW0 + stage);
        PipeBarrier<PIPE_MTE2>();
        DataCopyPad(wL, wGm[static_cast<uint64_t>(t) * TK], ExtBlock1(TK * 4),
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        for (uint32_t k = 0; k < TK; ++k) {
            const int32_t slot = slotGm.GetValue(static_cast<uint64_t>(t) * TK + k);   // 标量 GM 读 = 地址
            DataCopy(yL[k * HIDDEN], yGm[static_cast<uint64_t>(slot) * HIDDEN], Block1(HIDDEN * 2));
        }
        BufRelease<PIPE_MTE2>(PFR_BUF_ROW0 + stage);
    }

    __aicore__ inline void ComputeRow(uint32_t stage)
    {
        __ubuf__ bfloat16_t* yUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_Y);
        __ubuf__ bfloat16_t* outUb =
            reinterpret_cast<__ubuf__ bfloat16_t*>(UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_OUT);
        __ubuf__ float* wRowUb = reinterpret_cast<__ubuf__ float*>(UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_W);

        BufAcquire<PIPE_V>(PFR_BUF_ROW0 + stage);
        // ① **VF 里的权重取整**（新规：scalar 不参与数据计算）：fp32 → bf16(RNE) → fp32，原地写回。
        //    与 donor 的 "scalar 打包 bf16 位 → 展回 fp32" 舍入点**逐位相同**。
        __VEC_SCOPE__
        {
            RegTensor<float> wF;
            RegTensor<bfloat16_t> wB;
            uint32_t nk = TK;
            MaskReg mk = UpdateMask<float>(nk);
            LoadAlign(wF, wRowUb);
            Cast<bfloat16_t, float, M15M::NormDonor::castTraitB322B16>(wB, wF, mk);
            Cast<float, bfloat16_t, M15M::NormDonor::castTraitB162B32>(wF, wB, mk);
            StoreAlign(wRowUb, wF, mk);
        }
        __VEC_SCOPE__
        {
            RegTensor<float> acc;
            RegTensor<float> yF;
            RegTensor<float> wF;
            RegTensor<bfloat16_t> yB16;
            RegTensor<bfloat16_t> outB16;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t c = 0; c < CHUNKS_H; ++c) {
                const uint32_t off = static_cast<uint32_t>(c) * VL_F32;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(yB16, yUb + off);
                Cast<float, bfloat16_t, M15M::NormDonor::castTraitB162B32>(yF, yB16, maskAll);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(wF, wRowUb);
                Mul(acc, yF, wF, maskAll);
                if constexpr (TK > 1) FmaChunkP<1>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 2) FmaChunkP<2>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 3) FmaChunkP<3>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 4) FmaChunkP<4>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 5) FmaChunkP<5>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 6) FmaChunkP<6>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 7) FmaChunkP<7>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 8) FmaChunkP<8>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 9) FmaChunkP<9>(acc, yF, yB16, wF, yUb, wRowUb, maskAll, off);
                Cast<bfloat16_t, float, M15M::NormDonor::castTraitB322B16>(outB16, acc, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(outUb + off, outB16, maskAll);
            }
        }
        BufRelease<PIPE_V>(PFR_BUF_ROW0 + stage);
    }

    __aicore__ inline void CopyOut(uint32_t t, uint32_t stage)
    {
        LocalTensor<bfloat16_t> outL(TPosition::VECCALC, UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_OUT, HIDDEN);
        BufAcquire<PIPE_MTE3>(PFR_BUF_ROW0 + stage);
        DataCopy(outGm[static_cast<uint64_t>(t) * HIDDEN], outL, Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE3>(PFR_BUF_ROW0 + stage);
    }

private:
    uint32_t M = 0;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    AscendC::GlobalTensor<int32_t> slotGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;
};

// S9b' combine（**逐字复制** m15_moe_layer.h::CombineStage，视图按 MT 定尺）
class CombineP {
public:
    __aicore__ inline CombineP() {}

    __aicore__ inline void Init(__gm__ uint8_t* routed, __gm__ uint8_t* shdMlp, __gm__ uint8_t* sgate,
                                __gm__ uint8_t* sharedOut, __gm__ uint8_t* moeOut, uint32_t m)
    {
        M = m;
        routedGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(routed), static_cast<uint64_t>(MT) * HIDDEN);
        shdGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(shdMlp), static_cast<uint64_t>(MT) * HIDDEN);
        sgateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sgate), MT);
        sharedGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sharedOut), static_cast<uint64_t>(MT) * HIDDEN);
        moeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(moeOut), static_cast<uint64_t>(MT) * HIDDEN);
    }

    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        for (uint32_t t = bid; t < M; t += nAiv) {
            CopyInRow(t);
            ComputeRow();
            CopyOutRow(t);
        }
    }

private:
    __aicore__ inline void CopyInRow(uint32_t t)
    {
        LocalTensor<bfloat16_t> rtL(TPosition::VECCALC, UB_CB_ROUTED, HIDDEN);
        LocalTensor<bfloat16_t> shL(TPosition::VECCALC, UB_CB_SHARED, HIDDEN);
        LocalTensor<float> sgL(TPosition::VECCALC, UB_CB_G, 64);
        const uint64_t off = static_cast<uint64_t>(t) * HIDDEN;
        // 整行（MTE2 写 → V 就地改 → MTE3 读）必须共用一个 BufferID
        BufAcquire<PIPE_MTE2>(PFR_BUF_ROW0);
        PipeBarrier<PIPE_MTE2>();
        DataCopy(rtL, routedGm[off], Block1(HIDDEN * 2));
        DataCopy(shL, shdGm[off], Block1(HIDDEN * 2));
        DataCopyPad(sgL, sgateGm[t], ExtBlock1(4), DataCopyPadExtParams<float>{false, 0, 0, 0});
        BufRelease<PIPE_MTE2>(PFR_BUF_ROW0);
    }

    __aicore__ inline void ComputeRow()
    {
        __ubuf__ bfloat16_t* rtUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_CB_ROUTED);
        __ubuf__ bfloat16_t* shUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_CB_SHARED);
        __ubuf__ float* sgUb = reinterpret_cast<__ubuf__ float*>(UB_CB_G);

        BufAcquire<PIPE_V>(PFR_BUF_ROW0);   // 与 MTE2 写、MTE3 读共用同一 token
        __VEC_SCOPE__
        {
            RegTensor<float> g;
            RegTensor<float> one;
            RegTensor<float> shF;
            RegTensor<float> rtF;
            RegTensor<bfloat16_t> shB;
            RegTensor<bfloat16_t> rtB;
            RegTensor<bfloat16_t> outB;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            Duplicate(one, 1.0f);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(g, sgUb);
            Muls(g, g, -1.0f, maskAll);
            Exp(g, g, maskAll);
            Adds(g, g, 1.0f, maskAll);
            Div(g, one, g, maskAll);                             // sigmoid = 1/(1+exp(-x))
            for (uint16_t c = 0; c < CHUNKS_H; ++c) {
                const uint32_t off = static_cast<uint32_t>(c) * VL_F32;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(shB, shUb + off);
                Cast<float, bfloat16_t, M15M::NormDonor::castTraitB162B32>(shF, shB, maskAll);
                Mul(shF, shF, g, maskAll);
                Cast<bfloat16_t, float, M15M::NormDonor::castTraitB322B16>(outB, shF, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(shUb + off, outB, maskAll);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(rtB, rtUb + off);
                Cast<float, bfloat16_t, M15M::NormDonor::castTraitB162B32>(rtF, rtB, maskAll);
                Add(shF, shF, rtF, maskAll);
                Cast<bfloat16_t, float, M15M::NormDonor::castTraitB322B16>(outB, shF, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(rtUb + off, outB, maskAll);
            }
        }
        BufRelease<PIPE_V>(PFR_BUF_ROW0);
    }

    __aicore__ inline void CopyOutRow(uint32_t t)
    {
        LocalTensor<bfloat16_t> shL(TPosition::VECCALC, UB_CB_SHARED, HIDDEN);
        LocalTensor<bfloat16_t> moeL(TPosition::VECCALC, UB_CB_ROUTED, HIDDEN);
        const uint64_t off = static_cast<uint64_t>(t) * HIDDEN;
        BufAcquire<PIPE_MTE3>(PFR_BUF_ROW0);
        DataCopy(sharedGm[off], shL, Block1(HIDDEN * 2));
        DataCopy(moeGm[off], moeL, Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE3>(PFR_BUF_ROW0);
    }

private:
    uint32_t M = 0;
    AscendC::GlobalTensor<bfloat16_t> routedGm;
    AscendC::GlobalTensor<bfloat16_t> shdGm;
    AscendC::GlobalTensor<float> sgateGm;
    AscendC::GlobalTensor<bfloat16_t> sharedGm;
    AscendC::GlobalTensor<bfloat16_t> moeGm;
};

// ============================================================
// 5. 段序编排（S1–S10，按 `MT` 行分块；单一 `__mix__(1,2)` 启动内跑完一层的一块）
//    同步一律：核内 **BufferID**；核间 **CrossCoreSetFlag/WaitFlag**（mode 0 = 同类 all-to-all，
//    mode 2 = AIC↔配对 AIV）；**不用** SetFlag/WaitFlag 系列；除值依赖外不用 PIPE_S。
// ============================================================

using M15M::NormStage;
using M15M::PrecastGammaF32;

struct MoePrefillPtrs {
    __gm__ uint8_t* ws;
    __gm__ uint8_t* xLayer;     // [MT, HIDDEN] bf16（本 tile 的层输入 = 上一层残差流）
    __gm__ uint8_t* yLayer;     // S10 出口 [MT, HIDDEN] bf16
    __gm__ uint8_t* resZero;    // [MT, HIDDEN] bf16 全 0（S1 的残差入口）
    __gm__ uint8_t* gamma1;     // [HIDDEN] bf16
    __gm__ uint8_t* gamma2;     // [HIDDEN] bf16
    __gm__ uint8_t* routerWpad; // [E_PAD, HIDDEN] bf16（host 组板：512 专家 + 共享门（第 E 行）+ pad 行）
    __gm__ uint8_t* wGu;        // [E, GU_N, HIDDEN/2] u8（门控|up 拼接，MXFP4 packed）
    __gm__ uint8_t* sGu;        // [E, GU_N, HIDDEN/32] u8（e8m0 scale）
    __gm__ uint8_t* wDn;        // [E, HIDDEN, INTER/2] u8
    __gm__ uint8_t* sDn;        // [E, HIDDEN, INTER/32] u8
    __gm__ uint8_t* wGuShd;     // 共享专家（gate|up），[GU_N, HIDDEN/2] u8
    __gm__ uint8_t* sGuShd;
    __gm__ uint8_t* wDnShd;     // [HIDDEN, INTER/2] u8
    __gm__ uint8_t* sDnShd;
    uint32_t m;                 // 本 tile 的有效行数（1..MT）
    uint32_t topk;              // 运行期 top-k（真实档 10；仍按运行期参数、不退化成常量）
    uint32_t stageLimit;        // 段序截断（bring-up 定位用；8 = 全链）
};

class MoePrefillChain {
public:
    __aicore__ inline MoePrefillChain() {}

    __aicore__ inline void Init(const MoePrefillPtrs& a)
    {
        p = a;
        ws = a.ws;
        norm1.Init(a.xLayer, a.resZero, ws + PWS_XNORM, ws + PWS_RES1, M15M::UB_GAMMA1_F32, a.m);
        norm2.Init(ws + PWS_MOE, ws + PWS_RES1, a.yLayer, ws + PWS_RES2, M15M::UB_GAMMA2_F32, a.m);
        router.Init(a.xLayer, a.routerWpad, ws + PWS_LOGITS, a.m);
        topk.Init(ws + PWS_LOGITS, ws + PWS_IDS, ws + PWS_W, ws + PWS_SGATE, a.m);
        ig.Init(ws + PWS_IDS, ws + PWS_COUNTS, ws + PWS_OFFSETS, ws + PWS_PERM_SRC, ws + PWS_PERM_EXP,
                ws + PWS_INV, a.m, a.topk);
        perm.Init(ws + PWS_XNORM, ws + PWS_PERM_SRC, ws + PWS_COUNTS, ws + PWS_XSORT, E, a.m, a.topk);
        quantA.Init(ws + PWS_XSORT, nullptr, ws + PWS_AQ, ws + PWS_AS, ws + PWS_COUNTS, a.m, a.topk);
        quantAShd.Init(ws + PWS_XNORM, nullptr, ws + PWS_AQ_SHD, ws + PWS_AS_SHD, ws + PWS_COUNTS, a.m, a.topk);
        quantH.Init(ws + PWS_GU, ws + PWS_H, ws + PWS_HQ, ws + PWS_HS, ws + PWS_COUNTS, a.m, a.topk);
        quantHShd.Init(ws + PWS_GU_SHD, ws + PWS_H_SHD, ws + PWS_HQ_SHD, ws + PWS_HS_SHD, ws + PWS_COUNTS, a.m, a.topk);
        PrepareUnpermute(a);
        combine.Init(ws + PWS_ROUTED, ws + PWS_Y_SHD, ws + PWS_SGATE, ws + PWS_SHARED, ws + PWS_MOE, a.m);
    }

    // ---------------- AIV 侧 ----------------
    __aicore__ inline void ProcessAiv()
    {
        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nAiv = AscendC::GetBlockNum() * 2u;
        const bool isPrimary = (bid == 0u);
        const uint32_t sl = p.stageLimit;

        PrecastGammaF32(p.gamma1, M15M::UB_GAMMA1_F32);
        PrecastGammaF32(p.gamma2, M15M::UB_GAMMA2_F32);

        if (sl >= 1u) {                                       // S1：m6#1 Add+RMSNorm
            norm1.Run(bid, nAiv);
            BarrierAivStep<PIPE_MTE2>(0);
        }
        if (sl >= 2u) {                                       // S2b：top-k（等 AIC 的 mmad logits）
            AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE2, PIPE_MTE2>(M15PFR::PFR_FLAG_M2_RING[0]);
            topk.Run(bid, nAiv);
            // **全体 AIV 屏障，必须在 `ig.Run()` 之前**（r1 复审 P1-1）：S3 由 AIV0 独占并
            // **标量读回全量** ids/weights（S = M*TOPK 条），而 ids/weights 由**全部 AIV** 分行写出
            // （`nAiv = 2*numBlocks`）⇒ 读之前必须先对齐，否则 AIV0 会读到尚未落地的行
            // （静默错路由，后面 S4–S10 全错且不报错）。
            // ⚠ donor 里 router 与 idxGen **同在 `if (isPrimary)`**（`m15_moe_layer.h:2082-2090`）
            //   ⇒ 那里写者与读者同核、天然不需要屏障；B4 把 router 改成多核分派（交付义务②）后
            //   这是**新引入**的跨核依赖，必须显式补屏障。
            // wait 挂 PIPE_S：读者是标量 GM 读；set 侧是 MTE3（drain 落盘）。
            BarrierAivStep<PIPE_S>(1);
            if (isPrimary) {                                  // S3：索引生成（AIV0 单核）
                ig.Run();
            }
            BarrierAivStep<PIPE_S>(2);                        // S3 → S4：perm_src/counts 的标量值依赖
        }
        if (sl >= 3u) {                                       // S4：permute
            perm.Run(bid, nAiv);
            BarrierAivStep<PIPE_MTE2>(3);
        }
        if (sl >= 4u) {                                       // S5：A 侧量化（routed + shared）
            quantA.Run(bid, nAiv);
            quantAShd.Run(bid, nAiv);
            AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE2, PIPE_MTE3>(M15PFR::PFR_FLAG_M2_RING[1]);
        }
        if (sl >= 5u) {                                       // S7：SwiGLU + 量化（等 GU）
            AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE2, PIPE_MTE2>(M15PFR::PFR_FLAG_M2_RING[2]);
            quantH.Run(bid, nAiv);
            quantHShd.Run(bid, nAiv);
            AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE2, PIPE_MTE3>(M15PFR::PFR_FLAG_M2_RING[3]);
        }
        if (sl >= 6u) {                                       // S9a：unpermute 加权折叠（等 Y）
            AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE2, PIPE_MTE2>(M15PFR::PFR_FLAG_M2_RING[0]);
            RunUnpermute(bid, nAiv);
            BarrierAivStep<PIPE_MTE2>(4);
        }
        if (sl >= 7u) {                                       // S9b：combine（+sigmoid 门）
            combine.Run(bid, nAiv);
            BarrierAivStep<PIPE_MTE2>(5);
        }
        if (sl >= 8u) {                                       // S10：m6#2 Add+RMSNorm
            norm2.Run(bid, nAiv);
        }
    }

    // ---------------- AIC 侧 ----------------
    __aicore__ inline void ProcessAic()
    {
        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t numBlocks = AscendC::GetBlockNum();
        const uint32_t sl = p.stageLimit;
        if (sl < 2u) {
            return;
        }
        AscendC::GlobalTensor<uint32_t> countsGm;
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ws + PWS_COUNTS), E);
        AscendC::GlobalTensor<uint32_t> offGm;
        offGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ws + PWS_OFFSETS), E + 1u);

        // ---- S2：router 打分 = bf16 `Mmad`（N 方向按 nb 条带分给各 AIC）----
        for (uint32_t nb = bid; nb < RB_NBLK; nb += numBlocks) {
            router.RunTile(nb);
        }
        // 全体 AIC 对齐 → 逐个 AIC set mode-2（每个 AIC 覆盖其配对的 2 个 AIV ⇒ 全体 AIV 都被唤醒）
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE0, PIPE_FIX>(M15PFR::PFR_FLAG_AIC_RING[0]);
        AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE0, PIPE_S>(M15PFR::PFR_FLAG_AIC_RING[0]);
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE2, PIPE_MTE2>(M15PFR::PFR_FLAG_M2_RING[0]);

        if (sl < 5u) {
            return;
        }
        // ---- S6：grouped gate_up GEMM（槽 = 512 专家 + 1 共享；空专家按 counts 跳过）----
        AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE2, PIPE_S>(M15PFR::PFR_FLAG_M2_RING[1]);
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE0, PIPE_MTE2>(M15PFR::PFR_FLAG_AIC_RING[1]);
        AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE0, PIPE_S>(M15PFR::PFR_FLAG_AIC_RING[1]);
        RunGemm<true>(bid, numBlocks, countsGm, offGm);
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE0, PIPE_FIX>(M15PFR::PFR_FLAG_AIC_RING[2]);
        AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE0, PIPE_S>(M15PFR::PFR_FLAG_AIC_RING[2]);
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE2, PIPE_MTE2>(M15PFR::PFR_FLAG_M2_RING[2]);

        if (sl < 6u) {
            return;
        }
        // ---- S8：grouped down GEMM ----
        AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE2, PIPE_S>(M15PFR::PFR_FLAG_M2_RING[3]);
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE0, PIPE_MTE2>(M15PFR::PFR_FLAG_AIC_RING[3]);
        AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE0, PIPE_S>(M15PFR::PFR_FLAG_AIC_RING[3]);
        RunGemm<false>(bid, numBlocks, countsGm, offGm);
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE0, PIPE_FIX>(M15PFR::PFR_FLAG_AIC_RING[0]);
        AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE0, PIPE_S>(M15PFR::PFR_FLAG_AIC_RING[0]);
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE2, PIPE_MTE2>(M15PFR::PFR_FLAG_M2_RING[0]);
    }

private:
    // AIV 的全体 mode-0 屏障。**调用点只给"第几步"**（编译期常量），flagId 一律取自
    // `M15PFR::PF_AIV_BARRIER_SEQ`（见 `m15_moe_prefill_res.h` §5）—— 该表带"相邻两步不同 id"的
    // static_assert（r2 复审残余 2 的处置：原来那 4 元 ring 的环邻断言绑不住 6 步的执行序）。
    // wait 挂窄 pipe；需要标量值依赖的段传 PIPE_S。
    template <pipe_t WAIT_PIPE>
    __aicore__ inline void BarrierAivStep(uint32_t step)
    {
        const uint16_t id = M15PFR::PF_AIV_BARRIER_SEQ[step];
        AscendC::CrossCoreSetFlag<M15PFR::PFR_CC_MODE0, PIPE_MTE3>(id);
        AscendC::CrossCoreWaitFlag<M15PFR::PFR_CC_MODE0, WAIT_PIPE>(id);
    }

    // S6（GU=true）/ S8（GU=false）的 grouped MXFP4 GEMM：槽 = 专家；工作项 = (slot, mTile, nBlock)
    template <bool GU>
    __aicore__ inline void RunGemm(uint32_t bid, uint32_t numBlocks, AscendC::GlobalTensor<uint32_t>& countsGm,
                                   AscendC::GlobalTensor<uint32_t>& offGm)
    {
        constexpr uint32_t NBLK = GU ? (GU_N / M15M::BASE_N) : (HIDDEN / M15M::BASE_N);
        for (uint32_t slot = 0; slot < E + 1u; ++slot) {
            const bool shd = (slot == E);
            const uint32_t t = shd ? p.m : countsGm.GetValue(slot);
            if (t == 0u) {
                continue;   // 空专家：A/H 区未被写，无需计算
            }
            const uint32_t mTiles = (t + MT - 1u) / MT;
            for (uint32_t mt = 0; mt < mTiles; ++mt) {
                const uint32_t rest = t - mt * MT;
                const uint32_t rows = (rest < MT) ? rest : MT;
                const uint32_t rowBase = shd ? (mt * MT) : (offGm.GetValue(slot) + mt * MT);
                for (uint32_t nb = bid; nb < NBLK; nb += numBlocks) {
                    if constexpr (GU) {
                        gemmGu.SetItem(shd ? (ws + PWS_AQ_SHD + rowBase * (HIDDEN / 2)) : (ws + PWS_AQ + rowBase * (HIDDEN / 2)),
                                       shd ? (ws + PWS_AS_SHD + rowBase * GU_SCALE_STRIDE) : (ws + PWS_AS + rowBase * GU_SCALE_STRIDE),
                                       shd ? p.wGuShd : (p.wGu + static_cast<uint64_t>(slot) * GU_N * (HIDDEN / 2)),
                                       shd ? p.sGuShd : (p.sGu + static_cast<uint64_t>(slot) * GU_N * GU_SCALE_STRIDE),
                                       shd ? (ws + PWS_GU_SHD + rowBase * GU_N * 2) : (ws + PWS_GU + rowBase * GU_N * 2));
                        gemmGu.Run(nb, rows);
                    } else {
                        gemmDn.SetItem(shd ? (ws + PWS_HQ_SHD + rowBase * (INTER / 2)) : (ws + PWS_HQ + rowBase * (INTER / 2)),
                                       shd ? (ws + PWS_HS_SHD + rowBase * DN_SCALE_STRIDE) : (ws + PWS_HS + rowBase * DN_SCALE_STRIDE),
                                       shd ? p.wDnShd : (p.wDn + static_cast<uint64_t>(slot) * HIDDEN * (INTER / 2)),
                                       shd ? p.sDnShd : (p.sDn + static_cast<uint64_t>(slot) * HIDDEN * (INTER / 32)),
                                       shd ? (ws + PWS_Y_SHD + rowBase * HIDDEN * 2) : (ws + PWS_Y + rowBase * HIDDEN * 2));
                        gemmDn.Run(nb, rows);
                    }
                }
            }
        }
    }

    // topk 运行时 → 模板分发（`UnpermuteP<TK>` 的 K 链全展开；与 decode 档同款）
    __aicore__ inline void RunUnpermute(uint32_t bid, uint32_t nAiv)
    {
        switch (p.topk) {
            case 1: unperm1.Run(bid, nAiv); break;
            case 2: unperm2.Run(bid, nAiv); break;
            case 3: unperm3.Run(bid, nAiv); break;
            case 4: unperm4.Run(bid, nAiv); break;
            case 5: unperm5.Run(bid, nAiv); break;
            case 6: unperm6.Run(bid, nAiv); break;
            case 7: unperm7.Run(bid, nAiv); break;
            case 8: unperm8.Run(bid, nAiv); break;
            case 9: unperm9.Run(bid, nAiv); break;
            default: unperm10.Run(bid, nAiv); break;
        }
    }

    __aicore__ inline void PrepareUnpermute(const MoePrefillPtrs& a)
    {
        unperm1.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm2.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm3.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm4.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm5.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm6.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm7.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm8.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm9.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
        unperm10.Init(ws + PWS_Y, ws + PWS_INV, ws + PWS_W, ws + PWS_ROUTED, a.m);
    }

private:
    MoePrefillPtrs p;
    __gm__ uint8_t* ws = nullptr;

    // **S1 的残差是 bf16**（`docs/12` 小改 D 只把 S10 的残差改成 fp32）—— donor 里就是
    // `NormStage<false> norm1; NormStage<true> norm2;`（`m15_moe_layer.h:2267-2268`）。
    // 若把 norm1 写成 `<true>`，它会把 `resZero` 按 **fp32** 读（`MT*HIDDEN*4` 字节），
    // 而传入的零缓冲只有 `MT*HIDDEN*2` ⇒ **越界读**（设备档第一次跑链路臂时暴露的 bug）。
    NormStage<false> norm1;      // S1：x = 层输入，res = bf16 全 0 缓冲，resOut = fp32(x)
    NormStage<true> norm2;       // S10：x = moe_out，res = S1 的 fp32 resOut
    RouterMmadStage router;      // S2（AIC）
    RouterTopkStage topk;        // S2b（AIV）
    IndexGenP ig;                // S3（AIV0）
    PermuteP perm;               // S4
    VecQuantP<HIDDEN, GU_SCALE_STRIDE, false, true> quantA;      // S5 routed（Σt_e 行）
    VecQuantP<HIDDEN, GU_SCALE_STRIDE, false, false> quantAShd;  // S5 shared（m 行）
    VecQuantP<INTER, DN_SCALE_STRIDE, true, true> quantH;        // S7 routed（SwiGLU）
    VecQuantP<INTER, DN_SCALE_STRIDE, true, false> quantHShd;    // S7 shared
    UnpermuteP<1> unperm1;
    UnpermuteP<2> unperm2;
    UnpermuteP<3> unperm3;
    UnpermuteP<4> unperm4;
    UnpermuteP<5> unperm5;
    UnpermuteP<6> unperm6;
    UnpermuteP<7> unperm7;
    UnpermuteP<8> unperm8;
    UnpermuteP<9> unperm9;
    UnpermuteP<10> unperm10;
    CombineP combine;            // S9b
    MXFP4GemmItem<HIDDEN, GU_N, GU_SCALE_STRIDE> gemmGu;   // S6
    MXFP4GemmItem<INTER, HIDDEN, DN_SCALE_STRIDE> gemmDn;  // S8
};

}  // namespace M15MP

#endif  // M15_MOE_PREFILL_H
