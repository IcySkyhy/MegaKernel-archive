/**
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

namespace M15M {

using namespace AscendC;
using namespace AscendC::Reg;

// ============================================================
// 通用：pipe_t 常量（BufferID 同步封装）
// ============================================================

template <pipe_t pipe>
__aicore__ inline void BufAcquire(AscendC::MutexID id)
{
    AscendC::GetBufInternal<pipe, false>(id);   // acquire 立即获取
}

template <pipe_t pipe>
__aicore__ inline void BufRelease(AscendC::MutexID id)
{
    AscendC::RlsBufInternal<pipe, false>(id);    // 阻塞释放（mode=false = CANN ASC_LOCK_BLOCK 默认）
}

template <pipe_t pipe>
__aicore__ inline void MutexLock(AscendC::MutexID id)
{
    AscendC::Mutex::Lock<pipe>(id);
}

template <pipe_t pipe>
__aicore__ inline void MutexUnlock(AscendC::MutexID id)
{
    AscendC::Mutex::Unlock<pipe>(id);
}

// 显式块拷贝参数（小改 A：杜绝元素计数重载）
//   注意 DataCopyParams.blockLen 单位是 **32B 块**（CANN: blockLen = count / GetC0Count(sizeof(T))），
//   故入参统一为**字节数**，内部 /32 换算：bf16 行 HIDDEN → Block1(HIDDEN*2) = {1,160,0,0}；
//   fp32 resOut 行 → Block1(HIDDEN*4) = {1,320,0,0}；WROW_I32 → Block1(WROW_I32*4) = {1,2,0,0}。
__aicore__ inline AscendC::DataCopyParams Block1(uint32_t bytes)
{
    // 前置条件：bytes 必须是 32B 整数倍（当前 22 个点位全部整除）。非整数倍会静默少搬，
    // 必须改走 DataCopyPad —— 这里留防呆：违反时 Trap（device 报 aicore exception，便于定位）。
    if ((bytes & 31u) != 0u) {
        AscendC::Trap();
    }
    return AscendC::DataCopyParams{1, static_cast<uint16_t>(bytes / 32), 0, 0};
}

__aicore__ inline AscendC::DataCopyExtParams ExtBlock1(uint32_t bytes)
{
    return AscendC::DataCopyExtParams{1, bytes, 0, 0, 0};
}

// ============================================================
// 通用：bf16 / fp32 位运算与 cast trait
// ============================================================

__aicore__ inline float Bf16ToF32Bits(uint32_t bitsLow16)
{
    uint32_t x = bitsLow16 << 16;
    float f;
    __builtin_memcpy(&f, &x, 4);
    return f;
}

constexpr Reg::CastTrait castTraitB162B32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                            RoundMode::UNKNOWN};
constexpr Reg::CastTrait castTraitB322B16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                            RoundMode::CAST_RINT};
constexpr Reg::CastTrait castTraitRM_Round = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                             RoundMode::CAST_ROUND};


// ============================================================
// AIC：grouped MXFP4 GEMM 单 work item（m3_grouped_gemm 的 MXFP4GemmItem 逐行 lift）
//   K/N 编译期常量；A_SCAL_STRIDE = A 侧 scale 行距（字节）
// ============================================================

template <uint32_t K, uint32_t N, uint32_t A_SCAL_STRIDE>
class MXFP4GemmItem {
    static constexpr uint32_t PACKED_K = K / 2;
    static constexpr uint32_t SCALE_K = K / GROUP;
    static_assert(K % BASE_K == 0, "K must be divisible by BASE_K");
    static_assert(N % BASE_N == 0, "N must be divisible by BASE_N");

public:
    __aicore__ inline MXFP4GemmItem() {}

    // 每个 work item 调用：设置该槽位的 GM 基址（每对象只 Set 一次：本 item 是常驻成员）
    __aicore__ inline void SetItem(__gm__ uint8_t* a, __gm__ uint8_t* as, __gm__ uint8_t* b, __gm__ uint8_t* bs,
                                   __gm__ uint8_t* c)
    {
        aGM.SetGlobalBuffer((__gm__ fp4x2_e2m1_t*)a);
        asGM.SetGlobalBuffer((__gm__ fp8_e8m0_t*)as);
        bGM.SetGlobalBuffer((__gm__ fp4x2_e2m1_t*)b);
        bsGM.SetGlobalBuffer((__gm__ fp8_e8m0_t*)bs);
        cGM.SetGlobalBuffer((__gm__ bfloat16_t*)c);
    }

    // 计算一个 (nBlock, curM) tile：k 循环累加 L0C → Fixpipe 写 GM
    __aicore__ inline void Run(uint32_t nBlock, uint32_t curM)
    {
        const uint32_t curMAlign = ((curM + CUBE_BLOCK - 1) / CUBE_BLOCK) * CUBE_BLOCK;
        // 3510 quirk（同 m1）：Nd2Nz/Dn2Nz 行数=1 退化，计算侧提升为 >=2 行，仅 Fixpipe 按 curM 写出
        const uint32_t calcM = curM < 2 ? 2 : curM;
        const uint32_t kLoop = K / BASE_K;

        AscendC::LocalTensor<fp4x2_e2m1_t> a1Ping(AscendC::TPosition::A1, L1_OFF_A0, L1_A_DATA);
        AscendC::LocalTensor<fp4x2_e2m1_t> a1Pong(AscendC::TPosition::A1, L1_OFF_A1, L1_A_DATA);
        AscendC::LocalTensor<fp8_e8m0_t> as1Ping(AscendC::TPosition::A1, L1_OFF_AS0, L1_A_SCAL);
        AscendC::LocalTensor<fp8_e8m0_t> as1Pong(AscendC::TPosition::A1, L1_OFF_AS1, L1_A_SCAL);
        AscendC::LocalTensor<fp4x2_e2m1_t> b1Ping(AscendC::TPosition::B1, L1_OFF_B0, L1_B_DATA);
        AscendC::LocalTensor<fp4x2_e2m1_t> b1Pong(AscendC::TPosition::B1, L1_OFF_B1, L1_B_DATA);
        AscendC::LocalTensor<fp8_e8m0_t> bs1Ping(AscendC::TPosition::B1, L1_OFF_BS0, L1_B_SCAL);
        AscendC::LocalTensor<fp8_e8m0_t> bs1Pong(AscendC::TPosition::B1, L1_OFF_BS1, L1_B_SCAL);
        AscendC::LocalTensor<fp4x2_e2m1_t> a2Ping(AscendC::TPosition::A2, L0_OFF_0, BASE_M * BASE_K / 2);
        AscendC::LocalTensor<fp4x2_e2m1_t> a2Pong(AscendC::TPosition::A2, L0_OFF_1, BASE_M * BASE_K / 2);
        AscendC::LocalTensor<fp4x2_e2m1_t> b2Ping(AscendC::TPosition::B2, L0_OFF_0, BASE_K * BASE_N / 2);
        AscendC::LocalTensor<fp4x2_e2m1_t> b2Pong(AscendC::TPosition::B2, L0_OFF_1, BASE_K * BASE_N / 2);
        AscendC::LocalTensor<float> cL0C(AscendC::TPosition::CO1, 0, BASE_M * BASE_N);

        MutexLock<PIPE_M>(BUF_AIC_L0C);   // 挡住上一 item 的 Fixpipe 读，允许本 tile 累加
        for (uint32_t kBlock = 0; kBlock < kLoop; ++kBlock) {
            const uint32_t p = kBlock & 1;
            const AscendC::MutexID bufA = p ? BUF_AIC_A1 : BUF_AIC_A0;
            const AscendC::MutexID bufB = p ? BUF_AIC_B1 : BUF_AIC_B0;
            const AscendC::MutexID bufL0 = p ? BUF_AIC_L01 : BUF_AIC_L00;
            AscendC::LocalTensor<fp4x2_e2m1_t> a1 = p ? a1Pong : a1Ping;
            AscendC::LocalTensor<fp4x2_e2m1_t> b1 = p ? b1Pong : b1Ping;
            AscendC::LocalTensor<fp8_e8m0_t> as1 = p ? as1Pong : as1Ping;
            AscendC::LocalTensor<fp8_e8m0_t> bs1 = p ? bs1Pong : bs1Ping;
            AscendC::LocalTensor<fp4x2_e2m1_t> a2 = p ? a2Pong : a2Ping;
            AscendC::LocalTensor<fp4x2_e2m1_t> b2 = p ? b2Pong : b2Ping;

            MutexLock<PIPE_MTE2>(bufA);
            CopyInA(a1, as1, kBlock, calcM);
            MutexUnlock<PIPE_MTE2>(bufA);
            MutexLock<PIPE_MTE2>(bufB);
            CopyInB(b1, bs1, kBlock, nBlock);
            MutexUnlock<PIPE_MTE2>(bufB);

            MutexLock<PIPE_MTE1>(bufL0);
            MutexLock<PIPE_MTE1>(bufA);
            MutexLock<PIPE_MTE1>(bufB);
            LoadA(a1, as1, a2, curMAlign);
            LoadB(b1, bs1, b2);
            MutexUnlock<PIPE_MTE1>(bufA);
            MutexUnlock<PIPE_MTE1>(bufB);
            MutexUnlock<PIPE_MTE1>(bufL0);

            MutexLock<PIPE_M>(bufL0);
            AscendC::MmadParams mmadParams;
            mmadParams.m = calcM;
            mmadParams.n = BASE_N;
            mmadParams.k = BASE_K;
            mmadParams.cmatrixInitVal = (kBlock == 0);
            AscendC::MmadMx(cL0C, a2, b2, mmadParams);
            MutexUnlock<PIPE_M>(bufL0);
        }
        MutexUnlock<PIPE_M>(BUF_AIC_L0C);   // L0C 累加结果就绪
        MutexLock<PIPE_FIX>(BUF_AIC_L0C);
        CopyOut(cL0C, nBlock, curM, curMAlign);
        MutexUnlock<PIPE_FIX>(BUF_AIC_L0C);
    }

private:
    // GM → L1：A 的一个 baseK 大包（calcM 行 × BASE_K fp4）+ scale（Dn2Nz，b16 视图）
    __aicore__ inline void CopyInA(AscendC::LocalTensor<fp4x2_e2m1_t>& a1, AscendC::LocalTensor<fp8_e8m0_t>& as1,
                                  uint32_t kBlock, uint32_t curM)
    {
        constexpr uint32_t packedStepK = BASE_K / 2;
        AscendC::Nd2NzParams par;
        par.ndNum = 1;
        par.nValue = curM;
        par.dValue = packedStepK;
        par.srcNdMatrixStride = 0;
        par.srcDValue = PACKED_K;
        par.dstNzC0Stride = BASE_M;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1, aGM[kBlock * BASE_K], par);   // 偏移按 fp4 元素（4bit）计

        constexpr uint32_t stepScaleK = BASE_K / GROUP;
        AscendC::Dn2NzParams sp;
        sp.dnNum = 1;
        sp.dValue = curM;
        sp.nValue = stepScaleK / 2;
        sp.srcDnMatrixStride = 0;
        sp.srcDValue = A_SCAL_STRIDE / 2;
        sp.dstNzC0Stride = stepScaleK / 2;
        sp.dstNzNStride = 1;
        sp.dstNzMatrixStride = 0;
        const uint32_t scaleOffHalf = (kBlock * stepScaleK) / 2;
        AscendC::GlobalTensor<half> asGMB16;
        asGMB16.SetGlobalBuffer(((__gm__ half*)asGM.GetPhyAddr()) + scaleOffHalf, curM * (A_SCAL_STRIDE / 2));
        AscendC::DataCopy(as1.ReinterpretCast<half>(), asGMB16, sp);
    }

    // GM → L1：B 按 [N, K] 原始 layout 搬 baseN 行 × BASE_K fp4 + scale
    __aicore__ inline void CopyInB(AscendC::LocalTensor<fp4x2_e2m1_t>& b1, AscendC::LocalTensor<fp8_e8m0_t>& bs1,
                                  uint32_t kBlock, uint32_t nBlock)
    {
        constexpr uint32_t packedStepK = BASE_K / 2;
        AscendC::Nd2NzParams par;
        par.ndNum = 1;
        par.nValue = BASE_N;
        par.dValue = packedStepK;
        par.srcNdMatrixStride = 0;
        par.srcDValue = PACKED_K;
        par.dstNzC0Stride = BASE_N;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1, bGM[kBlock * BASE_K + nBlock * BASE_N * K], par);

        constexpr uint32_t stepScaleK = BASE_K / GROUP;
        AscendC::Dn2NzParams sp;
        sp.dnNum = 1;
        sp.dValue = BASE_N;
        sp.nValue = stepScaleK / 2;
        sp.srcDnMatrixStride = 0;
        sp.srcDValue = SCALE_K / 2;
        sp.dstNzC0Stride = stepScaleK / 2;
        sp.dstNzNStride = 1;
        sp.dstNzMatrixStride = 0;
        const uint32_t scaleOffHalf = (nBlock * BASE_N * SCALE_K + kBlock * stepScaleK) / 2;
        AscendC::GlobalTensor<half> bsGMB16;
        bsGMB16.SetGlobalBuffer(((__gm__ half*)bsGM.GetPhyAddr()) + scaleOffHalf, BASE_N * (SCALE_K / 2));
        AscendC::DataCopy(bs1.ReinterpretCast<half>(), bsGMB16, sp);
    }

    __aicore__ inline void LoadA(AscendC::LocalTensor<fp4x2_e2m1_t>& a1, AscendC::LocalTensor<fp8_e8m0_t>& as1,
                                AscendC::LocalTensor<fp4x2_e2m1_t>& a2, uint32_t curMAlign)
    {
        AscendC::LoadData2DParamsV2 lp;
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = curMAlign / CUBE_BLOCK;
        lp.kStep = BASE_K / 64;
        lp.srcStride = BASE_M / CUBE_BLOCK;
        lp.dstStride = curMAlign / CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        AscendC::LoadData2DMxParams mx;
        mx.xStartPosition = 0;
        mx.yStartPosition = 0;
        mx.xStep = curMAlign / CUBE_BLOCK;
        mx.yStep = BASE_K / 64;
        mx.srcStride = BASE_K / 64;
        mx.dstStride = BASE_K / 64;
        AscendC::LoadData(a2, a1[0], as1[0], lp, mx);
    }

    __aicore__ inline void LoadB(AscendC::LocalTensor<fp4x2_e2m1_t>& b1, AscendC::LocalTensor<fp8_e8m0_t>& bs1,
                                AscendC::LocalTensor<fp4x2_e2m1_t>& b2)
    {
        constexpr uint32_t nFracs = BASE_N / CUBE_BLOCK;
        AscendC::LoadData2DParamsV2 lp;
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = nFracs;
        lp.kStep = BASE_K / 64;
        lp.srcStride = BASE_N / CUBE_BLOCK;
        lp.dstStride = nFracs;
        lp.sid = 0;
        lp.ifTranspose = false;
        AscendC::LoadData2DMxParams mx;
        mx.xStartPosition = 0;
        mx.yStartPosition = 0;
        mx.xStep = nFracs;
        mx.yStep = BASE_K / 64;
        mx.srcStride = BASE_K / 64;
        mx.dstStride = BASE_K / 64;
        AscendC::LoadData(b2, b1[0], bs1[0], lp, mx);
    }

    __aicore__ inline void CopyOut(AscendC::LocalTensor<float>& cL0C, uint32_t nBlock, uint32_t curM,
                                  uint32_t curMAlign)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
        fp.nSize = BASE_N;
        fp.mSize = curM;
        fp.srcStride = curMAlign;
        fp.dstStride = N;
        fp.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe(cGM[nBlock * BASE_N], cL0C, fp);
    }

private:
    AscendC::GlobalTensor<fp4x2_e2m1_t> aGM;
    AscendC::GlobalTensor<fp8_e8m0_t> asGM;
    AscendC::GlobalTensor<fp4x2_e2m1_t> bGM;
    AscendC::GlobalTensor<fp8_e8m0_t> bsGM;
    AscendC::GlobalTensor<bfloat16_t> cGM;
};

}  // namespace
// ============================================================
// S1/S10：Add + RMSNorm 段（m6_rmsnorm 移植；小改 D：残差输入支持 fp32）
//   ① xAdd = f32(x) + [f32(res) | res fp32] → xFp32
//   ② resOut(fp32) = xAdd      ③ rstd = 1/sqrt(mean+eps)   ④ y = bf16(xAdd·rstd·gamma)
// ============================================================

namespace M15M {

using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t VL_F32 = 64;                 // 256B 向量寄存器 fp32 lane 数
constexpr uint16_t CHUNKS_H = HIDDEN / VL_F32;  // 每行 64-lane chunk 数 = 40
constexpr float RMS_EPS = 1e-6f;
constexpr float RMS_AVG_FACTOR = 1.0f / static_cast<float>(HIDDEN);
constexpr uint32_t FOLD_POINT = 1280;           // 二分 fold 点（m6 donor tiling binAddQuotient）
constexpr uint32_t REDUCE_TMP_STRIDE = 24;      // ceil(foldLoops(20)/BLK_B32(8))*8

// ---- donorn（ops-nn norm/norm_common/op_kernel/reduce_common_regbase*.h，m6 逐字 lift）----
namespace NormDonor {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t V_LENGTH = VL_F32;
constexpr uint16_t DICHOTOMY_ADD_COEFF = 2;

constexpr AscendC::Reg::CastTrait castTraitB162B32 = {AscendC::Reg::RegLayout::ZERO,
                                                      AscendC::Reg::SatMode::UNKNOWN,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait castTraitB322B16 = {AscendC::Reg::RegLayout::ZERO,
                                                      AscendC::Reg::SatMode::NO_SAT,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::CAST_RINT};

template <typename T, LoadDist FLOAT_LOAD_DIST = LoadDist::DIST_NORM,
          LoadDist NON_FLOAT_LOAD_DIST = LoadDist::DIST_UNPACK_B16>
__aicore__ inline void LoadRegForDtype(__ubuf__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<T, FLOAT_LOAD_DIST>(dst, src + offset);
    } else {
        RegTensor<T> srcReg;
        LoadAlign<T, NON_FLOAT_LOAD_DIST>(srcReg, src + offset);
        Cast<float, T, castTraitB162B32>(dst, srcReg, preg);
    }
}

template <typename T, int32_t LAST_LOOP_NUMS>
__aicore__ inline void CalculateSquareReduceSumCommon(__ubuf__ T* xPtr, __ubuf__ float* dstPtr,
                                                      __ubuf__ float* tmpPtr, uint16_t rows, uint32_t rowStride,
                                                      uint32_t reduceNum, uint32_t foldPoint, uint32_t tmpStride)
{
    uint16_t foldLoops = static_cast<uint16_t>((foldPoint + V_LENGTH - 1) / V_LENGTH);
    uint32_t lastNum = foldPoint / V_LENGTH;
    uint32_t tail = (reduceNum > foldPoint) ? reduceNum - foldPoint : 0;
    uint16_t tailCeilLoops = static_cast<uint16_t>((tail + V_LENGTH - 1) / V_LENGTH);
    uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>(foldLoops - tailCeilLoops);
    (void)firstFlodWithOutAddLoops;

    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t baseOffset = static_cast<uint32_t>(i) * rowStride;
            uint32_t tmpOffset = static_cast<uint32_t>(i) * tmpStride;
            uint32_t sregTail = tail;
            for (uint16_t j = 0; j < tailCeilLoops; ++j) {
                pregLoop = UpdateMask<float>(sregTail);
                uint32_t offset = static_cast<uint32_t>(j) * V_LENGTH + baseOffset;
                LoadRegForDtype<T>(xPtr, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                LoadRegForDtype<T>(xPtr + foldPoint, xFoldReg, pregFull, offset);
                Mul(xFoldReg, xFoldReg, xFoldReg, pregLoop);
                Add(sumReg, xReg, xFoldReg, pregFull);
                Reduce<ReduceType::SUM>(reduceReg, sumReg, pregFull);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tmpOffset + j, reduceReg, pregOne);
            }
            for (uint16_t j = 0; j < firstFlodWithOutAddLoops; ++j) {
                uint32_t offset = static_cast<uint32_t>(tailCeilLoops + j) * V_LENGTH + baseOffset;
                LoadRegForDtype<T>(xPtr, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                Reduce<ReduceType::SUM>(reduceReg, xReg, pregFull);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tmpOffset + tailCeilLoops + j,
                                                                     reduceReg, pregOne);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        if constexpr (LAST_LOOP_NUMS == 1) {
            MaskReg pregLast = UpdateMask<float>(lastNum);
            for (uint16_t i = 0; i < rows; ++i) {
                LoadAlign<float>(xReg, tmpPtr + static_cast<uint32_t>(i) * tmpStride);
                Reduce<ReduceType::SUM>(reduceReg, xReg, pregLast);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
            }
        } else if constexpr (LAST_LOOP_NUMS == DICHOTOMY_ADD_COEFF) {
            lastNum -= V_LENGTH;
            MaskReg pregLast = UpdateMask<float>(lastNum);
            for (uint16_t i = 0; i < rows; ++i) {
                uint32_t tmpOffset = static_cast<uint32_t>(i) * tmpStride;
                LoadAlign<float>(xReg, tmpPtr + tmpOffset);
                LoadAlign<float>(xFoldReg, tmpPtr + tmpOffset + V_LENGTH);
                ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                           pregLast);
                Add(sumReg, xReg, xFoldReg, pregFull);
                Reduce<ReduceType::SUM>(reduceReg, sumReg, pregFull);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSum(__ubuf__ T* xPtr, __ubuf__ float* dstPtr, __ubuf__ float* tmpPtr,
                                                uint16_t rows, uint32_t rowStride, uint32_t reduceNum,
                                                uint32_t foldPoint, uint32_t tmpStride)
{
    // 2560 = 2×1280 → 走 Common<1> 分支（与 m6 donor 调用一致）
    CalculateSquareReduceSumCommon<T, 1>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum, foldPoint, tmpStride);
}

template <bool NEED_MAX = true>
__aicore__ inline void ComputeRstdNewtonRaphsonReg(RegTensor<float>& var, RegTensor<float>& rstd, MaskReg& preg,
                                                   float epsilon)
{
    static constexpr float POS_INF = 3.40282366920938E+38;
    static constexpr float SCALAR1 = -0.5;
    static constexpr float SCALAR2 = 1.5;
    static constexpr float SCALAR3 = 0.5;
    static constexpr float SCALAR0 = -99.99;

    RegTensor<float> r;
    RegTensor<float> y;
    RegTensor<float> s;
    RegTensor<float> t;
    RegTensor<float> one;
    RegTensor<float> scalar1;
    RegTensor<float> t1;
    RegTensor<float> t3;
    RegTensor<float> t4;
    RegTensor<float> scalarInf;
    RegTensor<float> scalarZero;
    RegTensor<float> tmp;
    MaskReg cmpRegZero;
    MaskReg cmpRegInf;

    Duplicate(scalarInf, POS_INF, preg);
    Duplicate(scalarZero, float(0.0), preg);
    Duplicate(one, float(1.0), preg);
    Duplicate(scalar1, SCALAR3, preg);
    Duplicate(t1, SCALAR2, preg);
    Duplicate(s, float(1.0), preg);

    Adds(var, var, epsilon, preg);
    if constexpr (NEED_MAX) {
        Maxs(var, var, SCALAR0, preg);
    }
    Div(r, one, var, preg);
    Sqrt(y, r, preg);
    Muls(t, var, SCALAR1, preg);
    Mul(t, t, y, preg);
    Mul(tmp, t, y, preg);
    Add(t1, t1, tmp, preg);   // donor Mula
    Mul(rstd, y, t1, preg);
    Muls(t3, var, float(-1.0), preg);
    Mul(tmp, t3, r, preg);
    Add(s, s, tmp, preg);     // donor Mula
    Muls(t4, rstd, float(-1.0), preg);
    Mul(tmp, t4, rstd, preg);
    Add(r, r, tmp, preg);     // donor Mula
    Mul(tmp, var, r, preg);
    Add(s, s, tmp, preg);     // donor Mula
    Mul(s, s, rstd, preg);
    Mul(tmp, s, scalar1, preg);
    Add(rstd, rstd, tmp, preg);  // donor Mula
    Compares(cmpRegZero, var, POS_INF, preg);
    Select(rstd, scalarZero, rstd, cmpRegZero);
    Compares(cmpRegInf, var, float(0.0), preg);
    Select(rstd, scalarInf, rstd, cmpRegInf);
}

template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphson(__ubuf__ float* src, __ubuf__ float* dst, uint32_t rowCount,
                                                float epsilon, float avgFactor, uint32_t vectorLen)
{
    uint16_t loopRows = static_cast<uint16_t>((rowCount + vectorLen - 1) / vectorLen);
    __VEC_SCOPE__
    {
        RegTensor<float> var;
        RegTensor<float> rstd;
        MaskReg pregLoop;

        uint32_t sreg = rowCount;
        for (uint16_t i = 0; i < loopRows; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            LoadAlign<float>(var, src + i * vectorLen);
            if constexpr (NEED_AVG_FACTOR) {
                Muls(var, var, avgFactor, pregLoop);
            }
            ComputeRstdNewtonRaphsonReg<NEED_MAX>(var, rstd, pregLoop, epsilon);
            StoreAlign<float>(dst + i * vectorLen, rstd, pregLoop);
        }
    }
}

}  // namespace NormDonor

// ---- gamma bf16 → fp32 预转（每个 AIV 核一次；落在 UB_GAMMA1_F32 / UB_GAMMA2_F32）----
// m6 的 PrecomputeGammaF32 同款：整核只做一次，之后 S1/S10 每行少 40 次 Cast，
// 同时让 y 循环里每 chunk 只剩一次 fp32→bf16 Cast（规避 3510 连续 Cast quirk）。
__aicore__ inline void PrecastGammaF32(__gm__ uint8_t* gamma, uint32_t dstUbOff)
{
    AscendC::GlobalTensor<bfloat16_t> gGm;
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gamma), HIDDEN);
    LocalTensor<bfloat16_t> gbL(TPosition::VECCALC, UB_PERSIST, HIDDEN);

    BufAcquire<PIPE_MTE2>(BUF_AIV_GAMMA);
    PipeBarrier<PIPE_MTE2>();
    DataCopy(gbL, gGm, Block1(HIDDEN * 2));
    BufRelease<PIPE_MTE2>(BUF_AIV_GAMMA);

    __ubuf__ bfloat16_t* gbUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_PERSIST);
    __ubuf__ float* gfUb = reinterpret_cast<__ubuf__ float*>(dstUbOff);
    BufAcquire<PIPE_V>(BUF_AIV_GAMMA);
    __VEC_SCOPE__
    {
        RegTensor<bfloat16_t> gB16;
        RegTensor<float> gF;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t i = 0; i < CHUNKS_H; ++i) {
            uint32_t offset = static_cast<uint32_t>(i) * VL_F32;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gB16, gbUb + offset);
            Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(gF, gB16, maskAll);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(gfUb + offset, gF, maskAll);
        }
    }
    BufRelease<PIPE_V>(BUF_AIV_GAMMA);
}

// ============================================================
// S1/S10：NormStage（行切分；RES_F32 = 残差输入 fp32）
// ============================================================

template <bool RES_F32>
class NormStage {
public:
    __aicore__ inline NormStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* res, __gm__ uint8_t* y, __gm__ uint8_t* resOut,
                                uint32_t gammaF32UbOff, uint32_t m)
    {
        M = m;
        gammaOff = gammaF32UbOff;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), static_cast<uint64_t>(M_MAX) * HIDDEN);
        if constexpr (RES_F32) {
            resGmF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(res), static_cast<uint64_t>(M_MAX) * HIDDEN);
        } else {
            resGmB.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(res), static_cast<uint64_t>(M_MAX) * HIDDEN);
        }
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y), static_cast<uint64_t>(M_MAX) * HIDDEN);
        resOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(resOut), static_cast<uint64_t>(M_MAX) * HIDDEN);
    }

    // 行 r = bid + i*nAiv（行间无依赖）
    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        for (uint32_t row = bid; row < M; row += nAiv) {
            CopyInRow(row);
            ComputeRow();
            CopyOutRow(row);
        }
    }

private:
    __aicore__ inline void CopyInRow(uint32_t row)
    {
        LocalTensor<bfloat16_t> x1L(TPosition::VECCALC, UB_M6_X1, HIDDEN);
        const uint64_t off = static_cast<uint64_t>(row) * HIDDEN;
        BufAcquire<PIPE_MTE2>(BUF_AIV_ROW0);   // 挡上一行的 V 消费
        PipeBarrier<PIPE_MTE2>();              // 小改 B：行循环同 pipe 复用同一 UB 行缓冲
        DataCopy(x1L, xGm[off], Block1(HIDDEN * 2));
        if constexpr (RES_F32) {
            LocalTensor<float> resL(TPosition::VECCALC, UB_M6_RES, HIDDEN);
            DataCopy(resL, resGmF[off], Block1(HIDDEN * 4));   // fp32 残差（小改 D）
        } else {
            LocalTensor<bfloat16_t> resL(TPosition::VECCALC, UB_M6_RES, HIDDEN);
            DataCopy(resL, resGmB[off], Block1(HIDDEN * 2));
        }
        BufRelease<PIPE_MTE2>(BUF_AIV_ROW0);
    }

    // donor ①：残差加（fp32 寄存器）→ UB 留 xFp32（resOut fp32 直接由它 MTE3 写出）
    __aicore__ inline void CalculateXAdd(__ubuf__ bfloat16_t* x1Ub, __ubuf__ float* xfUb)
    {
        __VEC_SCOPE__
        {
            RegTensor<bfloat16_t> x1B16;
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t i = 0; i < CHUNKS_H; ++i) {
                uint32_t offset = static_cast<uint32_t>(i) * VL_F32;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(x1B16, x1Ub + offset);
                Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(x1, x1B16, maskAll);
                if constexpr (RES_F32) {
                    LoadAlign<float>(x2, reinterpret_cast<__ubuf__ float*>(UB_M6_RES) + offset);
                } else {
                    RegTensor<bfloat16_t> x2B16;
                    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(
                        x2B16, reinterpret_cast<__ubuf__ bfloat16_t*>(UB_M6_RES) + offset);
                    Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(x2, x2B16, maskAll);
                }
                Add(xSum, x1, x2, maskAll);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(xfUb + offset, xSum, maskAll);
            }
        }
    }

    // donor ④：y = (x·rstd)·gamma → bf16（gamma 已预转 fp32，每 chunk 仅一次 fp32→bf16 Cast）
    __aicore__ inline void CalculateY(__ubuf__ float* xfUb, __ubuf__ bfloat16_t* yUb, __ubuf__ float* rstdUb)
    {
        __ubuf__ float* gfUb = reinterpret_cast<__ubuf__ float*>(gammaOff);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> gReg;
            RegTensor<float> rstdReg;
            RegTensor<float> tReg;
            RegTensor<float> yReg;
            RegTensor<bfloat16_t> yB16;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdUb);
            for (uint16_t i = 0; i < CHUNKS_H; ++i) {
                uint32_t offset = static_cast<uint32_t>(i) * VL_F32;
                LoadAlign<float>(xReg, xfUb + offset);
                LoadAlign<float>(gReg, gfUb + offset);
                Mul(tReg, xReg, rstdReg, maskAll);
                Mul(yReg, tReg, gReg, maskAll);
                Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yB16, yReg, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(yUb + offset, yB16, maskAll);
            }
        }
    }

    __aicore__ inline void ComputeRow()
    {
        LocalTensor<float> xfL(TPosition::VECCALC, UB_M6_XF, HIDDEN);
        LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_M6_Y, HIDDEN);
        LocalTensor<float> tmpL(TPosition::VECCALC, UB_M6_TMP, 64);
        LocalTensor<float> redL(TPosition::VECCALC, UB_M6_RED, 64);
        LocalTensor<float> rstdL(TPosition::VECCALC, UB_M6_RSTD, 64);
        __ubuf__ bfloat16_t* x1Ub = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_M6_X1);
        __ubuf__ float* xfUb = reinterpret_cast<__ubuf__ float*>(xfL.GetPhyAddr());
        __ubuf__ bfloat16_t* yUb = reinterpret_cast<__ubuf__ bfloat16_t*>(yL.GetPhyAddr());
        __ubuf__ float* tmpUb = reinterpret_cast<__ubuf__ float*>(tmpL.GetPhyAddr());
        __ubuf__ float* redUb = reinterpret_cast<__ubuf__ float*>(redL.GetPhyAddr());
        __ubuf__ float* rstdUb = reinterpret_cast<__ubuf__ float*>(rstdL.GetPhyAddr());

        BufAcquire<PIPE_V>(BUF_AIV_ROW0);    // 等 x/res 行就位
        BufAcquire<PIPE_V>(BUF_AIV_OUT);     // 等上一行 MTE3 读完 xFp32/y 缓冲
        CalculateXAdd(x1Ub, xfUb);
        NormDonor::CalculateSquareReduceSum<float>(xfUb, redUb, tmpUb, 1, HIDDEN, HIDDEN, FOLD_POINT,
                                                   REDUCE_TMP_STRIDE);
        NormDonor::ComputeRstdNewtonRaphson<true, true>(redUb, rstdUb, 1, RMS_EPS, RMS_AVG_FACTOR, VL_F32);
        CalculateY(xfUb, yUb, rstdUb);
        BufRelease<PIPE_V>(BUF_AIV_ROW0);    // x/res 消费完，MTE2 可复用
        BufRelease<PIPE_V>(BUF_AIV_OUT);     // xFp32/y 写完（mode=false），MTE3 可读
    }

    __aicore__ inline void CopyOutRow(uint32_t row)
    {
        LocalTensor<float> xfL(TPosition::VECCALC, UB_M6_XF, HIDDEN);
        LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_M6_Y, HIDDEN);
        const uint64_t off = static_cast<uint64_t>(row) * HIDDEN;
        BufAcquire<PIPE_MTE3>(BUF_AIV_OUT);
        DataCopy(resOutGm[off], xfL, Block1(HIDDEN * 4));   // 残差出口 fp32
        DataCopy(yGm[off], yL, Block1(HIDDEN * 2));        // y bf16
        BufRelease<PIPE_MTE3>(BUF_AIV_OUT);
    }

private:
    uint32_t M = 0;
    uint32_t gammaOff = 0;
    AscendC::GlobalTensor<bfloat16_t> xGm;
    AscendC::GlobalTensor<bfloat16_t> resGmB;
    AscendC::GlobalTensor<float> resGmF;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    AscendC::GlobalTensor<float> resOutGm;
};

}  // namespace
// ============================================================
// S2：Router 段（m7_router_topk 移植：GEMV 向量 MAC → max-shift softmax →
//     Sort32 排序 + top-k 拆分（Extract 的 VF 内联）→ renorm），AIV0 单核
//     同一段内顺带算共享专家门裸点积 sgate[r]（sigmoid 在 S10 用向量做）
// ============================================================

namespace M15M {

using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t RT_LANES = 32;                 // Sort32 输入 lane 数（索引模板宽度）
constexpr float RT_NEG_BIG = -3.0e38f;

class RouterStage {
public:
    // [M91-#3] row-block 行数由 m15_moe_resources.h §4 的 RT_RB 给出（曾在此处 shadow 成 16；删除以免与资源表的 UB 定尺不一致）

    __aicore__ inline RouterStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* xIn, __gm__ uint8_t* routerW, __gm__ uint8_t* sgateW,
                                __gm__ uint8_t* logits, __gm__ uint8_t* ids, __gm__ uint8_t* weights,
                                __gm__ uint8_t* sgate, uint32_t m, uint32_t topk)
    {
        M = m;
        TOPK = topk;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xIn), static_cast<uint64_t>(M_MAX) * HIDDEN);
        rwGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(routerW), static_cast<uint64_t>(NUM_EXPERTS) * HIDDEN);
        sgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sgateW), HIDDEN);
        logitsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logits), static_cast<uint64_t>(M_MAX) * NUM_EXPERTS);
        idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids), static_cast<uint64_t>(M_MAX) * TOPK_MAX);
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights), static_cast<uint64_t>(M_MAX) * TOPK_MAX);
        sgateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sgate), M_MAX);
    }

    // m7 同款块结构：整块 x 行一次 MTE2 载入 → 逐行 V 计算写 UB staging → 整块一次 MTE3 出。
    // 块级粒度让 x 行缓冲的 MTE2→V 交接足以被 BufferID 阻塞释放（mode=false）覆盖（逐行粒度实测会被
    // MTE2 抢跑覆盖，导致所有行读到最后一行的 x）。
    __aicore__ inline void Run(uint32_t subLimit)
    {
        if (subLimit >= 2) {
            BuildIndexTemplate();
        }
        if (subLimit >= 3) {
            const uint32_t nblk = (M + RT_RB - 1) / RT_RB;
            for (uint32_t b = 0; b < nblk; ++b) {
                const uint32_t b0 = b * RT_RB;
                const uint32_t rows = (M - b0 < RT_RB) ? (M - b0) : RT_RB;
                CopyInXBlock(b0, rows);
                ComputeBlock(b0, rows);
                CopyOutBlock(b0, rows);
            }
        }
    }

private:
    // ---- [M91-#3] 权重流式：只保留 RT_EGRP 行 fp32 窗常驻（取用协议抄 m17_moe_real 的 Gemv）----
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

    __aicore__ inline void CopyInXBlock(uint32_t b0, uint32_t rows)
    {
        BufAcquire<PIPE_MTE2>(BUF_AIV_ROW0);   // 挡上一块的 V 消费
        PipeBarrier<PIPE_MTE2>();              // 小改 B
        for (uint32_t r = 0; r < rows; ++r) {
            LocalTensor<bfloat16_t> xL(TPosition::VECCALC, UB_RT_XB + r * HIDDEN * 2, HIDDEN);
            DataCopy(xL, xGm[static_cast<uint64_t>(b0 + r) * HIDDEN], Block1(HIDDEN * 2));
        }
        BufRelease<PIPE_MTE2>(BUF_AIV_ROW0);
    }

    // 整块 V 段：x 行全读 + staging 全写 都在同一 span 内，块尾一次性 release（mode=false）
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

    // [M91-#3] 对数行 pad：整行 64 lane 写 RT_NEG_BIG。改前的同值由 GemvRow 打包后的
    //   `Select(rowReg, v4, negInf, mE)` 写（lane >= NUM_EXPERTS 全覆盖）——同一常量、同一范围，
    //   只是搬到组循环之前（组循环只写 [e0, e0+ng)）。
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
        __ubuf__ float* rowUb = reinterpret_cast<__ubuf__ float*>(UB_RT_LOG + r * RT_ROWL * 4) + e0;
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

    __aicore__ inline void CopyOutBlock(uint32_t b0, uint32_t rows)
    {
        LocalTensor<float> sgAll(TPosition::VECCALC, UB_RT_SG, RT_RB);
        BufAcquire<PIPE_MTE3>(BUF_AIV_ROW1);
        for (uint32_t r = 0; r < rows; ++r) {
            // [M95-#4] 对数行 = RT_ROWL lane；top-k 结果 staging（IDS/WS）行距 = RT_WROW（固定 64）
            LocalTensor<float> logitsL(TPosition::VECCALC, UB_RT_LOG + r * RT_ROWL * 4, RT_ROWL);
            LocalTensor<int32_t> idsL(TPosition::VECCALC, UB_RT_IDS + r * RT_WROW * 4, RT_WROW);
            LocalTensor<float> wsL(TPosition::VECCALC, UB_RT_WS + r * RT_WROW * 4, RT_WROW);
            DataCopyPad(logitsGm[static_cast<uint64_t>(b0 + r) * NUM_EXPERTS], logitsL,
                        ExtBlock1(NUM_EXPERTS * 4));
            DataCopyPad(idsGm[static_cast<uint64_t>(b0 + r) * TOPK], idsL, ExtBlock1(TOPK * 4));
            DataCopyPad(wGm[static_cast<uint64_t>(b0 + r) * TOPK], wsL, ExtBlock1(TOPK * 4));
        }
        // 共享门裸点积（值在 lane0，整块一次紧凑拷出；sigmoid 由 S10 向量路径完成）
        DataCopyPad(sgateGm[b0], sgAll, ExtBlock1(rows * 4));
        BufRelease<PIPE_MTE3>(BUF_AIV_ROW1);
    }

private:
    uint32_t M = 0;
    uint32_t TOPK = 1;
    AscendC::GlobalTensor<bfloat16_t> xGm;
    AscendC::GlobalTensor<bfloat16_t> rwGm;
    AscendC::GlobalTensor<bfloat16_t> sgGm;
    AscendC::GlobalTensor<float> logitsGm;
    AscendC::GlobalTensor<int32_t> idsGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> sgateGm;
};

// ============================================================
// S3：routing 索引生成（m7 ↔ m8 之间的缺口 glue；m8#1 只消费 perm_src/counts，
//     m8#2 消费 inv_slot/w_tk_packed，没有任何算件产生它们 → 本段补齐）
//     计数排序（stable，按 expert id 分组）与 moe_block_ref.py::moe_permute 同语义；
//     标量实现（m*TOPK ≤ 256），UB 写 → MTE3 出（m3-AIV 同款 PIPE_S→MTE3 握手）
// ============================================================

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

class IndexGenStage {
public:
    __aicore__ inline IndexGenStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* ids, __gm__ uint8_t* weights, __gm__ uint8_t* counts,
                                __gm__ uint8_t* offsets, __gm__ uint8_t* permSrc, __gm__ uint8_t* permExp,
                                __gm__ uint8_t* inv, __gm__ uint8_t* wtk, uint32_t m, uint32_t topk)
    {
        M = m;
        TOPK = topk;
        idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids), static_cast<uint64_t>(M_MAX) * TOPK_MAX);
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights), static_cast<uint64_t>(M_MAX) * TOPK_MAX);
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts), NUM_EXPERTS);
        offsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets), NUM_EXPERTS + IG_DIAG_SLOT_END);
        permSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(permSrc), TOTAL_MAX);
        permExpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(permExp), TOTAL_MAX);
        invGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(inv), static_cast<uint64_t>(M_MAX) * TOPK_MAX);
        wtkGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(wtk), static_cast<uint64_t>(M_MAX) * 16);
    }

    __aicore__ inline uint32_t OobCount() const { return oobCountUb; }

    // subLimit：与 RouterStage 同一 bring-up 旋钮（运行时值；7/8/9 分别为索引数组/counts/诊断槽门控）
    __aicore__ inline void Run(uint32_t subLimit)
    {
        // router 的 MTE3 GM 写排空后再做标量 GM 读（同核次序）
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        const uint32_t S = M * TOPK;
        uint32_t badIds = 0;   // 脏 ids 计数：诊断槽上报 + 落位循环的短路条件
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

        __ubuf__ int32_t* srcUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_SRC);
        __ubuf__ int32_t* expUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_EXP);
        __ubuf__ int32_t* invUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_INV);
        __ubuf__ int32_t* wtkUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_WTK);

        MutexLock<PIPE_S>(BUF_AIV_IDX);
        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
            curUb[e] = offUb[e];
        }
        for (uint32_t t = 0; t < M; ++t) {
            for (uint32_t k = 0; k < TOPK; ++k) {
                const uint32_t s = t * TOPK + k;
                const int32_t e = idsGm.GetValue(s);
                if (e < 0 || e >= static_cast<int32_t>(NUM_EXPERTS) || badIds != 0) {
                    // 与计数循环对称的防御：脏 ids 时 cursor[e]/off[e]/invUb[s] 一律不越界
                    ++badIds;
                    continue;
                }
                const uint32_t pos = static_cast<uint32_t>(curUb[e]);
                curUb[e] = static_cast<int32_t>(pos + 1);
                srcUb[pos] = static_cast<int32_t>(t);
                expUb[pos] = e;
                // Y 布局是 [slot][M_MAX][HIDDEN]（槽位 padding）→ inv 存 padding 行号
                invUb[s] = static_cast<int32_t>(pos);   // [M40-6a] 紧凑行号（= 计数排序产出位置）
                // bf16(RNE) 权重位打包进 int32 低 16 位（m8 w_tk_packed 协议）
                const float f = wGm.GetValue(s);
                uint32_t bits;
                __builtin_memcpy(&bits, &f, 4);
                const uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
                wtkUb[t * 16 + k] = static_cast<int32_t>(rounded & 0xFFFFu);
            }
        }
        // [M105] 写侧释放：`BufRelease<PIPE_S>` = `RlsBufInternal<PIPE_S,false>`
        //   —— mode=false = CANN `ASC_LOCK_BLOCK` 默认「阻塞」；`true` 为 `NON_BLOCK`。两种模式都等本
        //   pipe 已发射指令落地，`true` 额外等此前同 id 的释放（更保守）。此处与 acquire 侧同模式。
        //   改前这里是 `MutexUnlock<PIPE_S>`（= `RlsBufInternal<PIPE_S,0>`，同为 mode 0；
        //   见 docs/05 §6 与 CANN kernel_common.h:138-160）。
        //   写侧 acquire 仍是 `MutexLock<PIPE_S>`（= `GetBufInternal<PIPE_S,0>`，与 `BufAcquire<PIPE_S>`
        //   是同一 token 获取）⇒ 成对不破（只放 release 而缺写侧 acquire 会挂死，见 `m15_ple.asc`
        //   文件头 §BUF 的教训）。
        BufRelease<PIPE_S>(BUF_AIV_IDX);
        if (subLimit < 7) {
            return;
        }

        // UB → GM（PIPE_S 写 → MTE3 读的 BufferID 握手，m3-AIV 同款）
        LocalTensor<int32_t> srcL(TPosition::VECCALC, UB_IG_SRC, S);
        LocalTensor<int32_t> expL(TPosition::VECCALC, UB_IG_EXP, S);
        LocalTensor<int32_t> invL(TPosition::VECCALC, UB_IG_INV, S);
        LocalTensor<int32_t> wtkL(TPosition::VECCALC, UB_IG_WTK, M * 16);
        LocalTensor<int32_t> cntL(TPosition::VECCALC, UB_IG_SCAL, NUM_EXPERTS);
        LocalTensor<int32_t> offL(TPosition::VECCALC, UB_IG_OFF, NUM_EXPERTS + 1);
        BufAcquire<PIPE_MTE3>(BUF_AIV_IDX);
        DataCopyPad(permSrcGm[0], srcL, ExtBlock1(S * 4));
        DataCopyPad(permExpGm[0], expL, ExtBlock1(S * 4));
        DataCopyPad(invGm[0], invL, ExtBlock1(S * 4));
        DataCopyPad(wtkGm[0], wtkL, ExtBlock1(M * 16 * 4));
        if (subLimit < 8) {
            BufRelease<PIPE_MTE3>(BUF_AIV_IDX);
            return;
        }
        DataCopyPad(countsGm[0], cntL, ExtBlock1(NUM_EXPERTS * 4));
        DataCopyPad(offsGm[0], offL, ExtBlock1((NUM_EXPERTS + 1) * 4));
        if (subLimit >= 9) {
            // [M84-6] 诊断槽：UB 源 32B 对齐（+128B）；GM 目的放 IG_DIAG_GM_SLOT
            //   （= SZ_OFFSETS 尾部 32B 槽起点，恒在真 offsets[0..E] 之后）
            // [M105] 这一读的 UB 源那 4 B 的**标量写**已提前到写侧 acquire 之前
            //   （见本函数开头 `diagUb[32] = …`）—— 此处只留 MTE3 那一读；改前写在这一读紧邻处、
            //   却排在 `BufAcquire<PIPE_MTE3>` **之后**（次序 token 之外），与它无次序保证。
            //   次序现由写侧 release（mode=false）`BufRelease<PIPE_S>(BUF_AIV_IDX)` 建立（见该行注释）。
            LocalTensor<int32_t> bL(TPosition::VECCALC, UB_IG_CNT, 64);
            DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], ExtBlock1(4));
        }
        BufRelease<PIPE_MTE3>(BUF_AIV_IDX);
    }

private:
    uint32_t M = 0;
    uint32_t TOPK = 1;
    uint32_t oobCountUb = 0;
    AscendC::GlobalTensor<int32_t> idsGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<int32_t> countsGm;
    AscendC::GlobalTensor<int32_t> offsGm;
    AscendC::GlobalTensor<int32_t> permSrcGm;
    AscendC::GlobalTensor<int32_t> permExpGm;
    AscendC::GlobalTensor<int32_t> invGm;
    AscendC::GlobalTensor<int32_t> wtkGm;
};

}  // namespace
// ============================================================
// S4：permute（m8_permute kernel #1 移植：x_sorted[i] = x[perm_src_token[i]]，
//     纯带宽逐行 gather，四级流水 + 小改 B 的 MTE2 排空）
// ============================================================

namespace M15M {

using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t PERM_STAGES = 4;

class PermuteStage {
public:
    __aicore__ inline PermuteStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* permSrc, __gm__ uint8_t* counts,
                                __gm__ uint8_t* xSorted, uint32_t numExperts, uint32_t m, uint32_t topk)
    {
        M = m;
        TOPK = topk;
        E = numExperts;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), static_cast<uint64_t>(M_MAX) * HIDDEN);
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(permSrc), TOTAL_MAX);
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts), numExperts);
        xSortedGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xSorted), static_cast<uint64_t>(TOTAL_MAX) * HIDDEN);
    }

    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        uint32_t total = 0;
        for (uint32_t e = 0; e < E; ++e) {
            total += static_cast<uint32_t>(countsGm.GetValue(e));
        }
        TOTAL = total;
        const uint32_t myRows = (total > bid) ? (total - bid - 1) / nAiv + 1 : 0;
        for (uint32_t s = 0; s < PERM_STAGES && s < myRows; ++s) {
            CopyIn(bid + s * nAiv, s);
        }
        for (uint32_t r = 0; r < myRows; ++r) {
            const uint32_t stage = r % PERM_STAGES;
            CopyOut(bid + r * nAiv, stage);
            if (r + PERM_STAGES < myRows) {
                CopyIn(bid + (r + PERM_STAGES) * nAiv, stage);
            }
        }
    }

    __aicore__ inline uint32_t Total() const { return TOTAL; }

private:
    __aicore__ inline void CopyIn(uint32_t row, uint32_t stage)
    {
        LocalTensor<bfloat16_t> bufL(TPosition::VECCALC, UB_PM_ROW + stage * HIDDEN * 2, HIDDEN);
        const int32_t src = srcGm.GetValue(row);   // 标量 GM 读（地址计算，值依赖合法）
        BufAcquire<PIPE_MTE2>(BUF_AIV_PERM0 + stage);
        PipeBarrier<PIPE_MTE2>();                  // 小改 B
        DataCopy(bufL, xGm[static_cast<uint64_t>(src) * HIDDEN], Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE2>(BUF_AIV_PERM0 + stage);
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t stage)
    {
        LocalTensor<bfloat16_t> bufL(TPosition::VECCALC, UB_PM_ROW + stage * HIDDEN * 2, HIDDEN);
        BufAcquire<PIPE_MTE3>(BUF_AIV_PERM0 + stage);
        DataCopy(xSortedGm[static_cast<uint64_t>(row) * HIDDEN], bufL, Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE3>(BUF_AIV_PERM0 + stage);
    }

private:
    uint32_t M = 0;
    uint32_t TOPK = 1;
    uint32_t E = 1;
    uint32_t TOTAL = 0;
    AscendC::GlobalTensor<bfloat16_t> xGm;
    AscendC::GlobalTensor<int32_t> srcGm;
    AscendC::GlobalTensor<int32_t> countsGm;
    AscendC::GlobalTensor<bfloat16_t> xSortedGm;
};

// ============================================================
// S5/S7：行级 MXFP4 量化（m5_swiglu_quant 的 SwiGLU 五元组 + MxQuant 三段式）
//   小改 C：全 VEC 路径，**无 PIPE_S 挂载**（m3-AIV 的标量量化器被替换）
//   SWIGLU=true 时先做 SwiGLU（源为 gate|up 拼接行）+ 写 swiglu 输出（H 抽点用）
//   PAD_SRC/PAD_DST：源/目标是「按槽位 MAXM 行 padding」的布局（routed 路径），
//   否则是紧凑 m 行布局（shared 专家路径）
// ============================================================

template <uint32_t K, uint32_t SCAL_STRIDE, bool SWIGLU, bool PAD_SRC, bool PAD_DST, bool ROUTED = false>   // [M40-6g] ROUTED = 紧凑槽 + counts/active_num 驱动的行数
class VecQuantStage {
public:
    static constexpr uint32_t TILE = 256;
    static constexpr uint32_t FULL_TILES = K / TILE;
    static constexpr uint32_t NTILES = (K % TILE == 0) ? FULL_TILES : FULL_TILES + 1;
    static constexpr uint32_t PACKED_K = K / 2;
    static constexpr uint32_t SRC_ROW = SWIGLU ? (2 * K) : K;   // gate|up vs 单段
    static constexpr uint32_t GP = K / GROUP;                   // 每行 scale 有效字节数

    __aicore__ inline VecQuantStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* swigluOut, __gm__ uint8_t* qx,
                                __gm__ uint8_t* scale, __gm__ uint8_t* counts, uint32_t m, uint32_t topk)
    {
        M = m;
        TOPK = topk;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(src),
                              static_cast<uint64_t>(TOTAL_MAX) * SRC_ROW);
        swigluGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(swigluOut),
                                 static_cast<uint64_t>(TOTAL_MAX) * K);
        qxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(qx), static_cast<uint64_t>(TOTAL_MAX) * PACKED_K);
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scale),
                               static_cast<uint64_t>(TOTAL_MAX) * (SCAL_STRIDE / 2));
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts), NUM_EXPERTS);
    }

    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        uint32_t startOff[NUM_EXPERTS + 1];
        startOff[0] = 0;
        for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
            startOff[e + 1] = startOff[e] + static_cast<uint32_t>(countsGm.GetValue(e));
        }
        // [M40-6g] 行数由 counts / active_num 驱动：紧凑槽下 routed 路的行数 = Σt_e
        //（等于 m*topk，但**不写成常量**，也不依赖 NUM_EXPERTS 的取值）；共享路仍是 m 行。
        uint32_t rows = M;
        if constexpr (ROUTED) {
            rows = startOff[NUM_EXPERTS];
        }
        for (uint32_t r = bid; r < rows; r += nAiv) {
            uint32_t slot = 0;
            uint32_t lr = r;
            if constexpr (ROUTED || PAD_DST) {
                while (slot < NUM_EXPERTS - 1 && r >= startOff[slot + 1]) {
                    ++slot;
                }
                lr = r - startOff[slot];
            }
            const uint32_t srcRow = PAD_SRC ? (slot * M_MAX + lr) : r;
            const uint32_t dstRow = ROUTED ? r : (PAD_DST ? (slot * M_MAX + lr) : r);
            for (uint32_t t = 0; t < NTILES; ++t) {
                // 尾 tile 回退窗口（K % TILE != 0 时），group 对齐 → 重叠区结果逐位一致
                const uint32_t t0 = (t < FULL_TILES) ? t * TILE : (K - TILE);
                ProcessTile(r, srcRow, dstRow, t0);
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

        // ---- MTE2：GM → UB ----
        BufAcquire<PIPE_MTE2>(BUF_AIV_ROW0);
        PipeBarrier<PIPE_MTE2>();   // 小改 B：tile 循环同 pipe 复用同一 UB tile
        DataCopy(gateL, srcGm[xOff], Block1(TILE * 2));
        if constexpr (SWIGLU) {
            DataCopy(upL, srcGm[xOff + K], Block1(TILE * 2));
        }
        BufRelease<PIPE_MTE2>(BUF_AIV_ROW0);

        // ---- V：SwiGLU（可选）→ MxQuant scale → MxQuant data（原地接力）----
        BufAcquire<PIPE_V>(BUF_AIV_ROW0);
        BufAcquire<PIPE_V>(BUF_AIV_OUT);
        if constexpr (SWIGLU) {
            SwigluComputeTile(gateUb, upUb, swigluUb);
            MxQuantComputeScale(swigluUb, scaleUb, halfUb);
            MxQuantComputeDataFP4(swigluUb, halfUb, qxUb);
        } else {
            MxQuantComputeScale(gateUb, scaleUb, halfUb);
            MxQuantComputeDataFP4(gateUb, halfUb, qxUb);
        }
        BufRelease<PIPE_V>(BUF_AIV_ROW0);
        BufRelease<PIPE_V>(BUF_AIV_OUT);

        // ---- MTE3：UB → GM ----
        BufAcquire<PIPE_MTE3>(BUF_AIV_OUT);
        if constexpr (SWIGLU) {
            DataCopy(swigluGm[hOff], swigluL, Block1(TILE * 2));
        }
        DataCopy(qxGm[qxOff], qxL, Block1(TILE / 2));   // int8：TILE/2 字节 = 4 个 32B 块
        DataCopyPad<uint16_t, PaddingMode::Compact>(scaleGm[sOff], scaleL, ExtBlock1(TILE / GROUP));
        BufRelease<PIPE_MTE3>(BUF_AIV_OUT);
    }

    // SwiGLU 五元组（donor VFSwiGlu；m5 逐字 lift）：silu(g) = g/(1+exp(-g))，y = silu(gate)*up
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
                Cast<float, bfloat16_t, castTraitB162B32>(g, tg, dataMask);
                Cast<float, bfloat16_t, castTraitB162B32>(u, tu, dataMask);
                Muls(v, g, -1.0f, dataMask);
                Exp(v, v, dataMask);
                Adds(v, v, 1.0f, dataMask);
                Div(v, g, v, dataMask);
                Mul(y, v, u, dataMask);
                Cast<bfloat16_t, float, castTraitB322B16>(ty, y, dataMask);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(swigluUb + j * VL, ty, dataMask);
            }
        }
    }

    // 以下两段为 m2_mxfp4_quant → m5 的 MxQuantComputeScale / MxQuantComputeDataFP4 逐字移植
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
            // 官方 add_rms_norm_dynamic_mx_quant_common.h:412-413 同源：组内非有限（±Inf/NaN）
            // 时把 halfScale 覆盖成 NAN_CUSTOMIZATION=0x7F81（bf16 NaN），使后续 Mul(±Inf, NaN)=NaN
            // -> Cast<fp4> = 0.0；否则 Mul(±Inf, 2^-126)=±Inf -> Cast 饱和成 ±6。
            // 官方 :401 Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask=0x7F80) 得到的
            // mask 在此复用（置位取 src0=halfScale，清位取 src1=nanRegTensor）。
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
            Cast<fp4x2_e2m1_t, bfloat16_t, castTraitRM_Round>(vdExp0FP4, vdExp0, dataMask1);
            Cast<fp4x2_e2m1_t, bfloat16_t, castTraitRM_Round>(vdExp1FP4, vdExp1, dataMask1);
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

}  // namespace
// ============================================================
// S9a：unpermute 加权折叠（m8_permute kernel #2 移植；topK 模板实例化 1..TOPK_MAX）
//   routed_out[t] = bf16( Σ_k fp32(bf16 w[t,k]) * fp32(y_sorted[inv[t,k]]) )
// ============================================================

namespace M15M {

using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t U_STAGES = 2;
constexpr uint32_t WROW_I32 = 16;     // 权重行 int32 槽位数（64B，DataCopy 32B 对齐）
constexpr uint32_t UB_U_OUT = TOPK_MAX * HIDDEN * 2;
constexpr uint32_t UB_U_WROW = UB_U_OUT + HIDDEN * 2;

template <uint32_t TK>
__aicore__ inline void FmaChunk(RegTensor<float>& acc, RegTensor<float>& yF, RegTensor<bfloat16_t>& yB16,
                                RegTensor<int32_t>& wRaw, RegTensor<float>& wF, __ubuf__ bfloat16_t* yUb,
                                __ubuf__ int32_t* wRowUb, MaskReg& maskAll, uint32_t off)
{
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(yB16, yUb + TK * HIDDEN + off);
    Cast<float, bfloat16_t, castTraitB162B32>(yF, yB16, maskAll);
    LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(wRaw, wRowUb + TK);
    ShiftLefts((RegTensor<uint32_t>&)wF, (RegTensor<uint32_t>&)wRaw, static_cast<int16_t>(16), maskAll);
    Mul(yF, yF, wF, maskAll);
    Add(acc, acc, yF, maskAll);
}

template <uint32_t TK>
class UnpermuteStage {
public:
    __aicore__ inline UnpermuteStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* ySorted, __gm__ uint8_t* invSlot, __gm__ uint8_t* wtk,
                                __gm__ uint8_t* out, uint32_t m)
    {
        M = m;
        // [M84-8] Y 视图上界 = 紧凑 Σt_e 上界 TOTAL_MAX 行（**不是** padded 的
        //   NUM_EXPERTS*M_MAX：两者在 E=4 相等只是巧合，E=512 时不等）
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ySorted), static_cast<uint64_t>(TOTAL_MAX) * HIDDEN);
        slotGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(invSlot), static_cast<uint64_t>(M_MAX) * TOPK_MAX);
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(wtk), static_cast<uint64_t>(M_MAX) * WROW_I32);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(M_MAX) * HIDDEN);
    }

    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        const uint32_t myRows = (M > bid) ? (M - bid - 1) / nAiv + 1 : 0;
        for (uint32_t r = 0; r < myRows; ++r) {
            const uint32_t t = bid + r * nAiv;
            const uint32_t stage = r % U_STAGES;
            CopyIn(t, stage);
            ComputeRow(stage);
            CopyOut(t, stage);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t t, uint32_t stage)
    {
        LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_UP_STAGE0 + stage * U_STAGE_B, TOPK_MAX * HIDDEN);
        LocalTensor<int32_t> wL(TPosition::VECCALC, UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_WROW, WROW_I32);
        BufAcquire<PIPE_MTE2>(BUF_AIV_ROW0 + stage);
        PipeBarrier<PIPE_MTE2>();   // 小改 B
        DataCopy(wL, wGm[static_cast<uint64_t>(t) * WROW_I32], Block1(WROW_I32 * 4));
        for (uint32_t k = 0; k < TK; ++k) {
            const int32_t slot = slotGm.GetValue(static_cast<uint64_t>(t) * TK + k);   // 标量 GM 读→地址
            DataCopy(yL[k * HIDDEN], yGm[static_cast<uint64_t>(slot) * HIDDEN], Block1(HIDDEN * 2));
        }
        BufRelease<PIPE_MTE2>(BUF_AIV_ROW0 + stage);
    }

    __aicore__ inline void ComputeRow(uint32_t stage)
    {
        __ubuf__ bfloat16_t* yUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_UP_STAGE0 + stage * U_STAGE_B);
        __ubuf__ bfloat16_t* outUb =
            reinterpret_cast<__ubuf__ bfloat16_t*>(UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_OUT);
        __ubuf__ int32_t* wRowUb =
            reinterpret_cast<__ubuf__ int32_t*>(UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_WROW);

        BufAcquire<PIPE_V>(BUF_AIV_ROW0 + stage);
        __VEC_SCOPE__
        {
            RegTensor<float> acc;
            RegTensor<float> yF;
            RegTensor<float> wF;
            RegTensor<int32_t> wRaw;
            RegTensor<bfloat16_t> yB16;
            RegTensor<bfloat16_t> outB16;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t c = 0; c < CHUNKS_H; ++c) {
                const uint32_t off = static_cast<uint32_t>(c) * VL_F32;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(yB16, yUb + off);
                Cast<float, bfloat16_t, castTraitB162B32>(yF, yB16, maskAll);
                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(wRaw, wRowUb);
                ShiftLefts((RegTensor<uint32_t>&)wF, (RegTensor<uint32_t>&)wRaw, static_cast<int16_t>(16), maskAll);
                Mul(acc, yF, wF, maskAll);   // k = 0 直写 acc
                if constexpr (TK > 1) FmaChunk<1>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 2) FmaChunk<2>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 3) FmaChunk<3>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 4) FmaChunk<4>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 5) FmaChunk<5>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 6) FmaChunk<6>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 7) FmaChunk<7>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 8) FmaChunk<8>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                if constexpr (TK > 9) FmaChunk<9>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
                Cast<bfloat16_t, float, castTraitB322B16>(outB16, acc, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(outUb + off, outB16, maskAll);
            }
        }
        BufRelease<PIPE_V>(BUF_AIV_ROW0 + stage);
    }

    __aicore__ inline void CopyOut(uint32_t t, uint32_t stage)
    {
        LocalTensor<bfloat16_t> outL(TPosition::VECCALC, UB_UP_STAGE0 + stage * U_STAGE_B + UB_U_OUT, HIDDEN);
        BufAcquire<PIPE_MTE3>(BUF_AIV_ROW0 + stage);
        DataCopy(outGm[static_cast<uint64_t>(t) * HIDDEN], outL, Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE3>(BUF_AIV_ROW0 + stage);
    }

private:
    uint32_t M = 0;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    AscendC::GlobalTensor<int32_t> slotGm;
    AscendC::GlobalTensor<int32_t> wGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;
};

// ============================================================
// S9b：combine（shared_out = sigmoid(sgate)·shared_mlp；moe_out = routed + shared_out）
//   sgate 的 sigmoid 在这里用向量做（S2 只出裸点积，避免 AIV 内标量浮点）
// ============================================================

class CombineStage {
public:
    __aicore__ inline CombineStage() {}

    __aicore__ inline void Init(__gm__ uint8_t* routed, __gm__ uint8_t* shdMlp, __gm__ uint8_t* sgate,
                                __gm__ uint8_t* sharedOut, __gm__ uint8_t* moeOut, uint32_t m)
    {
        M = m;
        routedGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(routed), static_cast<uint64_t>(M_MAX) * HIDDEN);
        shdGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(shdMlp), static_cast<uint64_t>(M_MAX) * HIDDEN);
        sgateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sgate), M_MAX);
        sharedGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sharedOut), static_cast<uint64_t>(M_MAX) * HIDDEN);
        moeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(moeOut), static_cast<uint64_t>(M_MAX) * HIDDEN);
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
        // 整行（MTE2 写 → V 就地改 → MTE3 读）必须共用一个 BufferID：
        // 否则下一行的 MTE2 会抢在 MTE3 读之前覆盖 UB_CB_ROUTED/SHARED
        BufAcquire<PIPE_MTE2>(BUF_AIV_ROW0);
        PipeBarrier<PIPE_MTE2>();   // 小改 B
        DataCopy(rtL, routedGm[off], Block1(HIDDEN * 2));
        DataCopy(shL, shdGm[off], Block1(HIDDEN * 2));
        // 单值 4B / 非 32B 对齐地址：必须走 DataCopyPad（非原子 DataCopy 会报地址未对齐）
        DataCopyPad(sgL, sgateGm[t], ExtBlock1(4), DataCopyPadExtParams<float>{false, 0, 0, 0});
        BufRelease<PIPE_MTE2>(BUF_AIV_ROW0);
    }

    __aicore__ inline void ComputeRow()
    {
        __ubuf__ bfloat16_t* rtUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_CB_ROUTED);
        __ubuf__ bfloat16_t* shUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_CB_SHARED);
        __ubuf__ float* sgUb = reinterpret_cast<__ubuf__ float*>(UB_CB_G);

        BufAcquire<PIPE_V>(BUF_AIV_ROW0);   // 与 MTE2 写、MTE3 读共用同一 token
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
            LoadAlign<float, LoadDist::DIST_BRC_B32>(g, sgUb);   // 门裸点积广播
            Muls(g, g, -1.0f, maskAll);
            Exp(g, g, maskAll);
            Adds(g, g, 1.0f, maskAll);
            Div(g, one, g, maskAll);                             // sigmoid = 1/(1+exp(-x))
            for (uint16_t c = 0; c < CHUNKS_H; ++c) {
                const uint32_t off = static_cast<uint32_t>(c) * VL_F32;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(shB, shUb + off);
                Cast<float, bfloat16_t, castTraitB162B32>(shF, shB, maskAll);
                Mul(shF, shF, g, maskAll);
                Cast<bfloat16_t, float, castTraitB322B16>(outB, shF, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(shUb + off, outB, maskAll);  // 原地写回
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(rtB, rtUb + off);
                Cast<float, bfloat16_t, castTraitB162B32>(rtF, rtB, maskAll);
                Add(shF, shF, rtF, maskAll);
                Cast<bfloat16_t, float, castTraitB322B16>(outB, shF, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(rtUb + off, outB, maskAll);  // 复用 routed 缓冲
            }
        }
        BufRelease<PIPE_V>(BUF_AIV_ROW0);
    }

    __aicore__ inline void CopyOutRow(uint32_t t)
    {
        // shared 结果原地写在 UB_CB_SHARED，moe 结果写在 UB_CB_ROUTED
        LocalTensor<bfloat16_t> shL(TPosition::VECCALC, UB_CB_SHARED, HIDDEN);
        LocalTensor<bfloat16_t> moeL(TPosition::VECCALC, UB_CB_ROUTED, HIDDEN);
        const uint64_t off = static_cast<uint64_t>(t) * HIDDEN;
        BufAcquire<PIPE_MTE3>(BUF_AIV_ROW0);
        DataCopy(sharedGm[off], shL, Block1(HIDDEN * 2));
        DataCopy(moeGm[off], moeL, Block1(HIDDEN * 2));
        BufRelease<PIPE_MTE3>(BUF_AIV_ROW0);
    }

private:
    uint32_t M = 0;
    AscendC::GlobalTensor<bfloat16_t> routedGm;
    AscendC::GlobalTensor<bfloat16_t> shdGm;
    AscendC::GlobalTensor<float> sgateGm;
    AscendC::GlobalTensor<bfloat16_t> sharedGm;
    AscendC::GlobalTensor<bfloat16_t> moeGm;
};

}  // namespace
// ============================================================
// MoE layer kernel：段序编排 + 跨核同步（单一 __mix__(1,2) 启动）
// ============================================================

namespace M15M {

using namespace AscendC;
using namespace AscendC::Reg;

// 段形状的编译期常量（gate_up: K=2560 N=1280；down: K=640 N=2560）
constexpr uint32_t GU_NBLK = GU_N / BASE_N;       // 5
constexpr uint32_t DN_NBLK = HIDDEN / BASE_N;     // 10
constexpr uint32_t NUM_SLOTS = NUM_EXPERTS + 1;   // routed 专家 + shared 专家
constexpr uint32_t GU_ITEMS = NUM_SLOTS * GU_NBLK;
constexpr uint32_t DN_ITEMS = NUM_SLOTS * DN_NBLK;

struct MoeLayerPtrs {
    __gm__ uint8_t* ws;
    __gm__ uint8_t* xLayer;     // 层输入激活 [M_MAX, HIDDEN] bf16
    __gm__ uint8_t* yLayer;     // S10 出口（M40 改造：写独立 GM 缓冲 = 层间残差流）
    __gm__ uint8_t* resZero;    // 全 0 残差（S1 的残差入口）
    __gm__ uint8_t* gamma1;
    __gm__ uint8_t* gamma2;
    __gm__ uint8_t* routerW;
    __gm__ uint8_t* sgateW;
    __gm__ uint8_t* wGu;
    __gm__ uint8_t* sGu;
    __gm__ uint8_t* wDn;
    __gm__ uint8_t* sDn;
    __gm__ uint8_t* wGuShd;
    __gm__ uint8_t* sGuShd;
    __gm__ uint8_t* wDnShd;
    __gm__ uint8_t* sDnShd;
    uint32_t m;
    uint32_t topk;
    uint32_t sliceMode;   // 0 = MoE slice 直接消费 golden x（逐 op 对齐 golden）
                          // 1 = 全层链（router 消费 S1 的 x_norm）
    uint32_t stageLimit;  // 段序截断（bring-up 定位用；8 = 全链）
    uint32_t subLimit;    // S2/S3 段内截断（bring-up 定位用；9 = 全开）
};

class MoeLayerChain {
public:
    __aicore__ inline MoeLayerChain() {}

    __aicore__ inline void Init(const MoeLayerPtrs& args)
    {
        p = args;
        ws = args.ws;

        // S1 m6#1：x = 层输入，res = 0，gamma1
        norm1.Init(args.xLayer, args.resZero, ws + WS_XNORM, ws + WS_RES1, UB_GAMMA1_F32, args.m);
        // S10 m6#2：x = moe_out，res = S1 的 fp32 resOut（小改 D），gamma2
        norm2.Init(ws + WS_MOE, ws + WS_RES1, args.yLayer, ws + WS_RES2, UB_GAMMA2_F32, args.m);

        __gm__ uint8_t* sliceSrc = args.sliceMode ? (ws + WS_XNORM) : args.xLayer;

        // S2/S3（AIV0）
        router.Init(sliceSrc, args.routerW, args.sgateW, ws + WS_LOGITS, ws + WS_IDS, ws + WS_W, ws + WS_SGATE,
                    args.m, args.topk);
        idxGen.Init(ws + WS_IDS, ws + WS_W, ws + WS_COUNTS, ws + WS_OFFSETS, ws + WS_PERM_SRC, ws + WS_PERM_EXP,
                    ws + WS_INV, ws + WS_WTK, args.m, args.topk);
        // S4 permute
        perm.Init(sliceSrc, ws + WS_PERM_SRC, ws + WS_COUNTS, ws + WS_XSORT, NUM_EXPERTS, args.m, args.topk);
        // S5 A 侧量化（routed + shared）
        quantA.Init(ws + WS_XSORT, nullptr, ws + WS_AQ, ws + WS_AS, ws + WS_COUNTS, args.m, args.topk);
        quantAShd.Init(sliceSrc, nullptr, ws + WS_AQ_SHD, ws + WS_AS_SHD, ws + WS_COUNTS, args.m, args.topk);
        // S7 SwiGLU + 量化（routed + shared）
        quantH.Init(ws + WS_GU, ws + WS_H, ws + WS_HQ, ws + WS_HS, ws + WS_COUNTS, args.m, args.topk);
        quantHShd.Init(ws + WS_GU_SHD, ws + WS_H_SHD, ws + WS_HQ_SHD, ws + WS_HS_SHD, ws + WS_COUNTS, args.m,
                       args.topk);
        // S8/S9
        PrepareUnpermute(args);
    }

    // ---- AIV 侧 ----
    __aicore__ inline void ProcessAiv()
    {
        const uint32_t bid = AscendC::GetBlockIdx();          // 0 .. 2*numBlocks-1
        const uint32_t nAiv = AscendC::GetBlockNum() * 2;
        const bool isPrimary = (bid == 0);

        // gamma bf16 → fp32 预转（每核一次）
        PrecastGammaF32(p.gamma1, UB_GAMMA1_F32);
        PrecastGammaF32(p.gamma2, UB_GAMMA2_F32);

        // ---------- S1：m6#1 ----------
        if (p.stageLimit >= 1) {
            norm1.Run(bid, nAiv);
            BarrierAiv<PIPE_MTE2>(0);   // 等 x_norm/res1 全体就位
        }

        // ---------- S2/S3：router + 索引生成（AIV0）----------
        if (p.stageLimit >= 2) {
            if (isPrimary) {
                router.Run(p.subLimit);
                if (p.subLimit >= 6) {
                    idxGen.Run(p.subLimit);
                }
            }
            BarrierAiv<PIPE_S>(1);      // perm_src/counts 的标量 GM 读需要 PIPE_S 值依赖
        }

        // ---------- S4：permute ----------
        if (p.stageLimit >= 3) {
            perm.Run(bid, nAiv);
            BarrierAiv<PIPE_MTE2>(2);
        }

        // ---------- S5：A 侧 MXFP4 全 VEC 量化 ----------
        if (p.stageLimit >= 4) {
            quantA.Run(bid, nAiv);
            quantAShd.Run(bid, nAiv);
            CrossCoreSetFlag<CC_MODE2, PIPE_MTE3>(FLAG_M2_RING[0]);   // A 就绪 → 配对 AIC
        }

        // ---------- S7：SwiGLU + MXFP4 量化（等 GU）----------
        if (p.stageLimit >= 5) {
            CrossCoreWaitFlag<CC_MODE2, PIPE_MTE2>(FLAG_M2_RING[1]);
            quantH.Run(bid, nAiv);
            quantHShd.Run(bid, nAiv);
            CrossCoreSetFlag<CC_MODE2, PIPE_MTE3>(FLAG_M2_RING[2]);   // H 就绪 → 配对 AIC
        }

        // ---------- S9a：unpermute 加权折叠（等 Y）----------
        if (p.stageLimit >= 6) {
            CrossCoreWaitFlag<CC_MODE2, PIPE_S>(FLAG_M2_RING[3]);     // inv_slot/w_tk 标量读
            RunUnpermute(bid, nAiv);
            BarrierAiv<PIPE_MTE2>(3);
        }

        // ---------- S9b：combine ----------
        if (p.stageLimit >= 7) {
            combine.Run(bid, nAiv);
            BarrierAiv<PIPE_MTE2>(0);
        }

        // ---------- S10：m6#2 ----------
        if (p.stageLimit >= 8) {
            norm2.Run(bid, nAiv);
        }
    }

    // ---- AIC 侧 ----
    __aicore__ inline void ProcessAic()
    {
        const uint32_t bid = AscendC::GetBlockIdx();          // 0 .. numBlocks-1
        const uint32_t numBlocks = AscendC::GetBlockNum();
        AscendC::GlobalTensor<uint32_t> countsGm;
        countsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ws + WS_COUNTS), NUM_EXPERTS);
        // [M40-6f] 紧凑槽起点（= S3 产出的 expert_offsets 前缀和）；absent 时由 counts 推算
        AscendC::GlobalTensor<uint32_t> offGm;
        offGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ws + WS_OFFSETS), NUM_EXPERTS + 1);

        if (p.stageLimit < 5) {
            return;
        }
        // ---------- 等 S5 的 A 就绪 ----------
        CrossCoreWaitFlag<CC_MODE2, PIPE_S>(FLAG_M2_RING[0]);
        CrossCoreSetFlag<CC_MODE0, PIPE_MTE2>(FLAG_AIC_SEG_RING[0]);
        CrossCoreWaitFlag<CC_MODE0, PIPE_S>(FLAG_AIC_SEG_RING[0]);

        // ---------- S6：grouped gate_up GEMM（槽位 = 4 routed + 1 shared）----------
        // [M40-6f] 工作项 = (expert, mTile, nBlock)：空专家由 counts 跳过（count 驱动）；
        // mTile 数由该专家的 t_e 推出（本实例 t_e <= BASE_M ⇒ mTiles 恒 1；prefill 自动分段）。
        for (uint32_t slot = 0; slot < NUM_SLOTS; ++slot) {
            const bool shd = (slot == NUM_EXPERTS);
            const uint32_t t = shd ? p.m : countsGm.GetValue(slot);
            if (t == 0) {
                continue;   // 空专家：A/H 区未被写，无需计算
            }
            const uint32_t mTiles = (t + BASE_M - 1) / BASE_M;
            for (uint32_t mt = 0; mt < mTiles; ++mt) {
                const uint32_t rows = ((t - mt * BASE_M) < BASE_M) ? (t - mt * BASE_M) : BASE_M;
                const uint32_t rowBase = shd ? (mt * BASE_M) : (offGm.GetValue(slot) + mt * BASE_M);
                for (uint32_t nb = bid; nb < GU_NBLK; nb += numBlocks) {
                    gemmGu.SetItem(shd ? (ws + WS_AQ_SHD + rowBase * (HIDDEN / 2)) : (ws + WS_AQ + rowBase * (HIDDEN / 2)),   // [M40-6d]
                        shd ? (ws + WS_AS_SHD + rowBase * GU_SCALE_STRIDE) : (ws + WS_AS + rowBase * GU_SCALE_STRIDE),
                        shd ? p.wGuShd : (p.wGu + slot * GU_N * (HIDDEN / 2)),
                        shd ? p.sGuShd : (p.sGu + slot * GU_N * GU_SCALE_STRIDE),
                        shd ? (ws + WS_GU_SHD + rowBase * GU_N * 2) : (ws + WS_GU + rowBase * GU_N * 2));
gemmGu.Run(nb, rows);
                }
            }
        }
        CrossCoreSetFlag<CC_MODE0, PIPE_FIX>(FLAG_AIC_SEG_RING[1]);
        CrossCoreWaitFlag<CC_MODE0, PIPE_S>(FLAG_AIC_SEG_RING[1]);
        CrossCoreSetFlag<CC_MODE2, PIPE_MTE2>(FLAG_M2_RING[1]);   // GU 就绪 → 配对 AIV

        if (p.stageLimit < 6) {
            return;
        }
        // ---------- 等 S7 的 H 就绪 ----------
        CrossCoreWaitFlag<CC_MODE2, PIPE_S>(FLAG_M2_RING[2]);
        CrossCoreSetFlag<CC_MODE0, PIPE_MTE2>(FLAG_AIC_SEG_RING[2]);
        CrossCoreWaitFlag<CC_MODE0, PIPE_S>(FLAG_AIC_SEG_RING[2]);

        // ---------- S8：grouped down GEMM ----------
        // [M40-6f] 工作项 = (expert, mTile, nBlock)：空专家由 counts 跳过（count 驱动）；
        // mTile 数由该专家的 t_e 推出（本实例 t_e <= BASE_M ⇒ mTiles 恒 1；prefill 自动分段）。
        for (uint32_t slot = 0; slot < NUM_SLOTS; ++slot) {
            const bool shd = (slot == NUM_EXPERTS);
            const uint32_t t = shd ? p.m : countsGm.GetValue(slot);
            if (t == 0) {
                continue;   // 空专家：A/H 区未被写，无需计算
            }
            const uint32_t mTiles = (t + BASE_M - 1) / BASE_M;
            for (uint32_t mt = 0; mt < mTiles; ++mt) {
                const uint32_t rows = ((t - mt * BASE_M) < BASE_M) ? (t - mt * BASE_M) : BASE_M;
                const uint32_t rowBase = shd ? (mt * BASE_M) : (offGm.GetValue(slot) + mt * BASE_M);
                for (uint32_t nb = bid; nb < DN_NBLK; nb += numBlocks) {
                    gemmDn.SetItem(shd ? (ws + WS_HQ_SHD + rowBase * (INTER / 2)) : (ws + WS_HQ + rowBase * (INTER / 2)),   // [M40-6e]
                        shd ? (ws + WS_HS_SHD + rowBase * DN_SCALE_STRIDE) : (ws + WS_HS + rowBase * DN_SCALE_STRIDE),
                        shd ? p.wDnShd : (p.wDn + slot * HIDDEN * (INTER / 2)),
                        shd ? p.sDnShd : (p.sDn + slot * HIDDEN * (INTER / GROUP)),
                        shd ? (ws + WS_Y_SHD + rowBase * HIDDEN * 2) : (ws + WS_Y + rowBase * HIDDEN * 2));
gemmDn.Run(nb, rows);
                }
            }
        }
        CrossCoreSetFlag<CC_MODE0, PIPE_FIX>(FLAG_AIC_SEG_RING[3]);
        CrossCoreWaitFlag<CC_MODE0, PIPE_S>(FLAG_AIC_SEG_RING[3]);
        CrossCoreSetFlag<CC_MODE2, PIPE_MTE2>(FLAG_M2_RING[3]);   // Y 就绪 → 配对 AIV
    }

private:
    // 全体 AIV mode 0 barrier（wait 挂窄 pipe；需标量值依赖的段用 PIPE_S）
    template <pipe_t WAIT_PIPE>
    __aicore__ inline void BarrierAiv(uint32_t slot)
    {
        CrossCoreSetFlag<CC_MODE0, PIPE_MTE3>(FLAG_AIV_SEG_RING[slot]);
        CrossCoreWaitFlag<CC_MODE0, WAIT_PIPE>(FLAG_AIV_SEG_RING[slot]);
    }

    // topK 运行时 → 模板分发（m8#2 同款：k 链全展开，下标皆编译期）
    // [M84-7] 真实 topk=10 ⇒ 展开到 1..9 + <10>（m13 只到 1..3 / <1..4>）
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

public:
    // Init 中把 topK 分发到模板实例（避免 Init 里做 switch）
    __aicore__ inline void PrepareUnpermute(const MoeLayerPtrs& args)
    {
        unperm1.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm2.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm3.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm4.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm5.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm6.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm7.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm8.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm9.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        unperm10.Init(ws + WS_Y, ws + WS_INV, ws + WS_WTK, ws + WS_ROUTED, args.m);
        combine.Init(ws + WS_ROUTED, ws + WS_Y_SHD, ws + WS_SGATE, ws + WS_SHARED, ws + WS_MOE, args.m);
    }

private:
    MoeLayerPtrs p;
    __gm__ uint8_t* ws = nullptr;

    NormStage<false> norm1;      // 残差 bf16（全 0）
    NormStage<true> norm2;       // 残差 fp32（小改 D）
    RouterStage router;
    IndexGenStage idxGen;
    PermuteStage perm;
    VecQuantStage<HIDDEN, GU_SCALE_STRIDE, false, false, false, true> quantA;   // [M40-6b] 紧凑目标+active_num 驱动
    VecQuantStage<HIDDEN, GU_SCALE_STRIDE, false, false, false> quantAShd;
    VecQuantStage<INTER, DN_SCALE_STRIDE, true, false, false, true> quantH;   // [M40-6c] 紧凑源+目标
    VecQuantStage<INTER, DN_SCALE_STRIDE, true, false, false> quantHShd;
    UnpermuteStage<1> unperm1;
    UnpermuteStage<2> unperm2;
    UnpermuteStage<3> unperm3;
    UnpermuteStage<4> unperm4;
    UnpermuteStage<5> unperm5;
    UnpermuteStage<6> unperm6;
    UnpermuteStage<7> unperm7;
    UnpermuteStage<8> unperm8;
    UnpermuteStage<9> unperm9;
    UnpermuteStage<10> unperm10;
    CombineStage combine;
    MXFP4GemmItem<HIDDEN, GU_N, GU_SCALE_STRIDE> gemmGu;
    MXFP4GemmItem<INTER, HIDDEN, DN_SCALE_STRIDE> gemmDn;
};

}  // namespace

#endif  // M15_MOE_LAYER_H
