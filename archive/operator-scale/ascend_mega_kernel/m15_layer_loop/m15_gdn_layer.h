/**
 * m14_gdn_layer.asc —— Mega kernel M22：GDN layer kernel 骨架
 * （图纸：docs/12-layer-integration.md §2/§4/§5 GDN 段序 + docs/10-gdn-analysis.md §1；约束 docs/05 §5/§6）
 *
 * 单一 __mix__(1,2) 启动内跑完整 GDN 层段序 S1-S7（decode m=1）：
 *
 *   S1 m6#1  Add+RMSNorm#1：x + res → x_norm(bf16) + res1(fp32)              [AIV 行切分]
 *   S2 m11   in_proj bf16 GEMM：K=2560 N=16480（权重 [16480,2560]）→ qkvzba [1,16480]bf16 [AIC N 切分]
 *   S3 m9    prolog：conv1d(K=4)+bias+SiLU → q/k l2norm（仅 q ×1/√128）→ gating(g/β)
 *            conv_state planar [3][10240] **in-place 原位更新**                  [AIV block 条带 80 块]
 *   S4 m4    recurrence：每 value head 对 [48,128,128] fp32 ssm_state **in-place RMW**
 *            （decay→delta→outer→matvec 寄存器单遍融合）→ o[48,128] fp32        [AIV head 条带 48]
 *   S5 m12   RMSNormGated：per-head(128) RMSNorm + ×gamma + ×sigmoid(z)
 *            （z 段 = S2 输出 qkvzba 的 [10240,16384) 元素，S2→S5 全程存活）      [AIV head 条带 48]
 *   S6 m11   out_proj bf16 GEMM：K=6144 N=2560 → [1,2560]bf16                [AIC N 切分]
 *   S7 m6#2  Add+RMSNorm#2：opout + res1(fp32 残差，小改 D) → y_final + res2     [AIV 行切分]
 *
 * 段边界同步（docs/12 §5 标准 mix 序列；flagId 用 GDN 槽 4-7）：
 *   AIV→AIC（S1 后 / S5 后）：全体 AIV 先到齐（mode-0 barrier，set 挂 MTE3 / wait 挂 MTE2，
 *     两 pipe 均 drain）→ 每个 AIV `set mode2(PIPE_MTE3)` → 每个 AIC `wait mode2(PIPE_S)`；
 *   AIC→AIV（in_proj 后 / out_proj 后）：全体 AIC `set mode0(PIPE_FIX)`（FIXP 写 GM 排空）
 *     + `wait mode0(PIPE_S)`（全体 AIC 对齐）→ 每个 AIC `set mode2(PIPE_MTE2)` →
 *     每个 AIV `wait mode2(PIPE_MTE2)`（数据依赖段挂 MTE2，标量依赖段挂 PIPE_S）。
 *   set 一律挂生产 pipe（FIX/MTE3），wait 用最窄 pipe；相邻同步点 flagId 必不同（旋转槽）。
 *
 * 算件来源（全部「复制改造」，原目录不动）：
 *   S1/S7  m6_rmsnorm（NormStage 模板 + donor 二分 fold reduce / NR rsqrt；fp32 残差 = 小改 D）
 *   S2/S6  m11_bf16_gemm（Bf16Gemm → Bf16GemmItem：Nd2Nz → LoadData2D → Mmad → Fixpipe F322BF16）
 *   S3     m9_gdn_prolog（GdnProlog：conv 增量 tap 累加 + l2norm + 全 head 向量化 gating）
 *   S4     m4_gdn_recurrent（GdnDecodeRec：寄存器 VF 单遍融合递推 + slab 乒乓软件流水）
 *   S5     m12_rmsnorm_gated（RmsNormGatedKernel → head 切分版；乘法次序/sigmoid 排布逐字保留）
 *
 * 小改清单（docs/12 §6 GDN 相关）：
 *   A. 元素计数 DataCopy 重载 → 显式 DataCopyParams/DataCopyExtParams（各段复制改造时逐点落实）
 *   B. 行循环 MTE2 背靠背复用同一 UB 行缓冲 → 补 PipeBarrier<PIPE_MTE2>
 *   D. m6 残差输入 fp32（S7 消费 S1 的 fp32 res1）
 *   G. m9↔m4 的 g/β stride-8 与 {H,1,0,0} 直读契约冻结（两段同一份代码，天然一致）
 *   本 mission 只覆盖 decode m=1 + 段间串行（不做跨段预取），见 README「已知限制」。
 */

#include "acl/acl.h"
#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "reg_compute/kernel_reg_compute_intf.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "m15_gdn_resources.h"

namespace {

using namespace AscendC;
using namespace AscendC::Reg;
using namespace M15G;

// ============================================================
// 通用：BufferID / Mutex 封装 + 显式块拷贝参数
// ============================================================

template <pipe_t pipe>
__aicore__ inline void BufAcquire(AscendC::MutexID id)
{
    AscendC::GetBufInternal<pipe, false>(id);   // acquire 立即获取
}

template <pipe_t pipe>
__aicore__ inline void BufRelease(AscendC::MutexID id)
{
    AscendC::RlsBufInternal<pipe, false>(id);   // release 用 `false`：CANN ASC_LOCK_BLOCK 默认（与 acquire 同模式）
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

// 显式块拷贝参数（小改 A：杜绝元素计数重载——其 DataCopyParams 只填 blockLen，
// blockCount/srcGap/dstGap 是未初始化栈垃圾，M16 实测踩雷）。
// 注意 DataCopyParams.blockLen 单位是 **32B 块**，故入参统一为字节数、内部 /32 换算。
__aicore__ inline AscendC::DataCopyParams Block1(uint32_t bytes)
{
    if ((bytes & 31u) != 0u) {
        AscendC::Trap();   // 非 32B 整数倍必须走 DataCopyPad，这里防呆
    }
    return AscendC::DataCopyParams{1, static_cast<uint16_t>(bytes / 32), 0, 0};
}

// ============================================================
// S1/S7（m6）：norm 段常量
// ============================================================

constexpr uint32_t VL_F32 = 64;                 // 256B 向量寄存器 fp32 lane 数
constexpr uint16_t CHUNKS_H = HIDDEN / VL_F32;  // 每行 64-lane chunk 数 = 40
constexpr float RMS_EPS = 1e-6f;
constexpr float RMS_AVG_FACTOR = 1.0f / static_cast<float>(HIDDEN);
constexpr uint32_t FOLD_POINT = 1280;           // 二分 fold 点（m6 donor tiling binAddQuotient）
constexpr uint32_t REDUCE_TMP_STRIDE = 24;      // ceil(foldLoops(20)/BLK_B32(8))*8

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
__aicore__ inline void CalculateSquareReduceSumLessThanVL(__ubuf__ T* xPtr, __ubuf__ float* dstPtr,
                                                          uint16_t rows, uint32_t rowStride, uint32_t reduceNum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        MaskReg pregLoop = UpdateMask<float>(reduceNum);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        for (uint16_t i = 0; i < rows; ++i) {
            LoadRegForDtype<T>(xPtr, xReg, pregLoop, static_cast<uint32_t>(i) * rowStride);
            Mul(xReg, xReg, xReg, pregLoop);
            Reduce<ReduceType::SUM>(sumReg, xReg, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, sumReg, pregOne);
        }
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(__ubuf__ T* xPtr, __ubuf__ float* dstPtr,
                                                             uint16_t rows, uint32_t rowStride,
                                                             uint32_t reduceNum)
{
    uint32_t tailLen = reduceNum - V_LENGTH;
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregTail = UpdateMask<float>(tailLen);
        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t baseOffset = static_cast<uint32_t>(i) * rowStride;
            LoadRegForDtype<T>(xPtr, xReg, pregFull, baseOffset);
            LoadRegForDtype<T>(xPtr + V_LENGTH, xFoldReg, pregTail, baseOffset);
            Mul(xReg, xReg, xReg, pregFull);
            Mul(xFoldReg, xFoldReg, xFoldReg, pregTail);
            ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                       pregTail);
            Add(sumReg, xReg, xFoldReg, pregFull);
            Reduce<ReduceType::SUM>(reduceReg, sumReg, pregFull);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
        }
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSum(__ubuf__ T* xPtr, __ubuf__ float* dstPtr, __ubuf__ float* tmpPtr,
                                                uint16_t rows, uint32_t rowStride, uint32_t reduceNum,
                                                uint32_t foldPoint, uint32_t tmpStride, uint32_t branchNum = 0)
{
    // donor（m6/m12 同源）的完整 dispatch，逐字保留：
    //   reduceNum ≤ 64        → LessThanVL（m12 的 128 维不落此支）
    //   reduceNum ≤ 128       → LessThanTwoVL（**m12/S5 的 128 维走这支**：无 tmp 往返、无 LocalMemBar）
    //   reduceNum ≤ 8192      → Common<1>（m6/S1·S7 的 2560 维 = 2×1280 fold 走这支，与 m6 donor 一致）
    //   否则                  → Common<2>
    uint32_t reduceBranchNum = branchNum == 0 ? reduceNum : branchNum;
    if (reduceBranchNum <= V_LENGTH) {
        CalculateSquareReduceSumLessThanVL<T>(xPtr, dstPtr, rows, rowStride, reduceNum);
    } else if (reduceBranchNum <= V_LENGTH + V_LENGTH) {
        CalculateSquareReduceSumLessThanTwoVL<T>(xPtr, dstPtr, rows, rowStride, reduceNum);
    } else if (reduceBranchNum <= V_LENGTH * V_LENGTH * DICHOTOMY_ADD_COEFF) {
        CalculateSquareReduceSumCommon<T, 1>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum, foldPoint, tmpStride);
    } else {
        CalculateSquareReduceSumCommon<T, DICHOTOMY_ADD_COEFF>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum,
                                                               foldPoint, tmpStride);
    }
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
// ============================================================
// S1/S7：NormStage（m6_rmsnorm 逐字 lift；行切分 + fp32 残差 = 小改 D）
// ============================================================
// 段内 BufferID 取 M15G 全局资源表（本段专属槽 BUF_NORM_*，与其它段完全错开）
constexpr uint32_t BUF_AIV_ROW0 = M15G::BUF_NORM_ROW;   // 行输入: MTE2 → V
constexpr uint32_t BUF_AIV_OUT = M15G::BUF_NORM_OUT;    // 行输出: V → MTE3

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
        BufRelease<PIPE_V>(BUF_AIV_OUT);     // xFp32/y 写完（release mode=false），MTE3 可读
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
// ============================================================
// S3：GDN prolog（m9_gdn_prolog 复制改造；纯 AIV，80 个 128ch block 条带划分）
//   conv1d(K=4,128ch/block, planar conv_state in-place) + bias + SiLU
//   → q/k per-key-head l2norm（仅 q 乘 1/√128）→ gating g/β（每 value head 一槽）
//   段内 BufferID：BUF_PL_*；UB 偏移：UB_PL_*（32B 对齐、13KB）
// ============================================================

namespace Prolog {
namespace {
using namespace AscendC;
using namespace AscendC::Reg;

// ---- 模型维度（Qwen3.8-Flash-Next GDN 层，docs/10 §1）----
constexpr uint32_t HD = 128;      // head 维度（q/k/v 均 128）
constexpr uint32_t KW = 4;        // conv1d 宽度 K=4
constexpr uint32_t ST = 3;        // conv_state 历史行数（K-1）
constexpr uint32_t CH = 10240;    // conv 通道数 = q2048|k2048|v6144
constexpr uint32_t NQB = 16;      // q head 块数（2048/128）
constexpr uint32_t QKB = 32;      // q+k 块数（4096/128）
constexpr uint32_t NBLK = 80;     // 总块数 = 10240/128（q 16 | k 16 | v 48）
constexpr uint32_t VH = 48;       // value head 数（g/β per-head）
constexpr uint32_t INW = 16480;   // in_proj 输出宽度
constexpr uint32_t XB_OFF = 16384;  // x 内 b48 段偏移（bf16 元素）
constexpr uint32_t XA_OFF = 16432;  // x 内 a48 段偏移

constexpr uint32_t VL = 64;                 // fp32 向量寄存器 lane 数
constexpr float EPS = 1e-6f;                // l2norm eps
constexpr float QSCALE = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float SPTH = 20.0f;               // softplus threshold（β=1）

// bf16->fp32 cast trait（m5 同款 castTraitB162B32Even；本核无 B32->B16 cast）
constexpr CastTrait castTraitB162B32Even = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                            RoundMode::UNKNOWN};

// ---- UB 静态布局（字节偏移；全部 32B 对齐；VECCALC 空间 248KB 内）----
// 每个 parity 的 block 输入/输出区（2816B）：
//   [0,768)  conv_state block（bf16 3 行 × 256B，行 j = 通道历史元素 j）
//   [768,1024) x block（bf16 128）
//   [1024,2048) w block（bf16 4 tap 行 × 256B，行 j = w[j][c]）
//   [2048,2304) bias block（bf16 128）
//   [2304,2816) OUT（fp32 128：conv+SiLU 结果；q/k 块原地被 l2norm 结果覆盖）
// state 写回：planar [3][10240] 下 new 行0/1/2 分别取本区 [256,512)/[512,768)/[768,1024)
// 各 256B（blockCount=1 单块拷贝 ×3，规避 3510 NZ 重排 quirk）。
constexpr uint32_t PAR_B = 2816;
constexpr uint32_t ST_OFF = 0;
constexpr uint32_t X_OFF = 768;
constexpr uint32_t W_OFF = 1024;
constexpr uint32_t B_OFF = 2048;
constexpr uint32_t O_OFF = 2304;
// 偏移一律取 M15G 全局资源表（m14_resources.h §4：SEG-GDN 窗内属于 m9 的 UB_PL_* 段）
constexpr uint32_t UB_BK0 = M15G::UB_PL_BK0;
constexpr uint32_t UB_BK1 = M15G::UB_PL_BK1;
// gating 区（a/b 为 bf16 staging + UNPACK 越界预读 slack；A_log/dt_bias 为 fp32 + slack）
constexpr uint32_t UB_AST = M15G::UB_PL_AST;   // a staging bf16 [64] + slack   256B
constexpr uint32_t UB_BST = M15G::UB_PL_BST;   // b staging bf16 [64] + slack   256B
constexpr uint32_t UB_AL = M15G::UB_PL_AL;     // A_log fp32 [64] + slack       512B
constexpr uint32_t UB_DT = M15G::UB_PL_DT;     // dt_bias fp32 [64] + slack     512B
constexpr uint32_t UB_GS0 = M15G::UB_PL_GS0;   // g slot [48,8] fp32（parity0） 1536B
constexpr uint32_t UB_BS0 = M15G::UB_PL_BS0;   // β slot（parity0）            1536B
constexpr uint32_t UB_GS1 = M15G::UB_PL_GS1;   // g slot（parity1）            1536B
constexpr uint32_t UB_BS1 = M15G::UB_PL_BS1;   // β slot（parity1）            1536B
static_assert(UB_BS1 + 1536 <= 248 * 1024, "UB footprint exceeds 248KB");
static_assert(UB_BK0 % 32 == 0 && UB_BK1 % 32 == 0 && UB_AST % 32 == 0 && UB_AL % 32 == 0 &&
                  UB_DT % 32 == 0 && UB_GS0 % 32 == 0 && UB_BS0 % 32 == 0 && UB_GS1 % 32 == 0 &&
                  UB_BS1 % 32 == 0,
              "UB offsets must be 32B aligned");

// ---- BufferID（层内核内 m9 专属槽 BUF_PL_*；每核用户可用 0-27）----
constexpr AscendC::MutexID BUF_BK0 = M15G::BUF_PL_BK0;  // block 输入区 parity0：MTE2 → VEC → MTE3
constexpr AscendC::MutexID BUF_BK1 = M15G::BUF_PL_BK1;  // block 输入区 parity1
constexpr AscendC::MutexID BUF_GB = M15G::BUF_PL_GB;    // gating 阵列：MTE2 → VEC

template <pipe_t pipe>
__aicore__ inline void BufAcquire(AscendC::MutexID id)
{
    AscendC::GetBufInternal<pipe, false>(id);
}
template <pipe_t pipe>
__aicore__ inline void BufRelease(AscendC::MutexID id)
{
    AscendC::RlsBufInternal<pipe, false>(id);
}

// ============================================================
// 寄存器 VF 计算（`__simd_vf__` 函数）
//   合规判据落在**被调函数**上：函数必须标 `__simd_vf__`，且函数体内只用寄存器 API；
//   **调用点不要求词法 `__VEC_SCOPE__`**（`__VEC_SCOPE__` 是词法入口、`__simd_vf__` 是函数属性，
//   两者等价合规；本仓 0 处 `asc_vf_call`，CANN 上游同样不在 VEC_SCOPE 内调用）。
//   依据：tower M44 裁决 ①（inbox `20260926-tower-all-tower-m44-vec-scope-sort32-mrgsort-host.md`）——
//   本段三个 `__simd_vf__` 函数（PrologBlock / EgExpAll / GdnHeadRecurrence）的调用点均为裸调用。
// ============================================================

// conv1d(K=4) + bias + SiLU + 后处理，128 通道整块融合（__simd_vf__ 不可传 RegTensor&，
// 故 conv/l2norm/gating 合为一个函数、y0/y1 全程驻留寄存器）：
//   公共段：donor ComputeConv1dUnroll 同款增量 tap 累加（acc = bias; acc += w_j·win_j），
//     bf16 行 UNPACK 加载 + 单次全行 Cast 立即消费（m5 已验证模式，避开 M10 Cast bug）；
//     win = [st 行0, st 行1, st 行2, x]（st 行 j = 通道最旧第 j 个历史样本；w 行 j 同序）；
//     SiLU: y = acc/(1+exp(−acc))，fp32 写 outUb。
//   q 块（b<NQB）  : l2norm(eps) × 1/√128 覆写 outUb；
//   k 块（b<32）   : l2norm(eps) 覆写 outUb（不乘 scale——模型语义，见文件头）；
//   v 块           : outUb 即 v 输出；另做 gating（内联于下方）写 g/β slot。
// l2norm: y = x·(1/sqrt(Σx²+eps))；rcp 先算出再 broadcast，q scale 在 broadcast 前折进 rcp。
__simd_vf__ inline void PrologBlock(uint32_t b, __ubuf__ bfloat16_t* stUb, __ubuf__ bfloat16_t* xUb,
                                    __ubuf__ bfloat16_t* wUb, __ubuf__ bfloat16_t* bUb, __ubuf__ float* outUb,
                                    __ubuf__ bfloat16_t* aSt, __ubuf__ bfloat16_t* bSt, __ubuf__ float* alUb,
                                    __ubuf__ float* dtUb, __ubuf__ float* gSlot, __ubuf__ float* bSlot)
{
    RegTensor<bfloat16_t> tb, tw;
    RegTensor<float> s, w, acc, t, y0, y1;
    MaskReg fullM = CreateMask<float, MaskPattern::ALL>();

#define CONV_TAP(WROW, SROW)                                                                        \
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tw, wUb + (WROW) * HD + e);                    \
    Cast<float, bfloat16_t, castTraitB162B32Even>(w, tw, fullM);                                    \
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tb, (SROW) + e);                               \
    Cast<float, bfloat16_t, castTraitB162B32Even>(s, tb, fullM);                                    \
    MulAddDst(acc, w, s, fullM)

    for (uint32_t c = 0; c < HD / VL; ++c) {
        const uint32_t e = c * VL;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tb, bUb + e);
        Cast<float, bfloat16_t, castTraitB162B32Even>(acc, tb, fullM);  // acc = bias
        CONV_TAP(0, stUb);
        CONV_TAP(1, stUb + HD);
        CONV_TAP(2, stUb + 2 * HD);
        CONV_TAP(3, xUb);
        // SiLU: y = acc / (1 + exp(-acc))
        Muls(t, acc, -1.0f, fullM);
        Exp(t, t, fullM);
        Adds(t, t, 1.0f, fullM);
        if (c == 0) {
            Div(y0, acc, t, fullM);
            StoreAlign<float>(outUb, y0, fullM);
        } else {
            Div(y1, acc, t, fullM);
            StoreAlign<float>(outUb + VL, y1, fullM);
        }
    }
#undef CONV_TAP

    if (b >= QKB) {  // v 块：gating（outUb 已经是 v 输出）
        // 全 head 向量化 gating：staging 对齐整 64-lane 加载（前 48 lane 有效），
        // 数学全程 64 lane（≥48 的垃圾 lane 结果不被读取）。
        // 3510 quirk（M16 实测）：向量 LoadAlign 地址必须 32B 对齐——按 head 偏移
        // （aSt+h，2B/4B 粒度）加载直接 vector core exception(507035)，故改为
        // 对齐加载 + one-hot 掩码 Reduce 抽取目标 lane。
        const uint32_t h = b - QKB;
        RegTensor<bfloat16_t> tg;
        RegTensor<int32_t> idx;
        RegTensor<float> aF, bF, dtF, alF, x, u, gAll, beAll, one, gS, bS;
        MaskReg cmp, oh;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tg, aSt);
        Cast<float, bfloat16_t, castTraitB162B32Even>(aF, tg, fullM);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tg, bSt);
        Cast<float, bfloat16_t, castTraitB162B32Even>(bF, tg, fullM);
        LoadAlign<float>(dtF, dtUb);
        LoadAlign<float>(alF, alUb);
        Add(x, aF, dtF, fullM);
        Exp(u, x, fullM);
        Adds(u, u, 1.0f, fullM);
        Log(u, u, fullM);  // log1p(e^x)（x≤20 分支；>20 时由 Select 修正）
        Compares<float, CMPMODE::LE>(cmp, x, SPTH, fullM);
        Select<float>(u, u, x, cmp);  // softplus(x)：x≤20 → log1p(e^x)，否则 x
        Exp(gAll, alF, fullM);
        Muls(gAll, gAll, -1.0f, fullM);
        Mul(gAll, gAll, u, fullM);  // g[all] = −exp(A_log)·softplus(x)
        Muls(beAll, bF, -1.0f, fullM);
        Exp(beAll, beAll, fullM);
        Adds(beAll, beAll, 1.0f, fullM);
        Duplicate(one, 1.0f, fullM);
        Div(beAll, one, beAll, fullM);  // β[all] = sigmoid(b)
        // one-hot 抽取本 head 的 lane 到 lane0（idx==h 的 lane 参与 Reduce）
        Arange<int32_t>(idx, 0);
        Compares<int32_t, CMPMODE::EQ>(oh, idx, static_cast<int32_t>(h), fullM);
        Reduce<ReduceType::SUM>(gS, gAll, oh);
        Reduce<ReduceType::SUM>(bS, beAll, oh);
        // DIST_FIRST_ELEMENT_B32 单元素落槽：mask 必须 VL1（full mask 按 32B 块重复
        // 落 8 元素；CANN softmax_impl.h:1117 同款用法 MaskPattern::VL1）
        MaskReg oneM = CreateMask<float, MaskPattern::VL1>();
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(gSlot + h * 8, gS, oneM);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(bSlot + h * 8, bS, oneM);
        return;
    }
    const bool qScale = (b < NQB);  // 仅 q 乘 1/√128
    RegTensor<float> m, r, one, bcast;
    Mul(m, y0, y0, fullM);
    Mul(r, y1, y1, fullM);
    Add(m, m, r, fullM);
    ReduceSum(r, m, fullM);  // lane0 = Σx²
    Adds(r, r, EPS, fullM);
    Sqrt(r, r, fullM);
    Duplicate(one, 1.0f, fullM);
    Div(r, one, r, fullM);  // lane0 = 1/sqrt(Σx²+eps)
    if (qScale) {
        Muls(r, r, QSCALE, fullM);
    }
    Duplicate(bcast, r, fullM);
    Mul(y0, y0, bcast, fullM);
    Mul(y1, y1, bcast, fullM);
    StoreAlign<float>(outUb, y0, fullM);
    StoreAlign<float>(outUb + VL, y1, fullM);
}

// ============================================================
// Kernel 实现
// ============================================================

class GdnProlog {
public:
    __aicore__ inline GdnProlog() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR convState, GM_ADDR w, GM_ADDR bias, GM_ADDR aLog, GM_ADDR dtBias,
                                GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), INW);
        csGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(convState), CH * ST);
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w), KW * CH);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bias), CH);
        alGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aLog), 64);   // host 填充 48 有效 + 16 零
        dtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dtBias), 64);
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q), NQB * HD);
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k), NQB * HD);
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v), VH * HD);
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g), VH * 8);    // [48,8] stride-8，仅 [h][0]
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta), VH * 8);
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = GetBlockIdx();
        const uint32_t nblk = GetBlockNum() * 2;   // mix(1,2)：AIV 视角 GetBlockNum()=AIC 数，AIV 线程数 = 2×
        if (bid >= NBLK) {
            return;  // AIV 超配（NBLK=80 > blk 时不会发生；blk>80 时多余 AIV 早退）
        }

        // 条带划分的最后一个 block（静态 ascending striping）
        const uint32_t lastB = bid + ((NBLK - 1 - bid) / nblk) * nblk;
        const bool ownsV = lastB >= QKB;  // 本 AIV 是否拥有 v block（需 gating 阵列）

        if (ownsV) {
            CopyInGating();
        }

        CopyInBlock(bid, 0);
        uint32_t slot = 0;
        for (uint32_t b = bid; b < NBLK; b += nblk, ++slot) {
            const uint32_t p = slot & 1;
            const uint32_t bn = b + nblk;
            if (bn < NBLK) {
                CopyInBlock(bn, p ^ 1);  // MTE2 预取下一 block（与下方 VEC/MTE3 重叠）
            }
            ComputeBlock(b, p);
            CopyOutBlock(b, p);
        }
    }

private:
    // MTE2：gating 权重阵列（一次）：A_log/dt_bias fp32[64]（48 有效），a/b 取 x 的 bf16 段。
    // 注意：一律显式 DataCopyParams——元素计数 DataCopy 重载经 DataCopyCheck 只填 blockLen，
    // blockCount/srcGap/dstGap 是未初始化栈垃圾（M16 实测踩雷：复制块数随机翻倍）。
    __aicore__ inline void CopyInGating()
    {
        LocalTensor<bfloat16_t> aL(TPosition::VECCALC, UB_AST, 64);
        LocalTensor<bfloat16_t> bL(TPosition::VECCALC, UB_BST, 64);
        LocalTensor<float> alL(TPosition::VECCALC, UB_AL, 128);  // 512B：64 数据 + 预读 slack
        LocalTensor<float> dtL(TPosition::VECCALC, UB_DT, 128);
        const AscendC::DataCopyParams cp96{1, 3, 0, 0};   // 48 bf16 = 96B（VEC 预读由 slack 覆盖）
        const AscendC::DataCopyParams cp256{1, 8, 0, 0};  // 64 fp32 = 256B
        BufAcquire<PIPE_MTE2>(BUF_GB);
        DataCopy(aL, xGm_[XA_OFF], cp96);
        DataCopy(bL, xGm_[XB_OFF], cp96);
        DataCopy(alL, alGm_, cp256);
        DataCopy(dtL, dtGm_, cp256);
        BufRelease<PIPE_MTE2>(BUF_GB);
    }

    // MTE2：block b 的 conv 输入（state planar 3 行 × 256B + x/w/bias）
    __aicore__ inline void CopyInBlock(uint32_t b, uint32_t p)
    {
        const uint32_t base = p ? UB_BK1 : UB_BK0;
        const uint32_t c0 = b * HD;
        LocalTensor<bfloat16_t> xL(TPosition::VECCALC, base + X_OFF, HD);
        LocalTensor<bfloat16_t> bL(TPosition::VECCALC, base + B_OFF, HD);

        const AscendC::MutexID bk = p ? BUF_BK1 : BUF_BK0;
        // 3510 quirk（M16 实测）：DataCopyParams blockCount>1 会走 NZ 格式重排而非线性 strided
        // 拷贝，故一律 blockCount=1 单块拷贝；conv_state 为 planar [3][10240]（donor
        // causal_conv1d 的 (cache, state_len, dim) 布局），state 行 j = csGm_[j*CH + c0]。
        const AscendC::DataCopyParams cp256{1, 8, 0, 0};  // 256B 单块
        BufAcquire<PIPE_MTE2>(bk);
        for (uint32_t j = 0; j < ST; ++j) {
            LocalTensor<bfloat16_t> sjL(TPosition::VECCALC, base + ST_OFF + j * HD * sizeof(bfloat16_t), HD);
            DataCopy(sjL, csGm_[static_cast<uint64_t>(j) * CH + c0], cp256);  // state 行 j
        }
        DataCopy(xL, xGm_[c0], cp256);
        DataCopy(bL, bGm_[c0], cp256);
        for (uint32_t j = 0; j < KW; ++j) {
            LocalTensor<bfloat16_t> wjL(TPosition::VECCALC, base + W_OFF + j * HD * sizeof(bfloat16_t), HD);
            DataCopy(wjL, wGm_[static_cast<uint64_t>(j) * CH + c0], cp256);  // w [4][10240] 行主序
        }
        BufRelease<PIPE_MTE2>(bk);
    }

    // VEC：conv+SiLU → q/k l2norm 或 v 直写 + gating；BUF_GB 首 v block acquire、末 v block release
    __aicore__ inline void ComputeBlock(uint32_t b, uint32_t p)
    {
        const uint32_t base = p ? UB_BK1 : UB_BK0;
        const AscendC::MutexID bk = p ? BUF_BK1 : BUF_BK0;
        const uint32_t nblk = GetBlockNum() * 2;   // mix(1,2)：AIV 视角 GetBlockNum()=AIC 数，AIV 线程数 = 2×
        const uint32_t bid = GetBlockIdx();
        const bool isV = b >= QKB;
        const bool firstV = isV && ((b == bid) || (b - nblk < QKB));
        const bool lastV = isV && (b + nblk >= NBLK);

        BufAcquire<PIPE_V>(bk);
        if (firstV) {
            BufAcquire<PIPE_V>(BUF_GB);
        }
        __VEC_SCOPE__
        {
            __ubuf__ bfloat16_t* stUb = reinterpret_cast<__ubuf__ bfloat16_t*>(base + ST_OFF);
            __ubuf__ bfloat16_t* xUb = reinterpret_cast<__ubuf__ bfloat16_t*>(base + X_OFF);
            __ubuf__ bfloat16_t* wUb = reinterpret_cast<__ubuf__ bfloat16_t*>(base + W_OFF);
            __ubuf__ bfloat16_t* bUb = reinterpret_cast<__ubuf__ bfloat16_t*>(base + B_OFF);
            __ubuf__ float* outUb = reinterpret_cast<__ubuf__ float*>(base + O_OFF);
            __ubuf__ bfloat16_t* aSt = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_AST);
            __ubuf__ bfloat16_t* bSt = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_BST);
            __ubuf__ float* alUb = reinterpret_cast<__ubuf__ float*>(UB_AL);
            __ubuf__ float* dtUb = reinterpret_cast<__ubuf__ float*>(UB_DT);
            __ubuf__ float* gSlot = reinterpret_cast<__ubuf__ float*>(p ? UB_GS1 : UB_GS0);
            __ubuf__ float* bSlot = reinterpret_cast<__ubuf__ float*>(p ? UB_BS1 : UB_BS0);
            PrologBlock(b, stUb, xUb, wUb, bUb, outUb, aSt, bSt, alUb, dtUb, gSlot, bSlot);
        }
        if (lastV) {
            BufRelease<PIPE_V>(BUF_GB);
        }
        BufRelease<PIPE_V>(bk);
    }

    // MTE3：conv_state 原位写回（[st 行1, st 行2, x] = 本区 [256,1024) 连续 768B）+
    //       q/k/v fp32 写出 + g/β 32B slot 写出（v block）
    __aicore__ inline void CopyOutBlock(uint32_t b, uint32_t p)
    {
        const uint32_t base = p ? UB_BK1 : UB_BK0;
        const uint32_t c0 = b * HD;
        const AscendC::MutexID bk = p ? BUF_BK1 : BUF_BK0;

        LocalTensor<bfloat16_t> nbL(TPosition::VECCALC, base + ST_OFF + HD * sizeof(bfloat16_t), HD);
        LocalTensor<float> outL(TPosition::VECCALC, base + O_OFF, HD);

        const AscendC::DataCopyParams cp256{1, 8, 0, 0};                        // 256B 单块
        const AscendC::DataCopyParams cp512{1, HD * sizeof(float) / 32, 0, 0};  // 512B 单块
        const AscendC::DataCopyParams cpSlot{1, 1, 0, 0};                       // 32B slot
        BufAcquire<PIPE_MTE3>(bk);
        // planar [3][10240] 原位写回：new 行0<-st 行1(UB[256,512))，new 行1<-st 行2(UB[512,768))，
        // new 行2<-x(UB[768,1024))；均 256B 单块（blockCount=1 规避 NZ 重排）
        DataCopy(csGm_[c0], nbL, cp256);
        DataCopy(csGm_[static_cast<uint64_t>(CH) + c0],
                 LocalTensor<bfloat16_t>(TPosition::VECCALC, base + ST_OFF + 2 * HD * sizeof(bfloat16_t), HD), cp256);
        DataCopy(csGm_[static_cast<uint64_t>(2 * CH) + c0],
                 LocalTensor<bfloat16_t>(TPosition::VECCALC, base + X_OFF, HD), cp256);
        if (b < NQB) {
            DataCopy(qGm_[static_cast<uint64_t>(b) * HD], outL, cp512);
        } else if (b < QKB) {
            DataCopy(kGm_[static_cast<uint64_t>(b - NQB) * HD], outL, cp512);
        } else {
            const uint32_t h = b - QKB;
            DataCopy(vGm_[static_cast<uint64_t>(h) * HD], outL, cp512);
            LocalTensor<float> gL(TPosition::VECCALC, (p ? UB_GS1 : UB_GS0) + h * 8 * sizeof(float), 8);
            LocalTensor<float> beL(TPosition::VECCALC, (p ? UB_BS1 : UB_BS0) + h * 8 * sizeof(float), 8);
            DataCopy(gGm_[static_cast<uint64_t>(h) * 8], gL, cpSlot);  // 32B slot，仅 [h][0] 有效
            DataCopy(betaGm_[static_cast<uint64_t>(h) * 8], beL, cpSlot);
        }
        BufRelease<PIPE_MTE3>(bk);
    }

private:
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<bfloat16_t> csGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> bGm_;
    GlobalTensor<float> alGm_;
    GlobalTensor<float> dtGm_;
    GlobalTensor<float> qGm_;
    GlobalTensor<float> kGm_;
    GlobalTensor<float> vGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> betaGm_;
};

}  // namespace
}  // namespace Prolog

// ============================================================
// S4：GDN decode 递推（m4_gdn_recurrent 复制改造；纯 AIV，head 条带划分）
//   per value head hv（key head hk = hv/3）：S ← e^g·S；v ← β·(v − S·k)；
//   S ← S + k⊗v；o = S·q —— 寄存器 VF 单遍融合，slab（128×128 fp32）原位 RMW + 乒乓流水。
//   段内 BufferID：BUF_RC_*；UB 偏移：UB_RC_*（138KB = 2×64KB slab + q/k/v/o + g/eg/β）
// ============================================================

namespace Recur {
namespace {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t ROWS = M15G::RC_ROWS;   // V 维（state 行数 / o、v 长度）
constexpr uint32_t COLS = M15G::RC_COLS;   // K 维（state 列数 / k、q 长度）
constexpr uint32_t VLEN = 64;    // 每个 256B 向量寄存器的 fp32 lane 数
constexpr uint32_t GROUP = M15G::VGROUP;  // NV = 3×NK：value head hv 使用 key head hv/3（模型 48=3×16）
constexpr uint32_t MAX_HEADS = M15G::M_MAX;  // g/beta UB 缓冲按 64 head 静态分配（host 保证 H ≤ 64）
constexpr uint32_t ST_ELEMS = ROWS * COLS;
constexpr uint32_t ST_BYTES = ST_ELEMS * sizeof(float);

// ---- UB 静态布局（字节偏移；全部 32B 对齐）----
// 偏移一律取 M15G 全局资源表（m14_resources.h §4：SEG-GDN 窗内 m4 专属的 UB_RC_* 段）
constexpr uint32_t UB_ST0 = M15G::UB_RC_ST0;                  // state slab ping：128×128 fp32 = 64KB
constexpr uint32_t UB_ST1 = M15G::UB_RC_ST1;                  // state slab pong                  64KB
constexpr uint32_t UB_Q0 = M15G::UB_RC_Q0;                    // q [128] fp32 = 512B（本 parity 的 key head）
constexpr uint32_t UB_K0 = M15G::UB_RC_K0;                    // k [128]
constexpr uint32_t UB_V0 = M15G::UB_RC_V0;                    // v [128]
constexpr uint32_t UB_O0 = M15G::UB_RC_O0;                    // o [128]
constexpr uint32_t UB_Q1 = M15G::UB_RC_Q1;
constexpr uint32_t UB_K1 = M15G::UB_RC_K1;
constexpr uint32_t UB_V1 = M15G::UB_RC_V1;
constexpr uint32_t UB_O1 = M15G::UB_RC_O1;
constexpr uint32_t UB_G = M15G::UB_RC_G;                      // g  [64][8] fp32（stride-8，仅 [h][0] 有效）2KB
constexpr uint32_t UB_EG = M15G::UB_RC_EG;                    // e^g（VEC 内 Exp 一次成）                      2KB
constexpr uint32_t UB_BETA = M15G::UB_RC_BETA;                // β  [64][8] fp32                              2KB
static_assert(UB_BETA + MAX_HEADS * 32 <= 248 * 1024, "UB footprint exceeds 248KB");
static_assert(UB_ST0 % 32 == 0 && UB_ST1 % 32 == 0 && UB_Q0 % 32 == 0 && UB_G % 32 == 0 &&
                  UB_EG % 32 == 0 && UB_BETA % 32 == 0,
              "UB offsets must be 32B aligned");

// ---- BufferID（层内核内 m4 专属槽 BUF_RC_*；每核用户可用 0-27）----
//   全局获取顺序一致（ST → IV → OV），无循环等待：
//     BUF_STx : MTE2 装载 slab → VEC 递推 → MTE3 原位写回（token 随 slab 全生命周期）
//     BUF_IVx : MTE2 装载 q/k/v → VEC 消费
//     BUF_OVx : VEC 写出 o → MTE3 写出 GM
//     BUF_GB  : MTE2 装载 g/β → VEC Exp 成 e^g（kernel 开头一次）
constexpr AscendC::MutexID BUF_ST0 = M15G::BUF_RC_ST0;
constexpr AscendC::MutexID BUF_ST1 = M15G::BUF_RC_ST1;
constexpr AscendC::MutexID BUF_IV0 = M15G::BUF_RC_IV0;
constexpr AscendC::MutexID BUF_IV1 = M15G::BUF_RC_IV1;
constexpr AscendC::MutexID BUF_OV0 = M15G::BUF_RC_OV0;
constexpr AscendC::MutexID BUF_OV1 = M15G::BUF_RC_OV1;
constexpr AscendC::MutexID BUF_GB = M15G::BUF_RC_GB;

// BufferID 语义封装：acquire/release 一律 mode=false（CANN `ASC_LOCK_BLOCK` 默认「阻塞」模式；
// M181 起 release 与 acquire 同模式）。
template <pipe_t pipe>
__aicore__ inline void BufAcquire(AscendC::MutexID id)
{
    AscendC::GetBufInternal<pipe, false>(id);
}
template <pipe_t pipe>
__aicore__ inline void BufRelease(AscendC::MutexID id)
{
    AscendC::RlsBufInternal<pipe, false>(id);
}

// ============================================================
// 寄存器 VF 递推（照 donor arch35 融合写法的单遍版）
// ============================================================

// g -> e^g：[64][8] fp32（每 head 一个 32B 槽位，[h][0] 有效），512 float = 8 个 64-lane
// tile 各一次 Exp（超出 numHeads 的槽位为废值，不读）
__simd_vf__ inline void EgExpAll(__ubuf__ float* egUb, __ubuf__ float* gUb)
{
    AscendC::Reg::RegTensor<float> g;
    AscendC::Reg::MaskReg fullM = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    for (uint32_t t = 0; t < MAX_HEADS * 8 / VLEN; ++t) {
        AscendC::Reg::LoadAlign(g, gUb + t * VLEN);
        AscendC::Reg::Exp(g, g, fullM);
        AscendC::Reg::StoreAlign(egUb + t * VLEN, g, fullM);
    }
}

// 单 head 递推：st(UB 128×128 fp32) 原位读改写；o(UB 128) 写出。
// 每行 i 完全在寄存器内完成四步融合（状态行只 Load 一次、Store 一次）：
//   decay   : s = e^g * S[i,:]                     （Mul，egB 为 e^g 广播寄存器）
//   delta   : w = Σ_j s[j]·k[j]; d = β·(v[i] − w)  （Mul+Add+ReduceSum，DIST_BRC_B32 取 v[i]）
//   outer   : S[i,:] += k·d                        （MulAddDst，随即 StoreAlign 回 slab）
//   matvec  : o[i] = Σ_j S_new[i,j]·q[j]           （Mul+Add+ReduceSum，DIST_FIRST_ELEMENT_B32 存 lane0）
__simd_vf__ inline void GdnHeadRecurrence(__ubuf__ float* stUb, __ubuf__ float* qUb, __ubuf__ float* kUb,
                                          __ubuf__ float* vUb, __ubuf__ float* oUb, __ubuf__ float* egP,
                                          __ubuf__ float* betaP)
{
    AscendC::Reg::RegTensor<float> k0, k1, q0, q1, egB, betaB;
    AscendC::Reg::RegTensor<float> s0, s1, t, u, r, wB, vB, dB;
    AscendC::Reg::MaskReg fullM = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::LoadAlign(k0, kUb);
    AscendC::Reg::LoadAlign(k1, kUb + VLEN);
    AscendC::Reg::LoadAlign(q0, qUb);
    AscendC::Reg::LoadAlign(q1, qUb + VLEN);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(egB, egP);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(betaB, betaP);

    for (uint32_t i = 0; i < ROWS; ++i) {
        __ubuf__ float* row = stUb + i * COLS;
        AscendC::Reg::LoadAlign(s0, row);
        AscendC::Reg::LoadAlign(s1, row + VLEN);
        // decay: S ← e^g·S
        AscendC::Reg::Mul(s0, s0, egB, fullM);
        AscendC::Reg::Mul(s1, s1, egB, fullM);
        // delta: w = (e^g·S)·k；d = β·(v[i] − w)
        AscendC::Reg::Mul(t, s0, k0, fullM);
        AscendC::Reg::Mul(u, s1, k1, fullM);
        AscendC::Reg::Add(t, t, u, fullM);
        AscendC::Reg::ReduceSum(r, t, fullM);   // lane0 = w
        AscendC::Reg::Duplicate(wB, r, fullM);  // 广播 w 到全 lane
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vB, vUb + i);
        AscendC::Reg::Sub(dB, vB, wB, fullM);
        AscendC::Reg::Mul(dB, dB, betaB, fullM);  // 全 lane = delta[i]
        // outer: S ← S + k⊗d（随即写回 slab）
        AscendC::Reg::MulAddDst(s0, k0, dB, fullM);
        AscendC::Reg::MulAddDst(s1, k1, dB, fullM);
        AscendC::Reg::StoreAlign(row, s0, fullM);
        AscendC::Reg::StoreAlign(row + VLEN, s1, fullM);
        // matvec: o[i] = S_new·q
        AscendC::Reg::Mul(t, s0, q0, fullM);
        AscendC::Reg::Mul(u, s1, q1, fullM);
        AscendC::Reg::Add(t, t, u, fullM);
        AscendC::Reg::ReduceSum(r, t, fullM);  // lane0 = o[i]
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(oUb + i, r, fullM);
    }
}

// ============================================================
// Kernel 实现
// ============================================================

class GdnDecodeRec {
public:
    __aicore__ inline GdnDecodeRec() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR state,
                                GM_ADDR out, uint32_t numHeads)
    {
        numHeads_ = numHeads;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nblk = AscendC::GetBlockNum() * 2;   // mix(1,2)：AIV 视角 GetBlockNum()=AIC 数，AIV 线程数 = 2×

        // ---- (1) g/β 一次装载 + e^g 预算（布局 [H,8] fp32，每 head 占 32B，仅 [h][0] 有效）----
        LocalTensor<float> gL(TPosition::VECCALC, UB_G, MAX_HEADS * 8);
        LocalTensor<float> betaL(TPosition::VECCALC, UB_BETA, MAX_HEADS * 8);
        LocalTensor<float> egL(TPosition::VECCALC, UB_EG, MAX_HEADS * 8);
        __ubuf__ float* gUb = reinterpret_cast<__ubuf__ float*>(gL.GetPhyAddr());
        __ubuf__ float* egUb = reinterpret_cast<__ubuf__ float*>(egL.GetPhyAddr());
        __ubuf__ float* betaUb = reinterpret_cast<__ubuf__ float*>(betaL.GetPhyAddr());
        {
            const AscendC::DataCopyParams gp{static_cast<uint16_t>(numHeads_), 1, 0, 0};  // H 块 × 32B
            BufAcquire<PIPE_MTE2>(BUF_GB);
            DataCopy(gL, gGm_, gp);
            DataCopy(betaL, betaGm_, gp);
            BufRelease<PIPE_MTE2>(BUF_GB);
            BufAcquire<PIPE_V>(BUF_GB);
            EgExpAll(egUb, gUb);
            BufRelease<PIPE_V>(BUF_GB);
        }

        // ---- (2) head 软件流水：prefetch(h+nblk, parity 1-p) ∥ compute(h, p) ∥ writeback(h, p) ----
        if (bid < numHeads_) {
            CopyInHead(bid, 0);
        }
        uint32_t slot = 0;
        for (uint32_t h = bid; h < numHeads_; h += nblk, ++slot) {
            const uint32_t p = slot & 1;
            const uint32_t hn = h + nblk;
            if (hn < numHeads_) {
                CopyInHead(hn, p ^ 1);  // MTE2 预取下一 head（与下方 VEC/MTE3 重叠）
            }
            ComputeHead(h, p, egUb, betaUb);
            CopyOutHead(h, p);
        }
    }

private:
    // MTE2：state slab（128 行 × 512B 连续）+ q/k/v（key head hk = h/3 的 q/k）
    __aicore__ inline void CopyInHead(uint32_t h, uint32_t p)
    {
        const uint32_t hk = h / GROUP;
        LocalTensor<float> stL(TPosition::VECCALC, p ? UB_ST1 : UB_ST0, ST_ELEMS);
        LocalTensor<float> qL(TPosition::VECCALC, p ? UB_Q1 : UB_Q0, COLS);
        LocalTensor<float> kL(TPosition::VECCALC, p ? UB_K1 : UB_K0, COLS);
        LocalTensor<float> vL(TPosition::VECCALC, p ? UB_V1 : UB_V0, ROWS);

        const AscendC::MutexID bufSt = p ? BUF_ST1 : BUF_ST0;
        const AscendC::MutexID bufIv = p ? BUF_IV1 : BUF_IV0;
        BufAcquire<PIPE_MTE2>(bufSt);
        const AscendC::DataCopyParams sp{ROWS, (COLS * static_cast<uint32_t>(sizeof(float))) / 32, 0, 0};
        DataCopy(stL, stateGm_[static_cast<uint64_t>(h) * ST_ELEMS], sp);
        BufRelease<PIPE_MTE2>(bufSt);

        BufAcquire<PIPE_MTE2>(bufIv);
        DataCopy(qL, qGm_[static_cast<uint64_t>(hk) * COLS], COLS);
        DataCopy(kL, kGm_[static_cast<uint64_t>(hk) * COLS], COLS);
        DataCopy(vL, vGm_[static_cast<uint64_t>(h) * ROWS], ROWS);
        BufRelease<PIPE_MTE2>(bufIv);
    }

    // VEC：单遍融合递推（slab 原位读改写，o 写 UB）
    __aicore__ inline void ComputeHead(uint32_t h, uint32_t p, __ubuf__ float* egUb, __ubuf__ float* betaUb)
    {
        LocalTensor<float> stL(TPosition::VECCALC, p ? UB_ST1 : UB_ST0, ST_ELEMS);
        LocalTensor<float> qL(TPosition::VECCALC, p ? UB_Q1 : UB_Q0, COLS);
        LocalTensor<float> kL(TPosition::VECCALC, p ? UB_K1 : UB_K0, COLS);
        LocalTensor<float> vL(TPosition::VECCALC, p ? UB_V1 : UB_V0, ROWS);
        LocalTensor<float> oL(TPosition::VECCALC, p ? UB_O1 : UB_O0, ROWS);
        __ubuf__ float* stUb = reinterpret_cast<__ubuf__ float*>(stL.GetPhyAddr());
        __ubuf__ float* qUb = reinterpret_cast<__ubuf__ float*>(qL.GetPhyAddr());
        __ubuf__ float* kUb = reinterpret_cast<__ubuf__ float*>(kL.GetPhyAddr());
        __ubuf__ float* vUb = reinterpret_cast<__ubuf__ float*>(vL.GetPhyAddr());
        __ubuf__ float* oUb = reinterpret_cast<__ubuf__ float*>(oL.GetPhyAddr());

        const AscendC::MutexID bufSt = p ? BUF_ST1 : BUF_ST0;
        const AscendC::MutexID bufIv = p ? BUF_IV1 : BUF_IV0;
        const AscendC::MutexID bufOv = p ? BUF_OV1 : BUF_OV0;
        BufAcquire<PIPE_V>(bufSt);   // 等 MTE2 slab 就位
        BufAcquire<PIPE_V>(bufIv);   // 等 MTE2 q/k/v 就位
        BufAcquire<PIPE_V>(bufOv);   // 等上一轮回 MTE3 读走 o
        // eg/β 为 [H][8] fp32 槽位（32B/head，仅 [h][0] 有效），BRC 按槽位首元素广播
        GdnHeadRecurrence(stUb, qUb, kUb, vUb, oUb, egUb + h * 8, betaUb + h * 8);
        BufRelease<PIPE_V>(bufOv);   // o 写完（release mode=false），MTE3 可写 GM
        BufRelease<PIPE_V>(bufIv);   // q/k/v 已消费，MTE2 可复用
        BufRelease<PIPE_V>(bufSt);   // slab 递推完（release mode=false），MTE3 可原位写回
    }

    // MTE3：state 原位写回 + o 写出
    __aicore__ inline void CopyOutHead(uint32_t h, uint32_t p)
    {
        LocalTensor<float> stL(TPosition::VECCALC, p ? UB_ST1 : UB_ST0, ST_ELEMS);
        LocalTensor<float> oL(TPosition::VECCALC, p ? UB_O1 : UB_O0, ROWS);

        const AscendC::MutexID bufSt = p ? BUF_ST1 : BUF_ST0;
        const AscendC::MutexID bufOv = p ? BUF_OV1 : BUF_OV0;
        BufAcquire<PIPE_MTE3>(bufSt);
        const AscendC::DataCopyParams sp{ROWS, (COLS * static_cast<uint32_t>(sizeof(float))) / 32, 0, 0};
        DataCopy(stateGm_[static_cast<uint64_t>(h) * ST_ELEMS], stL, sp);
        BufRelease<PIPE_MTE3>(bufSt);

        BufAcquire<PIPE_MTE3>(bufOv);
        DataCopy(outGm_[static_cast<uint64_t>(h) * ROWS], oL, ROWS);
        BufRelease<PIPE_MTE3>(bufOv);
    }

private:
    uint32_t numHeads_;
    GlobalTensor<float> qGm_;
    GlobalTensor<float> kGm_;
    GlobalTensor<float> vGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> betaGm_;
    GlobalTensor<float> stateGm_;
    GlobalTensor<float> outGm_;
};

}  // namespace
}  // namespace Recur

// ============================================================
// S5：RMSNormGated（m12_rmsnorm_gated 复制改造：**head 切分**版）
//
//   per head h（48 组 × 128 维，vllm variance dim=-1）：
//     var  = mean(o[h]²)（fp32 累加，donor ② 二分 fold）
//     rstd = 1/sqrt(var + 1e-6)（donor ③ NR rsqrt；NR 与精确 rsqrt 差 ~1 ulp fp32）
//     out  = bf16( ((o[h]·rstd)·gamma) · sigmoid(z[h]) )（RNE；乘法次序 = vllm norm_before_gate）
//   sigmoid(z) = 1/(1+exp(−z)) fp32（m5 实证排布：Muls(−1)/Exp/Adds(+1)/Div）。
//
// donor 的 m12 核是「单核 + 整行 6144」；层内核改为**每 AIV 一个 head**（48 head ≤ 56 AIV）：
//   per-head 输入输出仍是 512B(o)/256B(z/out) 的 32B 整数倍块，DataCopy 参数显式（小改 A）。
//   gamma（[128] bf16，48 head 共享）在 kernel 开头预转 fp32 到 UB_GAMMA_G_F32（每核一份）。
// ============================================================

namespace Gated {

using namespace AscendC;
using namespace AscendC::Reg;

// ---- 常量（与 m12 donor 同名同值；偏移取 M15G 资源表 SEG-GDN 窗内 m12 段）----
constexpr uint32_t HEADS_ = M15G::HEADS;                 // 48 value head
constexpr uint32_t HEAD_ = M15G::HEAD_D;                 // 128（vllm head_v_dim）
constexpr uint16_t CHUNK_PER_HEAD_ = HEAD_ / 64;        // 每 head 64-lane chunk 数 = 2（无尾块）
constexpr float EPS_ = 1e-6f;                           // config.rms_norm_eps
constexpr float AVG_FACTOR_ = 1.0f / static_cast<float>(M15G::HEAD_D);
constexpr uint32_t FOLD_POINT_ = M15G::HEAD_D / 2;
constexpr uint32_t TMP_STRIDE_ = 24;

constexpr uint32_t UB_O_ = M15G::UB_NG_O;                // fp32[128] 512B
constexpr uint32_t UB_Z_ = M15G::UB_NG_Z;                // bf16[128] 256B
constexpr uint32_t UB_Y_ = M15G::UB_NG_Y;                // bf16[128] 256B
constexpr uint32_t UB_TMP_ = M15G::UB_NG_TMP;            // fp32[64]
constexpr uint32_t UB_RED_ = M15G::UB_NG_RED;            // fp32[64]
constexpr uint32_t UB_RSTD_ = M15G::UB_NG_RSTD;          // fp32[64]

constexpr AscendC::MutexID BUF_X_ = M15G::BUF_NG_X;      // o/z head: MTE2 → V
constexpr AscendC::MutexID BUF_OUT_ = M15G::BUF_NG_OUT;  // out head: V → MTE3

class RmsNormGatedStage {
public:
    __aicore__ inline RmsNormGatedStage() {}

    // o: [HEADS, HEAD] fp32（m4 出口）；z: qkvzba 的 z 段（bf16，按 head 取 128 个）；
    // out: [HEADS, HEAD] bf16（out_proj 的 A 行）；gammaF32UbOff: 预转后的 fp32 gamma
    __aicore__ inline void Init(__gm__ uint8_t* o, __gm__ uint8_t* z, __gm__ uint8_t* out, uint32_t gammaF32UbOff)
    {
        gammaOff = gammaF32UbOff;
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(o), static_cast<uint64_t>(HEADS_) * HEAD_);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(z), static_cast<uint64_t>(M15G::M_MAX) * M15G::IN_N);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(M15G::M_MAX) * M15G::V_DIM);
    }

    // head h = bid + i*nAiv（head 间无依赖；48 head ≤ AIV 数时每核一个）
    __aicore__ inline void Run(uint32_t bid, uint32_t nAiv)
    {
        for (uint32_t h = bid; h < HEADS_; h += nAiv) {
            CopyInHead(h);
            ComputeHead(h);
            CopyOutHead(h);
        }
    }

private:
    __aicore__ inline void CopyInHead(uint32_t h)
    {
        LocalTensor<float> oL(TPosition::VECCALC, UB_O_, HEAD_);
        LocalTensor<bfloat16_t> zL(TPosition::VECCALC, UB_Z_, HEAD_);
        const uint64_t off = static_cast<uint64_t>(h) * HEAD_;
        // 显式 DataCopyParams（小改 A）：o head 512B = 16×32B；z head 256B = 8×32B
        // z 取自 qkvzba 的 z 段（元素偏移 M15G::Z_OFF + h*128）
        BufAcquire<PIPE_MTE2>(BUF_X_);   // 挡上一 head 的 V 消费
        PipeBarrier<PIPE_MTE2>();        // 小改 B：head 行循环背靠背复用同一 UB 行缓冲（同 pipe 不保序）
        DataCopy(oL, oGm[off], Block1(HEAD_ * sizeof(float)));
        DataCopy(zL, zGm[M15G::Z_OFF + off], Block1(HEAD_ * sizeof(bfloat16_t)));
        BufRelease<PIPE_MTE2>(BUF_X_);
    }

    // 门控输出（m12 donor CalculateGateY 逐字保留：乘法次序/sigmoid 四元组/RNE Cast）
    __aicore__ inline void CalculateGateY(__ubuf__ float* oUb, __ubuf__ bfloat16_t* zUb, __ubuf__ float* gfUb,
                                          __ubuf__ bfloat16_t* yUb, __ubuf__ float* rstdUb)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> oReg;
            RegTensor<float> gReg;
            RegTensor<float> rstdReg;
            RegTensor<float> zReg;
            RegTensor<float> vReg;
            RegTensor<float> tReg;
            RegTensor<float> yReg;
            RegTensor<float> oneReg;
            RegTensor<bfloat16_t> zB16;
            RegTensor<bfloat16_t> yB16;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            Duplicate(oneReg, 1.0f, maskAll);
            // rstd 恒在 rstdUb[0]（per-head 流水），DIST_BRC_B32 广播到全 lane
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdUb);
            for (uint16_t c = 0; c < CHUNK_PER_HEAD_; ++c) {
                uint32_t offset = c * 64;
                LoadAlign<float>(oReg, oUb + offset);
                LoadAlign<float>(gReg, gfUb + offset);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(zB16, zUb + offset);
                Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(zReg, zB16, maskAll);
                Muls(vReg, zReg, -1.0f, maskAll);
                Exp(vReg, vReg, maskAll);
                Adds(vReg, vReg, 1.0f, maskAll);
                Div(vReg, oneReg, vReg, maskAll);  // sigmoid(z)
                Mul(tReg, oReg, rstdReg, maskAll);
                Mul(tReg, tReg, gReg, maskAll);
                Mul(yReg, tReg, vReg, maskAll);
                Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yB16, yReg, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(yUb + offset, yB16, maskAll);
            }
        }
    }

    __aicore__ inline void ComputeHead(uint32_t)
    {
        LocalTensor<float> oL(TPosition::VECCALC, UB_O_, HEAD_);
        LocalTensor<bfloat16_t> zL(TPosition::VECCALC, UB_Z_, HEAD_);
        LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_Y_, HEAD_);
        LocalTensor<float> tmpL(TPosition::VECCALC, UB_TMP_, 64);
        LocalTensor<float> redL(TPosition::VECCALC, UB_RED_, 64);
        LocalTensor<float> rstdL(TPosition::VECCALC, UB_RSTD_, 64);
        __ubuf__ float* oUb = reinterpret_cast<__ubuf__ float*>(oL.GetPhyAddr());
        __ubuf__ bfloat16_t* zUb = reinterpret_cast<__ubuf__ bfloat16_t*>(zL.GetPhyAddr());
        __ubuf__ bfloat16_t* yUb = reinterpret_cast<__ubuf__ bfloat16_t*>(yL.GetPhyAddr());
        __ubuf__ float* gfUb = reinterpret_cast<__ubuf__ float*>(gammaOff);
        __ubuf__ float* tmpUb = reinterpret_cast<__ubuf__ float*>(tmpL.GetPhyAddr());
        __ubuf__ float* redUb = reinterpret_cast<__ubuf__ float*>(redL.GetPhyAddr());
        __ubuf__ float* rstdUb = reinterpret_cast<__ubuf__ float*>(rstdL.GetPhyAddr());

        BufAcquire<PIPE_V>(BUF_X_);    // 等 o/z head 就位
        BufAcquire<PIPE_V>(BUF_OUT_);  // 等上一 head MTE3 读完 out 缓冲
        // 128 维 = 2×64 lane → donor dispatch 走 LessThanTwoVL 分支（与 m12 donor 一致，
        // 无 tmp 往返 / 无 LocalMemBar）；mean 因子 1/128 走 NR 的 avgFactor
        NormDonor::CalculateSquareReduceSum<float>(oUb, redUb, tmpUb, 1, HEAD_, HEAD_, FOLD_POINT_, TMP_STRIDE_);
        NormDonor::ComputeRstdNewtonRaphson<true, true>(redUb, rstdUb, 1, EPS_, AVG_FACTOR_, 64);
        CalculateGateY(oUb, zUb, gfUb, yUb, rstdUb);
        BufRelease<PIPE_V>(BUF_X_);    // o/z 消费完，MTE2 可复用
        BufRelease<PIPE_V>(BUF_OUT_);  // out 写完（release mode=false），MTE3 可读
    }

    __aicore__ inline void CopyOutHead(uint32_t h)
    {
        LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_Y_, HEAD_);
        const uint64_t off = static_cast<uint64_t>(h) * HEAD_;
        BufAcquire<PIPE_MTE3>(BUF_OUT_);
        DataCopy(outGm[off], yL, Block1(HEAD_ * sizeof(bfloat16_t)));   // 256B = 8×32B
        BufRelease<PIPE_MTE3>(BUF_OUT_);
    }

private:
    uint32_t gammaOff = 0;
    AscendC::GlobalTensor<float> oGm;
    AscendC::GlobalTensor<bfloat16_t> zGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;
};

}  // namespace Gated
// ============================================================
// S2/S6：bf16 GEMM（m11_bf16_gemm 复制改造；单 AIC，N 方向条带划分）
//   流水：MTE2 Nd2Nz(GM→L1) ∥ MTE1 LoadData2D(L1→L0) ∥ M(Mmad) ∥ FIXP(F322BF16 → GM)
//   层内核里 Process() 改为 RunTile(nBlock)：mLoop==1（m ≤ 64），每个 AIC 只跑自己的 N tile。
// ============================================================

namespace Cube {
__aicore__ __inline__ constexpr uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ __inline__ constexpr uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
__aicore__ __inline__ constexpr uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ---- tile 与静态资源（一律取 M15G 全局资源表 m14_resources.h §1，单一来源）----
constexpr uint32_t CUBE_BLOCK = M15G::CUBE_BLOCK;  // cube 分形边长（bf16: 16 元素 = 32B）
constexpr uint32_t BASE_M = M15G::BASE_M;          // M 方向 tile；m ∈ [1,64] 单 tile，尾块用 curM mask
constexpr uint32_t BASE_K = M15G::BASE_K;          // K 方向 tile（两档 K 均整除）
constexpr uint32_t BASE_N = M15G::BASE_N;          // N 方向 tile（gcd(16480,2560) = 160）
// BASE_K/BASE_N 取值依据：L0B ping-pong 各 32KB（bf16 2B/元素）→ BASE_K×BASE_N ≤ 16384 元素；
// BASE_N 须为 16 的倍数且同时整除 in_proj N=16480 与 out_proj N=2560（gcd=160）→ 取最大 160。

// ---- L1 静态布局（字节偏移；A 区 [0,256KB)，B 区 [256KB,512KB)，与 m1/donor 一致）----
constexpr uint32_t L1_A_ELEMS = M15G::L1_A_ELEMS;  // 一个 A 大包：64×64 = 4096 元素 = 8KB
constexpr uint32_t L1_B_ELEMS = M15G::L1_B_ELEMS;  // 一个 B 大包：160×64 = 10240 元素 = 20KB
constexpr uint32_t L1_OFF_A0 = M15G::L1_OFF_A0;
constexpr uint32_t L1_OFF_A1 = M15G::L1_OFF_A1;
constexpr uint32_t L1_B_REGION = M15G::L1_B_REGION;
constexpr uint32_t L1_OFF_B0 = M15G::L1_OFF_B0;
constexpr uint32_t L1_OFF_B1 = M15G::L1_OFF_B1;
static_assert(L1_B_REGION + 2 * L1_B_ELEMS * 2 <= 512 * 1024, "A/B ping-pong L1 footprint exceeds 512KB");

// ---- L0 静态布局（字节偏移；L0A/L0B 各 64KB，ping/pong 各 32KB）----
constexpr uint32_t L0_PP_BYTES = M15G::L0_PP_BYTES;
constexpr uint32_t L0_OFF_0 = M15G::L0_OFF_0;
constexpr uint32_t L0_OFF_1 = M15G::L0_OFF_1;
static_assert(BASE_M * BASE_K * 2 <= L0_PP_BYTES, "A2 tile exceeds L0A half");  // bf16: 64×64 = 8KB
static_assert(BASE_K * BASE_N * 2 <= L0_PP_BYTES, "B2 tile exceeds L0B half");  // bf16: 64×160 = 20KB
// L0C（CO1）：64×160 fp32 = 40KB（≤ 256KB），偏移 0

// ---- BufferID（MutexID 有效范围 0..27，静态分配；单 AIC 核内同步；编号与 m1/m3 一致）----
constexpr AscendC::MutexID BUF_A0 = M15G::BUF_AIC_A0;     // A1 ping: MTE2 -> MTE1
constexpr AscendC::MutexID BUF_A1 = M15G::BUF_AIC_A1;     // A1 pong
constexpr AscendC::MutexID BUF_B0 = M15G::BUF_AIC_B0;     // B1 ping: MTE2 -> MTE1
constexpr AscendC::MutexID BUF_B1 = M15G::BUF_AIC_B1;     // B1 pong
constexpr AscendC::MutexID BUF_L0_0 = M15G::BUF_AIC_L00;  // L0A/L0B ping: MTE1 -> M
constexpr AscendC::MutexID BUF_L0_1 = M15G::BUF_AIC_L01;  // L0A/L0B pong
constexpr AscendC::MutexID BUF_L0C = M15G::BUF_AIC_L0C;   // L0C: M -> FIXP

// ============================================================
// Kernel 实现
// ============================================================

template <uint32_t K, uint32_t N>
class Bf16Gemm {
    static_assert(K % BASE_K == 0, "K must be divisible by BASE_K");
    static_assert(N % BASE_N == 0, "N must be divisible by BASE_N");

public:
    __aicore__ inline Bf16Gemm() {}

    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, uint32_t m)
    {
        aGMOri.SetGlobalBuffer((__gm__ bfloat16_t*)a);
        bGMOri.SetGlobalBuffer((__gm__ bfloat16_t*)b);
        cGMOri.SetGlobalBuffer((__gm__ bfloat16_t*)c);
        mTotal = m;
    }

    // 层内核（mission M22）单 tile 版：mLoop 必须为 1（m ≤ BASE_M = 64），N 方向由调用者给定
    // 一个 nBlock（各 AIC 按 item = bid, bid+numBlocks, ... 条带划分 N tile，与 m13 MoE 段同款）。
    __aicore__ inline void RunTile(uint32_t nBlock)
    {
        const uint32_t mLoop = CeilDiv(mTotal, BASE_M);  // m ≤ 64 时为 1
        const uint32_t kLoop = K / BASE_K;

        // L1 / L0 / L0C 静态 tensor（position + 字节偏移 + 元素数，自管理地址；
        // 注意 LocalTensor(position, addr, size) 的 addr 单位是字节、size 单位是元素）
        AscendC::LocalTensor<bfloat16_t> a1Ping(AscendC::TPosition::A1, L1_OFF_A0, L1_A_ELEMS);
        AscendC::LocalTensor<bfloat16_t> a1Pong(AscendC::TPosition::A1, L1_OFF_A1, L1_A_ELEMS);
        AscendC::LocalTensor<bfloat16_t> b1Ping(AscendC::TPosition::B1, L1_OFF_B0, L1_B_ELEMS);
        AscendC::LocalTensor<bfloat16_t> b1Pong(AscendC::TPosition::B1, L1_OFF_B1, L1_B_ELEMS);

        AscendC::LocalTensor<bfloat16_t> a2Ping(AscendC::TPosition::A2, L0_OFF_0, BASE_M * BASE_K);
        AscendC::LocalTensor<bfloat16_t> a2Pong(AscendC::TPosition::A2, L0_OFF_1, BASE_M * BASE_K);
        AscendC::LocalTensor<bfloat16_t> b2Ping(AscendC::TPosition::B2, L0_OFF_0, BASE_K * BASE_N);
        AscendC::LocalTensor<bfloat16_t> b2Pong(AscendC::TPosition::B2, L0_OFF_1, BASE_K * BASE_N);

        AscendC::LocalTensor<float> cL0C(AscendC::TPosition::CO1, 0, BASE_M * BASE_N);

        for (uint32_t mBlock = 0; mBlock < mLoop; ++mBlock) {
            const uint32_t curM = MinU32(mTotal - mBlock * BASE_M, BASE_M);  // 尾块 mask（动态 m 的核心处理）
            // 3510 实测（m1 契约）：Nd2Nz 行数为 1 时不做 NZ 切分（退化为 1D 拷贝），导致 m=1 数据错位。
            // 计算侧统一提升为 >=2 行（多读的 1 行由 host 保证可读，结果行不写出），仅在 Fixpipe 用 curM mask。
            const uint32_t calcM = curM < 2 ? 2 : curM;
            const uint32_t calcMAlign = AlignUp(calcM, CUBE_BLOCK);
            {   // 单个 N tile（nBlock 为入参；层内核不在此处循环 N）
                // M 先取 L0C 所有权：挡住上一 tile 的 Fixpipe 读、允许本 tile 累加
                AscendC::Mutex::Lock<PIPE_M>(BUF_L0C);
                for (uint32_t kBlock = 0; kBlock < kLoop; ++kBlock) {
                    const uint32_t p = kBlock & 1;
                    const AscendC::MutexID bufA = p ? BUF_A1 : BUF_A0;
                    const AscendC::MutexID bufB = p ? BUF_B1 : BUF_B0;
                    const AscendC::MutexID bufL0 = p ? BUF_L0_1 : BUF_L0_0;
                    AscendC::LocalTensor<bfloat16_t> a1 = p ? a1Pong : a1Ping;
                    AscendC::LocalTensor<bfloat16_t> b1 = p ? b1Pong : b1Ping;
                    AscendC::LocalTensor<bfloat16_t> a2 = p ? a2Pong : a2Ping;
                    AscendC::LocalTensor<bfloat16_t> b2 = p ? b2Pong : b2Ping;

                    // ---- MTE2：GM -> L1（Nd2Nz 数据），大包生产 ----
                    AscendC::Mutex::Lock<PIPE_MTE2>(bufA);
                    CopyInA(a1, kBlock, mBlock, calcM);
                    AscendC::Mutex::Unlock<PIPE_MTE2>(bufA);
                    AscendC::Mutex::Lock<PIPE_MTE2>(bufB);
                    CopyInB(b1, kBlock, nBlock);
                    AscendC::Mutex::Unlock<PIPE_MTE2>(bufB);

                    // ---- MTE1：L1 -> L0（LoadData 非 MX），消费 L1、生产 L0 ----
                    AscendC::Mutex::Lock<PIPE_MTE1>(bufL0);   // 等 M 释放 L0 缓冲
                    AscendC::Mutex::Lock<PIPE_MTE1>(bufA);    // 等 MTE2 数据就绪
                    AscendC::Mutex::Lock<PIPE_MTE1>(bufB);
                    LoadA(a1, a2, calcMAlign);
                    LoadB(b1, b2);
                    AscendC::Mutex::Unlock<PIPE_MTE1>(bufA);  // L1 可复用
                    AscendC::Mutex::Unlock<PIPE_MTE1>(bufB);
                    AscendC::Mutex::Unlock<PIPE_MTE1>(bufL0); // L0 数据就绪

                    // ---- M：Mmad 累加（L0C fp32），消费 L0 ----
                    AscendC::Mutex::Lock<PIPE_M>(bufL0);
                    AscendC::MmadParams mmadParams = {};    // 全字段 NSDMI 零初始化（M16 加固）
                    mmadParams.m = calcM;
                    mmadParams.n = BASE_N;
                    mmadParams.k = BASE_K;
                    mmadParams.cmatrixInitVal = (kBlock == 0);
                    AscendC::Mmad(cL0C, a2, b2, mmadParams);
                    AscendC::Mutex::Unlock<PIPE_M>(bufL0);
                }
                // ---- FIXP：L0C -> GM（F322BF16）----
                AscendC::Mutex::Unlock<PIPE_M>(BUF_L0C);      // L0C 累加结果就绪
                AscendC::Mutex::Lock<PIPE_FIX>(BUF_L0C);
                CopyOut(cL0C, mBlock, nBlock, curM, calcMAlign);
                AscendC::Mutex::Unlock<PIPE_FIX>(BUF_L0C);
            }
        }
    }

private:
    // GM -> L1：A 的一个 baseK 大包（calcM 行 × BASE_K bf16），Nd2Nz 落 L1（NZ 布局按满 tile 行距）
    __aicore__ inline void CopyInA(
        AscendC::LocalTensor<bfloat16_t>& a1, uint32_t kBlock, uint32_t mBlock, uint32_t curM)
    {
        AscendC::Nd2NzParams par = {};  // 全字段 NSDMI 零初始化后逐一赋值（M16 加固）
        par.ndNum = 1;
        par.nValue = curM;          // 行数（调用侧传 calcM≥2：3510 行数=1 退化 quirk，契约同 m1）
        par.dValue = BASE_K;        // 每行元素数（bf16 按元素计）
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;
        par.dstNzC0Stride = BASE_M; // L1 NZ 布局按满 tile 行距（与 LoadA 的 srcStride 配套）
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1, aGMOri[kBlock * BASE_K + mBlock * K * BASE_M], par);
    }

    // GM -> L1：B 按模型权重原始 layout [N, K] 存放，本次搬 baseN 行 × BASE_K bf16
    __aicore__ inline void CopyInB(AscendC::LocalTensor<bfloat16_t>& b1, uint32_t kBlock, uint32_t nBlock)
    {
        AscendC::Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = BASE_N;        // N 编译期整除 BASE_N，无尾块
        par.dValue = BASE_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;
        par.dstNzC0Stride = BASE_N;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1, bGMOri[kBlock * BASE_K + nBlock * BASE_N * K], par);
    }

    // L1 -> L0：A tile（非 MX 路径）
    __aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& a1, AscendC::LocalTensor<bfloat16_t>& a2, uint32_t calcMAlign)
    {
        AscendC::LoadData2DParamsV2 lp = {};  // 全字段 NSDMI 零初始化后逐一赋值（M16 加固）
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = calcMAlign / CUBE_BLOCK;   // M 轴分形数
        lp.kStep = BASE_K / CUBE_BLOCK;       // K 轴分形数（bf16: 16 元素 = 32B）
        lp.srcStride = BASE_M / CUBE_BLOCK;   // 源 K 方向相邻分形间隔（对应 Nd2Nz dstNzC0Stride=BASE_M）
        lp.dstStride = calcMAlign / CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        AscendC::LoadData(a2, a1[0], lp);
    }

    // L1 -> L0：B tile（N 轴 baseN 行，非 MX 路径）
    __aicore__ inline void LoadB(AscendC::LocalTensor<bfloat16_t>& b1, AscendC::LocalTensor<bfloat16_t>& b2)
    {
        AscendC::LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = BASE_N / CUBE_BLOCK;       // N 轴分形数
        lp.kStep = BASE_K / CUBE_BLOCK;
        lp.srcStride = BASE_N / CUBE_BLOCK;
        lp.dstStride = BASE_N / CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        AscendC::LoadData(b2, b1[0], lp);
    }

    // L0C -> GM：bf16 写回（curM 行 mask，尾块不越界）
    __aicore__ inline void CopyOut(
        AscendC::LocalTensor<float>& cL0C, uint32_t mBlock, uint32_t nBlock, uint32_t curM, uint32_t calcMAlign)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp = {};
        // 该结构体仅 reluScalar/vectorRelu/deqScalar 三个成员无 NSDMI，均显式清零（M16 加固）；
        // 其余字段 NSDMI 为 0/false，params=Nz2NdParams 默认（ndNum=1 即单矩阵语义）。
        fp.nSize = BASE_N;
        fp.mSize = curM;
        fp.srcStride = calcMAlign;
        fp.dstStride = N;
        fp.quantPre = QuantMode_t::F322BF16;
        fp.reluScalar = 0;
        fp.vectorRelu = 0;
        fp.deqScalar = 0;
        AscendC::Fixpipe(cGMOri[mBlock * BASE_M * N + nBlock * BASE_N], cL0C, fp);
    }

private:
    AscendC::GlobalTensor<bfloat16_t> aGMOri;
    AscendC::GlobalTensor<bfloat16_t> bGMOri;
    AscendC::GlobalTensor<bfloat16_t> cGMOri;
    uint32_t mTotal;
};
}  // namespace Cube

// ============================================================
// gamma bf16 → fp32 预转（每个 AIV 核一次，落 PERSIST 区）
//   m6/m12 的 PrecomputeGammaF32 同款：整核只做一次，之后 y 循环里每 chunk 只剩一次
//   fp32→bf16 Cast（规避 3510 相邻 Cast quirk）；bf16 staging 复用 UB_PERSIST。
// ============================================================

template <uint32_t NELEM>
__aicore__ inline void PrecastGammaN(__gm__ uint8_t* gamma, uint32_t dstUbOff, AscendC::MutexID id)
{
    AscendC::GlobalTensor<bfloat16_t> gGm;
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gamma), NELEM);
    LocalTensor<bfloat16_t> gbL(TPosition::VECCALC, UB_PERSIST, NELEM);

    BufAcquire<PIPE_MTE2>(id);
    PipeBarrier<PIPE_MTE2>();
    DataCopy(gbL, gGm, Block1(NELEM * 2));
    BufRelease<PIPE_MTE2>(id);

    __ubuf__ bfloat16_t* gbUb = reinterpret_cast<__ubuf__ bfloat16_t*>(UB_PERSIST);
    __ubuf__ float* gfUb = reinterpret_cast<__ubuf__ float*>(dstUbOff);
    constexpr uint16_t CHUNKS_G = static_cast<uint16_t>(NELEM / VL_F32);
    BufAcquire<PIPE_V>(id);
    __VEC_SCOPE__
    {
        RegTensor<bfloat16_t> gB16;
        RegTensor<float> gF;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t i = 0; i < CHUNKS_G; ++i) {
            uint32_t offset = static_cast<uint32_t>(i) * VL_F32;
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gB16, gbUb + offset);
            Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(gF, gB16, maskAll);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(gfUb + offset, gF, maskAll);
        }
    }
    BufRelease<PIPE_V>(id);
}

// ============================================================
// GDN 层链：段序 S1-S7（单一 __mix__(1,2) 启动）
// ============================================================

struct GdnLayerPtrs {
    __gm__ uint8_t* ws;
    __gm__ uint8_t* xLayer;     // S1 输入 x [M_MAX, HIDDEN] bf16（行 0 = 本 token）
    __gm__ uint8_t* resLayer;   // S1 残差 [M_MAX, HIDDEN] bf16
    __gm__ uint8_t* yLayer;     // S7 出口 y [M_MAX, HIDDEN] bf16（层循环残差流交接点；m15 新增）
    __gm__ uint8_t* gamma1;     // [HIDDEN] bf16
    __gm__ uint8_t* gamma2;     // [HIDDEN] bf16
    __gm__ uint8_t* gammaG;     // [HEAD_D] bf16（RMSNormGated 的 gamma）
    __gm__ uint8_t* wIn;        // [IN_N, HIDDEN] bf16（in_proj）
    __gm__ uint8_t* wOut;       // [OUT_N, OUT_K] bf16（out_proj）
    __gm__ uint8_t* convW;      // [KW, CH] bf16
    __gm__ uint8_t* convBias;   // [CH] bf16
    __gm__ uint8_t* aLog;       // [64] fp32（48 有效 + 尾零）
    __gm__ uint8_t* dtBias;     // [64] fp32
    __gm__ uint8_t* convState;  // [ST, CH] bf16（常驻 GM，in-place）
    __gm__ uint8_t* ssmState;   // [HEADS, HEAD_D, HEAD_D] fp32（常驻 GM，in-place）
    __gm__ uint8_t* xQkvzba;    // slice 模式：host 直供 qkvzba [M_MAX, IN_N] bf16
    uint32_t m;                 // decode 行数（本 mission 恒为 1）
    uint32_t sliceMode;         // 0 = 全链 S1→S7；1 = slice（host qkvzba 直入 S3，S3 起跑）
    uint32_t stageLimit;        // 段序截断（bring-up 定位用；7 = 全开）
};

class GdnLayerChain {
public:
    __aicore__ inline GdnLayerChain() {}

    __aicore__ inline void Init(const GdnLayerPtrs& args)
    {
        p = args;
        ws = args.ws;

        // S1：x = 层输入，res = 层残差（bf16），gamma1
        norm1.Init(args.xLayer, args.resLayer, ws + WS_XNORM, ws + WS_RES1, UB_GAMMA1_F32, args.m);
        // S7：x = out_proj 输出，res = S1 的 fp32 res1（小改 D），gamma2
        // y 出口 = args.yLayer（层循环的残差流 GM 缓冲；m14 原版落在 ws+WS_YFINAL）。
        // ws 内的 WS_YFINAL 区间在本接口下不再使用（m15 dump 直接取 GM 上的 h_out）。
        norm2.Init(ws + WS_OPOUT, ws + WS_RES1, args.yLayer, ws + WS_RES2, UB_GAMMA2_F32, args.m);

        // slice 模式下 S3 与 S5 的 qkvzba 都取 host 直供缓冲
        __gm__ uint8_t* qkvSrc = args.sliceMode ? args.xQkvzba : (ws + WS_QKVZBA);

        prolog.Init(qkvSrc, args.convState, args.convW, args.convBias, args.aLog, args.dtBias, ws + WS_Q,
                    ws + WS_K, ws + WS_V, ws + WS_G, ws + WS_BETA);
        rec.Init(ws + WS_Q, ws + WS_K, ws + WS_V, ws + WS_G, ws + WS_BETA, args.ssmState, ws + WS_O, HEADS);
        gated.Init(ws + WS_O, qkvSrc, ws + WS_OPIN, UB_GAMMA_G_F32);
        // S2/S6：A 的 row 1 由 host 按 max(m,2) 行分配（Nd2Nz 行数=1 退化 quirk 契约）
        gemmIn.Init(ws + WS_XNORM, args.wIn, ws + WS_QKVZBA, args.m);
        gemmOut.Init(ws + WS_OPIN, args.wOut, ws + WS_OPOUT, args.m);
    }

    // ---- AIV 侧：S1 / S3 / S4 / S5 / S7 + 段边界握手 ----
    __aicore__ inline void ProcessAiv()
    {
        const uint32_t bid = AscendC::GetBlockIdx();        // 0 .. 2*numBlocks-1（docs/05 §6 已确认）
        const uint32_t nAiv = AscendC::GetBlockNum() * 2;   // mix(1,2)：AIV 线程数 = 2×AIC 数

        // gamma bf16 → fp32 预转（每核一份，S1/S5/S7 共享）
        PrecastGammaN<HIDDEN>(p.gamma1, UB_GAMMA1_F32, BUF_PRECAST1);
        PrecastGammaN<HIDDEN>(p.gamma2, UB_GAMMA2_F32, BUF_PRECAST1);
        PrecastGammaN<HEAD_D>(p.gammaG, UB_GAMMA_G_F32, BUF_PRECAST_G);

        const bool loopAic = (p.sliceMode == 0);   // 是否需要 AIC 的 in_proj/out_proj

        // ---------- S1 + 段边界①（AIV → AIC：in_proj 输入 x_norm 就绪）----------
        if (loopAic) {
            if (p.stageLimit >= 1) {
                norm1.Run(bid, nAiv);
            }
            if (p.stageLimit >= 2) {
                // 全体 AIV 到齐（mode 0）之后才允许任一 AIV 通知 AIC：否则 AIC 可能在 x_norm
                // 写出前就开始搬 A（m=1 时只有 AIV0 有活，其余 AIV 会立刻越过 S1）。
                // wait 挂 PIPE_S 是 M10 规则的要求：本 wait **后面紧接一个 set**（跨核链式同步），
                // 而 CrossCoreSetFlag 是 drain 触发、窄 pipe wait 给不了后续 set 的次序；
                // 这里不需要 narrow-pipe 排空（x_norm 的可见性由 set 挂 PIPE_MTE3 的本地 drain 承担）。
                BarrierAiv<PIPE_S, FLAG_AIV_SEG_S1>();
                CrossCoreSetFlag<CC_MODE2, PIPE_MTE3>(FLAG_BOUND_IN);
                CrossCoreWaitFlag<CC_MODE2, PIPE_MTE2>(FLAG_BOUND_INPROJ);   // 等 in_proj 输出就绪
            }
        }

        // ---------- S3：prolog（conv_state in-place；读 qkvzba 的 q|k|v|b|a）----------
        // 段内 head/block 条带划分由 donor 自己的 Process() 内部完成（bid/nAiv 取自
        // GetBlockIdx/GetBlockNum*2，见 m9 donor Process）
        if (p.stageLimit >= 3) {
            prolog.Process();
        }

        // ---------- 段内 barrier：q/k/v/g/β 全体 AIV 就位 ----------
        if (p.stageLimit >= 4) {
            BarrierAiv<PIPE_MTE2, FLAG_AIV_SEG0>();
            rec.Process();   // S4：head 条带划分在 donor Process 内（bid/nAiv 同上）
        }

        // ---------- S4：递推（ssm_state in-place RMW）----------
        if (p.stageLimit >= 5) {
            BarrierAiv<PIPE_MTE2, FLAG_AIV_SEG1>();
            gated.Run(bid, nAiv);   // S5
        }

        // ---------- 段边界②（AIV → AIC：out_proj 输入就绪；AIC → AIV：输出就绪）----------
        if (loopAic && p.stageLimit >= 6) {
            // 与 S1 边界对称：全体 AIV 先到齐再通知（防御层；真正的 full rendezvous 由
            // AIC 侧的 mode-0 barrier 给出，见 README §3）。wait 挂 PIPE_S：后面紧接 set。
            BarrierAiv<PIPE_S, FLAG_AIV_SEG2>();
            CrossCoreSetFlag<CC_MODE2, PIPE_MTE3>(FLAG_BOUND_S5);
            CrossCoreWaitFlag<CC_MODE2, PIPE_MTE2>(FLAG_BOUND_OUTPROJ);
        }

        // ---------- S7：Add+RMSNorm#2（残差 = S1 的 fp32 res1）----------
        if (loopAic && p.stageLimit >= 7) {
            norm2.Run(bid, nAiv);
        }
    }

    // ---- AIC 侧：S2 in_proj / S6 out_proj（N 方向条带划分）----
    __aicore__ inline void ProcessAic()
    {
        const uint32_t bid = AscendC::GetBlockIdx();        // 0 .. numBlocks-1
        const uint32_t numBlocks = AscendC::GetBlockNum();  // AIC 数

        if (p.sliceMode != 0 || p.stageLimit < 2) {
            return;   // slice 模式：不经 AIC（host qkvzba 直入 S3）
        }

        // ---------- 等 S1 的 x_norm 就绪（配对 AIV 的 mode2 + 全体 AIC 对齐）----------
        CrossCoreWaitFlag<CC_MODE2, PIPE_S>(FLAG_BOUND_IN);
        CrossCoreSetFlag<CC_MODE0, PIPE_MTE2>(FLAG_AIC_SEG0A);
        CrossCoreWaitFlag<CC_MODE0, PIPE_S>(FLAG_AIC_SEG0A);

        // ---------- S2：in_proj（K=2560 N=16480；N tile 条带划分）----------
        for (uint32_t nt = bid; nt < IN_PROJ_NTILES; nt += numBlocks) {
            gemmIn.RunTile(nt);
        }
        CrossCoreSetFlag<CC_MODE0, PIPE_FIX>(FLAG_AIC_SEG0B);   // FIXP 写 GM 排空 + 全体 AIC 对齐
        CrossCoreWaitFlag<CC_MODE0, PIPE_S>(FLAG_AIC_SEG0B);
        CrossCoreSetFlag<CC_MODE2, PIPE_MTE2>(FLAG_BOUND_INPROJ);   // qkvzba 就绪 → 配对 AIV

        if (p.stageLimit < 6) {
            return;
        }

        // ---------- 等 S5 的 RMSNormGated 输出就绪 ----------
        CrossCoreWaitFlag<CC_MODE2, PIPE_S>(FLAG_BOUND_S5);
        CrossCoreSetFlag<CC_MODE0, PIPE_MTE2>(FLAG_AIC_SEG1A);
        CrossCoreWaitFlag<CC_MODE0, PIPE_S>(FLAG_AIC_SEG1A);

        // ---------- S6：out_proj（K=6144 N=2560）----------
        for (uint32_t nt = bid; nt < OUT_PROJ_NTILES; nt += numBlocks) {
            gemmOut.RunTile(nt);
        }
        CrossCoreSetFlag<CC_MODE0, PIPE_FIX>(FLAG_AIC_SEG1B);
        CrossCoreWaitFlag<CC_MODE0, PIPE_S>(FLAG_AIC_SEG1B);
        CrossCoreSetFlag<CC_MODE2, PIPE_MTE2>(FLAG_BOUND_OUTPROJ);   // 输出就绪 → 配对 AIV
    }

private:
    // 全体 AIV mode-0 barrier：set 挂 MTE3（排空本核写）、wait 挂最窄 pipe（排空对应队列）
    template <pipe_t WAIT_PIPE, uint16_t FLAG>
    __aicore__ inline void BarrierAiv()
    {
        CrossCoreSetFlag<CC_MODE0, PIPE_MTE3>(FLAG);
        CrossCoreWaitFlag<CC_MODE0, WAIT_PIPE>(FLAG);
    }

private:
    GdnLayerPtrs p;
    __gm__ uint8_t* ws = nullptr;

    NormStage<false> norm1;                       // S1（残差 bf16）
    NormStage<true> norm2;                        // S7（残差 fp32，小改 D）
    Prolog::GdnProlog prolog;                     // S3
    Recur::GdnDecodeRec rec;                      // S4
    Gated::RmsNormGatedStage gated;               // S5
    Cube::Bf16Gemm<IN_K, IN_OUT> gemmIn;          // S2（K=2560 N=16480）
    Cube::Bf16Gemm<OUT_K, OUT_N> gemmOut;         // S6（K=6144 N=2560）
};

}  // namespace

// ---- 层 kernel 入口（m14 m14_gdn_layer_kernel 的「层循环」签名改造）----
// 相对 m14 原入口的三处差异（**只有主体逻辑零改动**：GdnLayerChain 的段序/同步/算件一字未动）：
//   1. 出口 yLayer 独立成 GM 缓冲（层间残差流交接点），不再落在 ws 内；
//   2. 去掉 xQkvzba 入口参数 → 恒传 nullptr（层循环不需要 slice 模式的 host 直供 qkvzba）；
//   3. m / sliceMode / stageLimit 三个运行开关退化为编译期常量（decode m=1、全链 S1→S7）。
__global__ __mix__(1, 2) void m15_gdn_layer_kernel(
    __gm__ uint8_t* ws, __gm__ uint8_t* xLayer, __gm__ uint8_t* resLayer, __gm__ uint8_t* yLayer,
    __gm__ uint8_t* gamma1, __gm__ uint8_t* gamma2, __gm__ uint8_t* gammaG, __gm__ uint8_t* wIn,
    __gm__ uint8_t* wOut, __gm__ uint8_t* convW, __gm__ uint8_t* convBias, __gm__ uint8_t* aLog,
    __gm__ uint8_t* dtBias, __gm__ uint8_t* convState, __gm__ uint8_t* ssmState)
{
    AscendC::InitSocState();
    GdnLayerPtrs args;
    args.ws = ws;
    args.xLayer = xLayer;
    args.resLayer = resLayer;
    args.yLayer = yLayer;
    args.gamma1 = gamma1;
    args.gamma2 = gamma2;
    args.gammaG = gammaG;
    args.wIn = wIn;
    args.wOut = wOut;
    args.convW = convW;
    args.convBias = convBias;
    args.aLog = aLog;
    args.dtBias = dtBias;
    args.convState = convState;
    args.ssmState = ssmState;
    args.xQkvzba = nullptr;
    args.m = 1u;               // decode 单 token
    args.sliceMode = 0u;       // 全链 S1→S7
    args.stageLimit = 7u;      // 段序全开

    GdnLayerChain op;
    op.Init(args);
    if ASCEND_IS_AIV {
        op.ProcessAiv();
    }
    if ASCEND_IS_AIC {
        op.ProcessAic();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
