/**
 * m29_ple_support.h —— M161 PLE 单元目录的**自包含**支持件（不 include `m15_layer_loop/**`）
 *
 * 来源（逐字抄改，人类裁决「优先从现有代码库抄改」）：
 *   · BufferID 封装 / 显式块拷贝参数 / AIV·AIC barrier 封装：`m15_layer_loop/m15_hc_layer.h:57-114`
 *   · NormDonor（LoadRegForDtype / ComputeRstdNewtonRaphsonReg / SigmoidReg / cast traits）：
 *     `m15_layer_loop/m15_hc_layer.h:119-215`（其上游 = `m6_rmsnorm.asc`，本仓多处复用）
 *   · 形状 / tile / L0·L1 常量：`m15_layer_loop/m15_hc_resources.h:39-153`
 *
 * 为什么**复制**而不 include：本目录是「单元级」验证目录，要能**独立编译**（不因为 m15 在飞
 * 任务改 `m15_layer_loop/**` 而漂移）。复制的是纯 helper 与编译期常量，不含 hc 段体逻辑。
 * 逐字保真由 `evidence/` 的 sha256 见证（见 README §2）。
 */
#ifndef M29_PLE_SUPPORT_H
#define M29_PLE_SUPPORT_H

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "reg_compute/kernel_reg_compute_intf.h"

#include <cstdint>

namespace M29S {

using namespace AscendC;
using namespace AscendC::Reg;

// ============================================================
// §0 形状 / tile 常量（语义权威 = ple/PLE_SPEC.md §4.1；与 m15_hc_resources.h 同值）
// ============================================================
constexpr uint32_t HID = 2560;                  // hidden_size
constexpr uint32_t HC = 4;                      // hc_count
constexpr uint32_t HYPER = HID * HC;            // 10240 = 多流态宽度
constexpr uint32_t HE = 2560;                   // ple_embed_dim
constexpr uint32_t P = 8;                       // heads_per_ngram
constexpr uint32_t NGR = 3;                     // ngram_size
constexpr uint32_t NG = (NGR - 1) * P;          // 16 = 每 token 的 n-gram head 数
constexpr uint32_t HDIM = HE / NG;              // 160 = 每 head 的 embedding 维度
constexpr uint32_t NC = 2;                      // ngram_context_len = NGR-1
constexpr uint32_t KVW = HYPER + HID;           // 12800 = kv 投影输出宽
constexpr uint32_t KCONV = 4;                   // ple_conv_kernel_size
constexpr uint32_t DIL = NGR;                   // 3
constexpr uint32_t STLEN = (KCONV - 1) * DIL;   // 9 = short-conv 状态长度
constexpr float EPS = 1e-6f;                    // rms_norm_eps
constexpr int32_t EOS = 248044;
constexpr int64_t VOCAB_ROWS = 320001536LL;     // 词表总行数（128 × 2,500,012）
constexpr uint32_t VL = 64;                     // 256B/4B：fp32 SIMD lane 数
constexpr uint32_t NCH_H = HID / VL;            // 40 = 一路 2560 的 chunk 数
constexpr uint32_t FAIL_SLOTS = 128;            // 设备侧计数每核槽位数

// 真实 ngram 表几何（REAL_TABLE.md §2.2；跨分片换算的除数）
constexpr uint32_t NSHARDS = 128;
constexpr uint32_t ROWS_PER_SHARD = 2500012;
constexpr uint32_t ROW_BYTES = HDIM * 2u;       // 320 B / 行

// ---- AIC cube tile 常量（与 m15_hc_resources.h §1 同值）----
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t BASE_M = 64;
constexpr uint32_t BASE_K = 64;
constexpr uint32_t BASE_N = 160;
constexpr uint32_t KLOOP = HE / BASE_K;         // 40
constexpr uint32_t NTILES = KVW / BASE_N;       // 80
static_assert(HE % BASE_K == 0u, "HE must be divisible by BASE_K");
static_assert(KVW % BASE_N == 0u, "KVW must be divisible by BASE_N");
constexpr uint32_t L1_A_ELEMS = BASE_M * BASE_K;
constexpr uint32_t L1_B_ELEMS = BASE_N * BASE_K;
constexpr uint32_t L1_OFF_A0 = 0;
constexpr uint32_t L1_OFF_A1 = L1_OFF_A0 + L1_A_ELEMS * 2;
constexpr uint32_t L1_B_REGION = 256u * 1024u;
constexpr uint32_t L1_OFF_B0 = L1_B_REGION;
constexpr uint32_t L1_OFF_B1 = L1_B_REGION + L1_B_ELEMS * 2;
constexpr uint32_t L1_BYTES_TOTAL = 512u * 1024u;
static_assert(L1_B_REGION + 2u * L1_B_ELEMS * 2u <= L1_BYTES_TOTAL, "A/B ping-pong L1 too big");
constexpr uint32_t L0_PP_BYTES = 32u * 1024u;
constexpr uint32_t L0_OFF_0 = 0;
constexpr uint32_t L0_OFF_1 = L0_PP_BYTES;
static_assert(BASE_M * BASE_K * 2u <= L0_PP_BYTES, "A2 tile exceeds L0A half");
static_assert(BASE_K * BASE_N * 2u <= L0_PP_BYTES, "B2 tile exceeds L0B half");
static_assert(BASE_M * BASE_N * 4u <= 256u * 1024u, "L0C tile exceeds 256KB");

// ---- BufferID（MutexID 有效 0..27；本目录**独立启动** ⇒ 只用自定子集）----
// AIC cube（与 hc/GDN 同号，因段体互斥且成对配平；本单元只跑 PLE，无并发）
constexpr uint32_t BUF_AIC_A0 = 0;
constexpr uint32_t BUF_AIC_A1 = 1;
constexpr uint32_t BUF_AIC_B0 = 2;
constexpr uint32_t BUF_AIC_B1 = 3;
constexpr uint32_t BUF_AIC_L00 = 4;
constexpr uint32_t BUF_AIC_L01 = 5;
constexpr uint32_t BUF_AIC_L0C = 6;
using MutexID = AscendC::MutexID;

// ============================================================
// §1 BufferID / 显式块拷贝 / barrier 封装（m15_hc_layer.h:57-114 逐字）
// ============================================================
template <pipe_t pipe>
__aicore__ inline void BufAcquire(AscendC::MutexID id)
{
    AscendC::GetBufInternal<pipe, false>(id);
}
template <pipe_t pipe>
__aicore__ inline void BufRelease(AscendC::MutexID id)
{
    AscendC::RlsBufInternal<pipe, false>(id);   // 阻塞释放（mode=false = CANN 默认 ASC_LOCK_BLOCK，跨 pipe 交接）
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

__aicore__ inline AscendC::DataCopyParams Block1(uint32_t bytes)
{
    if ((bytes & 31u) != 0u) {
        AscendC::Trap();
    }
    return AscendC::DataCopyParams{1, static_cast<uint16_t>(bytes / 32), 0, 0};
}
__aicore__ inline AscendC::DataCopyExtParams ExtBlock1(uint32_t bytes)
{
    return AscendC::DataCopyExtParams{1, bytes, 0, 0, 0};
}

constexpr uint16_t CC_MODE0 = 0;
constexpr uint16_t CC_MODE2 = 2;

// AIV 段内 mode-0 barrier：全体 AIV 到齐（set 挂 MTE3：MTE3 写 GM 排空）
template <pipe_t WAIT_PIPE, uint16_t FLAG>
__aicore__ inline void BarrierAiv()
{
    AscendC::CrossCoreSetFlag<CC_MODE0, PIPE_MTE3>(FLAG);
    AscendC::CrossCoreWaitFlag<CC_MODE0, WAIT_PIPE>(FLAG);
}
// AIC 段内 mode-0 barrier：全体 AIC 到齐；SET_PIPE 取 PIPE_FIX 时表示 FIXP 写 GM 已排空
template <pipe_t WAIT_PIPE, pipe_t SET_PIPE, uint16_t FLAG>
__aicore__ inline void BarrierAic()
{
    AscendC::CrossCoreSetFlag<CC_MODE0, SET_PIPE>(FLAG);
    AscendC::CrossCoreWaitFlag<CC_MODE0, WAIT_PIPE>(FLAG);
}

__aicore__ __inline__ constexpr uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ __inline__ constexpr uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

// ============================================================
// §2 NormDonor（m15_hc_layer.h:119-215 逐字；被禁的仅 TPipe/TQue wrapper）
// ============================================================
namespace NormDonor {
using namespace AscendC;
using namespace AscendC::Reg;

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
    Maxs(var, var, SCALAR0, preg);
    Div(r, one, var, preg);
    Sqrt(y, r, preg);
    Muls(t, var, SCALAR1, preg);
    Mul(t, t, y, preg);
    Mul(tmp, t, y, preg);
    Add(t1, t1, tmp, preg);
    Mul(rstd, y, t1, preg);
    Muls(t3, var, float(-1.0), preg);
    Mul(tmp, t3, r, preg);
    Add(s, s, tmp, preg);
    Muls(t4, rstd, float(-1.0), preg);
    Mul(tmp, t4, rstd, preg);
    Add(r, r, tmp, preg);
    Mul(tmp, var, r, preg);
    Add(s, s, tmp, preg);
    Mul(s, s, rstd, preg);
    Mul(tmp, s, scalar1, preg);
    Add(rstd, rstd, tmp, preg);
    Compares(cmpRegZero, var, POS_INF, preg);
    Select(rstd, scalarZero, rstd, cmpRegZero);
    Compares(cmpRegInf, var, float(0.0), preg);
    Select(rstd, scalarInf, rstd, cmpRegInf);
}

// sigmoid(x) = 1/(1+exp(-x))（m5/m12 实证排布，fp32）
__aicore__ inline void SigmoidReg(RegTensor<float>& dst, RegTensor<float>& src, RegTensor<float>& one,
                                  MaskReg& preg)
{
    Muls(dst, src, -1.0f, preg);
    Exp(dst, dst, preg);
    Adds(dst, dst, 1.0f, preg);
    Div(dst, one, dst, preg);
}

}  // namespace NormDonor

}  // namespace M29S

#endif  // M29_PLE_SUPPORT_H
