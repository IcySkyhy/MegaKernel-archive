/**
 * m15_hc_layer.h —— **hc（hyper-connection）边界段 device 代码**，融合进 m15 per-layer kernel
 *
 * **本文件由 lift_hc_segment.py 机械生成，不要手改**（改上游 m20 后重跑脚本，再核对 --check
 * 输出的差异表）。抽取范围 = `m20_hyperconn/m20_hyperconn.asc` 的 **device 段**（内容锚点：首个顶格
 * `namespace {` 到入口 kernel `m20_hyperconn_kernel` 前最后一个 `}  // namespace`，**不写行号**）
 * ——入口与其后的 host 段（数据生成 / double 参考 / 判据 / dump）不抽取：融合后 hc 段不再自成
 * kernel，入口由 `m15_layer_kernel.h` 给出。
 *
 * 上游身份（可复核）：文件 sha256 = `b707507d4bc8dd7ec625a346660816b1d1fe9ef5f6b55d85f606fdfbe0cdebe5`
 * （last touched by `a0996ebab356defb743055f7e0178f9361d104c7` = 最后改动 m20 的 commit）；
 * **device 段 sha256 = `12fe5ca26d6f76e3e138a98405953925a5b912287d285e3ee64af635d80fa9a9`**（`--check` 的硬断言：device 段一变即失败）。
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

namespace M15H {

using namespace AscendC;
using namespace AscendC::Reg;

// ============================================================
// 通用：BufferID 封装 / 显式块拷贝参数 / CrossCore barrier 封装
// ============================================================

template <pipe_t pipe>
__aicore__ inline void BufAcquire(AscendC::MutexID id)
{
    AscendC::GetBufInternal<pipe, false>(id);   // acquire 立即获取
}

template <pipe_t pipe>
__aicore__ inline void BufRelease(AscendC::MutexID id)
{
    AscendC::RlsBufInternal<pipe, false>(id);    // 阻塞释放（mode=false = CANN ASC_LOCK_BLOCK 默认；跨 pipe 交接）
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

// 显式块拷贝参数（杜绝元素计数重载：其 DataCopyParams 只填 blockLen，其余字段未初始化）。
// DataCopyParams.blockLen 单位 = 32B 块，故入参统一为字节数、内部 /32 换算。
__aicore__ inline AscendC::DataCopyParams Block1(uint32_t bytes)
{
    if ((bytes & 31u) != 0u) {
        AscendC::Trap();   // 非 32B 整数倍必须走 DataCopyPad
    }
    return AscendC::DataCopyParams{1, static_cast<uint16_t>(bytes / 32), 0, 0};
}

__aicore__ inline AscendC::DataCopyExtParams ExtBlock1(uint32_t bytes)
{
    return AscendC::DataCopyExtParams{1, bytes, 0, 0, 0};
}

// AIV 段内 mode-0 barrier：全体 AIV 到齐（set 挂生产 pipe MTE3：MTE3 写 GM 排空）
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
__aicore__ __inline__ constexpr uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ============================================================
// NormDonor（m6_rmsnorm.asc 逐字 lift；被禁的仅 TPipe/TQue wrapper）
// ============================================================
namespace NormDonor {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t V_LENGTH = VL_F32;

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

// NR rsqrt（reduce_common_regbase_part1.h，m6 逐字 lift，Mula→Mul+Add）
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
    Add(t1, t1, tmp, preg);   // t1 = 1.5 + t*y
    Mul(rstd, y, t1, preg);
    Muls(t3, var, float(-1.0), preg);
    Mul(tmp, t3, r, preg);
    Add(s, s, tmp, preg);     // s = 1 + (-var)*r
    Muls(t4, rstd, float(-1.0), preg);
    Mul(tmp, t4, rstd, preg);
    Add(r, r, tmp, preg);     // r = r + (-rstd)*rstd
    Mul(tmp, var, r, preg);
    Add(s, s, tmp, preg);     // s = s + var*r
    Mul(s, s, rstd, preg);
    Mul(tmp, s, scalar1, preg);
    Add(rstd, rstd, tmp, preg);   // rstd = rstd + s*0.5
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

// ============================================================
// AIC：bf16 GEMM（m11 Bf16Gemm → 层内核 RunTile 版）
//   C[m, NSTRIDE] = A[m, K] · B[N, K]^T
//   B_DUAL=true 时 B 有两个来源（down | inject）：n-tile 落在 LOWRANK 内取 b0，否则取 b1
// ============================================================
template <uint32_t K, uint32_t NSTRIDE, bool B_DUAL>
class HcBf16Gemm {
    static_assert(K % BASE_K == 0, "K must be divisible by BASE_K");

public:
    __aicore__ inline HcBf16Gemm() {}

    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b0, __gm__ uint8_t* b1, __gm__ uint8_t* c,
                                uint32_t m)
    {
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a));
        b0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(b0));
        if constexpr (B_DUAL) {
            b1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(b1));
        }
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(c));
        mTotal = m;
    }

    // 层内核：mLoop 必须为 1（m ≤ BASE_M = 64），一个 n-tile 由调用者（按 bid 条带）给定
    __aicore__ inline void RunTile(uint32_t nt, uint32_t nSize)
    {
        const uint32_t kLoop = K / BASE_K;
        const uint32_t curM = MinU32(mTotal, BASE_M);
        // 3510 实测（m1 契约）：Nd2Nz 行数为 1 时退化为 1D 拷贝、m=1 数据错位。
        // 计算侧统一提升为 ≥2 行（多读的行由 host 保证可读），仅在 Fixpipe 用 curM mask。
        const uint32_t calcM = curM < 2 ? 2 : curM;
        const uint32_t calcMAlign = AlignUp(calcM, CUBE_BLOCK);

        AscendC::LocalTensor<bfloat16_t> a1Ping(TPosition::A1, L1_OFF_A0, L1_A_ELEMS);
        AscendC::LocalTensor<bfloat16_t> a1Pong(TPosition::A1, L1_OFF_A1, L1_A_ELEMS);
        AscendC::LocalTensor<bfloat16_t> b1Ping(TPosition::B1, L1_OFF_B0, L1_B_ELEMS);
        AscendC::LocalTensor<bfloat16_t> b1Pong(TPosition::B1, L1_OFF_B1, L1_B_ELEMS);

        AscendC::LocalTensor<bfloat16_t> a2Ping(TPosition::A2, L0_OFF_0, BASE_M * BASE_K);
        AscendC::LocalTensor<bfloat16_t> a2Pong(TPosition::A2, L0_OFF_1, BASE_M * BASE_K);
        AscendC::LocalTensor<bfloat16_t> b2Ping(TPosition::B2, L0_OFF_0, BASE_K * BASE_N);
        AscendC::LocalTensor<bfloat16_t> b2Pong(TPosition::B2, L0_OFF_1, BASE_K * BASE_N);

        AscendC::LocalTensor<float> cL0C(TPosition::CO1, 0, BASE_M * BASE_N);

        MutexLock<PIPE_M>(BUF_AIC_L0C);   // M 先取 L0C 所有权：挡住上一 tile 的 Fixpipe 读
        for (uint32_t kBlock = 0; kBlock < kLoop; ++kBlock) {
            const uint32_t p = kBlock & 1;
            const AscendC::MutexID bufA = p ? BUF_AIC_A1 : BUF_AIC_A0;
            const AscendC::MutexID bufB = p ? BUF_AIC_B1 : BUF_AIC_B0;
            const AscendC::MutexID bufL0 = p ? BUF_AIC_L01 : BUF_AIC_L00;
            AscendC::LocalTensor<bfloat16_t> a1 = p ? a1Pong : a1Ping;
            AscendC::LocalTensor<bfloat16_t> b1 = p ? b1Pong : b1Ping;
            AscendC::LocalTensor<bfloat16_t> a2 = p ? a2Pong : a2Ping;
            AscendC::LocalTensor<bfloat16_t> b2 = p ? b2Pong : b2Ping;

            // ---- MTE2：GM -> L1（Nd2Nz），大包生产 ----
            MutexLock<PIPE_MTE2>(bufA);
            CopyInA(a1, kBlock, calcM);
            MutexUnlock<PIPE_MTE2>(bufA);
            MutexLock<PIPE_MTE2>(bufB);
            CopyInB(b1, kBlock, nt);
            MutexUnlock<PIPE_MTE2>(bufB);

            // ---- MTE1：L1 -> L0（LoadData），消费 L1、生产 L0 ----
            MutexLock<PIPE_MTE1>(bufL0);
            MutexLock<PIPE_MTE1>(bufA);
            MutexLock<PIPE_MTE1>(bufB);
            LoadA(a1, a2, calcMAlign);
            LoadB(b1, b2);
            MutexUnlock<PIPE_MTE1>(bufA);
            MutexUnlock<PIPE_MTE1>(bufB);
            MutexUnlock<PIPE_MTE1>(bufL0);

            // ---- M：Mmad 累加（L0C fp32），消费 L0 ----
            MutexLock<PIPE_M>(bufL0);
            AscendC::MmadParams mp = {};
            mp.m = calcM;
            mp.n = BASE_N;
            mp.k = BASE_K;
            mp.cmatrixInitVal = (kBlock == 0);
            AscendC::Mmad(cL0C, a2, b2, mp);
            MutexUnlock<PIPE_M>(bufL0);
        }
        MutexUnlock<PIPE_M>(BUF_AIC_L0C);
        MutexLock<PIPE_FIX>(BUF_AIC_L0C);
        CopyOut(cL0C, nt, nSize, curM, calcMAlign);
        MutexUnlock<PIPE_FIX>(BUF_AIC_L0C);
    }

private:
    __aicore__ inline void CopyInA(AscendC::LocalTensor<bfloat16_t>& a1, uint32_t kBlock, uint32_t curM)
    {
        AscendC::Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = curM;             // 行数（调用侧传 calcM≥2）
        par.dValue = BASE_K;           // 每行元素数（bf16 按元素计）
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;
        par.dstNzC0Stride = BASE_M;    // L1 NZ 布局按满 tile 行距（与 LoadA 的 srcStride 配套）
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1, aGm[kBlock * BASE_K], par);
    }

    __aicore__ inline void CopyInB(AscendC::LocalTensor<bfloat16_t>& b1, uint32_t kBlock, uint32_t nt)
    {
        AscendC::Nd2NzParams par = {};
        par.ndNum = 1;
        par.dValue = BASE_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;
        par.dstNzC0Stride = BASE_N;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        // nt 落在 [0, LOWRANK/BASE_N) 内 → Wdown 的满 BASE_N 行；否则 → Winj 的 INJ_N 行
        if constexpr (B_DUAL) {
            if ((nt + 1) * BASE_N <= LOWRANK) {
                par.nValue = BASE_N;
                AscendC::DataCopy(b1, b0Gm[kBlock * BASE_K + nt * BASE_N * K], par);
            } else {
                // 读 16 行（checkpoint 只有 INJ_N=4 行，host 契约保证 ≥16 行可读且行 [4,16) 为 0）：
                // L0C 的列 [4,16) 由 0·x 精确得 0 → OH 的 padding 列 [324,336) 恒为 0，
                // 输出字节确定性成立。若只读 4 行，列 4..15 会取自 L1 残留（有限但随调度变化），
                // 输出虽不影响有效列却破坏重复运行的逐字节一致性。
                par.nValue = 16;
                AscendC::DataCopy(b1, b1Gm[kBlock * BASE_K], par);
            }
        } else {
            par.nValue = BASE_N;
            AscendC::DataCopy(b1, b0Gm[kBlock * BASE_K + nt * BASE_N * K], par);
        }
    }

    __aicore__ inline void LoadA(AscendC::LocalTensor<bfloat16_t>& a1, AscendC::LocalTensor<bfloat16_t>& a2,
                                 uint32_t calcMAlign)
    {
        AscendC::LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = calcMAlign / CUBE_BLOCK;
        lp.kStep = BASE_K / CUBE_BLOCK;
        lp.srcStride = BASE_M / CUBE_BLOCK;
        lp.dstStride = calcMAlign / CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        AscendC::LoadData(a2, a1[0], lp);
    }

    __aicore__ inline void LoadB(AscendC::LocalTensor<bfloat16_t>& b1, AscendC::LocalTensor<bfloat16_t>& b2)
    {
        AscendC::LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = BASE_N / CUBE_BLOCK;
        lp.kStep = BASE_K / CUBE_BLOCK;
        lp.srcStride = BASE_N / CUBE_BLOCK;
        lp.dstStride = BASE_N / CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        AscendC::LoadData(b2, b1[0], lp);
    }

    __aicore__ inline void CopyOut(AscendC::LocalTensor<float>& cL0C, uint32_t nt, uint32_t nSize, uint32_t curM,
                                   uint32_t calcMAlign)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp = {};
        // 该结构体仅 reluScalar/vectorRelu/deqScalar 三个成员无 NSDMI，均显式清零
        fp.nSize = nSize;
        fp.mSize = curM;
        fp.srcStride = calcMAlign;
        fp.dstStride = NSTRIDE;
        fp.quantPre = QuantMode_t::F322BF16;
        fp.reluScalar = 0;
        fp.vectorRelu = 0;
        fp.deqScalar = 0;
        AscendC::Fixpipe(cGm[nt * BASE_N], cL0C, fp);
    }

private:
    AscendC::GlobalTensor<bfloat16_t> aGm;
    AscendC::GlobalTensor<bfloat16_t> b0Gm;
    AscendC::GlobalTensor<bfloat16_t> b1Gm;
    AscendC::GlobalTensor<bfloat16_t> cGm;
    uint32_t mTotal;
};

}  // namespace

// ============================================================
// M36 段序实现（单一 __mix__(1,2)）
// ============================================================

namespace M15H {

struct HcPtrs {
    __gm__ uint8_t* hIn;     // 4 路残差流输入 H [M_MAX, HYPER] bf16
    __gm__ uint8_t* bo;      // pending block output [M_MAX, HID] bf16（mode 1/2 用）
    __gm__ uint8_t* ij;      // pending injection logits [M_MAX, IJ_STRIDE] bf16（mode 1/2 用）
    __gm__ uint8_t* wDown;   // input_mix_weight_down [LOWRANK, HYPER] bf16（checkpoint 原样）
    __gm__ uint8_t* wInj;    // block_inject_weight [>=16, HYPER] bf16（checkpoint 为 [INJ_N,HYPER]）
    __gm__ uint8_t* wUp;     // input_mix_weight_up [UP_N, UP_K] bf16（checkpoint 原样）
    __gm__ uint8_t* hcNorm;  // hc_norm [HYPER] bf16（per-branch mixed affine）
    __gm__ uint8_t* ws;      // GM workspace（偏移见 m15_hc_resources.h §4）
    uint32_t m;              // token 数（1..M_MAX）
    // ---- M58 新增（替换类 4）：IJ 源的行距（元素）----
    //   独立 [m,16] 平面（m20 的 host 形态）= IJ_STRIDE = 16；
    //   层内 handoff（IJ 直接取自上一个边界的 OH[:,320:324)）= OH_W = 336。
    //   两种行距都是 32B 倍数 ⇒ 每行 32B 搬运天然对齐（见 InjwStage 的按行搬运用法）。
    uint32_t ijStride;
    uint32_t mode;           // MODE_MIX / MODE_COMBINE_MIX / MODE_FINAL_MIX / MODE_COMBINE_ONLY
    uint32_t stageLimit;     // 段序截断（bring-up 定位；7 = 全开）
};

class HyperConnOp {
public:
    __aicore__ inline HyperConnOp() {}

    __aicore__ inline void Init(const HcPtrs& a)
    {
        p = a;
        ws = a.ws;
        hInGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.hIn));
        boGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.bo));
        ijGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.ij));
        hcNormGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.hcNorm));
        hcpGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws + WS_HCP));
        xnGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws + WS_XN));
        rstdGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + WS_RSTD));
        injwGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + WS_INJW));
        ohGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws + WS_OH));
        lsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws + WS_LS));
        gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws + WS_GATE));
        blkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws + WS_BLK));
        down.Init(ws + WS_XN, a.wDown, a.wInj, ws + WS_OH, a.m);
        up.Init(ws + WS_LS, a.wUp, nullptr, ws + WS_GATE, a.m);
    }

    // ------------------------------------------------------------
    // AIV：段序
    // ------------------------------------------------------------
    __aicore__ inline void ProcessAiv()
    {
        const uint32_t bid = AscendC::GetBlockIdx();       // 0 .. 2*numBlocks-1
        const uint32_t nAiv = AscendC::GetBlockNum() * 2;  // mix(1,2)：AIV 线程数 = 2×AIC 数
        const bool useCombine = (p.mode != MODE_MIX);

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

        if (p.stageLimit >= 1 && useCombine) {
            InjwStage(bid, nAiv);
        }
        if (p.stageLimit >= 2) {
            if (useCombine) {
                CombineStage(bid, nAiv);
            }
            // 段内 barrier：全体 AIV 到齐后 H' 才全量可见（S2 的组可能跨 AIV 的 S1 产出）
            BarrierAiv<PIPE_MTE2, FLAG_AV0>();
        }
        if (p.stageLimit >= 3) {
            if (useCombine) {
                NormStage(bid, nAiv, hcpGm);
            } else {
                NormStage(bid, nAiv, hInGm);
            }
            BarrierAiv<PIPE_MTE2, FLAG_AV1>();   // XN 全量落盘
        }
        if (p.stageLimit >= 4) {
            CrossCoreSetFlag<CC_MODE2, PIPE_MTE3>(FLAG_A2C_XN);
            CrossCoreWaitFlag<CC_MODE2, PIPE_MTE2>(FLAG_C2A_OH);
        }
        if (p.stageLimit >= 5) {
            SiluStage(bid, nAiv);
            BarrierAiv<PIPE_MTE2, FLAG_AV2>();   // LS 全量落盘
        }
        if (p.stageLimit >= 6) {
            CrossCoreSetFlag<CC_MODE2, PIPE_MTE3>(FLAG_A2C_LS);
            CrossCoreWaitFlag<CC_MODE2, PIPE_MTE2>(FLAG_C2A_GATE);
        }
        if (p.stageLimit >= 7) {
            GateMixStage(bid, nAiv);
        }
    }

    // ------------------------------------------------------------
    // AIC：两段 GEMM（N-tile 按 bid 条带；每个 AIC 只跑自己的 tile）
    // ------------------------------------------------------------
    __aicore__ inline void ProcessAic()
    {
        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nAic = AscendC::GetBlockNum();

        // M58 替换类 5：combine-only 不含任何 GEMM（S3 down 与 S5 up 都跳过）⇒ AIC 直接返回。
        // AIV 侧本档也不等待任何 mode-2 flag，故此早退不会造成对侧挂死。
        if (p.mode == MODE_COMBINE_ONLY) {
            return;
        }

        if (p.stageLimit >= 4) {
            CrossCoreWaitFlag<CC_MODE2, PIPE_S>(FLAG_A2C_XN);   // wait 后紧接 set → 挂 PIPE_S
            BarrierAic<PIPE_S, PIPE_MTE2, FLAG_AC0>();
            // mode 2（use_combine=false）不产出 injection：只跑 down 的 2 个 N-tile
            const uint32_t nDtiles = (p.mode == MODE_FINAL_MIX) ? (LOWRANK / BASE_N) : DN_NTILES;
            for (uint32_t nt = bid; nt < nDtiles; nt += nAic) {
                const uint32_t ns = (nDtiles == DN_NTILES && nt == DN_NTILES - 1) ? DN_LAST_NSIZE : BASE_N;
                down.RunTile(nt, ns);
            }
            // FIXP 写 GM 排空 + 全体 AIC 对齐（AIV 侧要读任意列，必须全体到齐）
            BarrierAic<PIPE_S, PIPE_FIX, FLAG_AC1>();
            CrossCoreSetFlag<CC_MODE2, PIPE_MTE2>(FLAG_C2A_OH);
        }
        if (p.stageLimit >= 6) {
            CrossCoreWaitFlag<CC_MODE2, PIPE_S>(FLAG_A2C_LS);
            BarrierAic<PIPE_S, PIPE_MTE2, FLAG_AC2>();
            for (uint32_t nt = bid; nt < UP_NTILES; nt += nAic) {
                up.RunTile(nt, BASE_N);
            }
            BarrierAic<PIPE_S, PIPE_FIX, FLAG_AC3>();
            CrossCoreSetFlag<CC_MODE2, PIPE_MTE2>(FLAG_C2A_GATE);
        }
    }

private:
    // ---- W0：injW = 2·sigmoid(IJ/HC)。
    //   实现要点（bring-up 实测 + tower 裁定「先隔离再入库」）：S1 里 injW 从 **V 自己写出**的
    //   32B 对齐槽做 BRC 广播。原实现是「W0 用 MTE3 把表写 GM、S1 用 MTE2 取回再 BRC」，该形态
    //   实测非确定性错读（部分 chunk 退化为 out = h）；**但该路径同时含两个可疑因子——
    //   (i) 同核 MTE3→GM→MTE2 回环没有任何跨 pipe 排序原语（生产者/消费者用了两个不同 BufferID）
    //   与 (ii) BRC 读 MTE2 搬进的 UB 槽的可见性——当时的修法同时去掉了两者，故本条《尚未隔离》，
    //   **不作为平台行为引用**（详见 README §4.1）。
    //   现状实现：每个 (token,stream) 一次 VL1 掩码单元素载入 + 单元素 32B 槽落盘，
    //   得到「每个 (token,stream) 一个 32B 槽、有效值在槽首」的全表供 S1 BRC。
    __aicore__ inline void InjwStage(uint32_t /*bid*/, uint32_t /*nAiv*/)
    {
        AscendC::LocalTensor<bfloat16_t> ijL(TPosition::VECCALC, UB_IW_IJ, M_MAX * IJ_STRIDE);
        AscendC::LocalTensor<float> tabL(TPosition::VECCALC, UB_IW_TAB, UB_IWTAB_SLOTS * INJW_SLOT);
        __ubuf__ bfloat16_t* ijUb = reinterpret_cast<__ubuf__ bfloat16_t*>(ijL.GetPhyAddr());
        __ubuf__ float* tabUb = reinterpret_cast<__ubuf__ float*>(tabL.GetPhyAddr());

        BufAcquire<PIPE_MTE2>(BUF_AIV_IJ);
        PipeBarrier<PIPE_MTE2>();
        // M58 替换类 4：IJ 源按**行距 p.ijStride** 逐行 32B 搬运（不再是一次整块拷贝）。
        // 两种行距（独立平面的 16、OH 列 [320,324) 的 336）都是 32B 倍数 ⇒ 每行 32B 天然
        // 对齐，不引入 DataCopyPad（docs/05 §6.1 的 32B 对齐要求）。
        for (uint16_t ijRow = 0; ijRow < static_cast<uint16_t>(M_MAX); ++ijRow) {
            AscendC::DataCopy(ijL[static_cast<uint32_t>(ijRow) * IJ_STRIDE],
                              ijGm[static_cast<uint32_t>(ijRow) * p.ijStride], Block1(IJ_STRIDE * 2));
        }
        BufRelease<PIPE_MTE2>(BUF_AIV_IJ);

        const uint16_t mU = static_cast<uint16_t>(p.m);
        BufAcquire<PIPE_V>(BUF_AIV_IJ);
        BufAcquire<PIPE_V>(BUF_AIV_SG);
        __VEC_SCOPE__
        {
            RegTensor<bfloat16_t> jb16;
            RegTensor<float> v, one, idx, zero, sel, red;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg cmpS;
            Duplicate(one, 1.0f, maskAll);
            Duplicate(zero, 0.0f, maskAll);
            Arange(idx, 0.0f);   // idx 各 lane = 该 lane 的序号（用于按 lane 选流）
            // M36 review P1：先把**整张表**按 32B 槽整槽清零。
            // 否则每个槽只写 4B（`DIST_FIRST_ELEMENT_B32` 只落槽首），槽的 [4,32) 保留 UB 残留 ⇒
            // dump 出来的 GM `injw` 张量含未初始化字节 ⇒ 同一二进制重复运行哈希不同、
            // 归档 manifest 永远复现不出。清零后「每槽 32B 全由本 kernel 写出」，确定性与可复现性成立。
            // （掩码用 VL8：8×fp32 = 32B = 恰好一个槽，不会溢出到相邻槽。）
            {
                // 掩码 VL8 ⇒ 每次只写 8 个 fp32（**不是** VL_F32=64），故步长必须取 INJW_SLOT。
                // （首版误用步长 VL_F32 ⇒ 只覆盖了槽 0、8、16… 共 32/256 个槽，槽 [1..7] 的尾部仍是残留，
                //   实测表现正是「槽 0 确定、槽 1-3 非确定」。）
                MaskReg mask8 = CreateMask<float, MaskPattern::VL8>();
                for (uint16_t c = 0; c < static_cast<uint16_t>(UB_IWTAB_SLOTS); ++c) {
                    StoreAlign<float, StoreDist::DIST_NORM_B32>(tabUb + static_cast<uint32_t>(c) * INJW_SLOT, zero,
                                                               mask8);
                }
            }
            for (uint16_t mi = 0; mi < mU; ++mi) {
                // 一行 IJ（32B，天然对齐）→ 4 个 sigmoid 落在 lane 0..3
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(jb16, ijUb + mi * IJ_STRIDE);
                Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(v, jb16, maskAll);
                Muls(v, v, 1.0f / static_cast<float>(HC), maskAll);
                NormDonor::SigmoidReg(v, v, one, maskAll);
                Muls(v, v, 2.0f, maskAll);
                // 按流抽取 lane s → 该流独有的 32B 槽首。
                // 注意：`StoreDist::DIST_FIRST_ELEMENT_B32` **按定义只写元素 0**（名字即语义；m6_rmsnorm
                // 的 6 处用法都是「Reduce 得标量 + VL1 掩码 + 本 dist」，与此一致）⇒ 用它做 lane 选择
                // 属**误用**（tower 裁定，非平台差异）——改为
                // 「掩码 Select 清零其它 lane → Reduce<SUM> 归约到 lane 0 → 单元素落盘」。
                // 另外 s 必须编译期常量展开（VF 内动态标量会触发
                // 「Unsupported scalar instruction in AIV loop」，docs/05 §6.3 #23 同类）。
                Compares(cmpS, idx, 0.0f, maskAll);
                Select(sel, v, zero, cmpS);
                Reduce<ReduceType::SUM>(red, sel, maskAll);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tabUb + (mi * HC + 0) * INJW_SLOT, red,
                                                                     maskOne);
                Compares(cmpS, idx, 1.0f, maskAll);
                Select(sel, v, zero, cmpS);
                Reduce<ReduceType::SUM>(red, sel, maskAll);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tabUb + (mi * HC + 1) * INJW_SLOT, red,
                                                                     maskOne);
                Compares(cmpS, idx, 2.0f, maskAll);
                Select(sel, v, zero, cmpS);
                Reduce<ReduceType::SUM>(red, sel, maskAll);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tabUb + (mi * HC + 2) * INJW_SLOT, red,
                                                                     maskOne);
                Compares(cmpS, idx, 3.0f, maskAll);
                Select(sel, v, zero, cmpS);
                Reduce<ReduceType::SUM>(red, sel, maskAll);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tabUb + (mi * HC + 3) * INJW_SLOT, red,
                                                                     maskOne);
            }
        }
        BufRelease<PIPE_V>(BUF_AIV_IJ);
        BufRelease<PIPE_V>(BUF_AIV_SG);

        BufAcquire<PIPE_MTE3>(BUF_AIV_SG);
        AscendC::DataCopy(injwGm[0], tabL, Block1(static_cast<uint32_t>(p.m) * HC * INJW_SLOT * 4));
        BufRelease<PIPE_MTE3>(BUF_AIV_SG);
    }

    // ---- S1：combine（4 路残差流进 → H' 出），item = 整行第 c 个 64 元素 chunk ----
    __aicore__ inline void CombineStage(uint32_t bid, uint32_t nAiv)
    {
        AscendC::LocalTensor<bfloat16_t> hL(TPosition::VECCALC, UB_S1_HB, CHUNK);
        AscendC::LocalTensor<bfloat16_t> boL(TPosition::VECCALC, UB_S1_BOB, CHUNK);
        AscendC::LocalTensor<bfloat16_t> obL(TPosition::VECCALC, UB_S1_OB, CHUNK);
        __ubuf__ bfloat16_t* hUb = reinterpret_cast<__ubuf__ bfloat16_t*>(hL.GetPhyAddr());
        __ubuf__ bfloat16_t* boUb = reinterpret_cast<__ubuf__ bfloat16_t*>(boL.GetPhyAddr());
        __ubuf__ bfloat16_t* obUb = reinterpret_cast<__ubuf__ bfloat16_t*>(obL.GetPhyAddr());
        AscendC::LocalTensor<float> tabL(TPosition::VECCALC, UB_IW_TAB, UB_IWTAB_SLOTS * INJW_SLOT);
        __ubuf__ float* iwTabUb = reinterpret_cast<__ubuf__ float*>(tabL.GetPhyAddr());

        const uint32_t nItems = p.m * NCH_D;
        for (uint32_t i = bid; i < nItems; i += nAiv) {
            const uint32_t mi = i / NCH_D;
            const uint32_t c = i - mi * NCH_D;
            const uint32_t s = c / NCH_H;          // 第几路残差流
            const uint32_t jj = c - s * NCH_H;     // 流内 chunk 号

            PipeBarrier<PIPE_MTE2>();   // 同 pipe 复用同一 UB 行缓冲（docs/05 §6.2 第 1 条）
            BufAcquire<PIPE_MTE2>(BUF_AIV_H);
            AscendC::DataCopy(hL, hInGm[mi * HYPER + c * CHUNK], Block1(CHUNK * 2));
            BufRelease<PIPE_MTE2>(BUF_AIV_H);
            BufAcquire<PIPE_MTE2>(BUF_AIV_BO);
            AscendC::DataCopy(boL, boGm[mi * HID + jj * CHUNK], Block1(CHUNK * 2));
            BufRelease<PIPE_MTE2>(BUF_AIV_BO);

            BufAcquire<PIPE_V>(BUF_AIV_H);
            BufAcquire<PIPE_V>(BUF_AIV_BO);
            BufAcquire<PIPE_V>(BUF_AIV_OB);
            __VEC_SCOPE__
            {
                RegTensor<float> w, hr, bor, t;
                RegTensor<bfloat16_t> ob16;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                LoadAlign<float, LoadDist::DIST_BRC_B32>(w, iwTabUb + (mi * HC + s) * INJW_SLOT);
                NormDonor::LoadRegForDtype<bfloat16_t>(hUb, hr, maskAll, 0);
                NormDonor::LoadRegForDtype<bfloat16_t>(boUb, bor, maskAll, 0);
                Mul(t, bor, w, maskAll);              // bo · injW[s]
                Add(t, hr, t, maskAll);               // h + bo·injW[s]
                Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(ob16, t, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(obUb, ob16, maskAll);
            }
            BufRelease<PIPE_V>(BUF_AIV_H);
            BufRelease<PIPE_V>(BUF_AIV_BO);
            BufRelease<PIPE_V>(BUF_AIV_OB);

            BufAcquire<PIPE_MTE3>(BUF_AIV_OB);
            AscendC::DataCopy(hcpGm[mi * HYPER + c * CHUNK], obL, Block1(CHUNK * 2));
            BufRelease<PIPE_MTE3>(BUF_AIV_OB);
        }
    }

    // ---- S2：per (token,stream) 的 GemmaRMSNorm（HID=2560 一组常驻 UB，纯 V 两遍）----
    __aicore__ inline void NormStage(uint32_t bid, uint32_t nAiv, AscendC::GlobalTensor<bfloat16_t>& srcGm)
    {
        AscendC::LocalTensor<bfloat16_t> xL(TPosition::VECCALC, UB_S2_XB, HID);
        AscendC::LocalTensor<bfloat16_t> wL(TPosition::VECCALC, UB_S2_WB, HID);
        AscendC::LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_S2_YB, HID);
        AscendC::LocalTensor<float> rsL(TPosition::VECCALC, UB_S2_RS, 8);
        __ubuf__ bfloat16_t* xUb = reinterpret_cast<__ubuf__ bfloat16_t*>(xL.GetPhyAddr());
        __ubuf__ bfloat16_t* wUb = reinterpret_cast<__ubuf__ bfloat16_t*>(wL.GetPhyAddr());
        __ubuf__ bfloat16_t* yUb = reinterpret_cast<__ubuf__ bfloat16_t*>(yL.GetPhyAddr());
        __ubuf__ float* rsUb = reinterpret_cast<__ubuf__ float*>(rsL.GetPhyAddr());

        const uint32_t nGroups = p.m * HC;
        for (uint32_t g = bid; g < nGroups; g += nAiv) {
            const uint32_t mi = g / HC;
            const uint32_t s = g - mi * HC;

            PipeBarrier<PIPE_MTE2>();
            BufAcquire<PIPE_MTE2>(BUF_AIV_X);
            AscendC::DataCopy(xL, srcGm[mi * HYPER + s * HID], Block1(HID * 2));
            BufRelease<PIPE_MTE2>(BUF_AIV_X);
            BufAcquire<PIPE_MTE2>(BUF_AIV_WB);
            AscendC::DataCopy(wL, hcNormGm[s * HID], Block1(HID * 2));
            BufRelease<PIPE_MTE2>(BUF_AIV_WB);

            BufAcquire<PIPE_V>(BUF_AIV_X);
            BufAcquire<PIPE_V>(BUF_AIV_WB);
            BufAcquire<PIPE_V>(BUF_AIV_Y);
            // pass A：平方和（依赖把 bf16 舍入后的值平方——与 triton 一致）
            __VEC_SCOPE__
            {
                RegTensor<float> acc, xr, t, red;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
                Duplicate(acc, 0.0f, maskAll);
                for (uint16_t c = 0; c < NCH_H; ++c) {
                    NormDonor::LoadRegForDtype<bfloat16_t>(xUb, xr, maskAll, c * CHUNK);
                    Mul(t, xr, xr, maskAll);
                    Add(acc, acc, t, maskAll);
                }
                Reduce<ReduceType::SUM>(red, acc, maskAll);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb, red, maskOne);
            }
            // pass A2：rstd = NR rsqrt(sum/HID + eps)
            __VEC_SCOPE__
            {
                RegTensor<float> v, rstd;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
                LoadAlign<float, LoadDist::DIST_BRC_B32>(v, rsUb);
                Muls(v, v, 1.0f / static_cast<float>(HID), maskAll);
                NormDonor::ComputeRstdNewtonRaphsonReg(v, rstd, maskAll, EPS);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb, rstd, maskOne);
            }
            // pass B：y = (x·rstd) + (x·rstd)·gamma → bf16（Gemma 仿射以 y + y·w 表达，同 triton）
            __VEC_SCOPE__
            {
                RegTensor<float> rstd, xr, wr, t, y;
                RegTensor<bfloat16_t> yb16;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, rsUb);
                for (uint16_t c = 0; c < NCH_H; ++c) {
                    NormDonor::LoadRegForDtype<bfloat16_t>(xUb, xr, maskAll, c * CHUNK);
                    NormDonor::LoadRegForDtype<bfloat16_t>(wUb, wr, maskAll, c * CHUNK);
                    Mul(t, xr, rstd, maskAll);
                    Mul(y, t, wr, maskAll);
                    Add(y, t, y, maskAll);
                    Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb16, y, maskAll);
                    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(yUb + c * CHUNK, yb16, maskAll);
                }
            }
            BufRelease<PIPE_V>(BUF_AIV_X);
            BufRelease<PIPE_V>(BUF_AIV_WB);
            BufRelease<PIPE_V>(BUF_AIV_Y);

            BufAcquire<PIPE_MTE3>(BUF_AIV_Y);
            AscendC::DataCopy(xnGm[mi * HYPER + s * HID], yL, Block1(HID * 2));
            BufRelease<PIPE_MTE3>(BUF_AIV_Y);
            // rstd 证据张量（4B）
            BufAcquire<PIPE_MTE3>(BUF_AIV_RS);
            AscendC::DataCopyPad(rstdGm[g], rsL, ExtBlock1(4));
            BufRelease<PIPE_MTE3>(BUF_AIV_RS);
        }
    }

    // ---- S4：silu(lora/HC)，item = lowrank 行的第 c 个 chunk ----
    __aicore__ inline void SiluStage(uint32_t bid, uint32_t nAiv)
    {
        AscendC::LocalTensor<bfloat16_t> lL(TPosition::VECCALC, UB_S4_LB, CHUNK);
        AscendC::LocalTensor<bfloat16_t> sL(TPosition::VECCALC, UB_S4_SB, CHUNK);
        __ubuf__ bfloat16_t* lUb = reinterpret_cast<__ubuf__ bfloat16_t*>(lL.GetPhyAddr());
        __ubuf__ bfloat16_t* sUb = reinterpret_cast<__ubuf__ bfloat16_t*>(sL.GetPhyAddr());

        const uint32_t nItems = p.m * NCH_R;
        for (uint32_t i = bid; i < nItems; i += nAiv) {
            const uint32_t mi = i / NCH_R;
            const uint32_t c = i - mi * NCH_R;

            PipeBarrier<PIPE_MTE2>();
            BufAcquire<PIPE_MTE2>(BUF_AIV_LB);
            AscendC::DataCopy(lL, ohGm[mi * OH_W + OH_LORA + c * CHUNK], Block1(CHUNK * 2));
            BufRelease<PIPE_MTE2>(BUF_AIV_LB);

            BufAcquire<PIPE_V>(BUF_AIV_LB);
            BufAcquire<PIPE_V>(BUF_AIV_SB);
            __VEC_SCOPE__
            {
                RegTensor<float> xr, u, sg, one, y;
                RegTensor<bfloat16_t> yb16;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                Duplicate(one, 1.0f, maskAll);
                NormDonor::LoadRegForDtype<bfloat16_t>(lUb, xr, maskAll, 0);
                Muls(u, xr, 1.0f / static_cast<float>(HC), maskAll);   // u = lora/HC
                NormDonor::SigmoidReg(sg, u, one, maskAll);
                Mul(y, u, sg, maskAll);                              // u·sigmoid(u)
                Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb16, y, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(sUb, yb16, maskAll);
            }
            BufRelease<PIPE_V>(BUF_AIV_LB);
            BufRelease<PIPE_V>(BUF_AIV_SB);

            BufAcquire<PIPE_MTE3>(BUF_AIV_SB);
            AscendC::DataCopy(lsGm[mi * LOWRANK + c * CHUNK], sL, Block1(CHUNK * 2));
            BufRelease<PIPE_MTE3>(BUF_AIV_SB);
        }
    }

    // ---- S6：gate mix，item = 输出行第 c 个 chunk（4 路并列读-算-累加）----
    __aicore__ inline void GateMixStage(uint32_t bid, uint32_t nAiv)
    {
        AscendC::LocalTensor<bfloat16_t> gL(TPosition::VECCALC, UB_S6_GB, HC * CHUNK);
        AscendC::LocalTensor<bfloat16_t> xL(TPosition::VECCALC, UB_S6_XB, HC * CHUNK);
        AscendC::LocalTensor<bfloat16_t> oL(TPosition::VECCALC, UB_S6_OB, CHUNK);
        __ubuf__ bfloat16_t* gUb = reinterpret_cast<__ubuf__ bfloat16_t*>(gL.GetPhyAddr());
        __ubuf__ bfloat16_t* xUb = reinterpret_cast<__ubuf__ bfloat16_t*>(xL.GetPhyAddr());
        __ubuf__ bfloat16_t* oUb = reinterpret_cast<__ubuf__ bfloat16_t*>(oL.GetPhyAddr());

        const uint32_t nItems = p.m * NCH_H;
        for (uint32_t i = bid; i < nItems; i += nAiv) {
            const uint32_t mi = i / NCH_H;
            const uint32_t c = i - mi * NCH_H;

            PipeBarrier<PIPE_MTE2>();
            BufAcquire<PIPE_MTE2>(BUF_AIV_GB);
            for (uint32_t s = 0; s < HC; ++s) {
                AscendC::DataCopy(gL[s * CHUNK], gateGm[mi * UP_N + s * HID + c * CHUNK], Block1(CHUNK * 2));
            }
            BufRelease<PIPE_MTE2>(BUF_AIV_GB);
            BufAcquire<PIPE_MTE2>(BUF_AIV_XB);
            for (uint32_t s = 0; s < HC; ++s) {
                AscendC::DataCopy(xL[s * CHUNK], xnGm[mi * HYPER + s * HID + c * CHUNK], Block1(CHUNK * 2));
            }
            BufRelease<PIPE_MTE2>(BUF_AIV_XB);

            BufAcquire<PIPE_V>(BUF_AIV_GB);
            BufAcquire<PIPE_V>(BUF_AIV_XB);
            BufAcquire<PIPE_V>(BUF_AIV_OB2);
            __VEC_SCOPE__
            {
                RegTensor<float> acc, gr, xr, sg, one, t;
                RegTensor<bfloat16_t> ob16;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                Duplicate(one, 1.0f, maskAll);
                Duplicate(acc, 0.0f, maskAll);
                // 累加顺序 s=0..HC-1 后统一 /HC —— 与 triton _hc_gate_mix_kernel 逐句一致
                for (uint16_t s = 0; s < static_cast<uint16_t>(HC); ++s) {
                    NormDonor::LoadRegForDtype<bfloat16_t>(gUb, gr, maskAll, s * CHUNK);
                    NormDonor::LoadRegForDtype<bfloat16_t>(xUb, xr, maskAll, s * CHUNK);
                    NormDonor::SigmoidReg(sg, gr, one, maskAll);
                    Mul(t, sg, xr, maskAll);
                    Add(acc, acc, t, maskAll);
                }
                Muls(acc, acc, 1.0f / static_cast<float>(HC), maskAll);
                Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(ob16, acc, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(oUb, ob16, maskAll);
            }
            BufRelease<PIPE_V>(BUF_AIV_GB);
            BufRelease<PIPE_V>(BUF_AIV_XB);
            BufRelease<PIPE_V>(BUF_AIV_OB2);

            BufAcquire<PIPE_MTE3>(BUF_AIV_OB2);
            AscendC::DataCopy(blkGm[mi * HID + c * CHUNK], oL, Block1(CHUNK * 2));
            BufRelease<PIPE_MTE3>(BUF_AIV_OB2);
        }
    }

private:
    HcPtrs p;
    __gm__ uint8_t* ws = nullptr;
    AscendC::GlobalTensor<bfloat16_t> hInGm;
    AscendC::GlobalTensor<bfloat16_t> boGm;
    AscendC::GlobalTensor<bfloat16_t> ijGm;
    AscendC::GlobalTensor<bfloat16_t> hcNormGm;
    AscendC::GlobalTensor<bfloat16_t> hcpGm;
    AscendC::GlobalTensor<bfloat16_t> xnGm;
    AscendC::GlobalTensor<float> rstdGm;
    AscendC::GlobalTensor<float> injwGm;
    AscendC::GlobalTensor<bfloat16_t> ohGm;
    AscendC::GlobalTensor<bfloat16_t> lsGm;
    AscendC::GlobalTensor<bfloat16_t> gateGm;
    AscendC::GlobalTensor<bfloat16_t> blkGm;
    HcBf16Gemm<DN_K, OH_W, true> down;
    HcBf16Gemm<UP_K, UP_N, false> up;
};

}  // namespace
