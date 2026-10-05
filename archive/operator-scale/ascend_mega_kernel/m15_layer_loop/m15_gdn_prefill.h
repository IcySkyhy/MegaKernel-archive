#ifndef M15_GDN_PREFILL_H
#define M15_GDN_PREFILL_H
/**
 * m15_layer_loop/m15_gdn_prefill.h —— GDN 线性注意力 **prefill 段**（chunk 扫描；M115 / Wave B1）
 *
 * 融合形态要求（docs/15 §M103-2.7）：include guard、**无 main()**、具名 namespace（`M15GP`，不用匿名
 * namespace 遮蔽）、只依赖设备头（`kernel_operator.h` + 本段资源头 `m15_gdn_resources.h`）、资源全部
 * 编译期静态分配、自带峰值 static_assert。挂载点补丁与融合清单一并在 `m15_gdn_prefill_host.h` 与本段
 * README 里给出（Wave C 的机械活）。
 *
 * 人类裁决（本段的选型依据，逐字）：「凡是涉及矩阵乘法的操作都需要用 mmad 实现，不管 M=1 或者是多大」；
 * 「只要是 mmad，就要用 cube 去做，没必要浪费时间去对比」⇒ 本段 **没有 VF 收缩实现、没有选型对比、
 * 没有 VF 备选支**：6 个收缩全部走 AIC 的 `Mmad`。非收缩项（cumsum / Γ=exp 组装 / β / 逐行缩放 /
 * `(I+A)⁻¹` 三角前代）按 docs/19 §10.2 的裁决映射留在 AIV 的 fp32 VF（不是矩阵乘法；且 §11.2 的
 * 精度分析要求这两条链留在 fp32）。
 *
 * dtype：**操作数与累加全 fp32**。依据 M106（`probe_cube_fp32/README.md` §3/§6）：3510 的
 * `Mmad(float,float,float)` 成立、默认模式下操作数保持全 fp32、累加为 fp32 宽度 ⇒ 判据形态不变
 * （继续对 fp64 紧容差参考）。代价：fp32 操作数下 Mmad 吞吐 = bf16 的 1/15.88（≈23 TFLOPS）。
 * 本段**不调 `asc_enable_hf32()`**（开了会把操作数舍到 10 位尾数，`docs/17` 的 ε≈5e-5 会失守）。
 * fp32 的 cube `C0_SIZE = 8`（bf16 是 16），全部装载参数按此口径。
 *
 * 数学（逐句对齐 m18_gdn_prefill 的已证实现与官方 golden；见 m18 README §1）：
 *   ĝ = chunk 内 inclusive cumsum(g)；egL = exp(ĝ[cv−1])
 *   Γs[i,j] = exp(ĝ_i−ĝ_j) (j<i)；Γi[i,j] = exp(ĝ_i−ĝ_j) (j≤i)   ← **指数差**，因子恒 ≤1、有限
 *   A[i,j] = β_i·Γs[i,j]·(k_i·k_j)（严格下三角）
 *   u = (I+A)⁻¹(β⊙v)；w = (I+A)⁻¹(β⊙exp(ĝ)⊙k)
 *   d = u − w·S₀；o = ((q·scale)·S₀)⊙_row exp(ĝ) + (Γi⊙((q·scale)·kᵀ))·d
 *   S₁ = egL·S₀ + kᵀ·(d⊙exp(ĝL−ĝ))
 * Γ 与 KT′ **不物化 ig=exp(−ĝ)**：chunk 内 |ĝ|>88.7 时 exp(ĝ)·exp(−ĝ) 是 0·inf=NaN，而 exp(ĝ_i−ĝ_j)
 * 恒有限；官方 FLA 同法（`chunk_o.py:119-120`、`chunk_delta_h.py:216-221`）。
 * 状态 **S 的 fp32 权威在 AIV 的 UB**（`GdnStateHome::AivUb`），且以 **ST = Sᵀ（[DV,DK]）** 的朝向存放
 * —— 这个朝向让 6 个收缩的 cube 操作数**零转置**（M3 的 A 与 M4 的 B 都是 ST 本身）。
 *
 * 三段 job（段间 mode 2 CrossCore；核内一律 BufferID）：
 *   J1  AIC：Nd2Nz(k,q) → M1 `kk = k·kᵀ`、M2 `qk = q·kᵀ` → Fixpipe 落 GM scratch
 *   J2  AIC：Nd2Nz(ST,w,q) → M3 `(w·S)ᵀ = ST·wᵀ`、M4 `q·S = q·STᵀ` → Fixpipe
 *   J3  AIC：Nd2Nz(ABM,DT,KT') → M5 `AB·d = ABM·DTᵀ`、M6 `(kᵀ·DP)ᵀ = DT·KT'ᵀ` → Fixpipe
 * `Mmad` 语义按 m11_bf16_gemm / probe_cube_fp32 的已证形态：`C[m][n] = Σ_k A[m][k]·B[n][k]`，
 * A、B 都以「行主序、K 连续」的 ND 矩阵给出，经 `Nd2Nz`（dstNzC0Stride = 行数）落 L1，
 * 再由 `LoadData2DParamsV2`(A) / `LoadData2DParams`(B) 进 L0A/L0B，`Mmad` 累加进 L0C，`Fixpipe` 落 GM。
 *
 * AIV 每个 chunk 只做 2 次 VF 转置（`u→uᵀ`、`k→kᵀ`）；`KT' = kᵀ` 每列乘 `exp(ĝL−ĝ[t])`，把状态的跨 chunk
 * 衰减折进操作数 ⇒ 状态更新退化成 `ST₁ = egL·ST₀ + SDT`（SDT = M6 的输出）。
 *
 * 同步纪律：核内一律 BufferID（`BufAcquire`/`BufRelease` 一律 mode=false，CANN `ASC_LOCK_BLOCK` 默认），**不用 set_flag/wait_flag
 * 系列**；同 pipe 背靠背复用同一 buffer 处按 docs/05 §6.2 加 `PipeBarrier<对应 PIPE>`；落 GM 一律走 DMA
 * （AIV 用 MTE3、AIC 用 Fixpipe），不做标量直写 GM（docs/05 §6.1 规则 ⓔ）。
 */

#include "kernel_operator.h"

#include "m15_gdn_resources.h"

#ifndef M15GP_J1_ONLY
#define M15GP_J1_ONLY 0
#endif
#ifndef M15GP_J1J2_ONLY
#define M15GP_J1J2_ONLY 0
#endif
#ifndef M15GP_J2_NZ_SKIP_FROM
#define M15GP_J2_NZ_SKIP_FROM 0   // >0：J2 的 Nd2Nz 从该 chunk 编号起跳过（诊断：L1 令牌跨次复用）
#endif
#ifndef M15GP_J2_NZ_WHICH
#define M15GP_J2_NZ_WHICH 0       // 0=三条全跳；1/2/3=只跳 q/ST/W；4/5/6=**只保留** q/ST/W（补集实验）
#endif
#ifndef M15GP_STAB_NAIVE
#define M15GP_STAB_NAIVE 0        // 负向对照 1：Γ 退回 exp(ĝ)·exp(−ĝ)（chunk 内 |ĝ|>88.7 ⇒ NaN）
#endif
#ifndef M15GP_STAB_BADIDX
#define M15GP_STAB_BADIDX 0       // 负向对照 2：Γ/KT′ 的指数差下标故意错一位（有限但错 ⇒ 判据变红）
#endif

namespace M15GP {

using namespace AscendC;
using namespace AscendC::Reg;

// ============================================================
// 0. 段接口
// ============================================================

/** 状态归属（必须进接口；docs/19 §4.4）。 */
enum class GdnStateHome : uint32_t {
    AivUb = 0,   // 本实现：fp32 权威常驻 AIV 的 UB（64KB），只首尾过 GM
    AicL1 = 1,   // 未实现（cube 侧就地缩放需 128×128 对角操作数，代价更大）
    Gm = 2,      // 未实现（官方 stage2 形态：每层 ~400MB GM 往返）
};
constexpr GdnStateHome kGdnStateHome = GdnStateHome::AivUb;

/**
 * 段参数（**不吃 `LayerArgs`**；docs/15 §M103-2.2 Wave B 共同契约第 1 条）。
 * GM 平面契约（每 head 行主序 fp32；`tp = align8(m)`）：
 *   q,k   [NK=16, m, 128]；v [48, m, 128]；g,β [48, tp]；h0,ht [48,128,128]；out [48, m, 128]
 *   scratch [nAic, GP_SLOT_BYTES] 本段私有工作区（每 AICore 一个 slot）
 * 越界读契约：AIC 的 Nd2Nz 对尾 chunk 固定读满 64 行 ⇒ q/k 平面末尾须留 ≥ 64 行可读且**置零**
 * （docs/05 §5.3：多读的行由 host 保证可读，且多算出的行必须确定 —— 置零保证 mmad 结果确定）。
 */
struct GdnPrefillArgs {
    __gm__ uint8_t* q;
    __gm__ uint8_t* k;
    __gm__ uint8_t* v;
    __gm__ uint8_t* g;
    __gm__ uint8_t* beta;
    __gm__ uint8_t* h0;
    __gm__ uint8_t* out;
    __gm__ uint8_t* ht;
    __gm__ uint8_t* scratch;
    uint32_t m;
    uint32_t heads;
    uint32_t tp;        // g/β 行 stride = align8(m)
    uint32_t qkStride;  // q/k 平面的**每 head 行 stride**（= m 时无补齐；host 若在每 head 尾部补行则 = m+padRows）
    float scale;
};

// ============================================================
// 1. 同步原语
// ============================================================

template <pipe_t P>
__aicore__ inline void BufAcquire(uint32_t id)
{
    AscendC::GetBufInternal<P, false>(static_cast<AscendC::MutexID>(id));
}
template <pipe_t P>
__aicore__ inline void BufRelease(uint32_t id)
{
    // mode=false：CANN `ASC_LOCK_BLOCK` 默认（阻塞）模式，与 acquire 侧同模式（M181 统一）
    AscendC::RlsBufInternal<P, false>(static_cast<AscendC::MutexID>(id));
}

constexpr uint8_t GP_CC_MODE2 = 2;   // AIC ↔ 其配对的两个 AIV
template <pipe_t P>
__aicore__ inline void CcSet(uint16_t id)
{
    AscendC::CrossCoreSetFlag<GP_CC_MODE2, P>(id);
}
template <pipe_t P>
__aicore__ inline void CcWait(uint16_t id)
{
    AscendC::CrossCoreWaitFlag<GP_CC_MODE2, P>(id);
}

constexpr uint32_t GP_VL = 64;   // fp32 单寄存器 lane 数（256B / 4B）

// ============================================================
// 2. VF 原语（非收缩项；形态与 m18_gdn_prefill.asc 的已证实现一致）
// ============================================================

__simd_vf__ inline void MemBarVL() { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }

__simd_vf__ inline void ZeroVF(__ubuf__ float* ptr, uint32_t count)
{
    RegTensor<float> z;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    Duplicate(z, 0.0f);
    for (uint16_t t = 0; t < static_cast<uint16_t>(count / GP_VL); ++t) {
        StoreAlign(ptr + static_cast<uint32_t>(t) * GP_VL, z, allF);
    }
}

__simd_vf__ inline void ZeroTailVF(__ubuf__ float* ptr, uint32_t from)
{
    RegTensor<float> z;
    RegTensor<int32_t> laneIdx;
    MaskReg allI = CreateMask<int32_t, MaskPattern::ALL>();
    MaskReg m;
    Arange(laneIdx, static_cast<int32_t>(0));
    Compares<int32_t, CMPMODE::GE>(m, laneIdx, static_cast<int32_t>(from), allI);
    Duplicate(z, 0.0f);
    StoreAlign(ptr, z, m);
}

// ĝ 与 eg = exp(ĝ)。ig = exp(−ĝ) 不再物化：Γ 与 KT′ 一律改用**指数差**（GammaStrictVF/BrcDiffExpVF）。
__simd_vf__ inline void CumSumExpVF(__ubuf__ float* gUb, __ubuf__ float* gcUb, __ubuf__ float* egUb)
{
    RegTensor<float> gc, shifted, tmp;
    RegTensor<int32_t> laneIdx, gidx;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    MaskReg allI = CreateMask<int32_t, MaskPattern::ALL>();
    MaskReg addMask;
    LoadAlign(gc, gUb);
    Arange(laneIdx, static_cast<int32_t>(0));
    for (int32_t stride = 1; stride < static_cast<int32_t>(M15G::GP_BT); stride <<= 1) {
        Adds<int32_t>(gidx, laneIdx, -stride, allI);
        Maxs<int32_t>(gidx, gidx, static_cast<int32_t>(0), allI);
        Gather<float, uint32_t>(shifted, gc, reinterpret_cast<RegTensor<uint32_t>&>(gidx));
        Compares<int32_t, CMPMODE::GE>(addMask, laneIdx, stride, allI);
        Add<float, MaskMergeMode::MERGING>(gc, gc, shifted, addMask);
    }
    StoreAlign<float, StoreDist::DIST_NORM>(gcUb, gc, allF);
    Exp<float, MaskMergeMode::ZEROING>(tmp, gc, allF);
    StoreAlign<float, StoreDist::DIST_NORM>(egUb, tmp, allF);
}

/**
 * Γ 行（严格下三角，lane j<r）：**指数差** Γ[i,j] = exp(ĝ_i−ĝ_j) —— 指数 ≤0 ⇒ 因子 ≤1、**恒有限**。
 * 不再造 exp(ĝ)·exp(−ĝ)：chunk 内 |ĝ|>88.7 时 eg 下溢 0、ig 上溢 inf ⇒ 0·inf=NaN。
 * 官方 FLA 同法：chunk_o.py `exp(b_g[:,None]-b_g[None,:])`。
 * `M15GP_STAB_NAIVE=1` 退回旧乘法（负向对照，应产生 NaN）；`M15GP_STAB_BADIDX=1` 下标错一位。
 */
__simd_vf__ inline void GammaStrictVF(__ubuf__ float* gcUb, __ubuf__ float* gamUb)
{
    RegTensor<float> gc, rowB, ig, out;
    RegTensor<int32_t> laneIdx;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    MaskReg allI = CreateMask<int32_t, MaskPattern::ALL>();
    LoadAlign(gc, gcUb);
    Arange(laneIdx, static_cast<int32_t>(0));
    for (uint16_t r = 0; r < M15G::GP_BT; ++r) {
        MaskReg m;
        Compares<int32_t, CMPMODE::LT>(m, laneIdx, static_cast<int32_t>(r), allI);
#if M15GP_STAB_BADIDX
        LoadAlign<float, LoadDist::DIST_BRC_B32>(rowB, gcUb + (r == 0u ? 1u : static_cast<uint32_t>(r) - 1u));
#else
        LoadAlign<float, LoadDist::DIST_BRC_B32>(rowB, gcUb + r);
#endif
#if M15GP_STAB_NAIVE
        Exp<float, MaskMergeMode::ZEROING>(rowB, rowB, allF);
        Muls<float>(ig, gc, -1.0f, allF);
        Exp<float, MaskMergeMode::ZEROING>(ig, ig, allF);
        Mul<float, MaskMergeMode::ZEROING>(out, rowB, ig, m);
#else
        Sub(rowB, rowB, gc, allF);
        Exp<float, MaskMergeMode::ZEROING>(out, rowB, m);
#endif
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gamUb + static_cast<uint32_t>(r) * M15G::GP_BT, out, allF);
    }
}

// Γ 行（含对角，j≤r）：同 GammaStrictVF 的指数差形式（对角项 exp(0)=1）。
__simd_vf__ inline void GammaInclVF(__ubuf__ float* gcUb, __ubuf__ float* gamUb)
{
    RegTensor<float> gc, rowB, ig, out;
    RegTensor<int32_t> laneIdx;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    MaskReg allI = CreateMask<int32_t, MaskPattern::ALL>();
    LoadAlign(gc, gcUb);
    Arange(laneIdx, static_cast<int32_t>(0));
    for (uint16_t r = 0; r < M15G::GP_BT; ++r) {
        MaskReg m;
        Compares<int32_t, CMPMODE::LE>(m, laneIdx, static_cast<int32_t>(r), allI);
#if M15GP_STAB_BADIDX
        LoadAlign<float, LoadDist::DIST_BRC_B32>(rowB, gcUb + (r == 0u ? 1u : static_cast<uint32_t>(r) - 1u));
#else
        LoadAlign<float, LoadDist::DIST_BRC_B32>(rowB, gcUb + r);
#endif
#if M15GP_STAB_NAIVE
        Exp<float, MaskMergeMode::ZEROING>(rowB, rowB, allF);
        Muls<float>(ig, gc, -1.0f, allF);
        Exp<float, MaskMergeMode::ZEROING>(ig, ig, allF);
        Mul<float, MaskMergeMode::ZEROING>(out, rowB, ig, m);
#else
        Sub(rowB, rowB, gc, allF);
        Exp<float, MaskMergeMode::ZEROING>(out, rowB, m);
#endif
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gamUb + static_cast<uint32_t>(r) * M15G::GP_BT, out, allF);
    }
}

// dst[t] = exp(ĝ_last − ĝ[t])（指数差；t ≤ last ⇒ ≤1、恒有限）。KT′ 列缩放，替代 egL·ig[t]。
__simd_vf__ inline void BrcDiffExpVF(__ubuf__ float* dUb, __ubuf__ float* gcUb, uint32_t last)
{
    RegTensor<float> gc, gB;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    LoadAlign(gc, gcUb);
#if M15GP_STAB_BADIDX
    last = (last == 0u) ? 1u : (last - 1u);
#endif
    LoadAlign<float, LoadDist::DIST_BRC_B32>(gB, gcUb + last);
    Sub(gB, gB, gc, allF);
    Exp<float, MaskMergeMode::ZEROING>(gB, gB, allF);
    StoreAlign(dUb, gB, allF);
}

// dst[r,:] = src[r,:]·s[r]，行宽 128
__simd_vf__ inline void CopyScale2VF(__ubuf__ float* dUb, uint32_t dS, __ubuf__ float* sUb, __ubuf__ float* srcUb,
                                     uint32_t sS)
{
    RegTensor<float> sr, a0, a1;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t r = 0; r < M15G::GP_BT; ++r) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(sr, sUb + r);
        LoadAlign(a0, srcUb + static_cast<uint32_t>(r) * sS);
        LoadAlign(a1, srcUb + static_cast<uint32_t>(r) * sS + GP_VL);
        Mul(a0, a0, sr, allF);
        Mul(a1, a1, sr, allF);
        StoreAlign(dUb + static_cast<uint32_t>(r) * dS, a0, allF);
        StoreAlign(dUb + static_cast<uint32_t>(r) * dS + GP_VL, a1, allF);
    }
}

// x[r,:] *= s[r]，行宽 128
__simd_vf__ inline void RowScale2VF(__ubuf__ float* xUb, uint32_t xS, __ubuf__ float* sUb)
{
    RegTensor<float> sr, a0, a1;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t r = 0; r < M15G::GP_BT; ++r) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(sr, sUb + r);
        LoadAlign(a0, xUb + static_cast<uint32_t>(r) * xS);
        LoadAlign(a1, xUb + static_cast<uint32_t>(r) * xS + GP_VL);
        Mul(a0, a0, sr, allF);
        Mul(a1, a1, sr, allF);
        StoreAlign(xUb + static_cast<uint32_t>(r) * xS, a0, allF);
        StoreAlign(xUb + static_cast<uint32_t>(r) * xS + GP_VL, a1, allF);
    }
}

// dUb = aUb ⊙ bUb（[BT] 向量）
__simd_vf__ inline void MulVecVF(__ubuf__ float* dUb, __ubuf__ float* aUb, __ubuf__ float* bUb)
{
    RegTensor<float> a, b;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    LoadAlign(a, aUb);
    LoadAlign(b, bUb);
    Mul(a, a, b, allF);
    StoreAlign(dUb, a, allF);
}

// dst[:] = src[:] · c（c 为运行期标量；[BT] 向量）
__simd_vf__ inline void VecScaleConstVF(__ubuf__ float* dUb, __ubuf__ float* sUb, float c)
{
    RegTensor<float> v;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    LoadAlign(v, sUb);
    Muls<float>(v, v, c, allF);
    StoreAlign(dUb, v, allF);
}

// dst[:] = src[0]（把单元素广播成整个 [BT] 向量，供后续 BRC 用对齐地址取）
__simd_vf__ inline void BrcScalarVF(__ubuf__ float* dUb, __ubuf__ float* srcUb)
{
    RegTensor<float> v;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    LoadAlign<float, LoadDist::DIST_BRC_B32>(v, srcUb);
    StoreAlign(dUb, v, allF);
}

// a[i,:] ⊙= b[i,:]，行宽 64
__simd_vf__ inline void MulRows1VF(__ubuf__ float* aUb, uint32_t aS, __ubuf__ float* bUb, uint32_t bS)
{
    RegTensor<float> av, bv;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t r = 0; r < M15G::GP_BT; ++r) {
        LoadAlign(av, aUb + static_cast<uint32_t>(r) * aS);
        LoadAlign(bv, bUb + static_cast<uint32_t>(r) * bS);
        Mul(av, av, bv, allF);
        StoreAlign(aUb + static_cast<uint32_t>(r) * aS, av, allF);
    }
}

// a[i,:] ⊙= (s[i]·b[i,:])，行宽 64（A = β_i·Γs[i,:]⊙kk[i,:]）
__simd_vf__ inline void MulRowBrc1VF(__ubuf__ float* aUb, uint32_t aS, __ubuf__ float* bUb, uint32_t bS,
                                     __ubuf__ float* sUb)
{
    RegTensor<float> sr, av, bv;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t r = 0; r < M15G::GP_BT; ++r) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(sr, sUb + r);
        LoadAlign(av, aUb + static_cast<uint32_t>(r) * aS);
        LoadAlign(bv, bUb + static_cast<uint32_t>(r) * bS);
        Mul(bv, bv, sr, allF);
        Mul(av, av, bv, allF);
        StoreAlign(aUb + static_cast<uint32_t>(r) * aS, av, allF);
    }
}

// x[r,:] *= c，行宽 64，rows 行（ABM = scale·(Γi⊙qk)）
__simd_vf__ inline void ConstScale1VF(__ubuf__ float* xUb, uint32_t xS, float c, uint16_t rows)
{
    RegTensor<float> a;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t r = 0; r < rows; ++r) {
        LoadAlign(a, xUb + static_cast<uint32_t>(r) * xS);
        Muls<float>(a, a, c, allF);
        StoreAlign(xUb + static_cast<uint32_t>(r) * xS, a, allF);
    }
}

// x[r,:] *= sv[:]（行宽 = 64 = 一个寄存器；KT' 每列乘 exp(ĝL−ĝ[t])）
__simd_vf__ inline void ColScaleVF(__ubuf__ float* xUb, uint32_t xS, __ubuf__ float* sv, uint16_t rows)
{
    RegTensor<float> s, a;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    LoadAlign(s, sv);
    for (uint16_t r = 0; r < rows; ++r) {
        LoadAlign(a, xUb + static_cast<uint32_t>(r) * xS);
        Mul(a, a, s, allF);
        StoreAlign(xUb + static_cast<uint32_t>(r) * xS, a, allF);
    }
}

// x[r,:] = brc[0]·x[r,:] + y[r,:]，行宽 128（ST₁ = egL·ST₀ + SDT，按 64 行一档）
__simd_vf__ inline void ScaleAddRowsBrc2VF(__ubuf__ float* xUb, uint32_t xS, __ubuf__ float* yUb, uint32_t yS,
                                           __ubuf__ float* brcUb, uint16_t rows)
{
    RegTensor<float> c, a0, a1, b0, b1;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    LoadAlign<float, LoadDist::DIST_BRC_B32>(c, brcUb);
    for (uint16_t r = 0; r < rows; ++r) {
        LoadAlign(a0, xUb + static_cast<uint32_t>(r) * xS);
        LoadAlign(a1, xUb + static_cast<uint32_t>(r) * xS + GP_VL);
        LoadAlign(b0, yUb + static_cast<uint32_t>(r) * yS);
        LoadAlign(b1, yUb + static_cast<uint32_t>(r) * yS + GP_VL);
        Mul(a0, a0, c, allF);
        Mul(a1, a1, c, allF);
        Add(a0, a0, b0, allF);
        Add(a1, a1, b1, allF);
        StoreAlign(xUb + static_cast<uint32_t>(r) * xS, a0, allF);
        StoreAlign(xUb + static_cast<uint32_t>(r) * xS + GP_VL, a1, allF);
    }
}

// x[r,:] += y[r,:]，行宽 128（o = o_part + AB·d）
__simd_vf__ inline void AddRows2VF(__ubuf__ float* xUb, uint32_t xS, __ubuf__ float* yUb, uint32_t yS, uint16_t rows)
{
    RegTensor<float> a0, a1, b0, b1;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t r = 0; r < rows; ++r) {
        LoadAlign(a0, xUb + static_cast<uint32_t>(r) * xS);
        LoadAlign(a1, xUb + static_cast<uint32_t>(r) * xS + GP_VL);
        LoadAlign(b0, yUb + static_cast<uint32_t>(r) * yS);
        LoadAlign(b1, yUb + static_cast<uint32_t>(r) * yS + GP_VL);
        Add(a0, a0, b0, allF);
        Add(a1, a1, b1, allF);
        StoreAlign(xUb + static_cast<uint32_t>(r) * xS, a0, allF);
        StoreAlign(xUb + static_cast<uint32_t>(r) * xS + GP_VL, a1, allF);
    }
}

/**
 * 下三角前代求解（WY：(I+A)X = RHS；A 严格下三角 [BT,BT] 行主序，X 行宽 128）：
 * X[i,:] -= Σ_{j<i} A[i,j]·X[j,:]。内层上界用**编译期常量** GP_BT（A 的上三角恒 0，多算项贡献 0）
 * —— 3510 实测：内层上界取外层归纳变量时硬件循环少执行一次（丢 j=i−1），见 docs/05 §6.2。
 * 非矩阵乘法（docs/19 §10.2 的裁决映射），留 fp32 VF。
 */
__simd_vf__ inline void TrilSolveVF(__ubuf__ float* xUb, uint32_t xS, __ubuf__ float* aUb, uint32_t aS,
                                    uint16_t rows)
{
    RegTensor<float> acc0, acc1, xj0, xj1, av, xi0, xi1;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t i = 0; i < rows; ++i) {
        Duplicate(acc0, 0.0f);
        Duplicate(acc1, 0.0f);
        for (uint16_t j = 0; j < M15G::GP_BT; ++j) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(av, aUb + static_cast<uint32_t>(i) * aS + j);
            LoadAlign(xj0, xUb + static_cast<uint32_t>(j) * xS);
            LoadAlign(xj1, xUb + static_cast<uint32_t>(j) * xS + GP_VL);
            MulAddDst(acc0, av, xj0, allF);
            MulAddDst(acc1, av, xj1, allF);
        }
        LoadAlign(xi0, xUb + static_cast<uint32_t>(i) * xS);
        LoadAlign(xi1, xUb + static_cast<uint32_t>(i) * xS + GP_VL);
        Sub(xi0, xi0, acc0, allF);
        Sub(xi1, xi1, acc1, allF);
        StoreAlign(xUb + static_cast<uint32_t>(i) * xS, xi0, allF);
        StoreAlign(xUb + static_cast<uint32_t>(i) * xS + GP_VL, xi1, allF);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

/**
 * 转置（UB→UB）：dst[c][r] = src[r][c]；src=[R,C]、dst=[C,R] 行主序；R 必须是 64 的倍数。
 * 输出按 64 lane 整寄存器写（32B 对齐 + 整寄存器写，docs/05 §6.3 表行 b/c）。
 * `Reg::Gather` 按元素索引、可寻址范围 = 本核 UB 窗口（M43 标定）；本段最大下标 16383。
 */
__simd_vf__ inline void TransposeVF(__ubuf__ float* src, __ubuf__ float* dst, uint32_t R, uint32_t C)
{
    RegTensor<float> v;
    RegTensor<int32_t> base, idx;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    MaskReg allI = CreateMask<int32_t, MaskPattern::ALL>();
    Arange(base, static_cast<int32_t>(0));
    for (uint16_t rb = 0; rb < static_cast<uint16_t>(R / GP_VL); ++rb) {
        for (uint16_t c = 0; c < static_cast<uint16_t>(C); ++c) {
            Muls<int32_t>(idx, base, static_cast<int32_t>(C), allI);
            Adds<int32_t>(idx, idx, static_cast<int32_t>(c + static_cast<uint32_t>(rb) * GP_VL * C), allI);
            Gather<float>(v, src, reinterpret_cast<RegTensor<uint32_t>&>(idx), allF);
            StoreAlign<float, StoreDist::DIST_NORM>(
                dst + static_cast<uint32_t>(c) * R + static_cast<uint32_t>(rb) * GP_VL, v, allF);
        }
    }
}

/** dst[c][r] = src[r][c] − sub[c][r]（`dᵀ = uᵀ − (w·S)ᵀ` 一趟完成，省一个 32KB 缓冲）。 */
__simd_vf__ inline void SubTransposeVF(__ubuf__ float* src, __ubuf__ float* sub, __ubuf__ float* dst, uint32_t R,
                                       uint32_t C)
{
    RegTensor<float> v, w;
    RegTensor<int32_t> base, idx;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    MaskReg allI = CreateMask<int32_t, MaskPattern::ALL>();
    Arange(base, static_cast<int32_t>(0));
    for (uint16_t rb = 0; rb < static_cast<uint16_t>(R / GP_VL); ++rb) {
        for (uint16_t c = 0; c < static_cast<uint16_t>(C); ++c) {
            Muls<int32_t>(idx, base, static_cast<int32_t>(C), allI);
            Adds<int32_t>(idx, idx, static_cast<int32_t>(c + static_cast<uint32_t>(rb) * GP_VL * C), allI);
            Gather<float>(v, src, reinterpret_cast<RegTensor<uint32_t>&>(idx), allF);
            LoadAlign(w, sub + static_cast<uint32_t>(c) * R + static_cast<uint32_t>(rb) * GP_VL);
            Sub(v, v, w, allF);
            StoreAlign<float, StoreDist::DIST_NORM>(
                dst + static_cast<uint32_t>(c) * R + static_cast<uint32_t>(rb) * GP_VL, v, allF);
        }
    }
}

// ============================================================
// 3. AIC 段：6 个收缩（fp32 mmad）
// ============================================================

__aicore__ inline void L1Nd2Nz(const AscendC::GlobalTensor<float>& gm, uint64_t elemOff, uint32_t rows,
                               uint32_t cols, uint32_t gmRowStride, AscendC::LocalTensor<float>& l1Dst)
{
    AscendC::Nd2NzParams par = {};
    par.ndNum = 1;
    par.nValue = rows;
    par.dValue = cols;
    par.srcNdMatrixStride = 0;
    par.srcDValue = gmRowStride;
    par.dstNzC0Stride = rows;
    par.dstNzNStride = 1;
    par.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Dst, gm[elemOff], par);
}

__aicore__ inline void L0LoadA(const AscendC::LocalTensor<float>& l0a, const AscendC::LocalTensor<float>& l1a,
                               uint32_t mm, uint32_t kk)
{
    AscendC::LoadData2DParamsV2 lp = {};
    lp.mStartPosition = 0;
    lp.kStartPosition = 0;
    lp.mStep = static_cast<uint16_t>(mm / M15G::GP_CUBE_M);
    lp.kStep = static_cast<uint16_t>(kk / M15G::GP_C0F);
    lp.srcStride = static_cast<int32_t>(mm / M15G::GP_CUBE_M);
    lp.dstStride = static_cast<uint16_t>(mm / M15G::GP_CUBE_M);
    lp.ifTranspose = false;
    lp.sid = 0;
    AscendC::LoadData(l0a, l1a, lp);
}

__aicore__ inline void L0LoadB(const AscendC::LocalTensor<float>& l0b, const AscendC::LocalTensor<float>& l1b,
                               uint32_t nn, uint32_t kk)
{
    AscendC::LoadData2DParams lb = {};
    lb.startIndex = 0;
    lb.repeatTimes = static_cast<uint8_t>((nn / M15G::GP_CUBE_M) * (kk / M15G::GP_C0F));
    lb.srcStride = 1;
    lb.dstGap = 0;
    lb.ifTranspose = false;
    AscendC::LoadData(l0b, l1b, lb);
}

__aicore__ inline void MmadF32(const AscendC::LocalTensor<float>& l0c, const AscendC::LocalTensor<float>& l0a,
                               const AscendC::LocalTensor<float>& l0b, uint32_t mm, uint32_t nn, uint32_t kk,
                               bool init)
{
    AscendC::MmadParams mp = {};
    mp.m = static_cast<uint16_t>(mm);
    mp.n = static_cast<uint16_t>(nn);
    mp.k = static_cast<uint16_t>(kk);
    mp.cmatrixInitVal = init;
    mp.cmatrixSource = false;
    AscendC::Mmad(l0c, l0a, l0b, mp);
}

__aicore__ inline void FixpipeGm(AscendC::GlobalTensor<float>& cGm, uint64_t elemOff,
                                 const AscendC::LocalTensor<float>& l0c, uint32_t mm, uint32_t nn,
                                 uint32_t dstStride)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp = {};
    fp.nSize = static_cast<uint16_t>(nn);
    fp.mSize = static_cast<uint16_t>(mm);
    fp.srcStride = static_cast<uint16_t>(mm);
    fp.dstStride = dstStride;
    fp.reluScalar = 0;
    fp.vectorRelu = 0;
    fp.deqScalar = 0;
    static constexpr AscendC::FixpipeConfig kFixGm(AscendC::CO2Layout::ROW_MAJOR, false);
    AscendC::Fixpipe<float, float, kFixGm>(cGm[elemOff], l0c, fp);
}

class GdnPrefillAic {
public:
    __aicore__ inline GdnPrefillAic() {}

    __aicore__ inline void Init(const GdnPrefillArgs& a, uint32_t bid)
    {
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.k));
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.q));
        scGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.scratch));
        m_ = a.m;
        heads_ = a.heads;
        qkStride_ = (a.qkStride == 0) ? a.m : a.qkStride;
        nAic_ = AscendC::GetBlockNum();
        if (nAic_ == 0) {
            nAic_ = 1;
        }
        bid_ = bid;
    }

    __aicore__ inline void Process()
    {
        if (bid_ >= M15G::GP_PAIRS) {
            return;   // 本 AICore 没有 head 对
        }
        const uint32_t nChunk = (m_ + M15G::GP_BT - 1) / M15G::GP_BT;
        for (uint32_t g = bid_; g < M15G::GP_PAIRS; g += nAic_) {
            for (uint32_t c = 0; c < nChunk; ++c) {
                t0_ = c * M15G::GP_BT;
                grp_ = g;
                CcWait<PIPE_MTE2>(M15G::GP_FLAG_GO1);
                Job1();
                CcSet<PIPE_FIX>(M15G::GP_FLAG_DONE1);
#if M15GP_J1_ONLY
                continue;   // 诊断收窄：只跑 J1（见 m23_gdn_prefill/README.md §4c）
#endif
                CcWait<PIPE_MTE2>(M15G::GP_FLAG_GO2);
                Job2();
                CcSet<PIPE_FIX>(M15G::GP_FLAG_DONE2);
#if M15GP_J1J2_ONLY
                continue;   // 诊断收窄：只跑 J1+J2
#endif
                CcWait<PIPE_MTE2>(M15G::GP_FLAG_GO3);
                Job3();
                CcSet<PIPE_FIX>(M15G::GP_FLAG_DONE3);
            }
        }
    }

private:
    // 每 head 的 L1 子偏移（字节）
    __aicore__ inline uint32_t L1Kq(uint32_t h) const
    {
        return M15G::GP_L1_KQ_K + h * (M15G::GP_L1_KQ_H * 4);
    }
    __aicore__ inline uint32_t L1St(uint32_t h) const { return M15G::GP_L1_ST_K + h * (M15G::GP_L1_ST_H * 4); }
    __aicore__ inline uint32_t L1W(uint32_t h) const { return M15G::GP_L1_W_K + h * (M15G::GP_L1_W_H * 4); }
    __aicore__ inline uint32_t L1J3(uint32_t h, uint32_t what) const
    {
        return M15G::GP_L1_J3_K + h * (M15G::GP_L1_J3_H * 4) + what * 4;
    }
    __aicore__ inline uint64_t Slot(uint32_t h) const
    {
        return static_cast<uint64_t>(bid_) * M15G::GP_SLOT_BYTES +
               static_cast<uint64_t>(h) * M15G::GP_SC_H_BYTES;
    }
    __aicore__ inline uint32_t Hv(uint32_t h) const { return 2u * grp_ + h; }

    /**
     * 一次 mmad：A/B 已在 L1；**L0 与 L0C 都由 BufferID 成对交接**，结果 Fixpipe 落 GM。
     * L0C 的交接是 **M ↔ FIXP 成对**：Mmad 前 M 取 L0C 所有权（挡上一 tile 的 Fixpipe 读 = WAR），
     * Mmad 后 M 释放（RAW：Fixpipe 必须等 Mmad 完成）—— 与仓内既有站点（m1 / m3 / m11 / m13 /
     * m14 / m15_gdn_layer / m15_hc_layer / m15_moe_layer / m17 / m20，共 10 处）同形；
     * 只用 PipeBarrier 不足以建立 M→FIX 与 FIX→M 的跨 pipe 依赖
     * （probe_cube_fp32 的实测：整张表滞后一轮）。
     */
    __aicore__ inline void DoMmad(uint32_t l1aOff, uint32_t aL1Elems, uint32_t l1bOff, uint32_t bL1Elems,
                                  uint32_t mm, uint32_t nn, uint32_t kk, uint64_t cOff, uint32_t cStride)
    {
        AscendC::LocalTensor<float> l1a(TPosition::A1, l1aOff, aL1Elems);
        AscendC::LocalTensor<float> l1b(TPosition::B1, l1bOff, bL1Elems);
        AscendC::LocalTensor<float> l0a(TPosition::A2, 0, M15G::GP_L0A_MAX_ELEMS);
        AscendC::LocalTensor<float> l0b(TPosition::B2, 0, M15G::GP_L0B_MAX_ELEMS);
        AscendC::LocalTensor<float> l0c(TPosition::CO1, 0, M15G::GP_L0C_MAX_ELEMS);

        BufAcquire<PIPE_MTE1>(M15G::GP_BUF_L0);
        BufAcquire<PIPE_MTE1>(M15G::GP_BUF_L1);   // 等 MTE2 把 L1 大包写完（release mode=false 的对侧 get）
        L0LoadA(l0a, l1a, mm, kk);
        L0LoadB(l0b, l1b, nn, kk);
        BufRelease<PIPE_MTE1>(M15G::GP_BUF_L1);
        BufRelease<PIPE_MTE1>(M15G::GP_BUF_L0);
        BufAcquire<PIPE_M>(M15G::GP_BUF_L0C);   // 新增：M 先取 L0C 所有权（挡上一 tile 的 Fixpipe 读 = WAR）
        BufAcquire<PIPE_M>(M15G::GP_BUF_L0);
        MmadF32(l0c, l0a, l0b, mm, nn, kk, true);
        BufRelease<PIPE_M>(M15G::GP_BUF_L0);
        BufRelease<PIPE_M>(M15G::GP_BUF_L0C);   // 新增：L0C 结果就绪（RAW：Fixpipe 必须等 Mmad 完成）
        BufAcquire<PIPE_FIX>(M15G::GP_BUF_L0C);
        FixpipeGm(scGm_, cOff, l0c, mm, nn, cStride);
        BufRelease<PIPE_FIX>(M15G::GP_BUF_L0C);
    }

    /** 一个 head 的操作数装载（J1/J2/J3 各自调用一次），全部走 BUF_L1 令牌。 */
    __aicore__ inline void Job1()
    {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t hk = Hv(h) / M15G::GP_GROUP;
            const uint64_t kqOff = static_cast<uint64_t>(hk) * qkStride_ * M15G::GP_DK +
                                   static_cast<uint64_t>(t0_) * M15G::GP_DK;
            BufAcquire<PIPE_MTE2>(M15G::GP_BUF_L1);
            {
                AscendC::LocalTensor<float> l1k(TPosition::A1, L1Kq(h), M15G::GP_BT * M15G::GP_DK);
                AscendC::LocalTensor<float> l1q(TPosition::A1, L1Kq(h) + M15G::GP_BT * M15G::GP_DK * 4,
                                                M15G::GP_BT * M15G::GP_DK);
                L1Nd2Nz(kGm_, kqOff, M15G::GP_BT, M15G::GP_DK, M15G::GP_DK, l1k);
                L1Nd2Nz(qGm_, kqOff, M15G::GP_BT, M15G::GP_DK, M15G::GP_DK, l1q);
            }
            BufRelease<PIPE_MTE2>(M15G::GP_BUF_L1);
            // M1: kk = k·kᵀ（A=k，B=k）
            DoMmad(L1Kq(h), M15G::GP_BT * M15G::GP_DK, L1Kq(h), M15G::GP_BT * M15G::GP_DK, M15G::GP_BT,
                   M15G::GP_BT, M15G::GP_DK, (Slot(h) + M15G::GP_SC_KK) / 4, M15G::GP_BT);
            // M2: qk = q·kᵀ（A=q，B=k）
            DoMmad(L1Kq(h) + M15G::GP_BT * M15G::GP_DK * 4, M15G::GP_BT * M15G::GP_DK, L1Kq(h),
                   M15G::GP_BT * M15G::GP_DK, M15G::GP_BT, M15G::GP_BT, M15G::GP_DK,
                   (Slot(h) + M15G::GP_SC_QK) / 4, M15G::GP_BT);
        }
    }

    __aicore__ inline void Job2()
    {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t hk = Hv(h) / M15G::GP_GROUP;
            const uint64_t kqOff = static_cast<uint64_t>(hk) * qkStride_ * M15G::GP_DK +
                                   static_cast<uint64_t>(t0_) * M15G::GP_DK;
            const uint64_t sl = Slot(h);
            const uint32_t ch = t0_ / M15G::GP_BT;
            const bool skipNz = (M15GP_J2_NZ_SKIP_FROM > 0 && ch >= M15GP_J2_NZ_SKIP_FROM);
            BufAcquire<PIPE_MTE2>(M15G::GP_BUF_L1);
            {
                AscendC::LocalTensor<float> l1q(TPosition::A1, L1Kq(h) + M15G::GP_BT * M15G::GP_DK * 4,
                                                M15G::GP_BT * M15G::GP_DK);
                AscendC::LocalTensor<float> l1st(TPosition::A1, L1St(h), M15G::GP_DV * M15G::GP_DK);
                AscendC::LocalTensor<float> l1w(TPosition::B1, L1W(h), M15G::GP_BT * M15G::GP_DK);
                // WHICH：0=三条全跳；1=只跳 q 重装；2=只跳 ST；3=只跳 W（见 README §4c5 的交叉验证表）
                if (!(skipNz && (M15GP_J2_NZ_WHICH == 0 || M15GP_J2_NZ_WHICH == 1 ||
                                 M15GP_J2_NZ_WHICH == 5 || M15GP_J2_NZ_WHICH == 6))) {
                    L1Nd2Nz(qGm_, kqOff, M15G::GP_BT, M15G::GP_DK, M15G::GP_DK, l1q);
                }
                if (!(skipNz && (M15GP_J2_NZ_WHICH == 0 || M15GP_J2_NZ_WHICH == 2 ||
                                 M15GP_J2_NZ_WHICH == 4 || M15GP_J2_NZ_WHICH == 6))) {
                    L1Nd2Nz(scGm_, (sl + M15G::GP_SC_ST) / 4, M15G::GP_DV, M15G::GP_DK, M15G::GP_DK, l1st);
                }
                if (!(skipNz && (M15GP_J2_NZ_WHICH == 0 || M15GP_J2_NZ_WHICH == 3 ||
                                 M15GP_J2_NZ_WHICH == 4 || M15GP_J2_NZ_WHICH == 5))) {
                    L1Nd2Nz(scGm_, (sl + M15G::GP_SC_W) / 4, M15G::GP_BT, M15G::GP_DK, M15G::GP_DK, l1w);
                }
            }
            BufRelease<PIPE_MTE2>(M15G::GP_BUF_L1);
            // M3: (w·S)ᵀ = ST·wᵀ（A=ST [M=DV,K=DK]，B=w [N=BT,K=DK]）
            DoMmad(L1St(h), M15G::GP_DV * M15G::GP_DK, L1W(h), M15G::GP_BT * M15G::GP_DK, M15G::GP_DV,
                   M15G::GP_BT, M15G::GP_DK, (Slot(h) + M15G::GP_SC_WT) / 4, M15G::GP_BT);
            // M4: q·S = q·STᵀ（A=q [M=BT,K=DK]，B=ST [N=DV,K=DK]）
            DoMmad(L1Kq(h) + M15G::GP_BT * M15G::GP_DK * 4, M15G::GP_BT * M15G::GP_DK, L1St(h),
                   M15G::GP_DV * M15G::GP_DK, M15G::GP_BT, M15G::GP_DV, M15G::GP_DK,
                   (Slot(h) + M15G::GP_SC_QS) / 4, M15G::GP_DV);
        }
    }

    __aicore__ inline void Job3()
    {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint64_t sl = Slot(h);
            BufAcquire<PIPE_MTE2>(M15G::GP_BUF_L1);
            {
                AscendC::LocalTensor<float> l1kt(TPosition::B1, L1J3(h, M15G::GP_L1_J3_KT) / 1,
                                                 M15G::GP_DK * M15G::GP_BT);
                AscendC::LocalTensor<float> l1dt(TPosition::A1, L1J3(h, M15G::GP_L1_J3_DT) / 1,
                                                 M15G::GP_DV * M15G::GP_BT);
                AscendC::LocalTensor<float> l1ab(TPosition::A1, L1J3(h, M15G::GP_L1_J3_ABM) / 1,
                                                 M15G::GP_BT * M15G::GP_BT);
                L1Nd2Nz(scGm_, (sl + M15G::GP_SC_KT) / 4, M15G::GP_DK, M15G::GP_BT, M15G::GP_BT, l1kt);
                L1Nd2Nz(scGm_, (sl + M15G::GP_SC_DT) / 4, M15G::GP_DV, M15G::GP_BT, M15G::GP_BT, l1dt);
                L1Nd2Nz(scGm_, (sl + M15G::GP_SC_ABM) / 4, M15G::GP_BT, M15G::GP_BT, M15G::GP_BT, l1ab);
            }
            BufRelease<PIPE_MTE2>(M15G::GP_BUF_L1);
            // M5: AB·d = ABM·DTᵀ（A=ABM [M=BT,K=BT]，B=DT [N=DV,K=BT]）
            DoMmad(L1J3(h, M15G::GP_L1_J3_ABM), M15G::GP_BT * M15G::GP_BT, L1J3(h, M15G::GP_L1_J3_DT),
                   M15G::GP_DV * M15G::GP_BT, M15G::GP_BT, M15G::GP_DV, M15G::GP_BT,
                   (Slot(h) + M15G::GP_SC_ABD) / 4, M15G::GP_DV);
            // M6: (kᵀ·DP)ᵀ = DT·KT'ᵀ（A=DT [M=DV,K=BT]，B=KT' [N=DK,K=BT]）
            DoMmad(L1J3(h, M15G::GP_L1_J3_DT), M15G::GP_DV * M15G::GP_BT, L1J3(h, M15G::GP_L1_J3_KT),
                   M15G::GP_DK * M15G::GP_BT, M15G::GP_DV, M15G::GP_DK, M15G::GP_BT,
                   (Slot(h) + M15G::GP_SC_SDT) / 4, M15G::GP_DK);
        }
    }

private:
    uint64_t m_ = 0;
    uint32_t heads_ = 0;
    uint32_t qkStride_ = 0;
    uint32_t nAic_ = 0;
    uint32_t bid_ = 0;
    uint32_t t0_ = 0;
    uint32_t grp_ = 0;
    AscendC::GlobalTensor<float> kGm_, qGm_, scGm_;
};

// ============================================================
// 3b. AIV 段：fp32 状态权威 + chunk 内 VF + 操作数上传/结果消费
// ============================================================
//
// 令牌（BufferID，核内）：
//   GP_BUF_BLK —— MTE2 ↔ V 的「一个 chunk 工作集」交接（m18 同款轮转）
//   GP_BUF_UP  —— **会被上传给 AIC 的所有 UB 缓冲**的生存期令牌（写者与 MTE3 读者成对）。
//                 「V 写 → 阻塞释放（mode=false）→ MTE3 搬」与「MTE3 读 → 下一个写者 acquire」，
//                 是 docs/05 §6.1 规则 ⓔ 的落盘通路；每处写者（V 或 MTE2）与 MTE3 都成对 acquire。
// 顺序纪律：任何 scope 一律先 BLK 后 UP（避免死锁）。

class GdnPrefillAiv {
public:
    __aicore__ inline GdnPrefillAiv() {}

    __aicore__ inline void Init(const GdnPrefillArgs& a, uint32_t bid)
    {
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.v));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.g));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.beta));
        h0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.h0));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.out));
        htGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.ht));
        scGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.scratch));
        m_ = a.m;
        heads_ = a.heads;
        qkStride_ = (a.qkStride == 0) ? a.m : a.qkStride;
        tp_ = a.tp;
        scale_ = a.scale;
        nAic_ = AscendC::GetBlockNum();   // mix(1,2) 下 AIV 视角返回 AIC 数（docs/05 §6.1）
        if (nAic_ == 0) {
            nAic_ = 1;
        }
        bid_ = bid;
    }

    __aicore__ inline void Process()
    {
        const uint32_t pair = bid_ >> 1;
        const uint32_t role = bid_ & 1u;
        if (pair >= M15G::GP_PAIRS) {
            return;   // 本 AICore 没有 head 对（head 数 48 < 2×28）
        }
        AscendC::LocalTensor<float> ubAll(TPosition::VECCALC, 0, M15G::GP_UB_BYTES / 4);
        ub_ = reinterpret_cast<__ubuf__ float*>(ubAll.GetPhyAddr());
        for (uint32_t g = pair; g < M15G::GP_PAIRS; g += nAic_) {
            const uint32_t hv = 2u * g + role;
            if (hv >= heads_) {
                continue;
            }
            ProcessHead(hv, g);
        }
    }

private:
    __aicore__ inline AscendC::LocalTensor<float> Lt(uint32_t byteOff, uint32_t elems)
    {
        AscendC::LocalTensor<float> t(TPosition::VECCALC, byteOff, elems);
        return t;
    }
    __aicore__ inline __ubuf__ float* P(uint32_t byteOff) { return ub_ + byteOff / 4; }

    __aicore__ inline void UpAcqV()
    {
        BufAcquire<PIPE_V>(M15G::GP_BUF_UP);
    }
    __aicore__ inline void UpRelV()
    {
        BufRelease<PIPE_V>(M15G::GP_BUF_UP);
    }
    __aicore__ inline void UpAcqMte2()
    {
        BufAcquire<PIPE_MTE2>(M15G::GP_BUF_UP);
    }
    __aicore__ inline void UpRelMte2()
    {
        BufRelease<PIPE_MTE2>(M15G::GP_BUF_UP);
    }
    __aicore__ inline void UpAcqMte3()
    {
        BufAcquire<PIPE_MTE3>(M15G::GP_BUF_UP);
    }
    __aicore__ inline void UpRelMte3()
    {
        BufRelease<PIPE_MTE3>(M15G::GP_BUF_UP);
    }

    __aicore__ inline void ProcessHead(uint32_t hv, uint32_t g)
    {
        const uint32_t hk = hv / M15G::GP_GROUP;
        const uint32_t nChunk = (m_ + M15G::GP_BT - 1) / M15G::GP_BT;
        const uint64_t slot = static_cast<uint64_t>(g) * M15G::GP_SLOT_BYTES +
                              static_cast<uint64_t>(hv & 1u) * M15G::GP_SC_H_BYTES;
        LoadState(hv);
        for (uint32_t c = 0; c < nChunk; ++c) {
            const uint32_t t0 = c * M15G::GP_BT;
            const uint32_t cv = (m_ - t0 < M15G::GP_BT) ? (m_ - t0) : M15G::GP_BT;
            RunChunk(hv, hk, cv, t0, slot);
        }
        StoreState(hv);
    }

    /** h0 [DK,DV] → ST = h0ᵀ（分两块 64 列窗，避免额外 64KB 缓冲）。 */
    __aicore__ inline void LoadState(uint32_t hv)
    {
        __ubuf__ float* stUb = P(M15G::GP_UB_ST);
        __ubuf__ float* tmp = P(M15G::GP_UB_K);
        const uint64_t hBase = static_cast<uint64_t>(hv) * M15G::GP_DK * M15G::GP_DV;
        for (uint32_t cb = 0; cb < 2; ++cb) {
            BufAcquire<PIPE_MTE2>(M15G::GP_BUF_BLK);
            UpAcqMte2();
            {
                // h0 的 [128 行, 64 列] 列窗（行距 128 元素）：128 块 × 256B，源块间空 8 块
                const AscendC::DataCopyParams hp{static_cast<uint16_t>(M15G::GP_DK), 8, 8, 0};
                DataCopy(Lt(M15G::GP_UB_K, M15G::GP_DK * GP_VL), h0Gm_[hBase + cb * GP_VL], hp);
            }
            UpRelMte2();
            BufRelease<PIPE_MTE2>(M15G::GP_BUF_BLK);
            BufAcquire<PIPE_V>(M15G::GP_BUF_BLK);
            UpAcqV();
            TransposeVF(tmp, stUb + cb * GP_VL * M15G::GP_DK, M15G::GP_DK, GP_VL);
            MemBarVL();
            UpRelV();
            BufRelease<PIPE_V>(M15G::GP_BUF_BLK);
        }
    }

    /** ht = STᵀ（同样分两块 64 列窗写回）。 */
    __aicore__ inline void StoreState(uint32_t hv)
    {
        __ubuf__ float* stUb = P(M15G::GP_UB_ST);
        __ubuf__ float* tmp = P(M15G::GP_UB_K);
        const uint64_t hBase = static_cast<uint64_t>(hv) * M15G::GP_DK * M15G::GP_DV;
        for (uint32_t cb = 0; cb < 2; ++cb) {
            BufAcquire<PIPE_V>(M15G::GP_BUF_BLK);
            UpAcqV();
            // ST 的行 stride 是 DK(=128)，本半块取 ST 的行 [cb*64, cb*64+64)（[64,128] 行主序，
            // 行 stride 恰为 DK）⇒ 转置成 [128,64] 正好是 ht 的列窗 [cb*64, cb*64+64)。
            // 原实现传 (R=DV, C=VL) 与源的 stride 不符（把 [128 行 stride 128] 当 [128 行 stride 64] 读）
            // ⇒ 末态整体错位（o 对而 ht 错，正是"写回路径"而非状态推进的形态）。
            TransposeVF(stUb + cb * GP_VL * M15G::GP_DK, tmp, GP_VL, M15G::GP_DK);
            MemBarVL();
            UpRelV();
            BufRelease<PIPE_V>(M15G::GP_BUF_BLK);
            BufAcquire<PIPE_MTE3>(M15G::GP_BUF_BLK);
            UpAcqMte3();
            {
                // 目的：ht 的 [128 行, 64 列] 列窗（行距 128 元素）：源连续、目的块间空 8 块
                const AscendC::DataCopyParams hp{static_cast<uint16_t>(M15G::GP_DK), 8, 0, 8};
                DataCopy(htGm_[hBase + cb * GP_VL], Lt(M15G::GP_UB_K, M15G::GP_DK * GP_VL), hp);
            }
            UpRelMte3();
            BufRelease<PIPE_MTE3>(M15G::GP_BUF_BLK);
        }
    }

    __aicore__ inline void RunChunk(uint32_t hv, uint32_t hk, uint32_t cv, uint32_t t0, uint64_t slot)
    {
        __ubuf__ float* stUb = P(M15G::GP_UB_ST);
        __ubuf__ float* kUb = P(M15G::GP_UB_K);
        __ubuf__ float* vuUb = P(M15G::GP_UB_VU);
        __ubuf__ float* rhUb = P(M15G::GP_UB_RH);
        __ubuf__ float* xxUb = P(M15G::GP_UB_XX);
        __ubuf__ float* qsUb = P(M15G::GP_UB_QS);
        __ubuf__ float* aUb = P(M15G::GP_UB_A);
        __ubuf__ float* scG = P(M15G::GP_UB_SC_G);
        __ubuf__ float* scB = P(M15G::GP_UB_SC_B);
        __ubuf__ float* scGC = P(M15G::GP_UB_SC_GC);
        __ubuf__ float* scEG = P(M15G::GP_UB_SC_EG);
        __ubuf__ float* scSG = P(M15G::GP_UB_SC_SG);
        __ubuf__ float* scSE = P(M15G::GP_UB_SC_SE);
        __ubuf__ float* scKT = P(M15G::GP_UB_SC_KT);
        __ubuf__ float* scL = P(M15G::GP_UB_SC_L);

        // ---------- (A) MTE2：k / v / g / β ----------
        PipeBarrier<PIPE_MTE2>();
        BufAcquire<PIPE_MTE2>(M15G::GP_BUF_BLK);
        UpAcqMte2();
        {
            const AscendC::DataCopyParams kp{static_cast<uint16_t>(cv), (M15G::GP_DK * 4) / 32, 0, 0};
            const AscendC::DataCopyParams vp{static_cast<uint16_t>(cv), (M15G::GP_DV * 4) / 32, 0, 0};
            const uint64_t kOff = static_cast<uint64_t>(hk) * qkStride_ * M15G::GP_DK +
                                  static_cast<uint64_t>(t0) * M15G::GP_DK;
            const uint64_t vOff =
                static_cast<uint64_t>(hv) * m_ * M15G::GP_DV + static_cast<uint64_t>(t0) * M15G::GP_DV;
            const uint32_t gLen = ((cv + 7) / 8) * 8;
            const AscendC::DataCopyParams gp{1, static_cast<uint16_t>(gLen / 8), 0, 0};
            DataCopy(Lt(M15G::GP_UB_K, M15G::GP_BT * M15G::GP_DK), kGm_[kOff], kp);
            DataCopy(Lt(M15G::GP_UB_VU, M15G::GP_BT * M15G::GP_DV), vGm_[vOff], vp);
            DataCopy(Lt(M15G::GP_UB_SC_G, M15G::GP_BT), gGm_[static_cast<uint64_t>(hv) * tp_ + t0], gp);
            DataCopy(Lt(M15G::GP_UB_SC_B, M15G::GP_BT), bGm_[static_cast<uint64_t>(hv) * tp_ + t0], gp);
        }
        UpRelMte2();
        BufRelease<PIPE_MTE2>(M15G::GP_BUF_BLK);
        // J1 只吃 GM 输入平面 ⇒ 立刻放行，让 AIC 与下面这段 VF 重叠（KK/QK 在 DONE1 之前不被写）
        CcSet<PIPE_V>(M15G::GP_FLAG_GO1);

        // ---------- (B) VF 前段：尾块清零 / kᵀ / ĝ,eg / Γs / 两个右端 ----------
        BufAcquire<PIPE_V>(M15G::GP_BUF_BLK);
        UpAcqV();
        if (cv < M15G::GP_BT) {
            ZeroTailVF(scG, cv);
            ZeroTailVF(scB, cv);
            ZeroVF(kUb + cv * M15G::GP_DK, (M15G::GP_BT - cv) * M15G::GP_DK);
            ZeroVF(vuUb + cv * M15G::GP_DV, (M15G::GP_BT - cv) * M15G::GP_DV);
            MemBarVL();
        }
        TransposeVF(kUb, xxUb, M15G::GP_BT, M15G::GP_DK);   // KT = kᵀ（[DK,BT]）
        MemBarVL();
        CumSumExpVF(scG, scGC, scEG);
        MemBarVL();
        BrcScalarVF(scL, scEG + (cv - 1));                  // scL = egL = exp(ĝL)（状态衰减，单因子安全）
        MemBarVL();
        BrcDiffExpVF(scKT, scGC, cv - 1);                   // scKT[t] = exp(ĝL−ĝ[t])（指数差）
        MemBarVL();
        ColScaleVF(xxUb, M15G::GP_BT, scKT, M15G::GP_DK);   // KT' = KT·diag(exp(ĝL−ĝ))
        MemBarVL();
        VecScaleConstVF(scSE, scEG, scale_);                // scSE = scale·eg（逐 lane）
        MemBarVL();
        GammaStrictVF(scGC, aUb);                           // Γs = exp(ĝ_i−ĝ_j)（严格下三角）
        MemBarVL();
        MulVecVF(scSG, scB, scEG);                          // sg = β⊙eg
        MemBarVL();
        CopyScale2VF(rhUb, M15G::GP_DK, scSG, kUb, M15G::GP_DK);   // w 的右端 = β⊙eg⊙k
        MemBarVL();
        CopyScale2VF(vuUb, M15G::GP_DV, scB, vuUb, M15G::GP_DV);   // u 的右端 = β⊙v（原地）
        MemBarVL();
        UpRelV();
        BufRelease<PIPE_V>(M15G::GP_BUF_BLK);

        // ---------- 上传 KT'（J3 的 B 操作数）----------
        BufAcquire<PIPE_MTE3>(M15G::GP_BUF_BLK);
        UpAcqMte3();
        DataCopy(scGm_[(slot + M15G::GP_SC_KT) / 4], Lt(M15G::GP_UB_XX, M15G::GP_DK * M15G::GP_BT), ktCp_);
        UpRelMte3();
        BufRelease<PIPE_MTE3>(M15G::GP_BUF_BLK);

        // ---------- 等 J1（kk / qk）----------
        CcWait<PIPE_MTE2>(M15G::GP_FLAG_DONE1);
        PipeBarrier<PIPE_MTE2>();
        BufAcquire<PIPE_MTE2>(M15G::GP_BUF_BLK);
        UpAcqMte2();
        {
            const AscendC::DataCopyParams ap{static_cast<uint16_t>(M15G::GP_BT), (M15G::GP_BT * 4) / 32, 0, 0};
            DataCopy(Lt(M15G::GP_UB_K, M15G::GP_BT * M15G::GP_BT), scGm_[(slot + M15G::GP_SC_KK) / 4], ap);
            DataCopy(Lt(M15G::GP_UB_K + M15G::GP_BT * M15G::GP_BT * 4, M15G::GP_BT * M15G::GP_BT),
                     scGm_[(slot + M15G::GP_SC_QK) / 4], ap);
        }
        UpRelMte2();
        BufRelease<PIPE_MTE2>(M15G::GP_BUF_BLK);
        BufAcquire<PIPE_V>(M15G::GP_BUF_BLK);
        UpAcqV();
        MulRowBrc1VF(aUb, M15G::GP_BT, kUb, M15G::GP_BT, scB);       // A = β_i·Γs[i,:]⊙kk[i,:]
        MemBarVL();
        TrilSolveVF(vuUb, M15G::GP_DV, aUb, M15G::GP_BT, M15G::GP_BT);   // u = (I+A)⁻¹(β⊙v)
        MemBarVL();
        TrilSolveVF(rhUb, M15G::GP_DK, aUb, M15G::GP_BT, M15G::GP_BT);   // w = (I+A)⁻¹(β⊙eg⊙k)
        MemBarVL();
        UpRelV();
        BufRelease<PIPE_V>(M15G::GP_BUF_BLK);

#if M15GP_J1_ONLY
        // 诊断收窄：只跑 J1 —— 到此为止所有 BufferID 令牌都已 release（见 README §4c）
        return;
#endif
        // ---------- 上传 ST + W（J2 的操作数）----------
        BufAcquire<PIPE_MTE3>(M15G::GP_BUF_BLK);
        UpAcqMte3();
        {
            const AscendC::DataCopyParams stp{static_cast<uint16_t>(M15G::GP_DV), (M15G::GP_DK * 4) / 32, 0, 0};
            const AscendC::DataCopyParams wp{static_cast<uint16_t>(M15G::GP_BT), (M15G::GP_DK * 4) / 32, 0, 0};
            DataCopy(scGm_[(slot + M15G::GP_SC_ST) / 4], Lt(M15G::GP_UB_ST, M15G::GP_DV * M15G::GP_DK), stp);
            DataCopy(scGm_[(slot + M15G::GP_SC_W) / 4], Lt(M15G::GP_UB_RH, M15G::GP_BT * M15G::GP_DK), wp);
        }
        UpRelMte3();
        BufRelease<PIPE_MTE3>(M15G::GP_BUF_BLK);
        CcSet<PIPE_MTE3>(M15G::GP_FLAG_GO2);

        // ---------- 等 J2（(w·S)ᵀ / q·S）----------
        CcWait<PIPE_MTE2>(M15G::GP_FLAG_DONE2);
        PipeBarrier<PIPE_MTE2>();
        BufAcquire<PIPE_MTE2>(M15G::GP_BUF_BLK);
        UpAcqMte2();
        {
            const AscendC::DataCopyParams wp{static_cast<uint16_t>(M15G::GP_DV), (M15G::GP_BT * 4) / 32, 0, 0};
            const AscendC::DataCopyParams qp{static_cast<uint16_t>(M15G::GP_BT), (M15G::GP_DV * 4) / 32, 0, 0};
            DataCopy(Lt(M15G::GP_UB_RH, M15G::GP_DV * M15G::GP_BT), scGm_[(slot + M15G::GP_SC_WT) / 4], wp);
            DataCopy(Lt(M15G::GP_UB_QS, M15G::GP_BT * M15G::GP_DV), scGm_[(slot + M15G::GP_SC_QS) / 4], qp);
        }
        UpRelMte2();
        BufRelease<PIPE_MTE2>(M15G::GP_BUF_BLK);
        BufAcquire<PIPE_V>(M15G::GP_BUF_BLK);
        UpAcqV();
        SubTransposeVF(vuUb, rhUb, xxUb, M15G::GP_BT, M15G::GP_DV);   // dᵀ = uᵀ − (w·S)ᵀ（覆盖 KT 区）
        MemBarVL();
        RowScale2VF(qsUb, M15G::GP_DV, scSE);                          // o_part = (scale·eg)_i·qS[i,:]
        MemBarVL();
        GammaInclVF(scGC, aUb);                                        // Γi = exp(ĝ_i−ĝ_j)（A 已消费）
        MemBarVL();
        MulRows1VF(aUb, M15G::GP_BT, kUb + M15G::GP_BT * M15G::GP_BT, M15G::GP_BT);   // ⊙qk
        MemBarVL();
        ConstScale1VF(aUb, M15G::GP_BT, scale_, M15G::GP_BT);          // ABM = scale·(Γi⊙qk)
        MemBarVL();
        UpRelV();
        BufRelease<PIPE_V>(M15G::GP_BUF_BLK);

#if M15GP_J1J2_ONLY
        return;   // 诊断收窄：只跑 J1+J2（此点所有 BufferID 令牌均已 release）
#endif
        // ---------- 上传 DT + ABM ----------
        BufAcquire<PIPE_MTE3>(M15G::GP_BUF_BLK);
        UpAcqMte3();
        DataCopy(scGm_[(slot + M15G::GP_SC_DT) / 4], Lt(M15G::GP_UB_XX, M15G::GP_DV * M15G::GP_BT), ktCp_);
        DataCopy(scGm_[(slot + M15G::GP_SC_ABM) / 4], Lt(M15G::GP_UB_A, M15G::GP_BT * M15G::GP_BT), abmCp_);
        UpRelMte3();
        BufRelease<PIPE_MTE3>(M15G::GP_BUF_BLK);
        CcSet<PIPE_MTE3>(M15G::GP_FLAG_GO3);

        // ---------- 等 J3（AB·d / (kᵀ·DP)ᵀ）----------
        CcWait<PIPE_MTE2>(M15G::GP_FLAG_DONE3);
        PipeBarrier<PIPE_MTE2>();
        BufAcquire<PIPE_MTE2>(M15G::GP_BUF_BLK);
        UpAcqMte2();
        {
            const AscendC::DataCopyParams op{static_cast<uint16_t>(M15G::GP_BT), (M15G::GP_DV * 4) / 32, 0, 0};
            DataCopy(Lt(M15G::GP_UB_K, M15G::GP_BT * M15G::GP_DV), scGm_[(slot + M15G::GP_SC_ABD) / 4], op);
        }
        UpRelMte2();
        BufRelease<PIPE_MTE2>(M15G::GP_BUF_BLK);
        BufAcquire<PIPE_V>(M15G::GP_BUF_BLK);
        UpAcqV();
        AddRows2VF(qsUb, M15G::GP_DV, kUb, M15G::GP_DV, M15G::GP_BT);   // o = o_part + AB·d
        MemBarVL();
        UpRelV();
        BufRelease<PIPE_V>(M15G::GP_BUF_BLK);
        BufAcquire<PIPE_MTE3>(M15G::GP_BUF_BLK);
        UpAcqMte3();
        {
            const AscendC::DataCopyParams opc{static_cast<uint16_t>(cv), (M15G::GP_DV * 4) / 32, 0, 0};
            const uint64_t oOff =
                static_cast<uint64_t>(hv) * m_ * M15G::GP_DV + static_cast<uint64_t>(t0) * M15G::GP_DV;
            DataCopy(oGm_[oOff], Lt(M15G::GP_UB_QS, M15G::GP_BT * M15G::GP_DV), opc);
        }
        UpRelMte3();
        BufRelease<PIPE_MTE3>(M15G::GP_BUF_BLK);

        // ---------- ST₁ = egL·ST₀ + SDT（SDT 分两块 64 行读入，复用 UB_K）----------
        for (uint32_t hb = 0; hb < 2; ++hb) {
            PipeBarrier<PIPE_MTE2>();
            BufAcquire<PIPE_MTE2>(M15G::GP_BUF_BLK);
            UpAcqMte2();
            {
                const AscendC::DataCopyParams sp{static_cast<uint16_t>(GP_VL), (M15G::GP_DK * 4) / 32, 0, 0};
                DataCopy(Lt(M15G::GP_UB_K, GP_VL * M15G::GP_DK),
                         scGm_[(slot + M15G::GP_SC_SDT + hb * GP_VL * M15G::GP_DK * 4) / 4], sp);
            }
            UpRelMte2();
            BufRelease<PIPE_MTE2>(M15G::GP_BUF_BLK);
            BufAcquire<PIPE_V>(M15G::GP_BUF_BLK);
            UpAcqV();
            ScaleAddRowsBrc2VF(stUb + hb * GP_VL * M15G::GP_DK, M15G::GP_DK, kUb, M15G::GP_DK, scL, GP_VL);
            MemBarVL();
            UpRelV();
            BufRelease<PIPE_V>(M15G::GP_BUF_BLK);
        }
    }

private:
    __ubuf__ float* ub_ = nullptr;
    uint32_t m_ = 0;
    uint32_t heads_ = 0;
    uint32_t qkStride_ = 0;
    uint32_t tp_ = 0;
    float scale_ = 1.0f;
    uint32_t nAic_ = 0;
    uint32_t bid_ = 0;
    const AscendC::DataCopyParams ktCp_{static_cast<uint16_t>(M15G::GP_DK), (M15G::GP_BT * 4) / 32, 0, 0};
    const AscendC::DataCopyParams abmCp_{static_cast<uint16_t>(M15G::GP_BT), (M15G::GP_BT * 4) / 32, 0, 0};
    AscendC::GlobalTensor<float> qGm_, kGm_, vGm_, gGm_, bGm_, h0Gm_, oGm_, htGm_, scGm_;
};

// ============================================================
// 4. 段入口（融合 TU 由 `if constexpr (PREFILL)` 分支调用；本段不自带 main()）
// ============================================================

__aicore__ inline void GdnPrefillAivEntry(const GdnPrefillArgs& a)
{
    const uint32_t bid = AscendC::GetBlockIdx();
    GdnPrefillAiv op;
    op.Init(a, bid);
    op.Process();
}

__aicore__ inline void GdnPrefillAicEntry(const GdnPrefillArgs& a)
{
    const uint32_t bid = AscendC::GetBlockIdx();
    GdnPrefillAic op;
    op.Init(a, bid);
    op.Process();
}

}  // namespace M15GP

#endif  // M15_GDN_PREFILL_H
