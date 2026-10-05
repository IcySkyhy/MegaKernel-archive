// m15_prefill_prolog.h —— M151：预填充 prolog 链路（S2 in_proj + S3 conv1d/l2norm/gating）的段体抽取
//
// 组成（全部为**抽取/薄壳**，不改 donor 数学）：
//   · M15PM：`m9_gdn_prolog.asc` 的 device 段体（常量 + `MtGdnProlog` 类；逐字抽取，
//     见 CONTRACT.md §3）。**只抽 device 部分**，不含 m9 的 host `main()`。
//   · `m15_pf_inproj_gemm_kernel`：S2 in_proj 薄壳，复用 `M15OP::OProjGemm<2560,16480>`
//     （`m15_attn_oproj.h`，m11 donor 的 lift；含负向对照 `GEMM_MODE_KMINUS1`）。
//   · `m15_pf_gdn_prolog_kernel`：S3 prolog 薄壳，逐参对齐 donor
//     `gdn_prolog_mt_kernel`（`m9_gdn_prolog.asc:844`）。
//
// 抽取等价性：本文件的 M15PM 命名空间内容 = `m9_gdn_prolog/m9_gdn_prolog.asc` 第 53..842 行
// 的逐字副本，外面套一层命名空间（避免与本 TU 其它头文件的匿名 namespace 常量撞名）。
// 复算脚本见 evidence/m151_prefill_prolog_wiring/verify_lift.sh。
//
// 人类约束：矩阵乘法（S2）走 cube（`__cube__`）；conv1d/l2norm/gating（S3）全 VF（donor 已是
// AIV-only + BufferID 同步）；本文件不引入 set_flag/wait_flag。
#ifndef M15_PREFILL_PROLOG_H
#define M15_PREFILL_PROLOG_H

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "m15_attn_oproj.h"   // M15OP::OProjGemm<K,N>（m11 donor lift）

// ---- M151 常量：预填充 prolog 的形状（与 CONTRACT.md §2/§3 一致）----
namespace M15PM {
constexpr uint32_t PF_IN_K = 2560u;    // HIDDEN：in_proj 的 K
constexpr uint32_t PF_IN_N = 16480u;   // in_proj 输出宽 = q2048|k2048|v6144|z6144|b48|a48
}  // namespace M15PM

// ============================================================
// M15PM：m9 的 device 段体（逐字抽取；外层命名空间是唯一包裹）
// ============================================================
namespace M15PM {
// ============================================================
// 编译期常量与静态资源表（UB 偏移 / BufferID 全部编译期静态分配）
// ============================================================

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
constexpr uint32_t UB_BK0 = 0;
constexpr uint32_t UB_BK1 = UB_BK0 + PAR_B;
// gating 区（a/b 为 bf16 staging + UNPACK 越界预读 slack；A_log/dt_bias 为 fp32 + slack）
constexpr uint32_t UB_AST = UB_BK1 + PAR_B;   // a staging bf16 [64] + slack   256B
constexpr uint32_t UB_BST = UB_AST + 256;     // b staging bf16 [64] + slack   256B
constexpr uint32_t UB_AL = UB_BST + 256;      // A_log fp32 [64] + slack       512B
constexpr uint32_t UB_DT = UB_AL + 512;       // dt_bias fp32 [64] + slack     512B
constexpr uint32_t UB_GS0 = UB_DT + 512;      // g slot [48,8] fp32（parity0） 1536B
constexpr uint32_t UB_BS0 = UB_GS0 + 1536;    // β slot（parity0）            1536B
constexpr uint32_t UB_GS1 = UB_BS0 + 1536;    // g slot（parity1）            1536B
constexpr uint32_t UB_BS1 = UB_GS1 + 1536;    // β slot（parity1）            1536B
static_assert(UB_BS1 + 1536 <= 248 * 1024, "UB footprint exceeds 248KB");
static_assert(UB_BK0 % 32 == 0 && UB_BK1 % 32 == 0 && UB_AST % 32 == 0 && UB_AL % 32 == 0 &&
                  UB_DT % 32 == 0 && UB_GS0 % 32 == 0 && UB_BS0 % 32 == 0 && UB_GS1 % 32 == 0 &&
                  UB_BS1 % 32 == 0,
              "UB offsets must be 32B aligned");

// ---- BufferID（每核用户可用 0-27，静态分配）----
constexpr AscendC::MutexID BUF_BK0 = 0;  // block 输入区 parity0：MTE2 → VEC → MTE3
constexpr AscendC::MutexID BUF_BK1 = 1;  // block 输入区 parity1
constexpr AscendC::MutexID BUF_GB = 2;   // gating 阵列：MTE2 → VEC（首个 v block 至末个 v block）

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
// 寄存器 VF 计算（__simd_vf__，全部在 __VEC_SCOPE__ 内调用）
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
        const uint32_t nblk = GetBlockNum();
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
        const uint32_t nblk = GetBlockNum();
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

__vector__ __global__ void gdn_prolog_kernel(GM_ADDR x, GM_ADDR convState, GM_ADDR w, GM_ADDR bias, GM_ADDR aLog,
                                             GM_ADDR dtBias, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta)
{
    GdnProlog op;
    op.Init(x, convState, w, bias, aLog, dtBias, q, k, v, g, beta);
    op.Process();
}

// ============================================================
// m>1 段体（MTP / prefill prolog）：同一 conv1d/l2norm/gating 数学，跨 token 窗口 +
// conv_state 环形交接，输出几何 = 相位 A 输入契约
// （docs/22-prefill-prolog-epilog-wiring.md §5.2/§5.3；m15_layer_kernel.h:643-648/:721）
//
// 与 m=1 段体的关系：几何不同（q/k 每 head 行 stride = m + 64，尾部 64 行置零；
// v [48,m,128]；g/β [48,align8(m)]），故为独立入口，m=1 入口一字未动。
//
// 语义决策（塔裁 2026-10-04，见 README §scale）：
//   **默认（qScaleOn=0）= 相位 A 契约**：q/k 均不乘 1/√128，scale 由相位 A 消费者施加
//   （m15_gdn_prefill.h:967/:1051；m15_layer_kernel.h:722）。qScaleOn=1 用于把 m=1 档切回
//   m9 独立模块的历史语义（q ×1/√128，m9_gdn_prolog.asc:75/:225）做交叉复现。
//
// 跨 token 语义：对 token t，窗口 win = [st0,st1,st2,x[t]]（st 为 conv_state 当前 3 行），
// 算完把 st 左移一格（st0<-st1, st1<-st2, st2<-x[t]）；m 个 token 跑完 conv_state_out =
// [old;x] 的最后 3 行（m≥3 时 = x[m-3..m-1]）——与 m21_layer_ref/ref/gdn.py:115-134 同义。
//
// 同步：全程 BufferID（get_buf/rls_buf，无 SetFlag/WaitFlag）；
//   MT_BUF_W（W/BIAS/AL/DT：MTE2→VEC）、MT_BUF_ST（state 3 行：MTE2→VEC→MTE3）、
//   MT_BUF_T0/1（每 token X/AST/BST/OUT：MTE2→VEC→MTE3，parity 双缓冲）、
//   MT_BUF_GB（g/β 行缓冲：VEC→MTE3）、MT_BUF_Z（pad 置零源：VEC→MTE3）。
// ============================================================

namespace {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t MT_PAD = 64u;    // PF_GDN_QK_PAD_ROWS（m15_layer_kernel.h:648）
constexpr uint32_t MT_MAX_M = 4097u;
constexpr uint32_t MT_MAX_TP = ((MT_MAX_M + 7u) / 8u) * 8u;      // 4104
constexpr uint32_t MT_MAX_TP64 = ((MT_MAX_TP + 63u) / 64u) * 64u;  // 4160

// UB 静态布局（字节偏移，全 32B 对齐）
constexpr uint32_t MT_W = 0;                                  // 4 行 w bf16 = 1024B
constexpr uint32_t MT_BIAS = MT_W + KW * HD * 2;              // 128 bf16 = 256B
constexpr uint32_t MT_AL = MT_BIAS + 256;                     // 64 fp32 = 256B（+slack）
constexpr uint32_t MT_DT = MT_AL + 512;                       // 512B
constexpr uint32_t MT_ST = MT_DT + 512;                       // state 3 行 = 768B
constexpr uint32_t MT_T0 = MT_ST + ST * HD * 2;               // 每 token 区 parity0
constexpr uint32_t MT_XO = 0;                                 // X 128 bf16 = 256B
constexpr uint32_t MT_AO = MT_XO + HD * 2;                    // a staging 64 bf16 = 128B→256B
constexpr uint32_t MT_BO = MT_AO + 256;                       // b staging 256B
constexpr uint32_t MT_OO = MT_BO + 256;                       // OUT 128 fp32 = 512B
constexpr uint32_t MT_TB = MT_OO + HD * 4;                    // 每 token 区合计 1280B
constexpr uint32_t MT_T1 = MT_T0 + MT_TB;
constexpr uint32_t MT_GROW = MT_T1 + MT_TB;                   // g 行缓冲 MT_MAX_TP64 fp32
constexpr uint32_t MT_BROW = MT_GROW + MT_MAX_TP64 * 4;
constexpr uint32_t MT_ZERO = MT_BROW + MT_MAX_TP64 * 4;       // pad 置零源 64*128 fp32
constexpr uint32_t MT_END = MT_ZERO + 64 * HD * 4;
static_assert(MT_END <= 248 * 1024, "MT UB footprint exceeds 248KB");
static_assert(MT_W % 32 == 0 && MT_BIAS % 32 == 0 && MT_AL % 32 == 0 && MT_DT % 32 == 0 && MT_ST % 32 == 0 &&
                  MT_T0 % 32 == 0 && MT_T1 % 32 == 0 && MT_GROW % 32 == 0 && MT_BROW % 32 == 0 &&
                  MT_ZERO % 32 == 0,
              "MT UB offsets must be 32B aligned");

constexpr AscendC::MutexID MT_BUF_W = 0;
constexpr AscendC::MutexID MT_BUF_ST = 1;
constexpr AscendC::MutexID MT_BUF_T0 = 2;
constexpr AscendC::MutexID MT_BUF_T1 = 3;
constexpr AscendC::MutexID MT_BUF_GB = 4;
constexpr AscendC::MutexID MT_BUF_Z = 5;

// pad 置零源的一次性清零（32KB = 64 行 × 128 fp32）
__simd_vf__ inline void MtZeroFill(__ubuf__ float* zp)
{
    RegTensor<float> zr;
    MaskReg fm = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zr, 0.0f, fm);
    for (uint32_t j = 0; j < 64 * HD; j += VL) {
        StoreAlign<float>(zp + j, zr, fm);
    }
}

// 单 token VF：conv(K=4)+bias+SiLU → (v: gating 存 g/β 行 | q/k: l2norm) → state 左移一格
__simd_vf__ inline void MtPrologTokenVF(uint32_t b, uint32_t t, uint32_t qScaleOn, uint32_t noShift,
                                        __ubuf__ bfloat16_t* st, __ubuf__ bfloat16_t* xUb, __ubuf__ bfloat16_t* wUb,
                                        __ubuf__ bfloat16_t* bUb, __ubuf__ float* outUb, __ubuf__ bfloat16_t* aSt,
                                        __ubuf__ bfloat16_t* bSt, __ubuf__ float* alUb, __ubuf__ float* dtUb,
                                        __ubuf__ float* gRow, __ubuf__ float* bRow)
{
    RegTensor<bfloat16_t> tb, tw, tr;
    RegTensor<float> s, w, acc, tv, y0, y1;
    MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
    MaskReg fullB = CreateMask<bfloat16_t, MaskPattern::ALL>();

#define MT_TAP(WROW, SROW)                                                                         \
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tw, wUb + (WROW) * HD + e);                    \
    Cast<float, bfloat16_t, castTraitB162B32Even>(w, tw, fullM);                                    \
    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tb, (SROW) + e);                               \
    Cast<float, bfloat16_t, castTraitB162B32Even>(s, tb, fullM);                                    \
    MulAddDst(acc, w, s, fullM)

    for (uint32_t c = 0; c < HD / VL; ++c) {
        const uint32_t e = c * VL;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tb, bUb + e);
        Cast<float, bfloat16_t, castTraitB162B32Even>(acc, tb, fullM);  // acc = bias
        MT_TAP(0, st);
        MT_TAP(1, st + HD);
        MT_TAP(2, st + 2 * HD);
        MT_TAP(3, xUb);
        Muls(tv, acc, -1.0f, fullM);
        Exp(tv, tv, fullM);
        Adds(tv, tv, 1.0f, fullM);
        if (c == 0) {
            Div(y0, acc, tv, fullM);
            StoreAlign<float>(outUb, y0, fullM);
        } else {
            Div(y1, acc, tv, fullM);
            StoreAlign<float>(outUb + VL, y1, fullM);
        }
    }
#undef MT_TAP

    if (b >= QKB) {  // v 块：gating，行内位置 t（gRow[t] / bRow[t]）
        const uint32_t h = b - QKB;
        RegTensor<bfloat16_t> tg;
        RegTensor<int32_t> idx;
        RegTensor<float> aF, bF, dtF, alF, u, gAll, beAll, one, gS, bS, zr;
        RegTensor<float> xr;
        MaskReg cmp, oh;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tg, aSt);
        Cast<float, bfloat16_t, castTraitB162B32Even>(aF, tg, fullM);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(tg, bSt);
        Cast<float, bfloat16_t, castTraitB162B32Even>(bF, tg, fullM);
        LoadAlign<float>(dtF, dtUb);
        LoadAlign<float>(alF, alUb);
        Add(xr, aF, dtF, fullM);
        Exp(u, xr, fullM);
        Adds(u, u, 1.0f, fullM);
        Log(u, u, fullM);
        Compares<float, CMPMODE::LE>(cmp, xr, SPTH, fullM);
        Select<float>(u, u, xr, cmp);
        Exp(gAll, alF, fullM);
        Muls(gAll, gAll, -1.0f, fullM);
        Mul(gAll, gAll, u, fullM);
        Muls(beAll, bF, -1.0f, fullM);
        Exp(beAll, beAll, fullM);
        Adds(beAll, beAll, 1.0f, fullM);
        Duplicate(one, 1.0f, fullM);
        Div(beAll, one, beAll, fullM);
        Arange<int32_t>(idx, 0);
        Compares<int32_t, CMPMODE::EQ>(oh, idx, static_cast<int32_t>(h), fullM);
        Reduce<ReduceType::SUM>(gS, gAll, oh);
        Reduce<ReduceType::SUM>(bS, beAll, oh);
        if (t == 0) {  // 行缓冲清零（pad 列 [m,tp) 与更远都保持 0）
            Duplicate(zr, 0.0f, fullM);
            for (uint32_t j = 0; j < MT_MAX_TP64; j += VL) {
                StoreAlign<float>(gRow + j, zr, fullM);
                StoreAlign<float>(bRow + j, zr, fullM);
            }
        }
        // 单元素落槽：Store（vstu 非对齐）——DIST_FIRST_ELEMENT_B32 要求 32B 对齐，t 是 4B 步长；
        // docs/18-vector-api-audit.md:408 指明单元素/非对齐落盘只在 Store/StoreUnAlign/Gather 一族。
        Store<float>(gRow + t, gS, 1);
        Store<float>(bRow + t, bS, 1);
    } else {  // q/k 块：l2norm；qScaleOn=1 时仅 q 乘 1/√128（m9 历史语义，交叉复现用）
        const bool qs = (b < NQB) && (qScaleOn != 0u);
        RegTensor<float> m2, r, one2, bcast;
        Mul(m2, y0, y0, fullM);
        Mul(r, y1, y1, fullM);
        Add(m2, m2, r, fullM);
        ReduceSum(r, m2, fullM);
        Adds(r, r, EPS, fullM);
        Sqrt(r, r, fullM);
        Duplicate(one2, 1.0f, fullM);
        Div(r, one2, r, fullM);
        if (qs) {
            Muls(r, r, QSCALE, fullM);
        }
        Duplicate(bcast, r, fullM);
        Mul(y0, y0, bcast, fullM);
        Mul(y1, y1, bcast, fullM);
        StoreAlign<float>(outUb, y0, fullM);
        StoreAlign<float>(outUb + VL, y1, fullM);
    }

    if (noShift == 0u) {  // conv_state 环形交接：st0<-st1, st1<-st2, st2<-x[t]
        LoadAlign<bfloat16_t>(tr, st + HD);
        StoreAlign<bfloat16_t>(st, tr, fullB);
        LoadAlign<bfloat16_t>(tr, st + 2 * HD);
        StoreAlign<bfloat16_t>(st + HD, tr, fullB);
        LoadAlign<bfloat16_t>(tr, xUb);
        StoreAlign<bfloat16_t>(st + 2 * HD, tr, fullB);
    }
}

class MtGdnProlog {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR convState, GM_ADDR w, GM_ADDR bias, GM_ADDR aLog, GM_ADDR dtBias,
                                GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, uint32_t m,
                                uint32_t qScaleOn, uint32_t noShift, uint32_t noPad)
    {
        m_ = m;
        qScaleOn_ = qScaleOn;
        noShift_ = noShift;
        noPad_ = noPad;
        tp_ = (m + 7u) / 8u * 8u;
        qkStride_ = m + MT_PAD;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), static_cast<uint64_t>(m) * INW);
        csGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(convState), CH * ST);
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w), KW * CH);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bias), CH);
        alGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aLog), 64);
        dtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dtBias), 64);
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q), static_cast<uint64_t>(NQB) * qkStride_ * HD);
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k), static_cast<uint64_t>(NQB) * qkStride_ * HD);
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v), static_cast<uint64_t>(VH) * m * HD);
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g), static_cast<uint64_t>(VH) * tp_);
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta), static_cast<uint64_t>(VH) * tp_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = GetBlockIdx();
        const uint32_t nblk = GetBlockNum();
        if (bid >= NBLK) {
            return;
        }
        // pad 置零源：VEC 一次清零（32KB），MTE3 全程持有只读
        BufAcquire<PIPE_V>(MT_BUF_Z);
        __VEC_SCOPE__
        {
            MtZeroFill(reinterpret_cast<__ubuf__ float*>(MT_ZERO));
        }
        BufRelease<PIPE_V>(MT_BUF_Z);
        BufAcquire<PIPE_MTE3>(MT_BUF_Z);  // 全程持有（每条 pad 拷贝的源）

        for (uint32_t b = bid; b < NBLK; b += nblk) {
            ProcessBlock(b);
        }
        BufRelease<PIPE_MTE3>(MT_BUF_Z);
    }

private:
    __aicore__ inline void CopyInBlockConst(uint32_t b)
    {
        const uint32_t c0 = b * HD;
        LocalTensor<bfloat16_t> wL(TPosition::VECCALC, MT_W, KW * HD);
        LocalTensor<bfloat16_t> biasL(TPosition::VECCALC, MT_BIAS, HD);
        LocalTensor<float> alL(TPosition::VECCALC, MT_AL, 64);
        LocalTensor<float> dtL(TPosition::VECCALC, MT_DT, 64);
        const AscendC::DataCopyParams cp256{1, 8, 0, 0};
        BufAcquire<PIPE_MTE2>(MT_BUF_W);
        for (uint32_t j = 0; j < KW; ++j) {
            DataCopy(wL[j * HD], wGm_[static_cast<uint64_t>(j) * CH + c0], cp256);
        }
        DataCopy(biasL, bGm_[c0], cp256);
        if (b >= QKB) {
            DataCopy(alL, alGm_, cp256);
            DataCopy(dtL, dtGm_, cp256);
        }
        BufRelease<PIPE_MTE2>(MT_BUF_W);
    }

    __aicore__ inline void CopyInState(uint32_t b)
    {
        const uint32_t c0 = b * HD;
        const AscendC::DataCopyParams cp256{1, 8, 0, 0};
        BufAcquire<PIPE_MTE2>(MT_BUF_ST);
        for (uint32_t j = 0; j < ST; ++j) {
            LocalTensor<bfloat16_t> sjL(TPosition::VECCALC, MT_ST + j * HD * sizeof(bfloat16_t), HD);
            DataCopy(sjL, csGm_[static_cast<uint64_t>(j) * CH + c0], cp256);
        }
        BufRelease<PIPE_MTE2>(MT_BUF_ST);
    }

    __aicore__ inline void CopyInToken(uint32_t b, uint32_t t, uint32_t p)
    {
        const uint32_t tb = p ? MT_T1 : MT_T0;
        const uint32_t c0 = b * HD;
        LocalTensor<bfloat16_t> xL(TPosition::VECCALC, tb + MT_XO, HD);
        const AscendC::DataCopyParams cp256{1, 8, 0, 0};
        const AscendC::DataCopyParams cp96{1, 3, 0, 0};
        const uint64_t base = static_cast<uint64_t>(t) * INW;
        BufAcquire<PIPE_MTE2>(p ? MT_BUF_T1 : MT_BUF_T0);
        DataCopy(xL, xGm_[base + c0], cp256);
        if (b >= QKB) {
            LocalTensor<bfloat16_t> aL(TPosition::VECCALC, tb + MT_AO, 64);
            LocalTensor<bfloat16_t> bL(TPosition::VECCALC, tb + MT_BO, 64);
            DataCopy(aL, xGm_[base + XA_OFF], cp96);
            DataCopy(bL, xGm_[base + XB_OFF], cp96);
        }
        BufRelease<PIPE_MTE2>(p ? MT_BUF_T1 : MT_BUF_T0);
    }

    __aicore__ inline void ComputeToken(uint32_t b, uint32_t t, uint32_t p)
    {
        const uint32_t tb = p ? MT_T1 : MT_T0;
        BufAcquire<PIPE_V>(p ? MT_BUF_T1 : MT_BUF_T0);
        __VEC_SCOPE__
        {
            MtPrologTokenVF(b, t, qScaleOn_, noShift_, reinterpret_cast<__ubuf__ bfloat16_t*>(MT_ST),
                            reinterpret_cast<__ubuf__ bfloat16_t*>(tb + MT_XO),
                            reinterpret_cast<__ubuf__ bfloat16_t*>(MT_W),
                            reinterpret_cast<__ubuf__ bfloat16_t*>(MT_BIAS),
                            reinterpret_cast<__ubuf__ float*>(tb + MT_OO),
                            reinterpret_cast<__ubuf__ bfloat16_t*>(tb + MT_AO),
                            reinterpret_cast<__ubuf__ bfloat16_t*>(tb + MT_BO),
                            reinterpret_cast<__ubuf__ float*>(MT_AL), reinterpret_cast<__ubuf__ float*>(MT_DT),
                            reinterpret_cast<__ubuf__ float*>(MT_GROW),
                            reinterpret_cast<__ubuf__ float*>(MT_BROW));
        }
        BufRelease<PIPE_V>(p ? MT_BUF_T1 : MT_BUF_T0);
    }

    __aicore__ inline void CopyOutToken(uint32_t b, uint32_t t, uint32_t p)
    {
        const uint32_t tb = p ? MT_T1 : MT_T0;
        LocalTensor<float> outL(TPosition::VECCALC, tb + MT_OO, HD);
        const AscendC::DataCopyParams cp512{1, 16, 0, 0};
        BufAcquire<PIPE_MTE3>(p ? MT_BUF_T1 : MT_BUF_T0);
        if (b < NQB) {
            DataCopy(qGm_[static_cast<uint64_t>(b) * qkStride_ * HD + static_cast<uint64_t>(t) * HD], outL, cp512);
        } else if (b < QKB) {
            DataCopy(kGm_[static_cast<uint64_t>(b - NQB) * qkStride_ * HD + static_cast<uint64_t>(t) * HD], outL,
                     cp512);
        } else {
            DataCopy(vGm_[static_cast<uint64_t>(b - QKB) * m_ * HD + static_cast<uint64_t>(t) * HD], outL, cp512);
        }
        BufRelease<PIPE_MTE3>(p ? MT_BUF_T1 : MT_BUF_T0);
    }

    __aicore__ inline void CopyOutState(uint32_t b)
    {
        const uint32_t c0 = b * HD;
        const AscendC::DataCopyParams cp256{1, 8, 0, 0};
        BufAcquire<PIPE_MTE3>(MT_BUF_ST);
        for (uint32_t j = 0; j < ST; ++j) {
            LocalTensor<bfloat16_t> sjL(TPosition::VECCALC, MT_ST + j * HD * sizeof(bfloat16_t), HD);
            DataCopy(csGm_[static_cast<uint64_t>(j) * CH + c0], sjL, cp256);
        }
        BufRelease<PIPE_MTE3>(MT_BUF_ST);
    }

    __aicore__ inline void CopyOutGB(uint32_t b)
    {
        const uint32_t h = b - QKB;
        LocalTensor<float> gL(TPosition::VECCALC, MT_GROW, MT_MAX_TP64);
        LocalTensor<float> bRL(TPosition::VECCALC, MT_BROW, MT_MAX_TP64);
        const AscendC::DataCopyParams cpGB{1, static_cast<uint16_t>(tp_ / 8), 0, 0};
        BufAcquire<PIPE_MTE3>(MT_BUF_GB);
        DataCopy(gGm_[static_cast<uint64_t>(h) * tp_], gL, cpGB);
        DataCopy(betaGm_[static_cast<uint64_t>(h) * tp_], bRL, cpGB);
        BufRelease<PIPE_MTE3>(MT_BUF_GB);
    }

    // q/k 尾部 pad 行 [m, m+64) 置零（相位 A 的 Nd2Nz 固读 64 行；m15_layer_kernel.h:643-648）
    __aicore__ inline void ZeroPadRows(uint32_t b)
    {
        const uint32_t hk = (b < NQB) ? b : (b - NQB);
        LocalTensor<float> zL(TPosition::VECCALC, MT_ZERO, 64 * HD);
        const AscendC::DataCopyParams cpPad{1, static_cast<uint16_t>(64 * HD * 4 / 32), 0, 0};
        if (b < NQB) {
            DataCopy(qGm_[static_cast<uint64_t>(hk) * qkStride_ * HD + static_cast<uint64_t>(m_) * HD], zL, cpPad);
        } else {
            DataCopy(kGm_[static_cast<uint64_t>(hk) * qkStride_ * HD + static_cast<uint64_t>(m_) * HD], zL, cpPad);
        }
    }

    __aicore__ inline void ProcessBlock(uint32_t b)
    {
        const bool isV = (b >= QKB);
        CopyInBlockConst(b);
        CopyInState(b);
        CopyInToken(b, 0, 0);
        // 全程持有的 VEC 侧令牌（W / state / g-β 行缓冲）
        BufAcquire<PIPE_V>(MT_BUF_W);
        BufAcquire<PIPE_V>(MT_BUF_ST);
        if (isV) {
            BufAcquire<PIPE_V>(MT_BUF_GB);
        }
        for (uint32_t t = 0; t < m_; ++t) {
            const uint32_t p = t & 1u;
            if (t + 1 < m_) {
                CopyInToken(b, t + 1, p ^ 1u);  // MTE2 预取
            }
            ComputeToken(b, t, p);
            CopyOutToken(b, t, p);
        }
        if (isV) {
            BufRelease<PIPE_V>(MT_BUF_GB);
        }
        BufRelease<PIPE_V>(MT_BUF_ST);
        BufRelease<PIPE_V>(MT_BUF_W);
        CopyOutState(b);
        if (isV) {
            CopyOutGB(b);
        } else if (noPad_ == 0u) {
            ZeroPadRows(b);
        }
    }

private:
    GlobalTensor<bfloat16_t> xGm_, csGm_, wGm_, bGm_;
    GlobalTensor<float> alGm_, dtGm_, qGm_, kGm_, vGm_, gGm_, betaGm_;
    uint32_t m_ = 0, tp_ = 0, qkStride_ = 0;
    uint32_t qScaleOn_ = 0, noShift_ = 0, noPad_ = 0;
};

}  // namespace
}  // namespace M15PM

// ============================================================
// 入口壳（全局作用域，便于 host `<<<...>>>` 启动）
// ============================================================

// S2 · in_proj（cube）：A `[m,2560]` bf16 / B=W `[16480,2560]` bf16（按 B^T）/ C `[m,16480]` bf16。
// 单核形态（m11 donor 原形：blockDim=1，内部循环全 M/N/K tile）；`mode` 透传 `GEMM_MODE_*`。
__global__ __cube__ void m15_pf_inproj_gemm_kernel(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c,
                                                   uint32_t m, uint32_t mode)
{
    AscendC::InitSocState();
    M15OP::OProjGemm<M15PM::PF_IN_K, M15PM::PF_IN_N> op;
    op.Init(a, b, c, m, mode);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}

// S3 · prolog（AIV）：逐参对齐 donor `gdn_prolog_mt_kernel`（`m9_gdn_prolog.asc:844-852`）。
// 启动 `<<<nAiv,0,stream>>>`；`qScaleOn=0` = 相位 A 契约；`noShift/noPad` 是负向对照开关。
__vector__ __global__ void m15_pf_gdn_prolog_kernel(__gm__ uint8_t* x, __gm__ uint8_t* convState, __gm__ uint8_t* w,
                                                    __gm__ uint8_t* bias, __gm__ uint8_t* aLog, __gm__ uint8_t* dtBias,
                                                    __gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* v,
                                                    __gm__ uint8_t* g, __gm__ uint8_t* beta, uint32_t m,
                                                    uint32_t qScaleOn, uint32_t noShift, uint32_t noPad)
{
    M15PM::MtGdnProlog op;
    op.Init(x, convState, w, bias, aLog, dtBias, q, k, v, g, beta, m, qScaleOn, noShift, noPad);
    op.Process();
}

#endif  // M15_PREFILL_PROLOG_H
