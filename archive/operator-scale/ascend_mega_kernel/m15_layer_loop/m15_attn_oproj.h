// ============================================================
// m15_attn_oproj.h —— M101：attention 输出段 = `×sigmoid(gate)` + `o_proj`
//
// 本文件回答 M97 `evidence/attn_wire/README.md` §5 第 8 项的前置（"`subOut` 在接线态下没有生产者"）：
// 子层出口 `subOut` 的生产者就是本段（`out = o_proj( attn·sigmoid(gate) )`）。**本 mission 只做段本身**
// ——把段挂进层 kernel 的交 M102（挂载点见 §5 未完成项）。
//
// ---------------------------------------------------------------------------
// 形状与乘法次序的**权威来源**（不猜；每条给 `文件:符号`）
//   · `docs/11-attn-analysis.md:39`   `o_proj [2560, 6144]`：24×256 → 2560
//   · `docs/11-attn-analysis.md:37`   `q_proj [12288, 2560]` = 24 头 ×(256 q + 256 gate)；gate 不 norm 不旋转
//   · `docs/11-attn-analysis.md:43`   `out = o_proj( attn(q,k,v) · sigmoid(gate) )`，scale = 256^-0.5
//   · `docs/11-attn-analysis.md:22`   箭头序 `► sigmoid(gate) ► o_proj`（与上式的括号一致：**先门控后 GEMM**）
//   · `slice_layer_manifest.py:78`    `("attn_o_proj", "self_attn.o_proj.weight", [2560, 6144], 2560, 6144)`
//   · `m11_bf16_gemm.asc:444`         donor 的 shapes 表里已有 `out_proj (K=6144, N=2560)` 实例
//   · `m15_attn_prolog.h:90-95`       prolog 的 `OUT_Q` / `OUT_GATE` 是**按头排布**的两块连续 bf16 平面
//   · `m10_attn_decode.asc`（抽出件的 `Combine()`）设备 out 寻址 = `outGM[(n2*GQA_G + row)*S2T]`
//     ⇒ 内存序 = 头序 `h = n2*12 + row`（GQA 相邻配对 `num_queries_per_kv = 24/2 = 12`）
//     ⇒ **attn 的输出平面与 gate 平面同序**，逐元素相乘无需任何重排。
//   **乘法次序的意义**：gate 的宽度 = 6144 = attn 输出宽度（不是 2560）⇒ 门控只能落在 o_proj **之前**；
//   这与上面 `docs/11` 的两处一致。若将来发现要落在之后，gate 宽度应为 2560 ⇒ 形状判据立刻红。
//
// ---------------------------------------------------------------------------
// 段划分（人类要求"模块化，暴露依赖哪些 pipeline、输出哪些 pipeline"，故**按核型各暴露一个 body**）：
//   §A `GateMulAiv`    —— AIV 段（`__simd_vf__`）：`t = attn · sigmoid(gate)`，逐元素，**无跨核同步**
//       依赖（消费）：`attn` GM（[m][6144] bf16）、`gate` GM（[m][6144] bf16，= prolog 的 OUT_GATE 平面）
//       产出：`t` GM（[m][6144] bf16）—— 下一段（GEMM）的 A 操作数
//       核间：无。核内 MutexID：`OP_BUF_GIN`(0) MTE2→V、`OP_BUF_GOUT`(1) V→MTE3（本段独占，可打平重编）
//   §B `OProjGemmAic` —— AIC 段（cube，抄 `m11_bf16_gemm.asc` 的 `Bf16Gemm` donor）：
//       依赖（消费）：`t` GM（A 操作数，[m][6144]）、`w` GM（B 操作数，**[N=2560, K=6144] 原始 layout**）
//       产出：`y` GM（[m][2560] bf16，RNE 落盘）—— 这就是子层出口 `subOut`
//       核间：无（两段在独立验证路里是两次 launch，靠 stream 同步；接进融合 kernel 时的跨核同步
//       由 M102 按打平表决定 —— 本段**不自带**任何 cross-core flagId，这是刻意的：段不知道自己被排在谁后面）
//       核内 MutexID：0..6（抄 m11 的 7 个；同样可打平重编）
//
// ---------------------------------------------------------------------------
// 为什么 AIV/AIC 各一个 body 而不是一个 mix body：m11 donor 是纯 AIC（`__global__ __cube__`），
//   `docs/11-attn-analysis.md:167` 又把 `sigmoid(gate)·out` 列为 AIV 的活 ⇒ 两段本来就分核。
//   本 mission 的独立验证路用**两次顺序 launch**（`__vector__` + `__cube__`），
//   于是段内**不需要任何跨核同步**、也就不可能因配对写错而死锁（跨核接线留给 M102）。
// ============================================================
#ifndef M15_ATTN_OPROJ_H
#define M15_ATTN_OPROJ_H

namespace M15OP {

using namespace AscendC;

// ---- 形状（上表逐条钉过）----
constexpr uint32_t OP_NH = 24;                       // num_attention_heads
constexpr uint32_t OP_HD = 256;                      // head_dim
constexpr uint32_t OP_K = OP_NH * OP_HD;             // 6144 = attn 输出宽 = o_proj 的 K = gate 宽
constexpr uint32_t OP_N = 2560;                      // hidden = o_proj 的 N
static_assert(OP_K == 6144u && OP_N == 2560u, "形状与 docs/11 §3.4 / slice_layer_manifest.py:78 不一致");

// 门控的三种口径（mode 由 host 传；**前两条是负向对照**，见 evidence/attn_core/README.md §4）
constexpr uint32_t GATE_MODE_CONTRACT = 0u;   // 正确：t = attn · sigmoid(gate)
constexpr uint32_t GATE_MODE_NO_GATE = 1u;    // 负向：跳过门控（t = attn）
constexpr uint32_t GATE_MODE_SIGN = 2u;       // 负向：sigmoid 取反号（σ(−g)）

// GEMM 的两种口径（同上）
constexpr uint32_t GEMM_MODE_CONTRACT = 0u;   // 正确
constexpr uint32_t GEMM_MODE_KMINUS1 = 1u;    // 负向：K 方向少累加一个 base 块

// ---- I/O 与精度规格（bf16；RNE 落盘与 m11 donor 同网格）----
constexpr uint32_t GATE_VL = 64;                       // fp32 一个向量寄存器的 lane 数（256B/4B）
constexpr uint32_t GATE_CHUNK_BYTES = GATE_VL * 2;     // 一个 chunk 的 bf16 字节数 = 128B（32B 整数倍）

constexpr AscendC::Reg::CastTrait castTraitB162B32 = {AscendC::Reg::RegLayout::ZERO,
                                                      AscendC::Reg::SatMode::UNKNOWN,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait castTraitB322B16 = {AscendC::Reg::RegLayout::ZERO,
                                                      AscendC::Reg::SatMode::NO_SAT,
                                                      AscendC::Reg::MaskMergeMode::ZEROING,
                                                      AscendC::RoundMode::CAST_RINT};

// ============================================================
// §A ×sigmoid(gate)（AIV）
// ============================================================
// UB 静态布局（字节；本段自管理，编译期定死）
constexpr uint32_t OP_UB_AT = 0;                       // bf16[64] 128B（attn chunk）
constexpr uint32_t OP_UB_GA = 128;                     // bf16[64] 128B（gate chunk）
constexpr uint32_t OP_UB_T = 256;                      // bf16[64] 128B（t chunk）
constexpr uint32_t OP_UB_END = 384;
static_assert(OP_UB_END <= 4096u, "门控段的 UB 用量（3×128B）");

// AIV 侧核内 MutexID（本段独占号段 0..1；融合时按打平表重编）
constexpr MutexID OP_BUF_GIN = 0;    // MTE2 生产 attn/gate chunk -> V 消费
constexpr MutexID OP_BUF_GOUT = 1;   // V 生产 t chunk -> MTE3 消费

// 核内同步一律 mode=false（与 `m15_attn_core.h` 的 `Acq/Rls` 同款；docs/05 M1 精炼）：
//   获取用 GetBuffImpl<pipe,false>；释放用 ReleaseBuffImpl<pipe,false> —— 两者一律 CANN
//   `ASC_LOCK_BLOCK` 默认「阻塞」模式（`true` = `NON_BLOCK`）。两种模式都等本 pipe 已发射指令
//   落地，`true` 额外等此前同 id 的释放 ⇒ `true` 更保守（不是"不等落地"）。
template <pipe_t p> __aicore__ __inline__ void OpAcq(MutexID id) { GetBuffImpl<p, false>(id); }
template <pipe_t p> __aicore__ __inline__ void OpRls(MutexID id) { ReleaseBuffImpl<p, false>(id); }

__aicore__ __inline__ __ubuf__ bfloat16_t* OpPhy(const LocalTensor<bfloat16_t>& t)
{
    return reinterpret_cast<__ubuf__ bfloat16_t*>(t.GetPhyAddr());
}

// 32B 整数倍的单块搬运参数（docs/05 M16：blockCount 恒为 1，避免 NZ 重排）
__aicore__ __inline__ DataCopyParams OpBlk(uint32_t bytes)
{
    DataCopyParams p;
    p.blockCount = 1;
    p.blockLen = (uint16_t)(bytes / 32u);
    p.srcStride = 0;
    p.dstStride = 0;
    return p;
}

// 契约口径：t = attn · σ(gate)。**直线代码**（VEC_SCOPE 内不放任何分支；
// σ 的四元组排布 = m15_gdn_layer.h:1290-1299 / m15_moe_layer.h:1954 的实证形态）。
__simd_vf__ inline void GateMulVf(__ubuf__ bfloat16_t* atUb, __ubuf__ bfloat16_t* gaUb,
                                 __ubuf__ bfloat16_t* tUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> a, g, s, one;
        RegTensor<bfloat16_t> ab, gb, tb;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(ab, atUb);
        Cast<float, bfloat16_t, castTraitB162B32>(a, ab, fullM);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gb, gaUb);
        Cast<float, bfloat16_t, castTraitB162B32>(g, gb, fullM);
        Muls(s, g, -1.0f, fullM);
        Exp(s, s, fullM);
        Adds(s, s, 1.0f, fullM);
        Duplicate(one, 1.0f, fullM);
        Div(s, one, s, fullM);          // σ(g)
        Mul(a, a, s, fullM);
        Cast<bfloat16_t, float, castTraitB322B16>(tb, a, fullM);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(tUb, tb, fullM);
    }
}

// 负向口径 ①：跳过门控（t = attn）—— 判据 OP-B/OP-C 必须变红
__simd_vf__ inline void GateBypassVf(__ubuf__ bfloat16_t* atUb, __ubuf__ bfloat16_t* gaUb,
                                     __ubuf__ bfloat16_t* tUb)
{
    using namespace AscendC::Reg;
    (void)gaUb;
    __VEC_SCOPE__
    {
        RegTensor<float> a;
        RegTensor<bfloat16_t> ab, tb;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(ab, atUb);
        Cast<float, bfloat16_t, castTraitB162B32>(a, ab, fullM);
        Cast<bfloat16_t, float, castTraitB322B16>(tb, a, fullM);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(tUb, tb, fullM);
    }
}

// 负向口径 ②：σ(−g) = 1/(1+e^{+g}) —— 判据 OP-B/OP-C 必须变红
__simd_vf__ inline void GateSignVf(__ubuf__ bfloat16_t* atUb, __ubuf__ bfloat16_t* gaUb,
                                   __ubuf__ bfloat16_t* tUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> a, g, s, one;
        RegTensor<bfloat16_t> ab, gb, tb;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(ab, atUb);
        Cast<float, bfloat16_t, castTraitB162B32>(a, ab, fullM);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gb, gaUb);
        Cast<float, bfloat16_t, castTraitB162B32>(g, gb, fullM);
        Exp(s, g, fullM);               // e^{+g}
        Adds(s, s, 1.0f, fullM);
        Duplicate(one, 1.0f, fullM);
        Div(s, one, s, fullM);          // = σ(−g)
        Mul(a, a, s, fullM);
        Cast<bfloat16_t, float, castTraitB322B16>(tb, a, fullM);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(tUb, tb, fullM);
    }
}

// AIV 段 body：把 [m×OP_K] 的扁平数组按 64 元素 chunk 轮转分给各 AIV（无跨行状态、无跨核同步）。
// 前置：调用方保证 `attn/gate/t` 三块对 AIV 可见（独立路里靠 stream 同步）。
__aicore__ inline void GateMulAiv(__gm__ uint8_t* attn, __gm__ uint8_t* gate, __gm__ uint8_t* t, uint32_t m,
                                  uint32_t mode)
{
    const uint32_t nChunk = m * (OP_K / GATE_VL);
    const uint32_t bid = GetBlockIdx();
    const uint32_t nAiv = GetBlockNum() * 2u;     // 与段同款的启动形态（见调用方）
    GlobalTensor<bfloat16_t> atGm;
    GlobalTensor<bfloat16_t> gaGm;
    GlobalTensor<bfloat16_t> tGm;
    atGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(attn));
    gaGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gate));
    tGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(t));
    LocalTensor<bfloat16_t> atL(TPosition::VECCALC, OP_UB_AT, GATE_VL);
    LocalTensor<bfloat16_t> gaL(TPosition::VECCALC, OP_UB_GA, GATE_VL);
    LocalTensor<bfloat16_t> tL(TPosition::VECCALC, OP_UB_T, GATE_VL);
    for (uint32_t c = bid; c < nChunk; c += nAiv) {
        const uint64_t off = (uint64_t)c * GATE_VL;
        OpAcq<PIPE_MTE2>(OP_BUF_GIN);
        DataCopy(atL, atGm[off], OpBlk(GATE_CHUNK_BYTES));
        DataCopy(gaL, gaGm[off], OpBlk(GATE_CHUNK_BYTES));
        OpRls<PIPE_MTE2>(OP_BUF_GIN);              // drain：MTE2 排空后 chunk 归 V
        OpAcq<PIPE_V>(OP_BUF_GIN);
        OpAcq<PIPE_V>(OP_BUF_GOUT);
        if (mode == GATE_MODE_NO_GATE) {
            GateBypassVf(OpPhy(atL), OpPhy(gaL), OpPhy(tL));
        } else if (mode == GATE_MODE_SIGN) {
            GateSignVf(OpPhy(atL), OpPhy(gaL), OpPhy(tL));
        } else {
            GateMulVf(OpPhy(atL), OpPhy(gaL), OpPhy(tL));
        }
        OpRls<PIPE_V>(OP_BUF_GIN);
        OpRls<PIPE_V>(OP_BUF_GOUT);                // drain：V 写完 t 才归 MTE3
        OpAcq<PIPE_MTE3>(OP_BUF_GOUT);
        DataCopy(tGm[off], tL, OpBlk(GATE_CHUNK_BYTES));
        OpRls<PIPE_MTE3>(OP_BUF_GOUT);
    }
}

// ============================================================
// §B o_proj GEMM（AIC）—— 抄 donor `m11_bf16_gemm.asc`（本仓库已合并、已实证的 bf16 单核 GEMM）
// ============================================================
// 与 donor 的关系（**这张表是"抄"的账**；除下表外逐字相同）：
//   1. 常量与类搬进 `namespace M15OP`（donor 在全局 namespace，融合 TU 里会与别的段撞名）
//   2. `class Bf16Gemm` → `class OProjGemm`
//   3. `Process()` 增加 `mode` 入参：`GEMM_MODE_KMINUS1` 时少累加一个 baseK 块（**负向对照**）
//   4. 去掉 donor 的寄存器名与函数名里的工程前缀冲突（无：donor 用的是 A1/B1/L0 等通用名，已随类进 namespace）
//   5. donor 的 `CopyInA/CopyInB/LoadA/LoadB/CopyOut` 逐字保留（同一 `Nd2NzParams`/`LoadData2DParamsV2`/
//      `MmadParams`/`FixpipeParamsArch3510` 字段值），仅函数体标点/缩进一致
constexpr uint32_t CUBE_BLOCK = 16;   // cube 分形边长（bf16: 16 元素 = 32B）
constexpr uint32_t BASE_M = 64;       // M 方向 tile（静态）；m ∈ [1, 64] 单 tile，尾块用 curM mask
constexpr uint32_t BASE_K = 64;       // K 方向 tile（K 必须能被 BASE_K 整除）
constexpr uint32_t BASE_N = 160;      // N 方向 tile（N 必须能被 BASE_N 整除）

constexpr uint32_t L1_A_ELEMS = BASE_M * BASE_K;
constexpr uint32_t L1_B_ELEMS = BASE_N * BASE_K;
constexpr uint32_t L1_OFF_A0 = 0;
constexpr uint32_t L1_OFF_A1 = L1_OFF_A0 + L1_A_ELEMS * 2;
constexpr uint32_t L1_B_REGION = 256 * 1024;
constexpr uint32_t L1_OFF_B0 = L1_B_REGION;
constexpr uint32_t L1_OFF_B1 = L1_B_REGION + L1_B_ELEMS * 2;
static_assert(L1_B_REGION + 2 * L1_B_ELEMS * 2 <= 512 * 1024, "A/B ping-pong L1 footprint exceeds 512KB");

constexpr uint32_t L0_PP_BYTES = 32 * 1024;
constexpr uint32_t L0_OFF_0 = 0;
constexpr uint32_t L0_OFF_1 = L0_PP_BYTES;
static_assert(BASE_M * BASE_K * 2 <= L0_PP_BYTES, "A2 tile exceeds L0A half");
static_assert(BASE_K * BASE_N * 2 <= L0_PP_BYTES, "B2 tile exceeds L0B half");

constexpr AscendC::MutexID BUF_A0 = 0;
constexpr AscendC::MutexID BUF_A1 = 1;
constexpr AscendC::MutexID BUF_B0 = 2;
constexpr AscendC::MutexID BUF_B1 = 3;
constexpr AscendC::MutexID BUF_L0_0 = 4;
constexpr AscendC::MutexID BUF_L0_1 = 5;
constexpr AscendC::MutexID BUF_L0C = 6;

__aicore__ __inline__ constexpr uint32_t OpCeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ __inline__ constexpr uint32_t OpAlignUp(uint32_t a, uint32_t b) { return OpCeilDiv(a, b) * b; }
__aicore__ __inline__ constexpr uint32_t OpMinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

template <uint32_t K, uint32_t N>
class OProjGemm {
    static_assert(K % BASE_K == 0, "K must be divisible by BASE_K");
    static_assert(N % BASE_N == 0, "N must be divisible by BASE_N");

public:
    __aicore__ inline OProjGemm() {}

    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, uint32_t m,
                               uint32_t mode = GEMM_MODE_CONTRACT)
    {
        aGMOri.SetGlobalBuffer((__gm__ bfloat16_t*)a);
        bGMOri.SetGlobalBuffer((__gm__ bfloat16_t*)b);
        cGMOri.SetGlobalBuffer((__gm__ bfloat16_t*)c);
        mTotal = m;
        mode_ = mode;
    }

    __aicore__ inline void Process()
    {
        const uint32_t mLoop = OpCeilDiv(mTotal, BASE_M);
        const uint32_t nLoop = N / BASE_N;
        const uint32_t kLoopAll = K / BASE_K;
        const uint32_t kLoop = (mode_ == GEMM_MODE_KMINUS1) ? (kLoopAll - 1u) : kLoopAll;  // 负向对照

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
            const uint32_t curM = OpMinU32(mTotal - mBlock * BASE_M, BASE_M);
            // 3510 实测（m1 契约，donor `m11_bf16_gemm.asc:122-124` 逐字）：Nd2Nz 行数为 1 时不做 NZ
            // 切分（退化为 1D 拷贝），导致 m=1 数据错位 ⇒ 计算侧统一提升为 >=2 行，结果行不写出。
            const uint32_t calcM = curM < 2 ? 2 : curM;
            const uint32_t calcMAlign = OpAlignUp(calcM, CUBE_BLOCK);
            for (uint32_t nBlock = 0; nBlock < nLoop; ++nBlock) {
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

                    AscendC::Mutex::Lock<PIPE_MTE2>(bufA);
                    CopyInA(a1, kBlock, mBlock, calcM);
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
                    mmadParams.n = BASE_N;
                    mmadParams.k = BASE_K;
                    mmadParams.cmatrixInitVal = (kBlock == 0);
                    AscendC::Mmad(cL0C, a2, b2, mmadParams);
                    AscendC::Mutex::Unlock<PIPE_M>(bufL0);
                }
                AscendC::Mutex::Unlock<PIPE_M>(BUF_L0C);
                AscendC::Mutex::Lock<PIPE_FIX>(BUF_L0C);
                CopyOut(cL0C, mBlock, nBlock, curM, calcMAlign);
                AscendC::Mutex::Unlock<PIPE_FIX>(BUF_L0C);
            }
        }
    }

private:
    __aicore__ inline void CopyInA(AscendC::LocalTensor<bfloat16_t>& a1, uint32_t kBlock, uint32_t mBlock,
                                   uint32_t curM)
    {
        AscendC::Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = curM;
        par.dValue = BASE_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;
        par.dstNzC0Stride = BASE_M;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1, aGMOri[kBlock * BASE_K + mBlock * K * BASE_M], par);
    }

    __aicore__ inline void CopyInB(AscendC::LocalTensor<bfloat16_t>& b1, uint32_t kBlock, uint32_t nBlock)
    {
        AscendC::Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = BASE_N;
        par.dValue = BASE_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;
        par.dstNzC0Stride = BASE_N;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1, bGMOri[kBlock * BASE_K + nBlock * BASE_N * K], par);
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

    __aicore__ inline void CopyOut(AscendC::LocalTensor<float>& cL0C, uint32_t mBlock, uint32_t nBlock,
                                   uint32_t curM, uint32_t calcMAlign)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp = {};
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
    uint32_t mode_;
};

// 编译期钉子：本段只用 (K=6144, N=2560) 这一个实例（`m11_bf16_gemm.asc:444` 的 out_proj 形状）
using OProjGemm6144x2560 = OProjGemm<OP_K, OP_N>;
static_assert(OP_K % BASE_K == 0 && OP_N % BASE_N == 0, "o_proj 形状必须能被 base 块整除");

// AIC 段 body（薄壳，供调用方在 AIC 分支里直接调）
__aicore__ inline void OProjGemmAic(__gm__ uint8_t* t, __gm__ uint8_t* w, __gm__ uint8_t* y, uint32_t m,
                                    uint32_t mode)
{
    OProjGemm6144x2560 op;
    op.Init(t, w, y, m, mode);
    op.Process();
}

}  // namespace M15OP

#endif  // M15_ATTN_OPROJ_H
