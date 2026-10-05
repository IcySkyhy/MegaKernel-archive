// ============================================================
// m15_attn_prolog_probe.h —— M88：prolog 的 **device 探针**（与 M82 的 `m15_attn_kv_probe.h` 同构）
//
// 为什么是探针而不是直接接进融合 kernel：把 prolog 接进 `M15L_FusedBody` 需要给
// `M15L_LAYER_HC_ARGS_DECL` 加参，而它**唯一的 host 调用点**在 `m15_hc_host.h:169-204`，
// 不在本 mission 的 scope（见 README 的「未完成项 1」）。探针把 prolog 的**数值语义**先钉死并可测。
//
// 结构（一次 launch 里 AIC 与 AIV 都有真活，**不是**占位直通）：
//   AIC：4 个 bf16 GEMM（q_proj / k_proj / v_proj / index_qk_proj）
//        MTE2 Nd2Nz(GM→L1) ∥ MTE1 LoadData(L1→L0) ∥ M Mmad ∥ FIXP F322BF16(→GM)
//        —— 流水骨架改自 `m15_gdn_layer.h:1403-1580` 的 `Cube::Bf16Gemm`：AP_BASE_N 160→128
//        （整除推导见 `m15_attn_prolog.h` §6），三个 GEMM 串成一条 N-块步进序列。
//   AIV：q/k 的 GemmaRMSNorm(256)+RoPE(64)、indexer q 的 GemmaRMSNorm(128)+RoPE(64)，
//        以及 gate / v / raw k 的**原样抄写**（R2/R13）。
//
// 【诊断闸】入口多一个 `dbg` 参数（host 侧环境变量 `M15_AP_DBG` 传入），用于把失败定位到哪一半：
//   dbg=0 全跑 | 1 跳 AIC 的 GEMM（flag 照发）| 2 跳 AIV 全部（不发 flag）| 3 全跳（冒烟）
//   dbg=4 AIC 只发两条 flag、AIV 全跳 | 5 AIV 只等一次 flag 后 return
//
// 同步（人类结构约束：核内 buffer id、核间 set cross core；除有值依赖外不用 PIPE_S）：
//   · 核内：`AscendC::Mutex::Lock/Unlock<PIPE_*>(BUF_*)`（AIC 的 L1/L0/L0C ping-pong）与
//     `BufAcquire/BufRelease`（AIV 的 UB 中转），**全部静态分配**（`m15_attn_prolog.h` §8 + M15G 的号段）。
//   · 核间：`CrossCoreSetFlag/WaitFlag`，**两条缺一不可**：
//       ① AIC mode-0 `AP_AIC_M0_OUT`：4 个 GEMM 的 FIXP 写 GM 全部排空 + 全体 AIC 对齐；
//       ② AIC→其配对 2 个 AIV 的 mode-2 `AP_A2V_GEMM`。
//     只有 ② 的话，AIV 只知道自己那对 AIC 好了。这与 `m15_gdn_layer.h:1758-1760` 的
//     `FLAG_AIC_SEG0B → FLAG_BOUND_INPROJ` 完全同构。
//     ✅ **M88 r2 定案：这一对本身是对的。** 当时的挂死是**一个花括号**：
//       r1 里 AIV 那条臂被整块写在 `if ASCEND_IS_AIC {` **内部**（全文件只有 `ASCEND_IS_AIC`、
//       没有 `ASCEND_IS_AIV`）⇒ AIV 核上整段被编译掉（与文件头自称的"AIC 与 AIV 都有真活"相反），
//       而 AIC 会执行属于 AIV 的那条 `CrossCoreWaitFlag` ⇒ **等一个只有 AIV 能置的 flag ⇒ 死锁**。
//       这逐个解释了 r1 的 rc 表：`dbg∈{2,3}` 不进那段 → rc=0；`dbg∈{0,1,4,5}` 进 → rc=124。
//       **r1 在 README §4 / 本文件 / m15_attn_prolog.h 里写的「mode-2 set 会阻塞 / 配对不符 / 根因未隔离」
//       是错的**（dbg=2 会发 flag 而 AIV 不消费却 rc=0，本身就否证了「set 阻塞」）；那组 rc 读数可复现，
//       但**当时的解释作废**。归位后 `runs=prolog` 由 rc=124 变 rc=1（见 evidence/attn_prolog_r2_*.log）。
//       姊妹文件的两段式写法见 `m15_attn_kv_probe.h` 的 `if ASCEND_IS_AIV {` 臂 / `m15_gdn_layer.h:1839,1842`。
//   · ⚠ **归位后又暴露了两处真缺陷**（挂死把它们盖住了，逐条见 README §4）：
//       (a) `Block1()` 的入参是**字节**（`m15_gdn_layer.h:91-97`，非 32 B 整数倍直接 `Trap`），
//           而 prolog 首版传的是"32 B 块数" ⇒ AIV 的每一次搬运都 `Trap`；
//       (b) 三段式 MTE2→V→MTE3 的 UB 交接**必须用两个 buffer id**（GDN 用
//           `BUF_AIV_ROW0`+`BUF_AIV_OUT`），用一个 id 串完三段时 V 的写与 MTE3 的读没被排序。
//
//   · `PIPE_S` 只出现在等 cross-core flag 的 wait 侧（与 GDN/hc/MoE 段一致）；核内数据依赖一律 buffer id，
//     **没有** set/wait flag 系列同步。
// ============================================================
#ifndef M15_ATTN_PROLOG_PROBE_H
#define M15_ATTN_PROLOG_PROBE_H

#include "m15_attn_prolog.h"

namespace M15AP {

using namespace AscendC;
using namespace AscendC::Reg;

// 与 `m15_gdn_layer.h:117-125` 的 NormDonor 同名同值（**拷贝**：本头文件不依赖 GDN 段的命名空间，
// 但两处必须一致，否则 norm 的落盘字节会不同）
constexpr AscendC::Reg::CastTrait AP_BF16_TO_F32 = {AscendC::Reg::RegLayout::ZERO,
                                                    AscendC::Reg::SatMode::UNKNOWN,
                                                    AscendC::Reg::MaskMergeMode::ZEROING,
                                                    RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait AP_F32_TO_BF16 = {AscendC::Reg::RegLayout::ZERO,
                                                    AscendC::Reg::SatMode::NO_SAT,
                                                    AscendC::Reg::MaskMergeMode::ZEROING,
                                                    RoundMode::CAST_RINT};

constexpr uint32_t VL = 64;                     // fp32 向量寄存器 lane 数（`m15_gdn_layer.h:103`）
constexpr uint32_t AP_BUF_IO = 0;               // AIV：两段式中转（MTE2 写 → MTE3 读；`AivCopy`）
// ⚠ 三段式（MTE2 写 → PIPE_V 读/算/写 → MTE3 读）**必须用两个 id**：`m15_gdn_layer.h:371-372`
//   的 `BUF_AIV_ROW0`（MTE2→V）与 `BUF_AIV_OUT`（V→MTE3）就是两段。M88 实测：用**一个** id
//   串完三段时，AIV 输出在 device 上时而全 0、时而毒值、时而正确（`Ap.repeat` 报告 6144/6144
//   不同）—— 即 V 的写与 MTE3 的读**没有被真正排序**。
constexpr uint32_t AP_BUF_IN = 1;               // AIV：输入面（in/w/cs）MTE2 → PIPE_V
constexpr uint32_t AP_BUF_OUT = 2;              // AIV：输出头 PIPE_V → MTE3

// ============================================================
// A1. bf16 GEMM（`Cube::Bf16Gemm` 的 AP_BASE_N=128 版）
// ============================================================
template <uint32_t K, uint32_t N>
class AttnGemm {
    static_assert(K % AP_BASE_K == 0u, "K 必须被 AP_BASE_K=64 整除");
    static_assert(N % AP_BASE_N == 0u, "N 必须被 AP_BASE_N=128 整除（无尾块）");

public:
    static constexpr uint32_t Blocks() { return N / AP_BASE_N; }

    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c)
    {
        aGMOri.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a));
        bGMOri.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(b));
        cGMOri.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(c));
    }

    __aicore__ inline void RunTile(uint32_t nBlock)
    {
        constexpr uint32_t kLoop = K / AP_BASE_K;
        LocalTensor<bfloat16_t> a1P(TPosition::A1, L1_A0, AP_L1_A_ELEMS);
        LocalTensor<bfloat16_t> a1G(TPosition::A1, L1_A1, AP_L1_A_ELEMS);
        LocalTensor<bfloat16_t> b1P(TPosition::B1, L1_B0, AP_L1_B_ELEMS);
        LocalTensor<bfloat16_t> b1G(TPosition::B1, L1_B1, AP_L1_B_ELEMS);
        LocalTensor<bfloat16_t> a2P(TPosition::A2, L0_A0, AP_BASE_M * AP_BASE_K);
        LocalTensor<bfloat16_t> a2G(TPosition::A2, L0_A1, AP_BASE_M * AP_BASE_K);
        LocalTensor<bfloat16_t> b2P(TPosition::B2, L0_B0, AP_BASE_K * AP_BASE_N);
        LocalTensor<bfloat16_t> b2G(TPosition::B2, L0_B1, AP_BASE_K * AP_BASE_N);
        LocalTensor<float> cL0C(TPosition::CO1, 0, AP_BASE_M * AP_BASE_N);

        Mutex::Lock<PIPE_M>(M15G::BUF_AIC_L0C);
        for (uint32_t kBlock = 0; kBlock < kLoop; ++kBlock) {
            const uint32_t p = kBlock & 1u;
            const MutexID bufA = p ? M15G::BUF_AIC_A1 : M15G::BUF_AIC_A0;
            const MutexID bufB = p ? M15G::BUF_AIC_B1 : M15G::BUF_AIC_B0;
            const MutexID bufL0 = p ? M15G::BUF_AIC_L01 : M15G::BUF_AIC_L00;
            LocalTensor<bfloat16_t> a1 = p ? a1G : a1P;
            LocalTensor<bfloat16_t> b1 = p ? b1G : b1P;
            LocalTensor<bfloat16_t> a2 = p ? a2G : a2P;
            LocalTensor<bfloat16_t> b2 = p ? b2G : b2P;

            Mutex::Lock<PIPE_MTE2>(bufA);
            CopyInA(a1, kBlock);
            Mutex::Unlock<PIPE_MTE2>(bufA);
            Mutex::Lock<PIPE_MTE2>(bufB);
            CopyInB(b1, kBlock, nBlock);
            Mutex::Unlock<PIPE_MTE2>(bufB);

            Mutex::Lock<PIPE_MTE1>(bufL0);
            Mutex::Lock<PIPE_MTE1>(bufA);
            Mutex::Lock<PIPE_MTE1>(bufB);
            LoadA(a1, a2);
            LoadB(b1, b2);
            Mutex::Unlock<PIPE_MTE1>(bufA);
            Mutex::Unlock<PIPE_MTE1>(bufB);
            Mutex::Unlock<PIPE_MTE1>(bufL0);

            Mutex::Lock<PIPE_M>(bufL0);
            MmadParams mp = {};
            mp.m = CALC_M;
            mp.n = AP_BASE_N;
            mp.k = AP_BASE_K;
            mp.cmatrixInitVal = (kBlock == 0u);
            Mmad(cL0C, a2, b2, mp);
            Mutex::Unlock<PIPE_M>(bufL0);
        }
        Mutex::Unlock<PIPE_M>(M15G::BUF_AIC_L0C);
        Mutex::Lock<PIPE_FIX>(M15G::BUF_AIC_L0C);
        CopyOut(cL0C, nBlock);
        Mutex::Unlock<PIPE_FIX>(M15G::BUF_AIC_L0C);
    }

    // m = 1：**3510 实测 quirk** —— Nd2Nz 行数为 1 时不切 NZ，数据错位，故计算侧统一按 2 行
    //（多读的第 1 行由 host 保证可读；结果只写第 0 行）。契约同 `m15_gdn_layer.h:1439-1443`。
    static constexpr uint32_t CALC_M = 2u;
    static constexpr uint32_t CALC_M_ALIGN = AP_CUBE_BLOCK * ((CALC_M + AP_CUBE_BLOCK - 1u) / AP_CUBE_BLOCK);

private:
    __aicore__ inline void CopyInA(LocalTensor<bfloat16_t>& a1, uint32_t kBlock)
    {
        Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = CALC_M;
        par.dValue = AP_BASE_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;
        par.dstNzC0Stride = AP_BASE_M;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        DataCopy(a1, aGMOri[kBlock * AP_BASE_K], par);
    }

    __aicore__ inline void CopyInB(LocalTensor<bfloat16_t>& b1, uint32_t kBlock, uint32_t nBlock)
    {
        Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = AP_BASE_N;
        par.dValue = AP_BASE_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;
        par.dstNzC0Stride = AP_BASE_N;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        DataCopy(b1, bGMOri[kBlock * AP_BASE_K + nBlock * AP_BASE_N * K], par);
    }

    __aicore__ inline void LoadA(LocalTensor<bfloat16_t>& a1, LocalTensor<bfloat16_t>& a2)
    {
        LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = CALC_M_ALIGN / AP_CUBE_BLOCK;
        lp.kStep = AP_BASE_K / AP_CUBE_BLOCK;
        lp.srcStride = AP_BASE_M / AP_CUBE_BLOCK;
        lp.dstStride = CALC_M_ALIGN / AP_CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        LoadData(a2, a1[0], lp);
    }

    __aicore__ inline void LoadB(LocalTensor<bfloat16_t>& b1, LocalTensor<bfloat16_t>& b2)
    {
        LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = AP_BASE_N / AP_CUBE_BLOCK;
        lp.kStep = AP_BASE_K / AP_CUBE_BLOCK;
        lp.srcStride = AP_BASE_N / AP_CUBE_BLOCK;
        lp.dstStride = AP_BASE_N / AP_CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        LoadData(b2, b1[0], lp);
    }

    __aicore__ inline void CopyOut(LocalTensor<float>& cL0C, uint32_t nBlock)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fp = {};
        fp.nSize = AP_BASE_N;
        fp.mSize = CALC_M - 1u;              // 只写第 0 行（m=1）
        fp.srcStride = CALC_M_ALIGN;
        fp.dstStride = N;
        fp.quantPre = QuantMode_t::F322BF16;
        fp.reluScalar = 0;
        fp.vectorRelu = 0;
        fp.deqScalar = 0;
        Fixpipe(cGMOri[nBlock * AP_BASE_N], cL0C, fp);
    }

    GlobalTensor<bfloat16_t> aGMOri;
    GlobalTensor<bfloat16_t> bGMOri;
    GlobalTensor<bfloat16_t> cGMOri;
};

// ============================================================
// A2. AIV：GemmaRMSNorm(D) + partial NeoX RoPE(64)
// ============================================================
template <uint32_t D>
class AivNormRope {
    static_assert(D == HD || D == IDX_D, "只支持 256（主干）/128（indexer）");
    static constexpr uint32_t CH = D / VL;          // 64-lane chunk 数：256→4、128→2

public:
    __aicore__ inline void Init(__gm__ bfloat16_t* inGm, __gm__ bfloat16_t* wGm, __gm__ bfloat16_t* csGm,
                                __gm__ bfloat16_t* outGm, uint32_t mode)
    {
        inG.SetGlobalBuffer(inGm, D);
        wG.SetGlobalBuffer(wGm, D);
        outG.SetGlobalBuffer(outGm, D);
        csG.SetGlobalBuffer(csGm, CS_ROW_ELEMS);
        mmode = mode;
    }

    __aicore__ inline void Run()
    {
        LocalTensor<bfloat16_t> inL(TPosition::VECCALC, UB_AP_IN, D);
        LocalTensor<bfloat16_t> wL(TPosition::VECCALC, UB_AP_WB, D);
        LocalTensor<bfloat16_t> csL(TPosition::VECCALC, UB_AP_CS, 128);
        LocalTensor<float> redL(TPosition::VECCALC, UB_AP_RED, VL);
        LocalTensor<bfloat16_t> outL(TPosition::VECCALC, UB_AP_OUT, D);

        {   // MTE2：头 + norm 权重 + cos/sin 行（三者都是「本核一次性读入」）
            BufAcquire<PIPE_MTE2>(AP_BUF_IN);
            PipeBarrier<PIPE_MTE2>();       // 同 pipe 复用同一 UB 输入面（GDN `CopyInRow` 同款）
            DataCopy(inL, inG, Block1(D * 2u));
            DataCopy(wL, wG, Block1(D * 2u));
            DataCopy(csL, csG, Block1(CS_ROW_BYTES));
            BufRelease<PIPE_MTE2>(AP_BUF_IN);
        }

        // ⚠ **必须让 PIPE_V 也拿一次同一个 buffer id**：MTE2 写完 → V 读/算/写 → MTE3 读。
        //   此前只在 MTE2/MTE3 两侧 acquire/release，VEC 的读与写**完全没被排序** ⇒ 实测同一个
        //   配置重跑给出不同结果（`Ap.repeat` 报告 6144/6144 不同、首次 launch 甚至是毒值）。
        //   与 `m15_gdn_layer.h` 的 `ComputeRow()`（`BufAcquire<PIPE_V>(BUF_AIV_ROW0)`）同款。
        __ubuf__ bfloat16_t* inUb = reinterpret_cast<__ubuf__ bfloat16_t*>(inL.GetPhyAddr());
        __ubuf__ bfloat16_t* wUb = reinterpret_cast<__ubuf__ bfloat16_t*>(wL.GetPhyAddr());
        __ubuf__ bfloat16_t* csUb = reinterpret_cast<__ubuf__ bfloat16_t*>(csL.GetPhyAddr());
        __ubuf__ float* redUb = reinterpret_cast<__ubuf__ float*>(redL.GetPhyAddr());
        __ubuf__ bfloat16_t* outUb = reinterpret_cast<__ubuf__ bfloat16_t*>(outL.GetPhyAddr());

        BufAcquire<PIPE_V>(AP_BUF_IN);     // 等 MTE2 把 in/w/cs 写进 UB
        BufAcquire<PIPE_V>(AP_BUF_OUT);    // 等上一轮 MTE3 读完 out 缓冲
        __VEC_SCOPE__
        {
            RegTensor<float> xr;
            RegTensor<float> acc;
            RegTensor<float> t;
            RegTensor<float> t2;
            RegTensor<float> var;
            RegTensor<float> one;
            RegTensor<float> rstd;
            RegTensor<float> wr;
            RegTensor<float> y;
            RegTensor<bfloat16_t> sb;
            RegTensor<bfloat16_t> yb;
            MaskReg all = CreateMask<float, MaskPattern::ALL>();
            MaskReg m1 = CreateMask<float, MaskPattern::VL1>();
            MaskReg m32 = CreateMask<float, MaskPattern::VL32>();

            // ---- 1. Σx²（fp32 累加；与 oracle 的"norm 在 fp32 里算"一致）----
            // 两遍：先只算 Σx²（chunk 用单寄存器轮转，**不 declare RegTensor 数组** ——
            // bisheng 对 RegTensor[N] 会报 "Unsupported Inst must be hoisted"），
            // 第二遍再重读同一段 UB 算 y。UB 带宽不是本段的瓶颈。
            Duplicate(acc, 0.0f, all);
            Duplicate(one, 1.0f, all);
            for (uint16_t c = 0; c < static_cast<uint16_t>(CH); ++c) {
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, inUb + c * VL);
                Cast<float, bfloat16_t, AP_BF16_TO_F32>(xr, sb, all);
                Mul(t, xr, xr, all);
                Add(acc, acc, t, all);
            }
            ReduceSum(var, acc, all);                        // lane0 = Σx²（其余 lane 不用）
            // ---- 2. rstd = 1/sqrt(mean + eps)：Sqrt+Div 的精确式，**不用 NR 近似**
            //      （用 NR 就必须把它的精度写进 ε；这里把这一项误差从判据里彻底去掉）
            Muls(var, var, 1.0f / static_cast<float>(D), m1);
            Adds(var, var, 1e-6f, m1);                       // config.rms_norm_eps
            Div(rstd, one, var, m1);
            Sqrt(rstd, rstd, m1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(redUb, rstd, m1);
        }

        // ---- 3. y = x * rstd * (1+w)（AP_MODE_PLAIN_NORM 时用 w，不打 +1）----
        // ⚠ **`rstd` 的 UB 往返必须跨 `__VEC_SCOPE__`**：上一步把 lane0 的 rstd 写进 UB，这一步用
        //   `DIST_BRC_B32` 广播回来，是同一条 PIPE_V 上的 store→load RAW。M88 r2 实测：放在**同一个**
        //   `__VEC_SCOPE__` 里时，设备输出是**该头正确值的「每 launch 一个常数负倍数」**
        //   （mode3 比值 -0.03665、mode0 比值 -0.13134，各自 8 个元素稳定到 4 位有效数字，
        //   但两次 launch 之间不同）—— 正是「rstd 这一个标量被读成垃圾」的指纹。
        //   donor 就是这么分两段的：`NormDonor::ComputeRstdNewtonRaphson`（写 rstdUb）与
        //   `CalculateGateY`（`LoadAlign<DIST_BRC_B32>` 读回）各是一个 `__VEC_SCOPE__`。
        __VEC_SCOPE__
        {
            RegTensor<float> xr;
            RegTensor<float> rstd;
            RegTensor<float> wr;
            RegTensor<float> y;
            RegTensor<bfloat16_t> sb;
            RegTensor<bfloat16_t> yb;
            MaskReg all = CreateMask<float, MaskPattern::ALL>();
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, redUb);   // 广播：所有 lane = rstd
            for (uint16_t c = 0; c < static_cast<uint16_t>(CH); ++c) {
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, wUb + c * VL);
                Cast<float, bfloat16_t, AP_BF16_TO_F32>(wr, sb, all);
                if (mmode != AP_MODE_PLAIN_NORM) {
                    Adds(wr, wr, 1.0f, all);
                }
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, inUb + c * VL);
                Cast<float, bfloat16_t, AP_BF16_TO_F32>(xr, sb, all);
                Mul(y, xr, rstd, all);
                Mul(y, y, wr, all);
                Cast<bfloat16_t, float, AP_F32_TO_BF16>(yb, y, all);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(outUb + c * VL, yb, all);
            }
        }

        // ---- 4. NeoX partial RoPE（只旋 [0,ROT)）----
        // ⚠ **必须另起一个 `__VEC_SCOPE__`**：这一步要把上一步 VF **刚写进 UB** 的 norm 输出
        //   重新读回寄存器，属于同一条 pipe 上的 store→load RAW。M88 r2 实测：放在**同一个**
        //   `__VEC_SCOPE__` 里时，norm 输出（mode=3 读数）逐字节正确，而带 rope 的读数
        //   时而全错、时而以不同数值全错 ⇒ 那次读回是**竞争**。donor 就是这么分两段的：
        //   `NormDonor::ComputeRstdNewtonRaphson` 与 `CalculateGateY` 各是一个 `__VEC_SCOPE__`
        //   （`m15_gdn_layer.h:529` / `:459` 附近）。
        if (mmode != AP_MODE_NO_ROPE) {
            __VEC_SCOPE__
            {
                RegTensor<float> x1;
                RegTensor<float> x2;
                RegTensor<float> cR;
                RegTensor<float> sR;
                RegTensor<float> o1;
                RegTensor<float> o2;
                RegTensor<float> t;
                RegTensor<float> t2;
                RegTensor<bfloat16_t> sb;
                RegTensor<bfloat16_t> yb;
                MaskReg m32 = CreateMask<float, MaskPattern::VL32>();
                // 过读说明：每次 `LoadAlign` 的 bf16 寄存器是 128 lane，故从 outUb+32 / csUb+32
                // 起读会多读 64 个 bf16；两段 UB 后面都还有别的段（`m15_attn_prolog.h` §7 的排布），
                // 读得到但不越 UB，且被 m32 mask 掉。
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, outUb + 0);
                Cast<float, bfloat16_t, AP_BF16_TO_F32>(x1, sb, m32);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, outUb + HALF);
                Cast<float, bfloat16_t, AP_BF16_TO_F32>(x2, sb, m32);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, csUb + 0);
                Cast<float, bfloat16_t, AP_BF16_TO_F32>(cR, sb, m32);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, csUb + HALF);
                Cast<float, bfloat16_t, AP_BF16_TO_F32>(sR, sb, m32);
                if (mmode == AP_MODE_ROPE_SIGN) {
                    Mul(t, x1, cR, m32);
                    Mul(t2, x2, sR, m32);
                    Add(o1, t, t2, m32);                     // o1 = x1·c + x2·s
                    Mul(t, x2, cR, m32);
                    Mul(t2, x1, sR, m32);
                    Muls(t2, t2, -1.0f, m32);
                    Add(o2, t, t2, m32);                     // o2 = x2·c − x1·s
                } else {
                    Mul(t, x1, cR, m32);
                    Mul(t2, x2, sR, m32);
                    Muls(t2, t2, -1.0f, m32);
                    Add(o1, t, t2, m32);                     // o1 = x1·c − x2·s
                    Mul(t, x2, cR, m32);
                    Mul(t2, x1, sR, m32);
                    Add(o2, t, t2, m32);                     // o2 = x2·c + x1·s
                }
                Cast<bfloat16_t, float, AP_F32_TO_BF16>(yb, o1, m32);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(outUb + 0, yb, m32);
                Cast<bfloat16_t, float, AP_F32_TO_BF16>(yb, o2, m32);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(outUb + HALF, yb, m32);
            }
        }

        BufRelease<PIPE_V>(AP_BUF_IN);     // in/w/cs 消费完，MTE2 可复用
        BufRelease<PIPE_V>(AP_BUF_OUT);    // outL 写完且 drain，MTE3 才可读
        BufAcquire<PIPE_MTE3>(AP_BUF_OUT);  // 等 PIPE_V 的 release（= outL 已写完）
        DataCopy(outG, outL, Block1(D * 2u));
        BufRelease<PIPE_MTE3>(AP_BUF_OUT);
    }

private:
    GlobalTensor<bfloat16_t> inG;
    GlobalTensor<bfloat16_t> wG;
    GlobalTensor<bfloat16_t> outG;
    GlobalTensor<bfloat16_t> csG;
    uint32_t mmode = 0;
};

// ============================================================
// A3. AIV：纯抄写（gate / v / raw k —— R2/R13 的"原样"通道）
// ============================================================
__aicore__ inline void AivCopy(__gm__ bfloat16_t* srcGm, __gm__ bfloat16_t* dstGm, uint32_t elems)
{
    const uint32_t bytes = elems * 2u;
    if ((bytes & 31u) != 0u) {
        AscendC::Trap();
    }
    LocalTensor<bfloat16_t> buf(TPosition::VECCALC, UB_AP_CPY, 512);
    GlobalTensor<bfloat16_t> s;
    GlobalTensor<bfloat16_t> d;
    s.SetGlobalBuffer(srcGm, elems);
    d.SetGlobalBuffer(dstGm, elems);
    BufAcquire<PIPE_MTE2>(AP_BUF_IO);
    DataCopy(buf, s, Block1(bytes));
    BufRelease<PIPE_MTE2>(AP_BUF_IO);
    BufAcquire<PIPE_MTE3>(AP_BUF_IO);
    DataCopy(d, buf, Block1(bytes));
    BufRelease<PIPE_MTE3>(AP_BUF_IO);
}

// ============================================================
// B. GEMM 的 N-块总表（4 个 GEMM 顺次编号；每个 AIC 按 core 步进取块）
// ============================================================
constexpr uint32_t GEMM_NBLOCKS = AttnGemm<AP_HIDDEN, QG_W>::Blocks() + AttnGemm<AP_HIDDEN, NKV * HD>::Blocks() +
                                  AttnGemm<AP_HIDDEN, NKV * HD>::Blocks() + AttnGemm<AP_HIDDEN, IDX_W>::Blocks();
static_assert(GEMM_NBLOCKS == 96u + 4u + 4u + 5u, "96+4+4+5 = 109 个 N-块");

// ============================================================
// C. 探针入口
// ============================================================
__aicore__ inline void m15_attn_prolog_probe_body(__gm__ uint8_t* wPlane, __gm__ uint8_t* xGm,
                                                  __gm__ uint8_t* csGm, __gm__ uint8_t* y0Gm,
                                                  __gm__ uint8_t* outGm, uint32_t pos, uint32_t mode,
                                                  uint32_t dbg)
{
    __gm__ bfloat16_t* w = reinterpret_cast<__gm__ bfloat16_t*>(wPlane);
    __gm__ bfloat16_t* y0 = reinterpret_cast<__gm__ bfloat16_t*>(y0Gm);
    __gm__ bfloat16_t* og = reinterpret_cast<__gm__ bfloat16_t*>(outGm);
    __gm__ bfloat16_t* cs = reinterpret_cast<__gm__ bfloat16_t*>(csGm) + static_cast<uint64_t>(pos) * CS_ROW_ELEMS;

    // 诊断闸（kernel 第 8 个参数；host 侧环境变量 M15_AP_DBG）。**每个取值都保证 set/wait 配平**，
    // 否则「AIV 等一个没人发的 flag」会退化成死锁、读数无法解释：
    //   dbg=0 全跑（AIC 4 个 GEMM + 两条 flag，AIV 等 + norm/rope/抄写）—— 契约档
    //   dbg=1 跳过 AIC 的 GEMM，两条 flag 照发，AIV 全跑 ⇒ 只测 AIV 半（输出是垃圾，但不断死）
    //   dbg=2 AIC 全跑，AIV **只等不干活** ⇒ 只测 AIC 半
    //   dbg=3 两边全跳（只 InitSocState + PipeBarrier）⇒ launch 冒烟
    //   dbg=4 AIC 只发两条 flag（不跑 GEMM），AIV **连 wait 都不做** ⇒ 单独验「mode-2 set 是否
    //         fire-and-forget」（若会阻塞则本档必超时）
    //   dbg=6 同 dbg=0 但 AIV **跳过 norm/rope**（只做抄写）⇒ 定位 VF 段
    //   dbg=7 同 dbg=0 但 AIV **跳过抄写**（只做 norm/rope）⇒ 定位 MTE 段
    const bool apAicGemm = (dbg == 0u || dbg == 2u || dbg == 6u || dbg == 7u);
    const bool apAicFlags = (dbg != 3u);
    const bool apAivNone = (dbg == 3u || dbg == 4u);
    const bool apAivWork = (dbg == 0u || dbg == 1u || dbg == 6u || dbg == 7u || dbg == 8u);
    //   dbg=8 同 dbg=0 但 AIV 0..23 **不抄 gate** ⇒ 单变量定位「q 头之后那次抄写是否干扰 out.q」
    const bool apGateCopy = (dbg != 8u);

    if ASCEND_IS_AIC {
      if (apAicGemm && mode != AP_MODE_NO_GEMM) {
        AttnGemm<AP_HIDDEN, QG_W> gq;
        AttnGemm<AP_HIDDEN, NKV * HD> gk;
        AttnGemm<AP_HIDDEN, NKV * HD> gv;
        AttnGemm<AP_HIDDEN, IDX_W> gi;
        gq.Init(xGm, wPlane + W_Q_OFF, y0Gm + Y0_QG * 2u);
        gk.Init(xGm, wPlane + W_K_OFF, y0Gm + Y0_K * 2u);
        gv.Init(xGm, wPlane + W_V_OFF, y0Gm + Y0_V * 2u);
        gi.Init(xGm, wPlane + W_IDX_OFF, y0Gm + Y0_IDX * 2u);

        const uint32_t bid = GetBlockIdx();
        const uint32_t nCores = GetBlockNum();          // mix(1,2)：AIC 侧 = blockDim
        for (uint32_t g = bid; g < GEMM_NBLOCKS; g += nCores) {
            if (g < 96u) {
                gq.RunTile(g);
            } else if (g < 100u) {
                gk.RunTile(g - 96u);
            } else if (g < 104u) {
                gv.RunTile(g - 100u);
            } else {
                gi.RunTile(g - 104u);
            }
        }
      }
      if (apAicFlags) {
        // ① 全体 AIC 的 mode-0 对齐：FIXP 写 GM 排空 + 所有核到齐
        CrossCoreSetFlag<AP_CC_M0, PIPE_FIX>(AP_AIC_M0_OUT);
        CrossCoreWaitFlag<AP_CC_M0, PIPE_S>(AP_AIC_M0_OUT);
        // ② 通知本核配对的 2 个 AIV（此刻 ① 已成立 ⇒ 这条 mode-2 蕴含"全体 AIC 完成"）
        CrossCoreSetFlag<AP_CC_M2, PIPE_MTE2>(AP_A2V_GEMM);
      }
    }

    if ASCEND_IS_AIV {
      if (!apAivNone) {
        const uint32_t bid = GetBlockIdx();             // 0 .. 2*numBlocks-1
        CrossCoreWaitFlag<AP_CC_M2, PIPE_MTE2>(AP_A2V_GEMM);
        if (!apAivWork) {
            return;
        }

        if (bid < NH) {                                  // 每头一个 AIV：q 头 bid
            AivNormRope<HD> q;
            q.Init(y0 + Y0_QG + bid * 2u * HD, w + W_QN_OFF / 2u, cs, og + OUT_Q + bid * HD, mode);
            if (dbg != 6u) { q.Run(); }
            // ⚠ 同一核上接着做两件互不相干的事（norm/rope 写 out.q；抄写读 y0 写 out.gate）。
            //   r2 实测：不插核内排空时 `Ap.out.q` **时而 PASS 时而 FAIL**（`dbg=7` 关掉抄写后稳定 PASS）
            //   ⇒ V 段的写与其后的 MTE2/MTE3 之间缺一次核内排空。这里插一条 `PIPE_ALL`（**核内**排空，
            //   不是跨核 flag；跨核仍走 mode-0/mode-2）。
            AscendC::PipeBarrier<PIPE_ALL>();
            if (dbg != 7u && dbg != 8u && mode != AP_MODE_NO_COPY) { AivCopy(y0 + Y0_QG + bid * 2u * HD + HD, og + OUT_GATE + bid * HD, HD); }
        } else if (bid == NH) {                          // k 头 0
            AivNormRope<HD> k;
            k.Init(y0 + Y0_K, w + W_KN_OFF / 2u, cs, og + AP_OUT_K, mode);
            if (dbg != 6u) { k.Run(); }
        } else if (bid == NH + 1u) {                     // k 头 1
            AivNormRope<HD> k;
            k.Init(y0 + Y0_K + HD, w + W_KN_OFF / 2u, cs, og + AP_OUT_K + HD, mode);
            if (dbg != 6u) { k.Run(); }
        } else if (bid == NH + 2u) {                     // v（2 头，1024 B）
            if (dbg != 7u) { AivCopy(y0 + Y0_V, og + OUT_V, 2u * HD); }
        } else if (bid >= NH + 3u && bid < NH + 3u + IDX_NH) {   // indexer q 的 4 个头
            const uint32_t h = bid - (NH + 3u);
            AivNormRope<IDX_D> i;
            i.Init(y0 + Y0_IDX + h * IDX_D, w + W_IQN_OFF / 2u, cs, og + OUT_QIDX + h * IDX_D, mode);
            if (dbg != 6u) { i.Run(); }
        } else if (bid == NH + 3u + IDX_NH) {            // raw k（R13：不 norm 不 rope）
            if (dbg != 7u) { AivCopy(y0 + Y0_IDX + IDX_Q, og + OUT_KRAW, IDX_D); }
        }
      }
    }
}

__global__ __mix__(1, 2) void m15_attn_prolog_probe_kernel(__gm__ uint8_t* wPlane, __gm__ uint8_t* xGm,
                                                           __gm__ uint8_t* csGm, __gm__ uint8_t* y0Gm,
                                                           __gm__ uint8_t* outGm, uint32_t pos, uint32_t mode,
                                                           uint32_t dbg)
{
    AscendC::InitSocState();
    m15_attn_prolog_probe_body(wPlane, xGm, csGm, y0Gm, outGm, pos, mode, dbg);
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace M15AP

#endif  // M15_ATTN_PROLOG_PROBE_H
