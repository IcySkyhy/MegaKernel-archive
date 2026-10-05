// m15_prefill_epilog.h -- M164: prefill epilog segment lift
//   (transpose -> S5 RMSNormGated -> S6 out_proj), device code only.
//
// Composition (all lifted/copied; no donor math rewritten):
//   * M15PE::Trans:  m28_gdn_epilog.asc lines 69..220 verbatim (standalone transpose+z kernel,
//                    incl. MUT_HEADSTRIDE negative control).
//   * M15PE::Chain:  m28_epilog_chain.asc lines 91..893 verbatim (ChainTransposeKernel +
//                    NormDonor + ChainS5Kernel with MUT_S5_NOGAMMA/MUT_S5_ZHEAD0 + S6::Bf16Gemm).
//   Entry shells below wrap them for host `<<<...>>>` launch.
//
// Lift-equivalence witness: evidence/m164_epilog_mount/verify_lift.py re-extracts the donor
// byte ranges and compares to the two marked regions (non-zero exit on mismatch).
//
// Human constraints: only S6 is a matmul -> cube (`Mmad`); Trans/S5 are pure DMA / elementwise-VF;
// scalar is control flow only. No set_flag/wait_flag, no cross-core flags: the three kernels are
// separate `__global__` launches on one stream, ordered by host `aclrtSynchronizeStream`.
#ifndef M15_PREFILL_EPILOG_H
#define M15_PREFILL_EPILOG_H

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "m15_attn_oproj.h"   // M15OP::OProjGemm<K,N> (donor lift; carries GEMM_MODE_KMINUS1)

// epilog shape constants (M159 section 2/3; same numbers as the donors)
namespace M15PE {
constexpr uint32_t PF_EPI_HEADS = 48u;    // value heads
constexpr uint32_t PF_EPI_HEAD = 128u;    // head dim
constexpr uint32_t PF_EPI_HIDDEN = 6144u; // S5 width = S6 K
constexpr uint32_t PF_EPI_OUT_N = 2560u;  // S6 N = hcAttnOut HID
}  // namespace M15PE

// ============================================================
// M15PE::Trans -- verbatim device region of m28_gdn_epilog.asc (lines 69..220)
// ============================================================
namespace M15PE {
namespace Trans {
// >>> M15PE_TRANS_BEGIN verbatim=m28_gdn_epilog.asc:69-220
namespace {
using namespace AscendC;

constexpr uint32_t HEADS = 48;             // value head 数（= M15G::HEADS）
constexpr uint32_t HEAD = 128;             // head 维（= M15G::HEAD_D）
constexpr uint32_t HIDDEN = HEADS * HEAD;  // 6144（m12 的 HIDDEN）
constexpr uint32_t IN_N = 16480;           // in_proj 输出宽度（qkvzba 行距）
constexpr uint32_t Z_OFF = 10240;          // z 段起始元素（= V_OFF + V_DIM）
constexpr uint32_t Z_DIM = 6144;           // z 段宽度（= HIDDEN）
constexpr uint32_t MAX_AIV = 28;           // 950PR 单芯片 AIV 组数（m8_permute 同款）

// ---- phase 1：o 转位的 token 分块（T_O × 128 fp32 = 98304 B）----
constexpr uint32_t T_O = 192;
// ---- phase 2：z 落位的 token 分块（T_Z × 6144 bf16 = 98304 B）----
constexpr uint32_t T_Z = 8;

// ---- UB 静态布局（字节偏移；两窗不重叠、均 32B 对齐，供 DataCopy）----
constexpr uint32_t UB_O = 0;                       // [T_O,128] fp32 = 98304B
constexpr uint32_t UB_O_BYTES = T_O * HEAD * 4;
constexpr uint32_t UB_Z = UB_O + UB_O_BYTES;       // [T_Z,6144] bf16 = 98304B
constexpr uint32_t UB_Z_BYTES = T_Z * Z_DIM * 2;
static_assert(UB_Z + UB_Z_BYTES <= 248 * 1024, "UB footprint exceeds 248KB");

// ---- BufferID（每核用户可用 0-27，静态分配）----
constexpr AscendC::MutexID BUF_O = 0;   // o 转位 tile: MTE2 -> MTE3
constexpr AscendC::MutexID BUF_Z = 1;   // z 落位 tile: MTE2 -> MTE3

// ---- 负向对照模式 ----
constexpr uint32_t MUT_NONE = 0;
constexpr uint32_t MUT_HEADSTRIDE = 1;  // phase 1 用错 head-stride（128 而非 m*128）

// BufferID 语义封装：acquire 立即获取；release 用阻塞模式（mode=false = CANN 默认 ASC_LOCK_BLOCK；true=NON_BLOCK），
// 阻塞至本 pipe 已发射访存（GM->UB 拷贝 / UB->GM 写）落地，对跨 pipe 消费者可见。
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
// Kernel：转位 + 落位（两个 phase 顺序执行；工作项按核 stride 划分）
// ============================================================

class GdnEpilogKernel {
public:
    __aicore__ inline GdnEpilogKernel() {}

    __aicore__ inline void Init(GM_ADDR wsO, GM_ADDR qkvzba, GM_ADDR oTok, GM_ADDR zTok, uint32_t m,
                                uint32_t mutant)
    {
        wsOGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsO), static_cast<uint64_t>(HEADS) * m * HEAD);
        qkvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(qkvzba), static_cast<uint64_t>(m) * IN_N);
        oTokGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(oTok), static_cast<uint64_t>(m) * HIDDEN);
        zTokGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(zTok), static_cast<uint64_t>(m) * Z_DIM);
        m_ = m;
        mutant_ = mutant;
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nblk = AscendC::GetBlockNum();

        // phase 1：o 转位。工作项 = (head, token-tile)，共 nTileO*HEADS 个。
        const uint32_t nTileO = (m_ + T_O - 1) / T_O;
        const uint32_t itemsO = nTileO * HEADS;
        for (uint32_t it = bid; it < itemsO; it += nblk) {
            const uint32_t h = it % HEADS;
            const uint32_t t0 = (it / HEADS) * T_O;
            const uint32_t rows = (m_ - t0 < T_O) ? (m_ - t0) : T_O;
            TransposeHead(h, t0, rows);
        }

        // phase 2：z 落位。工作项 = token-tile，共 ceil(m/T_Z) 个。
        const uint32_t itemsZ = (m_ + T_Z - 1) / T_Z;
        for (uint32_t it = bid; it < itemsZ; it += nblk) {
            const uint32_t t0 = it * T_Z;
            const uint32_t rows = (m_ - t0 < T_Z) ? (m_ - t0) : T_Z;
            PackZ(t0, rows);
        }
    }

private:
    // 一个 (head, token-tile) 工作项：源连续 [rows,128] fp32 → UB → 目的 strided 写。
    __aicore__ inline void TransposeHead(uint32_t h, uint32_t t0, uint32_t rows)
    {
        // 源 offset（元素）：正确 = h*m*128 + t0*128；mutant = h*128 + t0*128（错 head-stride）。
        const uint64_t srcOff =
            (mutant_ == MUT_HEADSTRIDE)
                ? (static_cast<uint64_t>(h) * HEAD + static_cast<uint64_t>(t0) * HEAD)
                : (static_cast<uint64_t>(h) * m_ * HEAD + static_cast<uint64_t>(t0) * HEAD);

        LocalTensor<float> tile(TPosition::VECCALC, UB_O, rows * HEAD);
        BufAcquire<PIPE_MTE2>(BUF_O);
        {
            // 源连续 rows*128 个 fp32 → 单块（blockLen 单位 32B）。
            const AscendC::DataCopyParams cpIn{static_cast<uint16_t>(1),
                                               static_cast<uint16_t>(rows * HEAD * 4 / 32), 0, 0};
            DataCopy(tile, wsOGm[srcOff], cpIn);
        }
        BufRelease<PIPE_MTE2>(BUF_O);

        BufAcquire<PIPE_MTE3>(BUF_O);
        {
            // 目的 rows 个 512B 块，块间 gap = (6144-128) 个 fp32 = 752 个 32B 单位。
            const AscendC::DataCopyParams cpOut{static_cast<uint16_t>(rows),
                                                static_cast<uint16_t>(HEAD * 4 / 32),
                                                static_cast<uint16_t>(0),
                                                static_cast<uint16_t>((HIDDEN - HEAD) * 4 / 32)};
            DataCopy(oTokGm[static_cast<uint64_t>(t0) * HIDDEN + static_cast<uint64_t>(h) * HEAD], tile, cpOut);
        }
        BufRelease<PIPE_MTE3>(BUF_O);
    }

    // 一个 token-tile 工作项：qkvzba 每行取 z 段 → UB（行距 IN_N）→ 连续写 zTok。
    __aicore__ inline void PackZ(uint32_t t0, uint32_t rows)
    {
        LocalTensor<bfloat16_t> tile(TPosition::VECCALC, UB_Z, rows * Z_DIM);
        BufAcquire<PIPE_MTE2>(BUF_Z);
        {
            // 源 rows 行、每行 6144 个 bf16 = 384 个 32B 单位，行间 gap = (16480-6144)*2/32 = 646。
            const AscendC::DataCopyParams cpIn{static_cast<uint16_t>(rows),
                                               static_cast<uint16_t>(Z_DIM * 2 / 32),
                                               static_cast<uint16_t>((IN_N - Z_DIM) * 2 / 32),
                                               static_cast<uint16_t>(0)};
            DataCopy(tile, qkvGm[static_cast<uint64_t>(t0) * IN_N + Z_OFF], cpIn);
        }
        BufRelease<PIPE_MTE2>(BUF_Z);

        BufAcquire<PIPE_MTE3>(BUF_Z);
        {
            // 目的连续 rows*6144 个 bf16（blockCount=rows, blockLen=384, 无 gap）。
            const AscendC::DataCopyParams cpOut{static_cast<uint16_t>(rows),
                                                static_cast<uint16_t>(Z_DIM * 2 / 32), 0, 0};
            DataCopy(zTokGm[static_cast<uint64_t>(t0) * Z_DIM], tile, cpOut);
        }
        BufRelease<PIPE_MTE3>(BUF_Z);
    }

private:
    uint32_t m_ = 0;
    uint32_t mutant_ = MUT_NONE;
    AscendC::GlobalTensor<float> wsOGm, oTokGm;
    AscendC::GlobalTensor<bfloat16_t> qkvGm, zTokGm;
};

} // namespace
// <<< M15PE_TRANS_END
}  // namespace Trans
}  // namespace M15PE

// ============================================================
// M15PE::Chain -- verbatim device region of m28_epilog_chain.asc (lines 91..893)
// ============================================================
namespace M15PE {
namespace Chain {
// >>> M15PE_CHAIN_BEGIN verbatim=m28_epilog_chain.asc:91-893
namespace {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t HEADS = 48;             // value head 数（= M15G::HEADS）
constexpr uint32_t HEAD = 128;             // head 维（= M15G::HEAD_D）
constexpr uint32_t HIDDEN = HEADS * HEAD;  // 6144（m12 的 HIDDEN = S6 的 K）
constexpr uint32_t IN_N = 16480;           // in_proj 输出宽度（qkvzba 行距）
constexpr uint32_t Z_OFF = 10240;          // z 段起始元素（= V_OFF + V_DIM）
constexpr uint32_t Z_DIM = 6144;           // z 段宽度（= HIDDEN）
constexpr uint32_t OUT_N = 2560;           // hcAttnOut hidden（= M15H::HID，S6 的 N）
constexpr uint32_t MAX_AIV = 28;           // 950PR 单芯片 AIV 组数（m8_permute 同款）

// BufferID 语义封装：acquire 立即获取；release 用阻塞模式（mode=false = CANN 默认
// ASC_LOCK_BLOCK；true=NON_BLOCK），阻塞至本 pipe 已发射访存落地，对跨 pipe 消费者可见（m28/m12 同款）。
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
// 段 ①（AIV，纯 DMA）：wsO head-major → token-major o；qkvzba z 段 → 连续 z
//   —— lift 自 `m28_gdn_epilog.asc`（M148）；去掉转位 mutant，链段转位恒为正确式。
// ============================================================

constexpr uint32_t T_O = 192;                      // o 转位 token 分块（192×128×4 = 98304 B）
constexpr uint32_t T_Z = 8;                        // z 落位 token 分块（8×6144×2 = 98304 B）
constexpr uint32_t UB_O = 0;                       // [T_O,128] fp32 = 98304B
constexpr uint32_t UB_O_BYTES = T_O * HEAD * 4;
constexpr uint32_t UB_Z = UB_O + UB_O_BYTES;       // [T_Z,6144] bf16 = 98304B
constexpr uint32_t UB_Z_BYTES = T_Z * Z_DIM * 2;
static_assert(UB_Z + UB_Z_BYTES <= 248 * 1024, "UB footprint exceeds 248KB");

constexpr AscendC::MutexID BUF_T_O = 0;   // o 转位 tile: MTE2 -> MTE3
constexpr AscendC::MutexID BUF_T_Z = 1;   // z 落位 tile: MTE2 -> MTE3

class ChainTransposeKernel {
public:
    __aicore__ inline ChainTransposeKernel() {}

    __aicore__ inline void Init(GM_ADDR wsO, GM_ADDR qkvzba, GM_ADDR o, GM_ADDR z, uint32_t m)
    {
        wsOGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsO), static_cast<uint64_t>(HEADS) * m * HEAD);
        qkvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(qkvzba), static_cast<uint64_t>(m) * IN_N);
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(o), static_cast<uint64_t>(m) * HIDDEN);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(z), static_cast<uint64_t>(m) * Z_DIM);
        m_ = m;
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nblk = AscendC::GetBlockNum();

        // phase 1：o 转位。工作项 = (head, token-tile)，共 ceil(m/T_O)*HEADS 个。
        const uint32_t nTileO = (m_ + T_O - 1) / T_O;
        const uint32_t itemsO = nTileO * HEADS;
        for (uint32_t it = bid; it < itemsO; it += nblk) {
            const uint32_t h = it % HEADS;
            const uint32_t t0 = (it / HEADS) * T_O;
            const uint32_t rows = (m_ - t0 < T_O) ? (m_ - t0) : T_O;
            TransposeHead(h, t0, rows);
        }

        // phase 2：z 落位。工作项 = token-tile，共 ceil(m/T_Z) 个。
        const uint32_t itemsZ = (m_ + T_Z - 1) / T_Z;
        for (uint32_t it = bid; it < itemsZ; it += nblk) {
            const uint32_t t0 = it * T_Z;
            const uint32_t rows = (m_ - t0 < T_Z) ? (m_ - t0) : T_Z;
            PackZ(t0, rows);
        }
    }

private:
    // 一个 (head, token-tile) 工作项：源连续 [rows,128] fp32 → UB → 目的 strided 写。
    __aicore__ inline void TransposeHead(uint32_t h, uint32_t t0, uint32_t rows)
    {
        // 源 offset（元素）：正确 = h*m*128 + t0*128（`m15_gdn_prefill.h:1090`）。
        const uint64_t srcOff = static_cast<uint64_t>(h) * m_ * HEAD + static_cast<uint64_t>(t0) * HEAD;

        LocalTensor<float> tile(TPosition::VECCALC, UB_O, rows * HEAD);
        BufAcquire<PIPE_MTE2>(BUF_T_O);
        {
            const AscendC::DataCopyParams cpIn{static_cast<uint16_t>(1),
                                               static_cast<uint16_t>(rows * HEAD * 4 / 32), 0, 0};
            DataCopy(tile, wsOGm[srcOff], cpIn);
        }
        BufRelease<PIPE_MTE2>(BUF_T_O);

        BufAcquire<PIPE_MTE3>(BUF_T_O);
        {
            const AscendC::DataCopyParams cpOut{static_cast<uint16_t>(rows),
                                                static_cast<uint16_t>(HEAD * 4 / 32),
                                                static_cast<uint16_t>(0),
                                                static_cast<uint16_t>((HIDDEN - HEAD) * 4 / 32)};
            DataCopy(oGm[static_cast<uint64_t>(t0) * HIDDEN + static_cast<uint64_t>(h) * HEAD], tile, cpOut);
        }
        BufRelease<PIPE_MTE3>(BUF_T_O);
    }

    // 一个 token-tile 工作项：qkvzba 每行取 z 段 → UB（行距 IN_N）→ 连续写 z。
    __aicore__ inline void PackZ(uint32_t t0, uint32_t rows)
    {
        LocalTensor<bfloat16_t> tile(TPosition::VECCALC, UB_Z, rows * Z_DIM);
        BufAcquire<PIPE_MTE2>(BUF_T_Z);
        {
            const AscendC::DataCopyParams cpIn{static_cast<uint16_t>(rows),
                                               static_cast<uint16_t>(Z_DIM * 2 / 32),
                                               static_cast<uint16_t>((IN_N - Z_DIM) * 2 / 32),
                                               static_cast<uint16_t>(0)};
            DataCopy(tile, qkvGm[static_cast<uint64_t>(t0) * IN_N + Z_OFF], cpIn);
        }
        BufRelease<PIPE_MTE2>(BUF_T_Z);

        BufAcquire<PIPE_MTE3>(BUF_T_Z);
        {
            const AscendC::DataCopyParams cpOut{static_cast<uint16_t>(rows),
                                                static_cast<uint16_t>(Z_DIM * 2 / 32), 0, 0};
            DataCopy(zGm[static_cast<uint64_t>(t0) * Z_DIM], tile, cpOut);
        }
        BufRelease<PIPE_MTE3>(BUF_T_Z);
    }

private:
    uint32_t m_ = 0;
    AscendC::GlobalTensor<float> wsOGm, oGm;
    AscendC::GlobalTensor<bfloat16_t> qkvGm, zGm;
};

// ============================================================
// 段 ②（AIV/VF）：S5 RMSNormGated（lift 自 `m12_rmsnorm_gated.asc`）
//   数学：per head(128) `var=Σo²/128`、`rstd=1/sqrt(var+1e-6)`、
//         `y = bf16( ((o·rstd)·gamma) · sigmoid(z))`（vllm norm_before_gate=True, act=sigmoid；
//         `m21_layer_ref/ref/gdn.py:183-196` 为同语义参考）。
//   唯一改动：`Process()` 的行循环按核 stride 划分（m12 原为单核 for(row=0..M)）。
// ============================================================

constexpr uint32_t VL_F32 = 64;                     // 256B 向量寄存器 fp32 lane 数
constexpr uint16_t CHUNK_PER_HEAD = HEAD / VL_F32;  // 2
constexpr float EPS = 1e-6f;                        // RMSNormGated eps = config.rms_norm_eps
constexpr float AVG_FACTOR = 1.0f / static_cast<float>(HEAD);
constexpr uint32_t FOLD_POINT = HEAD / 2;
constexpr uint32_t REDUCE_TMP_STRIDE = 24;

constexpr uint32_t UB_S5_O = 0;                     // fp32 [6144] = 24576B
constexpr uint32_t UB_S5_Z = UB_S5_O + 4 * HIDDEN;  // bf16 [6144] = 12288B
constexpr uint32_t UB_S5_Y = UB_S5_Z + 2 * HIDDEN;  // bf16 [6144] = 12288B
constexpr uint32_t UB_S5_GB = UB_S5_Y + 2 * HIDDEN; // bf16 [128] = 256B
constexpr uint32_t UB_S5_GF = UB_S5_GB + 256;       // fp32 [128] = 512B
constexpr uint32_t UB_S5_TMP = UB_S5_GF + 512;      // fp32 [64] = 256B
constexpr uint32_t UB_S5_RED = UB_S5_TMP + 256;     // fp32 [64]
constexpr uint32_t UB_S5_RSTD = UB_S5_RED + 256;    // fp32 [64]
static_assert(UB_S5_RSTD + 256 <= 248 * 1024, "S5 UB footprint exceeds 248KB");

constexpr AscendC::MutexID BUF_S5_X = 0;    // o/z 行: MTE2 -> V
constexpr AscendC::MutexID BUF_S5_OUT = 1;  // out 行: V -> MTE3
constexpr AscendC::MutexID BUF_S5_G = 2;    // gamma: MTE2 -> V（kernel 开头一次）

// 段 ② 负向对照模式
constexpr uint32_t MUT_NONE = 0;
constexpr uint32_t MUT_S5_NOGAMMA = 1;  // gamma 预转表写成全 1.0（权重未生效）
constexpr uint32_t MUT_S5_ZHEAD0 = 2;   // 所有 head 都用 head 0 的 z 做 sigmoid 门

// ---- donor：m6/m12 的 RegBase 规约 lift（逐字复用 `m12_rmsnorm_gated.asc:117-385`）----
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

template <typename T, StoreDist FLOAT_STORE_DIST = StoreDist::DIST_NORM,
          StoreDist NON_FLOAT_STORE_DIST = StoreDist::DIST_PACK_B32>
__aicore__ inline void StoreRegForDtype(__ubuf__ T* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign<T, FLOAT_STORE_DIST>(dst + offset, src, preg);
    } else {
        RegTensor<T> dstReg;
        Cast<T, float, castTraitB322B16>(dstReg, src, preg);
        StoreAlign<T, NON_FLOAT_STORE_DIST>(dst + offset, dstReg, preg);
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
                                                uint32_t foldPoint, uint32_t tmpStride, uint32_t branchNum = 0)
{
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

} // namespace NormDonor

class ChainS5Kernel {
public:
    __aicore__ inline ChainS5Kernel() {}

    __aicore__ inline void Init(GM_ADDR o, GM_ADDR z, GM_ADDR gamma, GM_ADDR out, uint32_t m, uint32_t mutant)
    {
        M = m;
        mutant_ = mutant;
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(o), static_cast<uint64_t>(M) * HIDDEN);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(z), static_cast<uint64_t>(M) * HIDDEN);
        gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gamma), HEAD);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(M) * HIDDEN);
    }

    __aicore__ inline void Process()
    {
        PrecomputeGammaF32();
        // M149 集成改动：行循环按核 stride 划分（m12 原为单核 for(row=0..M)）。
        // 行间无依赖，逐行数学与 m12 一字未动；只为 m=4097 档在多核上跑得完。
        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nblk = AscendC::GetBlockNum();
        for (uint32_t row = bid; row < M; row += nblk) {
            CopyInRow(row);
            ComputeRow();
            CopyOutRow(row);
        }
    }

private:
    __aicore__ inline void PrecomputeGammaF32()
    {
        LocalTensor<bfloat16_t> gbL(TPosition::VECCALC, UB_S5_GB, HEAD);
        LocalTensor<float> gfL(TPosition::VECCALC, UB_S5_GF, HEAD);
        __ubuf__ bfloat16_t* gbUb = reinterpret_cast<__ubuf__ bfloat16_t*>(gbL.GetPhyAddr());
        __ubuf__ float* gfUb = reinterpret_cast<__ubuf__ float*>(gfL.GetPhyAddr());

        const AscendC::DataCopyParams cpGamma{1, 256 / 32, 0, 0};
        BufAcquire<PIPE_MTE2>(BUF_S5_G);
        DataCopy(gbL, gammaGm, cpGamma);
        BufRelease<PIPE_MTE2>(BUF_S5_G);

        BufAcquire<PIPE_V>(BUF_S5_G);
        __VEC_SCOPE__
        {
            RegTensor<bfloat16_t> gB16;
            RegTensor<float> gF;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t i = 0; i < CHUNK_PER_HEAD; ++i) {
                uint32_t offset = i * VL_F32;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gB16, gbUb + offset);
                Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(gF, gB16, maskAll);
                if (mutant_ == MUT_S5_NOGAMMA) {
                    Duplicate(gF, 1.0f, maskAll);  // 负向对照：gamma 权重未生效
                }
                StoreAlign<float, StoreDist::DIST_NORM_B32>(gfUb + offset, gF, maskAll);
            }
        }
        BufRelease<PIPE_V>(BUF_S5_G);
    }

    __aicore__ inline void CopyInRow(uint32_t row)
    {
        LocalTensor<float> oL(TPosition::VECCALC, UB_S5_O, HIDDEN);
        LocalTensor<bfloat16_t> zL(TPosition::VECCALC, UB_S5_Z, HIDDEN);
        const uint64_t off = static_cast<uint64_t>(row) * HIDDEN;
        const AscendC::DataCopyParams cpO{1, 24576 / 32, 0, 0};
        const AscendC::DataCopyParams cpZ{1, 12288 / 32, 0, 0};
        BufAcquire<PIPE_MTE2>(BUF_S5_X);
        DataCopy(oL, oGm[off], cpO);
        DataCopy(zL, zGm[off], cpZ);
        BufRelease<PIPE_MTE2>(BUF_S5_X);
    }

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
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdUb);
            for (uint16_t c = 0; c < CHUNK_PER_HEAD; ++c) {
                uint32_t offset = c * VL_F32;
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

    __aicore__ inline void ComputeRow()
    {
        LocalTensor<float> oL(TPosition::VECCALC, UB_S5_O, HIDDEN);
        LocalTensor<bfloat16_t> zL(TPosition::VECCALC, UB_S5_Z, HIDDEN);
        LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_S5_Y, HIDDEN);
        LocalTensor<float> gfL(TPosition::VECCALC, UB_S5_GF, HEAD);
        LocalTensor<float> tmpL(TPosition::VECCALC, UB_S5_TMP, 64);
        LocalTensor<float> redL(TPosition::VECCALC, UB_S5_RED, 64);
        LocalTensor<float> rstdL(TPosition::VECCALC, UB_S5_RSTD, 64);
        __ubuf__ float* oUb = reinterpret_cast<__ubuf__ float*>(oL.GetPhyAddr());
        __ubuf__ bfloat16_t* zUb = reinterpret_cast<__ubuf__ bfloat16_t*>(zL.GetPhyAddr());
        __ubuf__ bfloat16_t* yUb = reinterpret_cast<__ubuf__ bfloat16_t*>(yL.GetPhyAddr());
        __ubuf__ float* gfUb = reinterpret_cast<__ubuf__ float*>(gfL.GetPhyAddr());
        __ubuf__ float* tmpUb = reinterpret_cast<__ubuf__ float*>(tmpL.GetPhyAddr());
        __ubuf__ float* redUb = reinterpret_cast<__ubuf__ float*>(redL.GetPhyAddr());
        __ubuf__ float* rstdUb = reinterpret_cast<__ubuf__ float*>(rstdL.GetPhyAddr());

        BufAcquire<PIPE_V>(BUF_S5_X);
        BufAcquire<PIPE_V>(BUF_S5_OUT);
        for (uint32_t h = 0; h < HEADS; ++h) {
            __ubuf__ float* headO = oUb + h * HEAD;
            NormDonor::CalculateSquareReduceSum<float>(headO, redUb, tmpUb, 1, HEAD, HEAD, FOLD_POINT,
                                                       REDUCE_TMP_STRIDE);
            NormDonor::ComputeRstdNewtonRaphson<true, true>(redUb, rstdUb, 1, EPS, AVG_FACTOR, VL_F32);
            // 负向对照 MUT_S5_ZHEAD0：所有 head 都用 head 0 的 z。
            const uint32_t zHeadOff = (mutant_ == MUT_S5_ZHEAD0) ? 0u : (h * HEAD);
            CalculateGateY(headO, zUb + zHeadOff, gfUb, yUb + h * HEAD, rstdUb);
        }
        BufRelease<PIPE_V>(BUF_S5_X);
        BufRelease<PIPE_V>(BUF_S5_OUT);
    }

    __aicore__ inline void CopyOutRow(uint32_t row)
    {
        LocalTensor<bfloat16_t> yL(TPosition::VECCALC, UB_S5_Y, HIDDEN);
        const uint64_t off = static_cast<uint64_t>(row) * HIDDEN;
        const AscendC::DataCopyParams cpY{1, 12288 / 32, 0, 0};
        BufAcquire<PIPE_MTE3>(BUF_S5_OUT);
        DataCopy(outGm[off], yL, cpY);
        BufRelease<PIPE_MTE3>(BUF_S5_OUT);
    }

private:
    uint32_t M = 0;
    uint32_t mutant_ = MUT_NONE;
    GlobalTensor<float> oGm;
    GlobalTensor<bfloat16_t> zGm;
    GlobalTensor<bfloat16_t> gammaGm;
    GlobalTensor<bfloat16_t> outGm;
};

// ============================================================
// 段 ③（AIC/cube）：S6 out_proj（lift 自 `m11_bf16_gemm.asc`）
//   C[m,2560] = A[m,6144] · B[2560,6144]^T（B 按模型原始 [N,K] layout 存放）。
//   C 的行距 = N = 2560 元素 ⇒ 与 `hcAttnOut`（HID=2560，`m15_hc_resources.h:40`）一致。
// ============================================================

namespace S6 {

__aicore__ __inline__ constexpr uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ __inline__ constexpr uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
__aicore__ __inline__ constexpr uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t BASE_M = 64;
constexpr uint32_t BASE_K = 64;
constexpr uint32_t BASE_N = 160;

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

    __aicore__ inline void Process()
    {
        const uint32_t mLoop = CeilDiv(mTotal, BASE_M);
        const uint32_t nLoop = N / BASE_N;
        const uint32_t kLoop = K / BASE_K;

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
            const uint32_t curM = MinU32(mTotal - mBlock * BASE_M, BASE_M);
            const uint32_t calcM = curM < 2 ? 2 : curM;
            const uint32_t calcMAlign = AlignUp(calcM, CUBE_BLOCK);
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
    __aicore__ inline void CopyInA(
        AscendC::LocalTensor<bfloat16_t>& a1, uint32_t kBlock, uint32_t mBlock, uint32_t curM)
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

    __aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& a1, AscendC::LocalTensor<bfloat16_t>& a2, uint32_t calcMAlign)
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

    __aicore__ inline void CopyOut(
        AscendC::LocalTensor<float>& cL0C, uint32_t mBlock, uint32_t nBlock, uint32_t curM, uint32_t calcMAlign)
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
};

} // namespace S6

} // namespace
// <<< M15PE_CHAIN_END
}  // namespace Chain
}  // namespace M15PE

// ============================================================
// Launch shells (global scope; host `<<<...>>>`)
// ============================================================
// (1) transpose + z pack (AIV, pure DMA). `mutant`: 0 = contract, 1 = MUT_HEADSTRIDE.
__vector__ __global__ void m15_pf_epilog_transpose_kernel(GM_ADDR wsO, GM_ADDR qkvzba, GM_ADDR o, GM_ADDR z,
                                                          uint32_t m, uint32_t mutant)
{
    M15PE::Trans::GdnEpilogKernel op;
    op.Init(wsO, qkvzba, o, z, m, mutant);
    op.Process();
}

// (2) S5 RMSNormGated (AIV/VF). `mutant`: 0 = clean, 1 = MUT_S5_NOGAMMA, 2 = MUT_S5_ZHEAD0.
__vector__ __global__ void m15_pf_epilog_s5_kernel(GM_ADDR o, GM_ADDR z, GM_ADDR gamma, GM_ADDR y, uint32_t m,
                                                   uint32_t mutant)
{
    M15PE::Chain::ChainS5Kernel op;
    op.Init(o, z, gamma, y, m, mutant);
    op.Process();
}

// (3) S6 out_proj (AIC/cube): C[m,2560] = A[m,6144] * B[2560,6144]^T; Fixpipe dstStride = N = 2560 elems
//     / F322BF16 == hcAttnOut ROW_HID (5120 B) -> writes hcAttnOut directly, no extra copy.
__global__ __cube__ void m15_pf_epilog_s6_kernel(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c,
                                                 uint32_t m)
{
    AscendC::InitSocState();
    M15PE::Chain::S6::Bf16Gemm<M15PE::PF_EPI_HIDDEN, M15PE::PF_EPI_OUT_N> op;
    op.Init(a, b, c, m);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}

// (3-neg) S6 negative control: same GEMM via M15OP::OProjGemm with GEMM_MODE_KMINUS1
//         (drops one base-K block; mutation donor m15_attn_oproj.h:63/287).
__global__ __cube__ void m15_pf_epilog_s6_mut_kernel(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c,
                                                     uint32_t m)
{
    AscendC::InitSocState();
    M15OP::OProjGemm<M15PE::PF_EPI_HIDDEN, M15PE::PF_EPI_OUT_N> op;
    op.Init(a, b, c, m, M15OP::GEMM_MODE_KMINUS1);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}

#endif  // M15_PREFILL_EPILOG_H
