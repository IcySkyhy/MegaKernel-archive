// ============================================================
// m15_attn_prefill_host.h —— M116（Wave B2）：prefill 前端的 **host 侧**（独立参考 + 发射器）
//
// 两个用途，逐条对应 M103-2.7 第 2 条「融合清单」与任务书的「独立验证路」：
//   ① **独立参考**（fp64，本文件自建，**不读任何设备产物**）：GEMM / GemmaRMSNorm+partial RoPE /
//      环行（含 3×int64 位置尾）/ 主 KV 行 / 压缩行（池化→norm→rope@组首）。
//   ② **发射器**：`PfFillArgs`（把一份纯 POD 的 `PfLaunchSpec` 翻成 `M15PF::PfChunkArgs`）——
//      Wave C/D 把 `LayerArgs` 的 13 个 prefill 字段接上时是机械活。本 mission **不改**
//      `m15_layer_kernel.h`（Wave B 不碰挂载点文件）。
//
// 【独立性（docs/17 §1.2 / §1.3）】
//   · 规则来源 = 官方源码（`qwen4_exp/common/qsa_cache.py` 的门控与容量、`nvidia/ops/qsa.py:397-433`
//     的池化语义、`indexer_qsa.py:342-367` 的"池化→norm→rope"顺序）+ `m15_attn_prolog.h` 的 R1–R13。
//     **不是**从设备产物反推。
//   · 输入来源 = ① 声明输入（验证工程的 `Fixture` 生成，字节可核）。
//   · 设备侧复用 M88 的 `AivNormRope`；本文件的参考是**另写的 fp64 实现** ⇒ 不是 S4（同一实现算两遍）。
//   · 判据口径按 `docs/17 §7`：`m=4097` **不得**对官方 QSA 输出。本文件全部期望值自建。
// ============================================================
#ifndef M15_ATTN_PREFILL_HOST_H
#define M15_ATTN_PREFILL_HOST_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#include "m15_attn_prefill.h"   // 设备段体 + 它带进来的 M15AP / M15Kv / M15AC 常量

namespace M15PFH {

using namespace M15AP;
using namespace M15Kv;
using namespace M15AC;

// ============================================================
// 0. bf16 / 误差工具
// ============================================================
inline double Bf2D(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return static_cast<double>(f);
}

inline uint16_t F32ToBf16(float v)   // RNE（与设备 CAST_RINT / Fixpipe 同口径）
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

inline uint16_t D2Bf(double v) { return F32ToBf16(static_cast<float>(v)); }

// docs/17 §1.1：BfUlp(v) = 2^(e−7)，e = binade 指数（2^e ≤ |v| < 2^(e+1)）
inline double BfUlp(double v)
{
    const double a = (v < 0.0) ? -v : v;
    if (a == 0.0) {
        return 0.0;
    }
    int e = 0;
    std::frexp(a, &e);
    return std::ldexp(1.0, (e - 1) - 7);
}

// T3 的 ε（norm+rope 段）：M88 的逐项推导 = 7 次 fp32 舍入 × 2 倍安全系数 = 14·2^-24。
constexpr double PF_EPS_T3 = 14.0 * (2.0 / 16777216.0);
// GEMM 累加的 ε：K·2^-24（M88 的 T2′ 口径；K = AP_HIDDEN = 2560）
constexpr double PF_EPS_ACC = 2560.0 * (2.0 / 16777216.0);
constexpr double PF_ROPE_THETA = 10000000.0;   // config.rope_parameters.rope_theta
constexpr double PF_RMS_EPS = 1e-06;           // config.rms_norm_eps

// ============================================================
// 1. cos/sin 表（R8：`base^(2j/rotary_dim)`；行 = [cos(32) | sin(32)]，bf16）
// ============================================================
struct PfCsTable {
    std::vector<uint16_t> bf;   // [nPos][CS_ROW_ELEMS]

    void Build(uint32_t nPos)
    {
        bf.assign(static_cast<size_t>(nPos) * CS_ROW_ELEMS, 0u);
        for (uint32_t p = 0u; p < nPos; ++p) {
            for (uint32_t j = 0u; j < HALF; ++j) {
                const double inv = 1.0 / std::pow(PF_ROPE_THETA, (2.0 * j) / static_cast<double>(ROT));
                const double ang = static_cast<double>(p) * inv;
                bf[static_cast<size_t>(p) * CS_ROW_ELEMS + j] = D2Bf(std::cos(ang));
                bf[static_cast<size_t>(p) * CS_ROW_ELEMS + HALF + j] = D2Bf(std::sin(ang));
            }
        }
    }
    const uint16_t* Row(uint32_t p) const { return bf.data() + static_cast<size_t>(p) * CS_ROW_ELEMS; }
};

// ============================================================
// 2. GEMM 一行的 fp64 参考：`y0[colOff..colOff+n) = x · Wᵀ`
// ============================================================
// `wPlane` 的 bf16 布局与设备一致（`m15_attn_prolog.h` §3 的 8 个 role 偏移表，role 内行主序 (out,in)）。
// `wRoleOffBytes` = 该 role 在平面里的字节偏移；`colOff` 是 role 内的输出列号。
inline void PfGemmRowRef(const uint16_t* xRow, const uint16_t* wPlane, uint32_t wRoleOffBytes, uint32_t colOff,
                         uint32_t nCols, std::vector<double>& out)
{
    const uint16_t* role = wPlane + wRoleOffBytes / 2u;
    out.assign(nCols, 0.0);
    for (uint32_t j = 0u; j < nCols; ++j) {
        const uint16_t* wRow = role + static_cast<size_t>(colOff + j) * AP_HIDDEN;
        double acc = 0.0;
        for (uint32_t k = 0u; k < AP_HIDDEN; ++k) {
            acc += Bf2D(xRow[k]) * Bf2D(wRow[k]);
        }
        out[j] = acc;
    }
}

// 同一行的 Σ|terms| = Σ_k |x_k|·|W_nk|（T2′ 的 ε 用；取绝对值后与 x 无关地被复用）
inline void PfGemmRowTerms(const uint16_t* xRow, const uint16_t* wPlane, uint32_t wRoleOffBytes,
                           uint32_t colOff, uint32_t nCols, std::vector<double>& out)
{
    const uint16_t* role = wPlane + wRoleOffBytes / 2u;
    out.assign(nCols, 0.0);
    std::vector<double> xAbs(AP_HIDDEN);
    for (uint32_t k = 0u; k < AP_HIDDEN; ++k) {
        const double v = Bf2D(xRow[k]);
        xAbs[k] = (v < 0.0) ? -v : v;
    }
    for (uint32_t j = 0u; j < nCols; ++j) {
        const uint16_t* wRow = role + static_cast<size_t>(colOff + j) * AP_HIDDEN;
        double s = 0.0;
        for (uint32_t k = 0u; k < AP_HIDDEN; ++k) {
            const double wv = Bf2D(wRow[k]);
            s += xAbs[k] * ((wv < 0.0) ? -wv : wv);
        }
        out[j] = s;
    }
}

// ============================================================
// 3. GemmaRMSNorm(D) + partial NeoX RoPE(64)（fp64 参考；R3 / R9 / R12）
// ============================================================
// R3：`y = x · rsqrt(mean(x²)+eps) · (1+w)`；AP_MODE_PLAIN_NORM 时用 `w`（负向对照）
// R9：`o1 = x1·c − x2·s`、`o2 = x2·c + x1·s`，NeoX 配对 (j, j+32)，`[ROT,D)` 直通
// R12：indexer 的 128 维复用同一张表、只旋前 64 维
// ⚠ **RoPE 的输入必须是 norm 之后的值**（设备侧从 `outUb` 读回，`AivNormRope` 的三个 `__VEC_SCOPE__`
//    就是按这个顺序切的）⇒ 本参考也先算 norm、再在 norm 的结果上旋转。
inline void PfNormRopeRef(const uint16_t* in, const uint16_t* w, const uint16_t* csRow, uint32_t dim,
                          uint32_t mode, std::vector<uint16_t>& out, std::vector<double>* termsOut)
{
    std::vector<double> xs(dim);
    std::vector<double> wv(dim);
    double acc = 0.0;
    for (uint32_t i = 0u; i < dim; ++i) {
        xs[i] = Bf2D(in[i]);
        acc += xs[i] * xs[i];
        const double wd = Bf2D(w[i]) + ((mode == AP_MODE_PLAIN_NORM) ? 0.0 : 1.0);
        wv[i] = wd;
    }
    const double rstd = 1.0 / std::sqrt(acc / static_cast<double>(dim) + PF_RMS_EPS);
    // ⚠ **norm 的输出必须先落 bf16 再旋转**：设备的 `AivNormRope` 把 norm 结果写进 UB 的 bf16 缓冲
    //    （`StoreAlign<DIST_PACK_B32>`），下一个 `__VEC_SCOPE__` 的 RoPE 是从那块**bf16**读回来算的
    //    （M88 r2 的"两个 scope"结构就是为了这个顺序）。参考若在**未舍入**的 fp64 norm 结果上旋转，
    //    差异会是**半个 bf16 ulp 级**（M116 实测：~1.2e-4 绝对差，高于任何 fp32 舍入界）——
    //    那是"参考的算术序列与设备不同"，不是设备的错。
    std::vector<double> yn(dim);
    for (uint32_t i = 0u; i < dim; ++i) {
        yn[i] = Bf2D(D2Bf(xs[i] * rstd * wv[i]));
    }
    out.assign(dim, 0u);
    if (termsOut != nullptr) {
        // Σ|terms|：norm 的两项乘积 + （rot 段）旋转的两项乘积。这是**上界**（不追求紧），
        // 它只承担"把 bf16 量化之外的差异卡在 fp32 舍入量级内"这一件事。
        termsOut->assign(dim, 0.0);
    }
    for (uint32_t i = 0u; i < dim; ++i) {
        double y = yn[i];
        double t = std::fabs(xs[i]) * std::fabs(wv[i]) * std::fabs(rstd) + std::fabs(yn[i]);
        if ((i < ROT) && (mode != AP_MODE_NO_ROPE)) {
            const uint32_t j = (i < HALF) ? i : (i - HALF);
            const double c = Bf2D(csRow[j]);
            const double s = Bf2D(csRow[HALF + j]);
            const double x1 = (i < HALF) ? yn[i] : yn[i - HALF];
            const double x2 = (i < HALF) ? yn[i + HALF] : yn[i];
            const bool flip = (mode == AP_MODE_ROPE_SIGN);
            if (i < HALF) {
                y = (flip) ? (x1 * c + x2 * s) : (x1 * c - x2 * s);
            } else {
                y = (flip) ? (x2 * c - x1 * s) : (x2 * c + x1 * s);
            }
            t += std::fabs(x1) * std::fabs(c) + std::fabs(x2) * std::fabs(s);
        }
        if (termsOut != nullptr) {
            (*termsOut)[i] = t;
        }
        out[i] = D2Bf(y);
    }
}

// ============================================================
// 4. 环行（raw k 128 bf16 + 3×int64 位置尾）与主 KV 一行
// ============================================================
inline void PfRingRowRef(const uint16_t* rawk, uint32_t pos, std::vector<uint8_t>& row280)
{
    row280.assign(RING_ROW_BYTES, 0u);
    std::memcpy(row280.data(), rawk, RING_KEY_DIM * ELEM_BYTES);
    for (uint32_t i = 0u; i < RING_TAIL_INT64; ++i) {
        const uint64_t v = static_cast<uint64_t>(pos);
        std::memcpy(row280.data() + RING_KEY_DIM * ELEM_BYTES + i * 8u, &v, 8u);
    }
}

// 主 KV 一行：[h0K | h0V | h1K | h1V]（各 256 bf16 = 512 B）—— `AC_IN_KV_ROW_BYTES` 的布局
inline void PfMkvRowRef(const uint16_t* kNorm, const uint16_t* vRaw, std::vector<uint8_t>& row2048)
{
    row2048.assign(AC_IN_KV_ROW_BYTES, 0u);
    for (uint32_t h = 0u; h < KV_HEADS; ++h) {
        // 槽内偏移经宏算（与设备同源）：head 平面步长 = 2*KV_HEAD_DIM 个 bf16（K 512 B ‖ V 512 B）
        const uint64_t kOff = static_cast<uint64_t>(h) * (2u * KV_HEAD_DIM) * ELEM_BYTES;
        const uint64_t vOff = kOff + KV_HEAD_CONTENT_BYTES;
        std::memcpy(row2048.data() + kOff, kNorm + h * KV_HEAD_DIM, KV_HEAD_CONTENT_BYTES);
        std::memcpy(row2048.data() + vOff, vRaw + h * KV_HEAD_DIM, KV_HEAD_CONTENT_BYTES);
    }
}

// ============================================================
// 5. 压缩行的池化（官方 `ops/qsa.py:397-433`：fp32 累加、结果落 bf16）
// ============================================================
inline void PfPoolRef(const uint16_t* const members[COMP_TOKENS_PER_STATE], bool mean,
                      std::vector<uint16_t>& pooled)
{
    pooled.assign(RING_KEY_DIM, 0u);
    for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
        double s = 0.0;
        for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
            s += Bf2D(members[i][c]);
        }
        const double v = mean ? (s / static_cast<double>(COMP_TOKENS_PER_STATE)) : s;
        pooled[c] = F32ToBf16(static_cast<float>(v));
    }
}

// ============================================================
// 6. 门控（官方规则的**可执行复算**；与设备的候选/写入门控逐条同源）
// ============================================================
inline uint32_t PfCandidates(uint32_t start, uint32_t rows, uint32_t outP[AC_MAX_GROUPS])
{
    uint32_t n = 0u;
    for (uint32_t g = 0u; g < AC_MAX_GROUPS; ++g) {
        const uint32_t p = (start / COMP_TOKENS_PER_STATE + g + 1u) * COMP_TOKENS_PER_STATE - 1u;
        if (p >= start && p < start + rows && M15KV_COMP_ROW_WRITTEN(p)) {
            outP[n++] = p;
        }
    }
    return n;
}

// 环写入门控：只写"本 chunk 末尾 capacity 行"（`qsa_cache.py:144-149` 的 keep 条件）
inline bool PfRingRowWritten(uint32_t q, uint32_t chunkStart, uint32_t chunkEnd)
{
    return (q >= chunkStart) && (q + RING_ROWS_PER_BLOCK >= chunkEnd);
}

// 组首位置（`ops/qsa.py:436`：`pos - 4 + 1` 的组首形式）
inline uint32_t PfGroupFirstPos(uint32_t p) { return M15KV_COMP_GROUP_OF(p) * COMP_TOKENS_PER_STATE; }

// ============================================================
// 7. 发射器（独立验证路 / Wave C-D 的机械活）
// ============================================================
// ⚠ **只支持恒等 block_table（单请求）**：所有 cache 地址只由 `pos` 经 `M15KV_KV_*` 算出
//    （Wave A 的字段契约 + M82 的 D2）。多请求 / 非连续位置**不在 B2 的范围**（显式写窄）。
struct PfLaunchSpec {
    __gm__ uint8_t* x;
    __gm__ uint8_t* w;
    __gm__ uint8_t* cs;
    __gm__ uint8_t* posTbl;   // 可空
    __gm__ uint8_t* y0;
    __gm__ uint8_t* out;
    __gm__ uint8_t* in;
    __gm__ uint8_t* kv;
    __gm__ uint8_t* comp;
    __gm__ uint8_t* ring;
    __gm__ uint8_t* pack;
    __gm__ uint8_t* packSeed;
    __gm__ uint8_t* pooled;
    __gm__ uint8_t* flag;
    uint32_t rowStart;
    uint32_t seqRows;
    uint32_t chunkRows;
    uint32_t nChunks;
    uint32_t posBase;
    uint32_t mut;
    uint32_t acMode;
};

__aicore__ inline void PfFillArgs(M15PF::PfChunkArgs& a, const PfLaunchSpec& s)
{
    a.x = s.x;
    a.w = s.w;
    a.cs = s.cs;
    a.posTbl = s.posTbl;
    a.y0 = s.y0;
    a.out = s.out;
    a.in = s.in;
    a.kv = s.kv;
    a.comp = s.comp;
    a.ring = s.ring;
    a.pack = s.pack;
    a.packSeed = s.packSeed;
    a.pooled = s.pooled;
    a.flag = s.flag;
    a.rowStart = s.rowStart;
    a.seqRows = s.seqRows;
    a.chunkRows = s.chunkRows;
    a.nChunks = s.nChunks;
    a.posBase = s.posBase;
    a.mut = s.mut;
    a.acMode = s.acMode;
}

}  // namespace M15PFH

#endif  // M15_ATTN_PREFILL_HOST_H
