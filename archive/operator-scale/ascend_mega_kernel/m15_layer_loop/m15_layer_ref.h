#include <string>
#include <vector>

// ---- bf16 编解码（RNE，与 tools/golden/moe_block_ref.py 同规则）----
static float H_Bf16ToF32(uint16_t h)
{
    uint32_t x = static_cast<uint32_t>(h) << 16;
    float f;
    __builtin_memcpy(&f, &x, 4);
    return f;
}

static uint16_t H_F32ToBf16(float f)
{
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    const uint32_t r = (x + 0x7FFFu + ((x >> 16) & 1u)) >> 16;   // RNE
    return static_cast<uint16_t>(r);
}

static uint32_t H_Hash(uint32_t i, uint32_t j, uint32_t salt)
{
    uint32_t x = i * 0x9E3779B9u + j * 0x85EBCA6Bu + salt * 0xC2B2AE35u;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

static float H_RandU(uint32_t i, uint32_t j, uint32_t salt)   // [-1, 1)
{
    return (static_cast<float>(H_Hash(i, j, salt) >> 8) / 8388608.0f) - 1.0f;
}

static void H_GenBf16(std::vector<uint16_t>& v, uint32_t salt, float amp, uint32_t off = 0)
{
    for (size_t k = 0; k < v.size(); ++k) {
        v[k] = H_F32ToBf16(H_RandU(static_cast<uint32_t>(k) + off, 0u, salt) * amp);
    }
}

static void H_GenF32(std::vector<float>& v, uint32_t salt, float amp, float bias, uint32_t off = 0)
{
    for (size_t k = 0; k < v.size(); ++k) {
        v[k] = H_RandU(static_cast<uint32_t>(k) + off, 1u, salt) * amp + bias;
    }
}

static bool H_WriteBin(const char* path, const void* buf, size_t bytes)
{
    FILE* f = fopen(path, "wb");
    if (f == nullptr) {
        printf("[m15][FAIL] cannot open %s for write\n", path);
        return false;
    }
    const size_t n = fwrite(buf, 1, bytes, f);
    fclose(f);
    return n == bytes;
}

// ---- 单层权重（host 侧；层循环里每层一份，与 device 一一对应）----
// m14 的 M14Params 是「一次启动一份参数（含激活/state 初值）」；层循环里每层只有权重不同，
// 激活与 state 由循环自己持有 → 这里只保留权重字段（slim 版）。
struct M15LayerW {
    std::vector<uint16_t> wIn;       // [IN_N, HIDDEN] bf16（84.4MB；行序 q|k|v|z|b|a）
    std::vector<uint16_t> wOut;      // [OUT_N, OUT_K] bf16（31.5MB）
    std::vector<uint16_t> convW;     // [KW, CH] bf16
    std::vector<uint16_t> convBias;  // [CH] bf16
    std::vector<float> aLog;         // [64] fp32（48 有效 + 16 零）
    std::vector<float> dtBias;       // [64] fp32
    std::vector<uint16_t> gamma1;    // [HIDDEN] bf16
    std::vector<uint16_t> gamma2;    // [HIDDEN] bf16
    std::vector<uint16_t> gammaG;    // [HEAD_D] bf16
    uint32_t layer = 0;              // 层号（日志/dump 用）
    bool real = false;               // 是否来自真实 checkpoint 切片
};

static void H_AllocLayerW(M15LayerW& P)
{
    using namespace M15G;
    P.wIn.resize(static_cast<size_t>(IN_N) * HIDDEN);
    P.wOut.resize(static_cast<size_t>(OUT_N) * OUT_K);
    P.convW.resize(static_cast<size_t>(KW) * CH);
    P.convBias.resize(CH);
    P.aLog.assign(64, 0.0f);
    P.dtBias.assign(64, 0.0f);
    P.gamma1.resize(HIDDEN);
    P.gamma2.resize(HIDDEN);
    P.gammaG.resize(HEAD_D);
}

// 合成权重（bring-up / 无 manifest 时的退路；与 m14 H_GenParams 同口径，但按层号加盐 →
// 层间权重互不相同，能暴露「循环取错层权重」）。saltBase 由层号派生。
static void H_GenLayerW(M15LayerW& P, uint32_t layer)
{
    using namespace M15G;
    const uint32_t s = 1000u * (layer + 1);
    H_GenBf16(P.wIn, s + 1u, 1.0f);
    H_GenBf16(P.wOut, s + 2u, 1.0f);
    H_GenBf16(P.convW, s + 3u, 1.0f);
    H_GenBf16(P.convBias, s + 4u, 0.5f);
    H_GenBf16(P.gamma1, s + 5u, 0.5f);
    H_GenBf16(P.gamma2, s + 6u, 0.5f);
    H_GenBf16(P.gammaG, s + 7u, 0.5f);
    for (size_t k = 0; k < P.gamma1.size(); k += 97) {
        P.gamma1[k] = H_F32ToBf16(1.0f);
    }
    for (size_t k = 0; k < P.gamma2.size(); k += 89) {
        P.gamma2[k] = H_F32ToBf16(1.0f);
    }
    // A_log > 0（g = −exp(A_log)·softplus ≤ 0，decay ∈ (0,1]）；dt_bias 覆盖 softplus 两分支
    H_GenF32(P.aLog, s + 8u, 0.5f, 1.0f);
    for (uint32_t h = 0; h < HEADS; ++h) {
        P.aLog[h] = 0.05f + 0.5f * (H_RandU(h, 3u, s + 8u) + 1.0f);
        P.dtBias[h] = (h % 4 == 0) ? (21.0f + 2.0f * H_RandU(h, 4u, s + 9u)) : (2.0f * H_RandU(h, 5u, s + 9u));
    }
    P.layer = layer;
    P.real = false;
}

// ---- dump manifest 写入（check_ref.py 的唯一契约，见 README §5 / 文件头）----

static std::string H_WsTensorLine(const char* name, uint32_t off, uint32_t bytes, const char* dtype,
                                  const char* shape)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "tensor=%s ws_off=%u bytes=%u dtype=%s shape=%s\n", name, off, bytes, dtype, shape);
    return std::string(buf);
}

// ============================================================
// 参考实现（组合各算件公式；fp32 段的乘法按 fp32 次序、累加用 double）
// ============================================================

// S1/S7：Add + RMSNorm（残差 bf16 或 fp32）。resOut = fp32(x + res)（位级可复现）；
// y = bf16((xsum·rstd)·gamma)（kernel 用 NR rsqrt，与精确 rsqrt 差 ~1 ulp fp32 → 远小于 bf16 ulp）
static void H_RefNorm(const uint16_t* x, const uint16_t* resB, const float* resF, const uint16_t* gamma, uint32_t rows,
                      float* resOut, uint16_t* y)
{
    using namespace M15G;
    for (uint32_t row = 0; row < rows; ++row) {
        const size_t base = static_cast<size_t>(row) * HIDDEN;
        std::vector<float> xf(HIDDEN);
        double ss = 0.0;
        for (uint32_t j = 0; j < HIDDEN; ++j) {
            const float xa = H_Bf16ToF32(x[base + j]);
            const float ra = (resB != nullptr) ? H_Bf16ToF32(resB[base + j]) : resF[base + j];
            const float s = xa + ra;   // IEEE fp32 加（double 无法引入额外舍入）
            xf[j] = s;
            ss += static_cast<double>(s) * static_cast<double>(s);
        }
        const double rstd = 1.0 / std::sqrt(ss / static_cast<double>(HIDDEN) + 1e-6);
        for (uint32_t j = 0; j < HIDDEN; ++j) {
            const float t = xf[j] * static_cast<float>(rstd);   // kernel: Mul(x, rstd)
            const float g = H_Bf16ToF32(gamma[j]);
            const float yv = t * g;                             // kernel: Mul(t, gamma)
            resOut[base + j] = xf[j];
            y[base + j] = H_F32ToBf16(yv);
        }
    }
}

// S2/S6：C = bf16(A[m,K] · W[N,K]^T)，double 累加（kernel 为 fp32 L0C 累加 + F322BF16 RNE）
static void H_RefGemm(const uint16_t* a, const uint16_t* w, uint32_t m, uint32_t n, uint32_t k, uint16_t* c)
{
    for (uint32_t row = 0; row < m; ++row) {
        const uint16_t* aRow = a + static_cast<size_t>(row) * k;
        uint16_t* cRow = c + static_cast<size_t>(row) * n;
        for (uint32_t nn = 0; nn < n; ++nn) {
            const uint16_t* wRow = w + static_cast<size_t>(nn) * k;
            double acc = 0.0;
            for (uint32_t kk = 0; kk < k; ++kk) {
                acc += static_cast<double>(H_Bf16ToF32(aRow[kk])) * static_cast<double>(H_Bf16ToF32(wRow[kk]));
            }
            cRow[nn] = H_F32ToBf16(static_cast<float>(acc));
        }
    }
}

// S3：prolog（conv1d K=4 + bias + SiLU → q/k l2norm(仅 q ×1/√128) → gating）
//   conv_state 为 planar [ST][CH]，w 行 j 配最旧样本；写回 = 纯 bf16 搬移（位级）
static void H_RefProlog(const uint16_t* xq, const uint16_t* csIn, const uint16_t* convW, const uint16_t* convBias,
                        const float* alog, const float* dtbias, float* q, float* k, float* v, float* g, float* beta,
                        uint16_t* csOut)
{
    using namespace M15G;
    const double qscale = 1.0 / std::sqrt(128.0);
    std::vector<float> blk(BN);
    for (uint32_t b = 0; b < NBLK_PL; ++b) {
        for (uint32_t c = 0; c < BN; ++c) {
            const uint32_t ch = b * BN + c;
            double acc = static_cast<double>(H_Bf16ToF32(convBias[ch]));
            for (uint32_t j = 0; j < ST; ++j) {
                acc += static_cast<double>(H_Bf16ToF32(convW[static_cast<size_t>(j) * CH + ch])) *
                       static_cast<double>(H_Bf16ToF32(csIn[static_cast<size_t>(j) * CH + ch]));
            }
            acc += static_cast<double>(H_Bf16ToF32(convW[static_cast<size_t>(ST) * CH + ch])) *
                   static_cast<double>(H_Bf16ToF32(xq[ch]));
            const float yv = static_cast<float>(acc);
            blk[c] = yv / (1.0f + std::exp(-yv));   // SiLU
        }
        if (b < 16) {   // q head（每 128ch 一个 head）
            double ss = 0.0;
            for (uint32_t c = 0; c < BN; ++c) {
                ss += static_cast<double>(blk[c]) * static_cast<double>(blk[c]);
            }
            const float r = static_cast<float>(1.0 / std::sqrt(ss + 1e-6)) * static_cast<float>(qscale);
            for (uint32_t c = 0; c < BN; ++c) {
                q[static_cast<size_t>(b) * BN + c] = blk[c] * r;
            }
        } else if (b < 32) {
            double ss = 0.0;
            for (uint32_t c = 0; c < BN; ++c) {
                ss += static_cast<double>(blk[c]) * static_cast<double>(blk[c]);
            }
            const float r = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));
            for (uint32_t c = 0; c < BN; ++c) {
                k[static_cast<size_t>(b - 16) * BN + c] = blk[c] * r;
            }
        } else {
            const uint32_t h = b - 32;
            for (uint32_t c = 0; c < BN; ++c) {
                v[static_cast<size_t>(h) * BN + c] = blk[c];
            }
            const float a = H_Bf16ToF32(xq[A_OFF + h]);
            const float bb = H_Bf16ToF32(xq[B_OFF + h]);
            const double x = static_cast<double>(a) + static_cast<double>(dtbias[h]);
            // kernel: Exp → Adds(+1) → Log（log(1+e^x)），x > 20 由 Select 换成 x
            const double sp = (x <= 20.0) ? std::log(1.0 + std::exp(x)) : x;
            g[static_cast<size_t>(h) * 8] = static_cast<float>(-std::exp(static_cast<double>(alog[h])) * sp);
            beta[static_cast<size_t>(h) * 8] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(bb))));
        }
    }
    // conv_state 原位左移：new[0]=old[1]，new[1]=old[2]，new[2]=x（纯 bf16 搬移）
    for (uint32_t j = 0; j < ST - 1; ++j) {
        __builtin_memcpy(csOut + static_cast<size_t>(j) * CH, csIn + static_cast<size_t>(j + 1) * CH,
                         static_cast<size_t>(CH) * sizeof(uint16_t));
    }
    for (uint32_t c = 0; c < CH; ++c) {
        csOut[static_cast<size_t>(ST - 1) * CH + c] = xq[c];
    }
}

// S4：递推（per value head；g/β 为 stride-8 槽位，仅 [h][0] 有效）
static void H_RefRecur(const float* q, const float* k, const float* v, const float* g, const float* beta,
                       const float* ssmIn, float* ssmOut, float* o)
{
    using namespace M15G;
    std::vector<double> row(HEAD_D);
    for (uint32_t h = 0; h < HEADS; ++h) {
        const uint32_t hk = h / VGROUP;
        const double eg = std::exp(static_cast<double>(g[static_cast<size_t>(h) * 8]));
        for (uint32_t i = 0; i < HEAD_D; ++i) {
            const size_t off = (static_cast<size_t>(h) * HEAD_D + i) * HEAD_D;
            for (uint32_t j = 0; j < HEAD_D; ++j) {
                row[j] = static_cast<double>(ssmIn[off + j]) * eg;   // decay
            }
            double w = 0.0;
            for (uint32_t j = 0; j < HEAD_D; ++j) {
                w += row[j] * static_cast<double>(k[static_cast<size_t>(hk) * HEAD_D + j]);
            }
            const double d = static_cast<double>(beta[static_cast<size_t>(h) * 8]) *
                             (static_cast<double>(v[static_cast<size_t>(h) * HEAD_D + i]) - w);
            double oi = 0.0;
            for (uint32_t j = 0; j < HEAD_D; ++j) {
                row[j] += static_cast<double>(k[static_cast<size_t>(hk) * HEAD_D + j]) * d;   // outer
                oi += row[j] * static_cast<double>(q[static_cast<size_t>(hk) * HEAD_D + j]);  // matvec
                ssmOut[off + j] = static_cast<float>(row[j]);
            }
            o[static_cast<size_t>(h) * HEAD_D + i] = static_cast<float>(oi);
        }
    }
}

// S5：RMSNormGated（per head 128 维；乘法次序 = (o·rstd)·gamma·sigmoid(z)）
static void H_RefGated(const float* o, const uint16_t* zqkvzba, const uint16_t* gammaG, uint16_t* out)
{
    using namespace M15G;
    for (uint32_t h = 0; h < HEADS; ++h) {
        const size_t ob = static_cast<size_t>(h) * HEAD_D;
        double ss = 0.0;
        for (uint32_t j = 0; j < HEAD_D; ++j) {
            ss += static_cast<double>(o[ob + j]) * static_cast<double>(o[ob + j]);
        }
        const double rstd = 1.0 / std::sqrt(ss / static_cast<double>(HEAD_D) + 1e-6);
        for (uint32_t j = 0; j < HEAD_D; ++j) {
            const float z = H_Bf16ToF32(zqkvzba[Z_OFF + ob + j]);
            const float sig = 1.0f / (1.0f + std::exp(-z));
            const float t = o[ob + j] * static_cast<float>(rstd);
            const float t2 = t * H_Bf16ToF32(gammaG[j]);
            out[ob + j] = H_F32ToBf16(t2 * sig);
        }
    }
}

// ============================================================
// ws / 张量视图 + 比对工具
// ============================================================

template <typename T>
static const T* H_WsPtr(const std::vector<uint8_t>& ws, size_t off)
{
    return reinterpret_cast<const T*>(ws.data() + off);
}

static int H_U16Dist(uint16_t a, uint16_t b)
{
    const int ia = (a & 0x8000u) ? -static_cast<int>(a & 0x7FFFu) : static_cast<int>(a);
    const int ib = (b & 0x8000u) ? -static_cast<int>(b & 0x7FFFu) : static_cast<int>(b);
    return ia > ib ? ia - ib : ib - ia;
}

// bf16 网格判据：|got − exp| ≤ maxUlp 个 bf16 ulp（同网格位型距离；另报逐位一致占比）
static bool H_CmpBf16Grid(const char* tag, const uint16_t* got, const uint16_t* exp, size_t n, int maxUlp = 1)
{
    size_t bad = 0;
    size_t first = 0;
    size_t exact = 0;
    int worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const int d = H_U16Dist(got[i], exp[i]);
        if (d > worst) {
            worst = d;
        }
        if (got[i] == exp[i]) {
            ++exact;
        }
        if (d > maxUlp) {
            if (bad == 0) {
                first = i;
            }
            ++bad;
        }
    }
    const double exPct = 100.0 * static_cast<double>(exact) / static_cast<double>(n);
    if (bad == 0) {
        printf("[m15]   %-34s PASS (bf16 网格 ≤%d ulp; max ulp %d; bit 一致 %.4f%%, n=%zu)\n", tag, maxUlp, worst,
               exPct, n);
        return true;
    }
    printf("[m15]   %-34s FAIL (%zu/%zu 超 %d ulp, max ulp %d, bit 一致 %.4f%%; first idx %zu got 0x%04x exp 0x%04x)\n",
           tag, bad, n, maxUlp, worst, exPct, first, got[first], exp[first]);
    return false;
}

// fp32 组合容差判据：|got − exp| ≤ rtol·|exp| + atol（判定）+ 报告「最差容差占用」
static bool H_CmpF32Rel(const char* tag, const float* got, const float* exp, size_t n, double rtol = 1e-5,
                        double atol = 1e-6)
{
    size_t bad = 0;
    size_t first = 0;
    double worstAbs = 0.0;
    double worstUtil = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double a = std::fabs(static_cast<double>(got[i]) - static_cast<double>(exp[i]));
        const double budget = rtol * static_cast<double>(std::fabs(exp[i])) + atol;
        const double util = a / budget;   // ≤1 即通过
        if (a > worstAbs) {
            worstAbs = a;
        }
        if (util > worstUtil) {
            worstUtil = util;
        }
        if (!(util <= 1.0)) {
            if (bad == 0) {
                first = i;
            }
            ++bad;
        }
    }
    if (bad == 0) {
        printf("[m15]   %-34s PASS (rel %.0e+%.0e; 最差容差占用 %.3f, maxAbs %.3e, n=%zu)\n", tag, rtol, atol,
               worstUtil, worstAbs, n);
        return true;
    }
    printf("[m15]   %-34s FAIL (%zu/%zu 超容差; 最差占用 %.3f, maxAbs %.3e; first idx %zu got %.9g exp %.9g)\n", tag,
           bad, n, worstUtil, worstAbs, first, static_cast<double>(got[first]), static_cast<double>(exp[first]));
    return false;
}

static bool H_CmpF32Bit(const char* tag, const float* got, const float* exp, size_t n)
{
    size_t bad = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != exp[i]) {
            if (bad == 0) {
                first = i;
            }
            ++bad;
        }
    }
    if (bad == 0) {
        printf("[m15]   %-34s PASS (fp32 逐位一致, n=%zu)\n", tag, n);
        return true;
    }
    printf("[m15]   %-34s FAIL (%zu/%zu 位型不同; first idx %zu got %.9g exp %.9g)\n", tag, bad, n, first,
           static_cast<double>(got[first]), static_cast<double>(exp[first]));
    return false;
}

static bool H_CmpU16Bit(const char* tag, const uint16_t* got, const uint16_t* exp, size_t n)
{
    size_t bad = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != exp[i]) {
            if (bad == 0) {
                first = i;
            }
            ++bad;
        }
    }
    if (bad == 0) {
        printf("[m15]   %-34s PASS (bf16 位级一致, n=%zu)\n", tag, n);
        return true;
    }
    printf("[m15]   %-34s FAIL (%zu/%zu 字节不同; first idx %zu got 0x%04x exp 0x%04x)\n", tag, bad, n, first,
           got[first], exp[first]);
    return false;
}

// fp32 位型统计（不判定，只报告：state 的「位级」视角）
//   注：位型距离在 |x|→0 处无意义（跨零会给出巨大数值），故 ulp 距离只在 |exp| ≥ 1e-6 的子集上统计。
static void H_UlpReport(const char* tag, const float* got, const float* exp, size_t n)
{
    size_t exact = 0;
    size_t nBig = 0;
    size_t bigExact = 0;
    size_t bigWithin1 = 0;
    uint32_t worstBig = 0;
    for (size_t i = 0; i < n; ++i) {
        bool same = false;
        uint32_t d = 0;
        {
            uint32_t a;
            uint32_t b;
            __builtin_memcpy(&a, &got[i], 4);
            __builtin_memcpy(&b, &exp[i], 4);
            same = (a == b);
            if (!same) {
                const int32_t ia = (a & 0x80000000u) ? -static_cast<int32_t>(a & 0x7FFFFFFFu) : static_cast<int32_t>(a);
                const int32_t ib = (b & 0x80000000u) ? -static_cast<int32_t>(b & 0x7FFFFFFFu) : static_cast<int32_t>(b);
                d = static_cast<uint32_t>(ia > ib ? ia - ib : ib - ia);
            }
        }
        if (same) {
            ++exact;
        }
        if (std::fabs(static_cast<double>(exp[i])) >= 1e-6) {
            ++nBig;
            if (same) {
                ++bigExact;
            }
            if (d <= 1) {
                ++bigWithin1;
            }
            if (d > worstBig) {
                worstBig = d;
            }
        }
    }
    printf("[m15]   %-34s (报告) 逐位一致 %.4f%%; |exp|>=1e-6 子集: 位级一致 %.4f%% / ≤1ulp %.4f%% / max ulp %u (n=%zu)\n",
           tag, 100.0 * static_cast<double>(exact) / static_cast<double>(n),
           100.0 * static_cast<double>(bigExact) / static_cast<double>(nBig > 0 ? nBig : 1),
           100.0 * static_cast<double>(bigWithin1) / static_cast<double>(nBig > 0 ? nBig : 1), worstBig, n);
}

// 传播诊断（**不判定**，只报数）：设备 vs 纯参考链（未用 bf16 锚点）。
// 差异来源已解释：S2/S6 是 bf16 GEMM，设备的 fp32 累加次序不可复现（输出仅在 bf16 网格上
// ≤1 ulp）；log 域 g 与近相消的 conv 段会把该 1 ulp 放大成下游的相对偏差。
static void H_ReportDevF32(const char* tag, const float* got, const float* exp, size_t n)
{
    double worstAbs = 0.0;
    double worstRel = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double a = std::fabs(static_cast<double>(got[i]) - static_cast<double>(exp[i]));
        const double r = a / (std::fabs(static_cast<double>(exp[i])) + 1e-6);
        if (a > worstAbs) {
            worstAbs = a;
        }
        if (r > worstRel) {
            worstRel = r;
        }
    }
    printf("[m15]   %-34s (诊断, 不判定) maxAbs %.3e maxRel %.3e\n", tag, worstAbs, worstRel);
}

static void H_ReportDevBf16(const char* tag, const uint16_t* got, const uint16_t* exp, size_t n)
{
    size_t exact = 0;
    int worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const int d = H_U16Dist(got[i], exp[i]);
        if (d > worst) {
            worst = d;
        }
        if (got[i] == exp[i]) {
            ++exact;
        }
    }
    printf("[m15]   %-34s (诊断, 不判定) maxUlp %d bit 一致 %.4f%%\n", tag, worst,
           100.0 * static_cast<double>(exact) / static_cast<double>(n));
}

// ============================================================
// 参考链（一个 step 的全链 / 分段）+ device 视图
// ============================================================

struct H_Ref {
    std::vector<uint16_t> xnorm;    // [M_MAX*HIDDEN]
    std::vector<uint16_t> qkvzba;   // [M_MAX*IN_N]
    std::vector<uint16_t> opin;     // [M_MAX*V_DIM]
    std::vector<uint16_t> opout;    // [M_MAX*HIDDEN]
    std::vector<uint16_t> yfinal;   // [M_MAX*HIDDEN]
    std::vector<float> res1;        // [M_MAX*HIDDEN]
    std::vector<float> res2;        // [M_MAX*HIDDEN]
    std::vector<float> q;           // [NK*HEAD_D]
    std::vector<float> k;
    std::vector<float> v;           // [HEADS*HEAD_D]
    std::vector<float> g;           // [HEADS*8]
    std::vector<float> beta;        // [HEADS*8]
    std::vector<float> o;           // [HEADS*HEAD_D]
    std::vector<uint16_t> cs;       // [ST*CH]
    std::vector<float> ssm;         // [HEADS*HEAD_D*HEAD_D]
};

static void H_RefAlloc(H_Ref& R)
{
    using namespace M15G;
    R.xnorm.assign(static_cast<size_t>(M_MAX) * HIDDEN, 0);
    R.qkvzba.assign(static_cast<size_t>(M_MAX) * IN_N, 0);
    R.opin.assign(static_cast<size_t>(M_MAX) * V_DIM, 0);
    R.opout.assign(static_cast<size_t>(M_MAX) * HIDDEN, 0);
    R.yfinal.assign(static_cast<size_t>(M_MAX) * HIDDEN, 0);
    R.res1.assign(static_cast<size_t>(M_MAX) * HIDDEN, 0.0f);
    R.res2.assign(static_cast<size_t>(M_MAX) * HIDDEN, 0.0f);
    R.q.assign(static_cast<size_t>(NK) * HEAD_D, 0.0f);
    R.k.assign(static_cast<size_t>(NK) * HEAD_D, 0.0f);
    R.v.assign(static_cast<size_t>(HEADS) * HEAD_D, 0.0f);
    R.g.assign(static_cast<size_t>(HEADS) * 8, 0.0f);
    R.beta.assign(static_cast<size_t>(HEADS) * 8, 0.0f);
    R.o.assign(static_cast<size_t>(HEADS) * HEAD_D, 0.0f);
    R.cs.assign(static_cast<size_t>(ST) * CH, 0);
    R.ssm.assign(static_cast<size_t>(HEADS) * HEAD_D * HEAD_D, 0.0f);
}

// 单层「状态演化」参考步（层循环口径）：cs/ssm 在 R 内原地推进（自携带），跨 token 复用。
//
// qkvAnchor：**bf16 锚点** = 设备自己产出的 qkvzba 字节。理由（沿用 m14 §5.2 的口径）：
//   两个 bf16 GEMM 的 fp32 L0C 累加次序在 host 不可复现，其输出只有「bf16 网格 ≤1 ulp」契约；
//   而 conv 段存在近相消，1 ulp 的输入差会被放大成下游 ~1e-3 的相对差。故 S3 的输入一律取
//   设备 bf16 字节 —— 这也让本步**不需要跑 in_proj/out_proj 的 host GEMM**
//   （层循环的状态判据只涉及 S3/S4 两段；42M MAC × 36 层 × 3 token 的 double 参考是多余开销）。
// csIn：可选「状态输入锚点」；nullptr = 用 R 自己的自携带状态（判定口径）。
// 返回 false = 无锚点（此时不构成判定，调用方应跳过）。
static bool H_RefLayerStateStep(const M15LayerW& W, H_Ref& R, const uint16_t* qkvAnchor,
                                const uint16_t* csIn = nullptr)
{
    using namespace M15G;
    if (qkvAnchor == nullptr) {
        return false;
    }
    std::vector<uint16_t> csPrev(static_cast<size_t>(ST) * CH);
    if (csIn != nullptr) {
        __builtin_memcpy(csPrev.data(), csIn, csPrev.size() * sizeof(uint16_t));
    } else {
        csPrev = R.cs;
    }
    H_RefProlog(qkvAnchor, csPrev.data(), W.convW.data(), W.convBias.data(), W.aLog.data(), W.dtBias.data(),
                R.q.data(), R.k.data(), R.v.data(), R.g.data(), R.beta.data(), R.cs.data());
    std::vector<float> ssmIn(R.ssm);
    H_RefRecur(R.q.data(), R.k.data(), R.v.data(), R.g.data(), R.beta.data(), ssmIn.data(), R.ssm.data(), R.o.data());
    return true;
}
