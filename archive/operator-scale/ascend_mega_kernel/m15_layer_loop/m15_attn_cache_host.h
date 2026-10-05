// ============================================================
// m15_attn_cache_host.h —— M98：cache 填数学的 **host 侧独立参考 + 段级判据 + 负向对照**
//
// 只被 `m15_attn_kv_host.h` 在**末尾** include（本文件用到那里的 Ctx / H_Guard / H_CmpBytes /
// H_CmpMustDiffer / H_Sync / H_H2D / H_D2H），因此它处在 `namespace M15Run` 内。
//
// ---------------------------------------------------------------------------
// 【判据清单（判定项 = 计入 PASS/FAIL；guard = 前置断言，单列）】
//   ① T1 逐字节（整数/位域档，docs/17 §1.1）：
//      · `Ac.ring.row`      环平面 1,120 B 整块：门控放行的末尾 4 行逐字节（4 槽环 ⇒ 无"未写入槽留毒值"形态，见 README §4.5）
//      · `Ac.pack.row`      packed 两行 16,416 B 逐字节（行距 8,208 B ≡ 16 (mod 32)）
//      · `Ac.mainkv.kv`     主 KV 的 4 个 (head, K/V) 槽各 512 B 逐字节 + 其余仍是毒值
//                           （官方几何 = `m15_attn_kv.h` 的 `M15KV_KV_*`，M98 已按塔裁 A 更正）
//      · `Ac.ob.ring`       off-by-one 档的环平面（4 行写满且槽绕回 1,2,3,0）
//   ② T3（含 Rsqrt 与 bf16 量化；docs/17 §1.1 的 T3 档，ε 逐项来源见打印行）：
//      · `Ac.comp.row.g*`   压缩行逐元素 `|dev−ref| ≤ 16·2^-24·Σ|terms| + 1.0·ulp(ref)`，越界**逐个**打印
//      · `Ac.comp.nospurious` 除期望行外，压缩平面全为毒值
//      · `Ac.ob.comp`       off-by-one 档里组 1023 的压缩行
//   ③ 门控（T4）：`Ac.gate.ring/.comp/.read/.mask` —— host 按**同一条官方规则**独立复算，与设备 flag 相等
//   ④ 非空洞：`Ac.nonvac.expect`（比对面非零非退化、两行互异）/`Ac.nonvac.sens`（换输入必换输出）/
//             `Ac.det.repeat`（同档两次逐字节一致 —— 拿逐字节当判据的前提）
//   ⑤ 负向对照（**同一批判据**必须被打破；判据函数有 quiet 模式，被打破时只计"该判据是否变红"这一条）：
//      · `Ac.neg.sum`（池化不除 4）、`Ac.neg.plaimnorm`（乘 w 不乘 1+w）、`Ac.neg.ropelast`（RoPE 用组末位置）、
//        `Ac.neg.noringread`（跨 chunk 成员不读环）—— 四条都打 `Ac.comp.row.g5`
//      · `Ac.neg.noringstor`（**变体五**：填环那段不执行）—— 打 `Ac.ring.row`
//      · `Ac.neg.nocompstor`（**变体五**：填压缩行那段不执行）—— 打 `Ac.comp.row.g5`
//      · `Ac.neg.offbyone`（边界判据写成 `pos % 4 == 0`）—— 打 `Ac.ob.open`（开放组不得被写）
//      · `Ac.neg.ringsplit`（**正对照**：280 B 拆 256 B + 24 B 两次搬）—— 环平面必须与契约档**逐字节相同**
//      · `Ac.dcpad.required`（**不用 DataCopyPad 会怎样**：`Block1(280)`）—— 期望 launch **非正常结束**
//
// 【参考的独立性（docs/17 §1.2 / §1.3）】
//   · 规则来源 = **官方源码**（`indexer_qsa.py:342-367` 的"池化→norm→rope"顺序、`ops/qsa.py:397-433`
//     的 fp32 累加/除 4/落 bf16、`ops/qsa.py:401-427` 的成员来源、R3/R9 的 norm/rope 语义），
//     不是从设备产物反推。
//   · 输入来源 = ① 声明输入（本文件 host 生成）；**没有任何设备产物进入期望值**。
//   · defd norm/rope 的设备实现复用 M88 的 `AivNormRope`（已合入、T3 差异 0），host 参考是**另写的
//     double 实现** ⇒ 不是 S4（同一实现算两遍）。
// ============================================================
#ifndef M15_ATTN_CACHE_HOST_H
#define M15_ATTN_CACHE_HOST_H

namespace AcH {

namespace MAC = ::M15AC;

// ---- 两个档（chunk 窗口）----
constexpr uint32_t AC_CASE_M_START = 22u;      // 主档：跨 chunk 成员（位置 20/21 在环里）+ chunk 内成员
constexpr uint32_t AC_CASE_M_ROWS = 8u;        //        组 g=5（p=23）、g=6（p=27）⇒ 压缩页 1 的行 1/2
constexpr uint32_t AC_CASE_OB_START = 4093u;   // off-by-one 档：组 g=1023（p=4095）完成；4096 属开放组
constexpr uint32_t AC_CASE_OB_ROWS = 4u;       //                环 4 行写满且槽绕回（1,2,3,0）
constexpr uint8_t AC_POISON = 0xCDu;           // 毒值：漏写/多写都会现形

// ---- 声明输入的盐 ----
constexpr uint32_t AC_SALT_RAWK = 77101u;
constexpr uint32_t AC_SALT_KV = 77102u;
constexpr uint32_t AC_SALT_W = 77103u;
constexpr double AC_ROPE_THETA = 10000000.0;   // config.rope_parameters.rope_theta
constexpr double AC_RMS_EPS = 1e-06;           // config.rms_norm_eps
// T3 的 εAcc：16 次 fp32 舍入（池化 3 加 + Σx² 4 + ReduceSum 1 + rstd 3 + y 2 + rope 3），每次 ≤0.5·2^-24
constexpr double AC_T3_EPS_ACC = 16.0 * 5.9604644775390625e-08;   // 16 × 2^-24

// ---------- bf16 / 位型工具（RNE 直接复用本 TU 的 H_F32ToBf16，不新造第二份）----------
static inline double AcBf16ToD(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return static_cast<double>(f);
}

static inline uint16_t AcBf16Of(double v)
{
    return H_F32ToBf16(static_cast<float>(v));
}

// docs/17 §1.1：BfUlp(v) = 2^(e−7)，e = binade 指数（2^e ≤ |v| < 2^(e+1)）
static inline double AcBfUlp(double v)
{
    const double a = (v < 0.0) ? -v : v;
    if (a == 0.0) {
        return 0.0;
    }
    int e = 0;
    std::frexp(a, &e);   // a = m·2^e, m ∈ [0.5,1) ⇒ binade 指数 = e−1
    return std::ldexp(1.0, (e - 1) - 7);
}

// ---------- 候选组（host 侧独立复算"哪几条组该写压缩行"）----------
// 契约规则：组完成位置 p 落在 chunk 内 **且** `M15KV_COMP_ROW_WRITTEN(p)`（官方 `qsa_cache.py:179-182`）
static uint32_t H_AcCandidatesPre(uint32_t start, uint32_t rows, uint32_t outP[MAC::AC_MAX_GROUPS])
{
    using namespace M15Kv;
    uint32_t n = 0u;
    for (uint32_t g = 0u; g < MAC::AC_MAX_GROUPS; ++g) {
        const uint32_t p = (start / COMP_TOKENS_PER_STATE + g + 1u) * COMP_TOKENS_PER_STATE - 1u;
        if (p >= start && p < start + rows && M15KV_COMP_ROW_WRITTEN(p)) {
            outP[n++] = p;
        }
    }
    return n;
}

// OFFBYONE 档：把边界判据换成 `pos % 4 == 0`（差一位的经典形态），候选由 chunk 内的行给出
static uint32_t H_AcCandidates(uint32_t start, uint32_t rows, uint32_t mode, uint32_t outP[MAC::AC_MAX_GROUPS])
{
    using namespace M15Kv;
    if (mode != MAC::AC_MODE_OFFBYONE) {
        return H_AcCandidatesPre(start, rows, outP);
    }
    uint32_t n = 0u;
    for (uint32_t r = 0u; r < rows && n < MAC::AC_MAX_GROUPS; ++r) {
        const uint32_t q = start + r;
        if ((q % COMP_TOKENS_PER_STATE) == 0u) {
            outP[n++] = q;
        }
    }
    return n;
}

// ============================================================
// 1. 声明输入（①；host 生成，内容全链绑位置 ⇒ "换输入必换输出"可判）
// ============================================================
// 环行形态的一行：128 bf16 raw k ‖ 12 bf16 = 3×int64 位置尾（共 280 B）
// 位置尾 = 三个恒等轴的位置（R4/R5：文本路径上 MRoPE 三轴置换恒等）
static void AcRawKRow(uint32_t pos, uint32_t salt, uint16_t out[M15Kv::RING_HEAD_SIZE])
{
    using namespace M15Kv;
    for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
        out[c] = H_F32ToBf16(H_RandU(pos, c, salt) * 1.0f);
    }
    int64_t tail[RING_TAIL_INT64];
    for (uint32_t j = 0u; j < RING_TAIL_INT64; ++j) {
        tail[j] = static_cast<int64_t>(pos);
    }
    std::memcpy(out + RING_KEY_DIM, tail, sizeof(tail));
}

// 主 KV 的一行（本 chunk 第 0 行的 k/v；[h0K|h0V|h1K|h1V]，与输入的声明布局一致）
static void AcKvRow(uint32_t pos, uint32_t salt, uint16_t* out)
{
    using namespace M15Kv;
    for (uint32_t h = 0u; h < KV_HEADS; ++h) {
        for (uint32_t kv = 0u; kv < 2u; ++kv) {
            for (uint32_t d = 0u; d < KV_HEAD_DIM; ++d) {
                out[(static_cast<size_t>(h) * 2u + kv) * KV_HEAD_DIM + d] =
                    H_F32ToBf16(H_RandU(pos, h * 4096u + kv * 2048u + d, salt));
            }
        }
    }
}

// cos/sin 表（R8：`base^(2j/rotary_dim)`；行 = [cos(32)|sin(32)] bf16，与 M88 的表同布局）
static void AcBuildCs(std::vector<uint8_t>& cs)
{
    using namespace M15AP;
    cs.assign(MAC::AC_CS_BYTES, 0u);
    uint16_t* rows = reinterpret_cast<uint16_t*>(cs.data());
    for (uint32_t p = 0u; p < CS_NPOS; ++p) {
        for (uint32_t j = 0u; j < HALF; ++j) {
            const double inv = std::pow(AC_ROPE_THETA, -static_cast<double>(j) / static_cast<double>(HALF));
            const double ang = static_cast<double>(p) * inv;
            rows[static_cast<size_t>(p) * CS_ROW_ELEMS + j] = H_F32ToBf16(static_cast<float>(std::cos(ang)));
            rows[static_cast<size_t>(p) * CS_ROW_ELEMS + HALF + j] =
                H_F32ToBf16(static_cast<float>(std::sin(ang)));
        }
    }
}

// packed 行内容（选择语义不在本段；这里钉的是"行距/传输"⇒ 内容由 host 给）
static int32_t AcPackValue(uint32_t row, uint32_t col)
{
    using namespace M15Kv;
    if (col < 16u) {
        return static_cast<int32_t>(row * 1000u + col);
    }
    if (col >= PACK_TAIL_COL && col < PACK_TAIL_COL + 2u) {
        return static_cast<int32_t>(row * 1000u + 4000u + (col - PACK_TAIL_COL));
    }
    if (col == PACK_COUNT_COL) {
        return static_cast<int32_t>(30u + row);
    }
    return static_cast<int32_t>(PACK_FILL);
}

// ============================================================
// 2. 独立参考（double；只在对齐点做 bf16 舍入）
// ============================================================
// 池化值离**最近的 bf16 舍入边界（格点中点）**有多远（单位 = 该值处的 fp32 ulp）。
// ⚠ 中点 = `half = u/2` 的**奇数**倍；`u/2` 的偶数倍是格点本身（离中点最远）。
// 早先一版把它写成"离最近的 (u/2) 整数倍" ⇒ 会把「正好落在格点上」的元素误报成平局
// （实测 69/128 被误报），这里按奇偶显式取。
static double AcTieDistFp32Ulp(double x)
{
    const double u = AcBfUlp(x);
    if (u <= 0.0 || x == 0.0) {
        return 1e30;
    }
    const double half = u * 0.5;
    const double r = x / half;
    const long long k = static_cast<long long>(std::floor(r));
    long long c1 = ((k % 2LL) == 0LL) ? (k - 1LL) : k;
    long long c2 = ((k % 2LL) == 0LL) ? (k + 1LL) : k;
    const double d1 = std::fabs(x - static_cast<double>(c1) * half);
    const double d2 = std::fabs(x - static_cast<double>(c2) * half);
    const double dist = (d1 < d2) ? d1 : d2;
    int e = 0;
    std::frexp(std::fabs(x), &e);
    const double f32ulp = std::ldexp(1.0, (e - 1) - 23);
    return (f32ulp > 0.0) ? (dist / f32ulp) : 1e30;
}

struct AcCompRef {
    uint16_t out[M15Kv::RING_KEY_DIM];
    double terms[M15Kv::RING_KEY_DIM];   // 每元素的 Σ|terms| 量级（T3 的 ε 用）
    double pooled[M15Kv::RING_KEY_DIM];  // 诊断：池化后的实值
    double y[M15Kv::RING_KEY_DIM];       // 诊断：norm 后的 fp32 值
    double cA[32];                       // 诊断：cos 表行（bf16→double）
    double sA[32];                       // 诊断：sin 表行
    bool tied[M15Kv::RING_KEY_DIM];      // 池化值是否落在 bf16 舍入平局点上（设备可合法取另一侧）
    double flipTerm[M15Kv::RING_KEY_DIM];// **推导出的合法歧义项**：输出对该元素所依赖的"平局输入"的灵敏度×1 格
    uint16_t member[4][M15Kv::RING_KEY_DIM];  // 诊断：4 个成员的 raw k（bf16 位型）
    double xr[M15Kv::RING_KEY_DIM];           // 诊断：池化落 bf16 后的值（= x）
    double rstd = 0.0;                        // 诊断
    double wd[M15Kv::RING_KEY_DIM];           // 诊断：norm 权重的 double 值
    uint16_t wraw[M15Kv::RING_KEY_DIM];       // 诊断：norm 权重的 bf16 位型
};

// 参考链（规则来源逐条见文件头；顺序 = 官方 `indexer_qsa.py:342-367`）
//   pooled[c] = bf16(Σ_{i<4} raw_k[i][c] / 4)   ← 官方 pooled 张量 dtype = raw_keys.dtype = bf16
//   y[c]      = x[c]·rstd·(1+w[c])              ← R3（GemmaRMSNorm：乘 (1+w)，不是乘 w）
//   o1 = y[j]·cos − y[j+32]·sin；o2 = y[j+32]·cos + y[j]·sin（仅 [0,64)；[64,128) 直通）  ← R9
static void AcRefCompRow(const uint16_t raws[4][M15Kv::RING_KEY_DIM], const uint16_t* w, const uint16_t* cosRow,
                         AcCompRef& ref)
{
    using namespace M15Kv;
    using namespace M15AP;
    double x[RING_KEY_DIM];
    for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
        double acc = 0.0;
        for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
            acc += AcBf16ToD(raws[i][c]);
        }
        ref.pooled[c] = acc / static_cast<double>(COMP_TOKENS_PER_STATE);
        for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
            ref.member[i][c] = raws[i][c];
        }
        x[c] = AcBf16ToD(AcBf16Of(ref.pooled[c]));   // 池化结果落 bf16（官方 pooled 的 dtype）
        ref.xr[c] = x[c];
    }
    double ss = 0.0;
    for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
        ss += x[c] * x[c];
    }
    const double rstd = 1.0 / std::sqrt(ss / static_cast<double>(RING_KEY_DIM) + AC_RMS_EPS);
    ref.rstd = rstd;
    double y[RING_KEY_DIM];
    bool tiedY[RING_KEY_DIM];
    for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
        ref.wd[c] = AcBf16ToD(w[c]);
        ref.wraw[c] = w[c];
        const double ye = x[c] * rstd * (1.0 + ref.wd[c]);
        // ⚠ **第二个 bf16 舍入点（官方语义）**：`gemma_rmsnorm` 的输出沿用输入 dtype（`pooled` 是 bf16
        //   ⇒ 输出 bf16，`indexer_qsa.py:354-358` + `ops/qsa.py:1001-1005`），而 `apply_qsa_rope` 随后
        //   吃的是这个 **bf16** 值（`indexer_qsa.py:359-367`）。设备侧同构：AivNormRope 先写 bf16 到 UB，
        //   rope 段再把它读回来。早先一版参考漏了这一点（用未量化的 fp32 y 去旋）——这正是本判据
        //   2/128 越界的来源（量化扰动 ≈ 0.4%·|y|·(|c|+|s|)，与输出 bf16 格距同量级 ⇒ 少数近中点的元素翻边）。
        y[c] = AcBf16ToD(AcBf16Of(ye));
        ref.y[c] = y[c];
        // "平局"判定（两个舍入点都要判）：离最近 bf16 中点 ≤ 32 fp32 ulp ⇒ 设备的 fp32 累加（≤3 次舍入）
        // 可能落到另一侧，从而合法地取相邻格点
        ref.tied[c] = (AcTieDistFp32Ulp(ref.pooled[c]) <= 32.0);
        tiedY[c] = (AcTieDistFp32Ulp(ye) <= 32.0);
    }
    for (uint32_t j = 0u; j < HALF; ++j) {
        const double cj = AcBf16ToD(cosRow[j]);
        const double sj = AcBf16ToD(cosRow[HALF + j]);
        ref.cA[j] = cj;
        ref.sA[j] = sj;
        const double o1 = y[j] * cj - y[j + HALF] * sj;
        const double o2 = y[j + HALF] * cj + y[j] * sj;
        ref.out[j] = AcBf16Of(o1);
        ref.out[j + HALF] = AcBf16Of(o2);
        ref.terms[j] = std::fabs(y[j] * cj) + std::fabs(y[j + HALF] * sj);
        ref.terms[j + HALF] = std::fabs(y[j + HALF] * cj) + std::fabs(y[j] * sj);
        // **合法歧义项（推导）**：若输入池化格点 x[i] 落在平局点上，设备可合法取相邻格点。
        //   ∂o/∂x_i = rstd·(1+w_i)·(cos 或 sin)。下面逐元素把"它实际依赖的两个输入"的歧义加起来。
        const double g1 = rstd * (1.0 + AcBf16ToD(w[j]));
        const double g2 = rstd * (1.0 + AcBf16ToD(w[j + HALF]));
        const double up1 = AcBfUlp(x[j]);            // 相邻 bf16 格点间距 = 一次"翻边"的幅度
        const double up2 = AcBfUlp(x[j + HALF]);
        const double uy1 = AcBfUlp(y[j]);            // norm 输出（bf16）那一级的格距
        const double uy2 = AcBfUlp(y[j + HALF]);
        ref.flipTerm[j] = (ref.tied[j] ? std::fabs(g1 * cj) * up1 : 0.0) +
                          (ref.tied[j + HALF] ? std::fabs(g2 * sj) * up2 : 0.0) +
                          (tiedY[j] ? std::fabs(cj) * uy1 : 0.0) +
                          (tiedY[j + HALF] ? std::fabs(sj) * uy2 : 0.0);
        ref.flipTerm[j + HALF] = (ref.tied[j] ? std::fabs(g1 * sj) * up1 : 0.0) +
                                 (ref.tied[j + HALF] ? std::fabs(g2 * cj) * up2 : 0.0) +
                                 (tiedY[j] ? std::fabs(sj) * uy1 : 0.0) +
                                 (tiedY[j + HALF] ? std::fabs(cj) * uy2 : 0.0);
    }
    for (uint32_t c = 2u * HALF; c < RING_KEY_DIM; ++c) {
        ref.out[c] = AcBf16Of(y[c]);
        ref.terms[c] = std::fabs(y[c]);
        const double g = rstd * (1.0 + AcBf16ToD(w[c]));
        ref.flipTerm[c] = (ref.tied[c] ? (std::fabs(g) * AcBfUlp(x[c])) : 0.0) +
                          (tiedY[c] ? AcBfUlp(y[c]) : 0.0);
    }
}

// T3 判据。`count=false` 时**静默**（只返回结论）——负向对照用它验证"同一条判据被打破"
static bool AcJudgeCompRow(Ctx& C, const char* tag, const uint8_t* got, const AcCompRef& ref, bool count)
{
    using namespace M15Kv;
    uint32_t nExceed = 0u;
    uint32_t worst = 0u;
    double worstRatio = 0.0;
    double maxAbs = 0.0;
    double maxRel = 0.0;
    uint32_t nLe1Ulp = 0u;
    for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
        const uint16_t gb = *reinterpret_cast<const uint16_t*>(got + static_cast<size_t>(c) * 2u);
        const double gd = AcBf16ToD(gb);
        const double rd = AcBf16ToD(ref.out[c]);
        const double d = std::fabs(gd - rd);
        const double ulp = AcBfUlp(rd);
        // 判据：ε_strict·Σ|terms| + （推导出的）平局合法歧义项 + 1.0·ulp(out)
        const double bound = AC_T3_EPS_ACC * ref.terms[c] + ref.flipTerm[c] + ulp;
        if (d > maxAbs) {
            maxAbs = d;
        }
        if (ulp > 0.0 && d <= ulp) {
            nLe1Ulp++;
        }
        if (rd != 0.0 && d / std::fabs(rd) > maxRel) {
            maxRel = d / std::fabs(rd);
        }
        const double ratio = (bound > 0.0) ? (d / bound) : (d > 0.0 ? 1e30 : 0.0);
        if (ratio > worstRatio) {
            worstRatio = ratio;
            worst = c;
        }
        if (d > bound) {
            nExceed++;
        }
    }
    const bool ok = (nExceed == 0u);
    if (count) {
        C.checks++;
        printf("[m15]   %-30s %s (T3 ε=16×2^-24；越界 %u/%u；报告项：≤1ulp %u/%u、maxAbs=%.3e、maxRel=%.3e、"
               "最大占用=%.3f@c%u)\n",
               tag, ok ? "PASS" : "FAIL", nExceed, RING_KEY_DIM, nLe1Ulp, RING_KEY_DIM, maxAbs, maxRel, worstRatio,
               worst);
        if (!ok) {
            // 行级诊断：池化值离最近 bf16 中点的距离分布（单位 = fp32 ulp）——用于判定
            // 「设备与参考在池化的 bf16 舍入平局点上各取一侧」这一机理是否解释了越界元素
            double minTie = 1e30;
            uint32_t nNear = 0u;
            uint32_t nearIdx[8];
            for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
                const double t = AcTieDistFp32Ulp(ref.pooled[c]);
                if (t < minTie) {
                    minTie = t;
                }
                if (t <= 32.0) {
                    if (nNear < 8u) {
                        nearIdx[nNear] = c;
                    }
                    nNear++;
                }
            }
            printf("[m15]     行级：池化值离最近 bf16 中点最近 = %.2f fp32 ulp（≤32 fp32 ulp 的元素 %u 个",
                   minTie, nNear);
            for (uint32_t i = 0u; i < nNear && i < 8u; ++i) {
                printf(" c%u(%.2f)", nearIdx[i], AcTieDistFp32Ulp(ref.pooled[nearIdx[i]]));
            }
            printf("）\n");
            uint32_t shown = 0u;
            for (uint32_t c = 0u; c < RING_KEY_DIM && shown < 6u; ++c) {
                const uint16_t gb = *reinterpret_cast<const uint16_t*>(got + static_cast<size_t>(c) * 2u);
                const double gd = AcBf16ToD(gb);
                const double rd = AcBf16ToD(ref.out[c]);
                const double bound = AC_T3_EPS_ACC * ref.terms[c] + ref.flipTerm[c] + AcBfUlp(rd);
                if (std::fabs(gd - rd) > bound) {
                    const uint32_t partner = (c < 32u) ? (c + 32u) : (c - 32u);
                    const uint32_t j = (c < 32u) ? c : (c - 32u);
                    printf("[m15]     c=%u got=0x%04x(ref=0x%04x) dev=%.6e ref=%.6e |Δ|=%.3e > bound=%.3e\n",
                           c, gb, ref.out[c], gd, rd, std::fabs(gd - rd), bound);
                    printf("[m15]       中间量：rstd=%.12e；x(c)=%.12e x(配对%u)=%.12e；w(c)=0x%04x"
                           "(%.9e) w(配对)=0x%04x(%.9e)\n",
                           ref.rstd, ref.xr[c], partner, ref.xr[partner],
                           ref.wraw[c], ref.wd[c], ref.wraw[partner], ref.wd[partner]);
                    printf("[m15]       y(c)=%.12e y(配对)=%.12e cos[j%u]=%.12e sin=%.12e"
                           "（tie(c)=%.2f tie(配对)=%.2f fp32ulp；flipTerm=%.3e）\n",
                           ref.y[c], ref.y[partner], j, ref.cA[j], ref.sA[j], AcTieDistFp32Ulp(ref.pooled[c]),
                           AcTieDistFp32Ulp(ref.pooled[partner]), ref.flipTerm[c]);
                    printf("[m15]       池化 4 成员（列 %u）：0x%04x 0x%04x 0x%04x 0x%04x；精确均值=%.12e\n", c,
                           ref.member[0][c], ref.member[1][c], ref.member[2][c], ref.member[3][c],
                           ref.pooled[c]);
                    shown++;
                }
            }
            C.fails++;
        }
    }
    return ok;
}

// ============================================================
// 3. 设备缓冲 / 跑档 / 读回
// ============================================================
struct AcBufs {
    void* inDev = nullptr;
    void* csDev = nullptr;
    void* pooledDev = nullptr;
    void* ringDev = nullptr;
    void* compDev = nullptr;
    void* packDev = nullptr;
    void* packSeedDev = nullptr;
    void* mkvDev = nullptr;
    void* flagDev = nullptr;
    std::vector<uint8_t> in;        // 输入面（①）
    std::vector<uint8_t> cs;        // cos/sin 表（①）
    std::vector<uint8_t> packSeed;  // packed 行内容（①）
    std::vector<uint8_t> ringInit;  // 环初值（①：上一 chunk 的写入）
};

static bool H_AcAlloc(Ctx& C, AcBufs& B)
{
    struct Item {
        void** p;
        size_t bytes;
        const char* tag;
    };
    const Item items[] = {
        {&B.inDev, static_cast<size_t>(MAC::AC_IN_BYTES), "ac_in"},
        {&B.csDev, static_cast<size_t>(MAC::AC_CS_BYTES), "ac_cs"},
        {&B.pooledDev, static_cast<size_t>(MAC::AC_POOLED_BYTES), "ac_pooled"},
        {&B.ringDev, static_cast<size_t>(MAC::AC_RING_BYTES), "ac_ring"},
        {&B.compDev, static_cast<size_t>(MAC::AC_COMP_BYTES), "ac_comp"},
        {&B.packDev, static_cast<size_t>(MAC::AC_PACK_BYTES), "ac_pack"},
        {&B.packSeedDev, static_cast<size_t>(MAC::AC_PACK_SEED_BYTES), "ac_pack_seed"},
        {&B.mkvDev, static_cast<size_t>(MAC::AC_MAIN_KV_BYTES), "ac_mainkv"},
        {&B.flagDev, static_cast<size_t>(MAC::AC_FLAG_LANES) * 4u, "ac_flag"},
    };
    for (const Item& it : items) {
        if (aclrtMalloc(it.p, it.bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || *it.p == nullptr) {
            printf("[m15][FAIL] aclrtMalloc(%s, %zu B) failed\n", it.tag, it.bytes);
            return false;
        }
    }
    return true;
}

static void H_AcFree(AcBufs& B)
{
    void* ps[] = {B.inDev, B.csDev, B.pooledDev, B.ringDev, B.compDev, B.packDev, B.packSeedDev, B.mkvDev,
                  B.flagDev};
    for (void* p : ps) {
        if (p != nullptr) {
            aclrtFree(p);
        }
    }
}

// 输入面：本 chunk 每一行的 raw k（环行形态）+ k_layernorm 权重 + 主 KV 行
static void H_AcBuildInput(AcBufs& B, uint32_t start, uint32_t rows, uint32_t rawkSalt, uint32_t kvSalt)
{
    using namespace M15Kv;
    B.in.assign(MAC::AC_IN_BYTES, 0u);
    uint16_t* raws = reinterpret_cast<uint16_t*>(B.in.data() + MAC::AC_IN_RAWK_OFF);
    for (uint32_t r = 0u; r < rows; ++r) {
        AcRawKRow(start + r, rawkSalt, raws + static_cast<size_t>(r) * MAC::AC_IN_RAWK_STRIDE);
    }
    uint16_t* w = reinterpret_cast<uint16_t*>(B.in.data() + MAC::AC_IN_W_OFF);
    for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
        w[c] = H_F32ToBf16(H_RandU(c, 7u, AC_SALT_W) * 0.5f);   // (1+w) ∈ [0.5,1.5)
    }
    uint16_t* kv = reinterpret_cast<uint16_t*>(B.in.data() + MAC::AC_IN_KV_OFF);
    for (uint32_t r = 0u; r < MAC::AC_MAX_ROWS; ++r) {
        AcKvRow(start + r, kvSalt, kv + static_cast<size_t>(r) * (MAC::AC_IN_KV_ROW_BYTES / 2u));
    }
}

// 环初值 = 上一 chunk 的写入（①）：填"契约档会读到的槽"（位置 < chunk 起点的成员）
static void H_AcBuildRingInit(AcBufs& B, uint32_t start, uint32_t rows, uint32_t rawkSalt)
{
    using namespace M15Kv;
    B.ringInit.assign(MAC::AC_RING_BYTES, AC_POISON);
    uint16_t row[RING_HEAD_SIZE];
    uint32_t candP[MAC::AC_MAX_GROUPS];
    const uint32_t n = H_AcCandidatesPre(start, rows, candP);
    for (uint32_t c = 0u; c < n; ++c) {
        for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
            const uint32_t q = candP[c] - (COMP_TOKENS_PER_STATE - 1u) + i;
            if (q >= start) {
                continue;
            }
            AcRawKRow(q, rawkSalt, row);
            std::memcpy(B.ringInit.data() + static_cast<size_t>(M15KV_RING_SLOT(q)) * RING_ROW_BYTES, row,
                        RING_ROW_BYTES);
        }
    }
}

static void H_AcBuildPackSeed(AcBufs& B)
{
    using namespace M15Kv;
    B.packSeed.assign(MAC::AC_PACK_SEED_BYTES, 0u);
    int32_t* pr = reinterpret_cast<int32_t*>(B.packSeed.data());
    for (uint32_t r = 0u; r < MAC::AC_PACK_ROWS; ++r) {
        for (uint32_t c = 0u; c < PACK_COLS; ++c) {
            pr[static_cast<size_t>(r) * PACK_COLS + c] = AcPackValue(r, c);
        }
    }
}

static void H_AcPushConst(Ctx& C, AcBufs& B)
{
    H_H2D(C, B.inDev, B.in.data(), B.in.size(), "ac_in");
    H_H2D(C, B.csDev, B.cs.data(), B.cs.size(), "ac_cs");
    H_H2D(C, B.packSeedDev, B.packSeed.data(), B.packSeed.size(), "ac_packseed");
}

static void H_AcPoison(Ctx& C, AcBufs& B)
{
    aclrtMemset(B.ringDev, MAC::AC_RING_BYTES, AC_POISON, MAC::AC_RING_BYTES);
    aclrtMemset(B.compDev, MAC::AC_COMP_BYTES, AC_POISON, MAC::AC_COMP_BYTES);
    aclrtMemset(B.packDev, MAC::AC_PACK_BYTES, AC_POISON, MAC::AC_PACK_BYTES);
    aclrtMemset(B.mkvDev, MAC::AC_MAIN_KV_BYTES, AC_POISON, MAC::AC_MAIN_KV_BYTES);
    aclrtMemset(B.flagDev, MAC::AC_FLAG_LANES * 4u, AC_POISON, MAC::AC_FLAG_LANES * 4u);
}

struct AcOut {
    std::vector<uint8_t> pooled;
    std::vector<uint8_t> ring;
    std::vector<uint8_t> comp;
    std::vector<uint8_t> pack;
    std::vector<uint8_t> mkv;
    std::vector<int32_t> flag;
};

static void H_AcReadBack(Ctx& C, AcBufs& B, AcOut& out, const char* tag)
{
    out.pooled.assign(MAC::AC_POOLED_BYTES, 0u);
    out.ring.assign(MAC::AC_RING_BYTES, 0u);
    out.comp.assign(MAC::AC_COMP_BYTES, 0u);
    out.pack.assign(MAC::AC_PACK_BYTES, 0u);
    out.mkv.assign(MAC::AC_MAIN_KV_BYTES, 0u);
    out.flag.assign(MAC::AC_FLAG_LANES, 0);
    H_D2H(C, out.pooled.data(), B.pooledDev, out.pooled.size(), tag);
    H_D2H(C, out.ring.data(), B.ringDev, out.ring.size(), tag);
    H_D2H(C, out.comp.data(), B.compDev, out.comp.size(), tag);
    H_D2H(C, out.pack.data(), B.packDev, out.pack.size(), tag);
    H_D2H(C, out.mkv.data(), B.mkvDev, out.mkv.size(), tag);
    H_D2H(C, out.flag.data(), B.flagDev, out.flag.size() * sizeof(int32_t), tag);
}

// 跑一档：毒化全部平面 → 环复位成初值（①）→ 启动 →（同步）→ 读回。返回同步是否成功
static bool H_AcRun(Ctx& C, AcBufs& B, uint32_t start, uint32_t rows, uint32_t mode, AcOut& out, const char* tag)
{
    H_AcPushConst(C, B);
    H_AcPoison(C, B);
    // 环初值必须在毒化**之后**复位（否则被毒值盖掉）
    H_H2D(C, B.ringDev, B.ringInit.data(), B.ringInit.size(), "ac_ring_init");
    MAC::m15_attn_cache_kernel<<<C.numBlocks, 0, C.stream>>>(
        reinterpret_cast<uint8_t*>(B.inDev), reinterpret_cast<uint8_t*>(B.csDev),
        reinterpret_cast<uint8_t*>(B.pooledDev), reinterpret_cast<uint8_t*>(B.ringDev),
        reinterpret_cast<uint8_t*>(B.compDev), reinterpret_cast<uint8_t*>(B.packDev),
        reinterpret_cast<uint8_t*>(B.packSeedDev), reinterpret_cast<uint8_t*>(B.mkvDev),
        reinterpret_cast<uint8_t*>(B.flagDev), start, rows, mode);
    if (!H_Sync(C, tag)) {
        return false;
    }
    H_AcReadBack(C, B, out, tag);
    return H_Sync(C, tag);
}

// ============================================================
// 4. 期望平面（**全部由声明输入 + 官方规则独立算出**，与设备产物无关）
// ============================================================
// 环平面期望：官方门控覆盖到的行 = 输入行原样；其余 = 毒值
static void H_AcRingExpect(const AcBufs& B, uint32_t start, uint32_t rows, std::vector<uint8_t>& exp)
{
    using namespace M15Kv;
    exp.assign(MAC::AC_RING_BYTES, AC_POISON);
    const uint32_t chunkEnd = start + rows;
    for (uint32_t r = 0u; r < rows; ++r) {
        const uint32_t q = start + r;
        if (q + RING_ROWS_PER_BLOCK < chunkEnd) {
            continue;
        }
        const uint8_t* src =
            B.in.data() + MAC::AC_IN_RAWK_OFF + static_cast<size_t>(r) * MAC::AC_IN_RAWK_STRIDE_BYTES;
        std::memcpy(exp.data() + static_cast<size_t>(M15KV_RING_SLOT(q)) * RING_ROW_BYTES, src, RING_ROW_BYTES);
    }
}

// 主 KV 期望（**官方几何**；其余字节必须是毒值）
static void H_AcMkvExpect(const AcBufs& B, uint32_t start, std::vector<uint8_t>& exp)
{
    using namespace M15Kv;
    exp.assign(MAC::AC_MAIN_KV_BYTES, AC_POISON);
    const uint32_t physBlk = start / KV_BLOCK_TOKENS;
    const uint32_t slot = start % KV_BLOCK_TOKENS;
    const uint8_t* row = B.in.data() + MAC::AC_IN_KV_OFF;
    for (uint32_t h = 0u; h < KV_HEADS; ++h) {
        for (uint32_t kv = 0u; kv < 2u; ++kv) {
            const uint64_t off = M15KV_KV_BYTE_OFF_PHYS(physBlk, slot, h, kv, 0u);
            const uint8_t* src = row + (static_cast<size_t>(h) * 2u + kv) * KV_HEAD_DIM * ELEM_BYTES;
            if (off + KV_HEAD_DIM * ELEM_BYTES <= exp.size()) {
                std::memcpy(exp.data() + off, src, KV_HEAD_DIM * ELEM_BYTES);
            }
        }
    }
}

struct AcCompExpect {
    uint32_t p = 0u;
    uint64_t off = 0u;
    AcCompRef ref;
};

// 压缩行期望：每条候选的 4 个成员（chunk 内 → 输入行；否则 → 环初值）→ 参考
static void H_AcCompExpect(const AcBufs& B, uint32_t start, uint32_t rows, std::vector<AcCompExpect>& out)
{
    using namespace M15Kv;
    using namespace M15AP;
    out.clear();
    uint32_t candP[MAC::AC_MAX_GROUPS];
    const uint32_t n = H_AcCandidatesPre(start, rows, candP);
    const uint16_t* w = reinterpret_cast<const uint16_t*>(B.in.data() + MAC::AC_IN_W_OFF);
    for (uint32_t c = 0u; c < n; ++c) {
        const uint32_t p = candP[c];
        const uint32_t first = p - (COMP_TOKENS_PER_STATE - 1u);
        uint16_t raws[4][RING_KEY_DIM];
        for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
            const uint32_t q = first + i;
            if (q >= start && q < start + rows) {
                const uint8_t* src =
                    B.in.data() + MAC::AC_IN_RAWK_OFF +
                    static_cast<size_t>(q - start) * MAC::AC_IN_RAWK_STRIDE_BYTES;
                std::memcpy(raws[i], src, RING_KEY_DIM * ELEM_BYTES);   // 参考只需要 raw k 的 128 个
            } else {
                std::memcpy(raws[i], B.ringInit.data() + static_cast<size_t>(M15KV_RING_SLOT(q)) * RING_ROW_BYTES,
                            RING_KEY_DIM * ELEM_BYTES);
            }
        }
        AcCompExpect e;
        e.p = p;
        const uint32_t grp = M15KV_COMP_GROUP_OF(p);
        e.off = M15KV_COMP_ROW_OFF_PHYS(grp / COMP_ROWS_PER_BLOCK, grp);
        const uint16_t* csRow = reinterpret_cast<const uint16_t*>(B.cs.data()) +
                               static_cast<size_t>(grp * COMP_TOKENS_PER_STATE) * CS_ROW_ELEMS;
        AcRefCompRow(raws, w, csRow, e.ref);
        out.push_back(e);
    }
}

// 门控账（host 按同一条官方规则独立复算）
struct AcGate {
    uint32_t ring = 0u;
    uint32_t comp = 0u;
    uint32_t read = 0u;
    uint32_t mask = 0u;
};

static AcGate H_AcGateExpect(uint32_t start, uint32_t rows, uint32_t mode)
{
    using namespace M15Kv;
    AcGate g;
    const uint32_t chunkEnd = start + rows;
    if (mode != MAC::AC_MODE_NO_RING_STORE) {
        for (uint32_t r = 0u; r < rows; ++r) {
            if (start + r + RING_ROWS_PER_BLOCK >= chunkEnd) {
                g.ring++;
            }
        }
    }
    uint32_t candP[MAC::AC_MAX_GROUPS];
    const uint32_t n = H_AcCandidates(start, rows, mode, candP);
    if (mode != MAC::AC_MODE_NO_COMP_STORE) {
        g.comp = n;
    }
    for (uint32_t c = 0u; c < n; ++c) {
        const uint32_t grp = M15KV_COMP_GROUP_OF(candP[c]);
        g.mask |= (1u << (grp % COMP_ROWS_PER_BLOCK));
        for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
            if (candP[c] - (COMP_TOKENS_PER_STATE - 1u) + i < start) {
                g.read++;
            }
        }
    }
    return g;
}

// ---------- 判据小工具 ----------
static bool AcBytesEqual(const uint8_t* a, const uint8_t* b, size_t n)
{
    return std::memcmp(a, b, n) == 0;
}

// 非空洞用的段级判据：某一段必须**与全毒值平面不同**（不是数"非毒字节个数"——随机字节里
// 恰好等于毒值 0xCD 是正常现象，用计数会误报）
static bool AcSegmentDiffersFromPoison(const std::vector<uint8_t>& plane, uint64_t off, uint32_t bytes)
{
    for (uint32_t i = 0u; i < bytes; ++i) {
        if (plane[off + i] != AC_POISON) {
            return true;
        }
    }
    return false;
}

// 计数型逐字节判据（与 H_CmpBytes 同族，但本文件自己实现以打印本段的上下文）
static bool AcCheckBytes(Ctx& C, const char* tag, const uint8_t* got, const uint8_t* exp, size_t n, bool count)
{
    const bool ok = AcBytesEqual(got, exp, n);
    if (count) {
        C.checks++;
        if (ok) {
            printf("[m15]   %-30s PASS (逐字节一致, n=%zu)\n", tag, n);
        } else {
            size_t bad = 0u, first = 0u;
            for (size_t i = 0u; i < n; ++i) {
                if (got[i] != exp[i]) {
                    if (bad == 0u) {
                        first = i;
                    }
                    bad++;
                }
            }
            printf("[m15]   %-30s FAIL (%zu/%zu 字节不同; first off %zu got 0x%02x exp 0x%02x)\n", tag, bad, n,
                   first, got[first], exp[first]);
            C.fails++;
        }
    }
    return ok;
}

static bool AcCheckEq(Ctx& C, const char* tag, uint32_t got, uint32_t exp)
{
    const bool ok = (got == exp);
    C.checks++;
    printf("[m15]   %-30s %s (got=%u exp=%u)\n", tag, ok ? "PASS" : "FAIL", got, exp);
    if (!ok) {
        C.fails++;
    }
    return ok;
}

// 压缩平面"没有多写"：除期望行外必须全是毒值
static bool AcJudgeNoSpurious(Ctx& C, const char* tag, const AcOut& out, const std::vector<AcCompExpect>& exps,
                              bool count)
{
    using namespace M15Kv;
    uint32_t stray = 0u;
    uint64_t firstStray = 0u;
    for (uint64_t off = 0u; off < MAC::AC_COMP_BYTES; ++off) {
        if (out.comp[off] == AC_POISON) {
            continue;
        }
        bool inRow = false;
        for (const AcCompExpect& e : exps) {
            if (off >= e.off && off < e.off + COMP_ROW_BYTES) {
                inRow = true;
                break;
            }
        }
        if (!inRow) {
            if (stray == 0u) {
                firstStray = off;
            }
            stray++;
        }
    }
    // 期望行自身也不能整行是毒值（"没写"的一种形态）
    uint32_t emptyRows = 0u;
    for (const AcCompExpect& e : exps) {
        bool anyNonPoison = false;
        for (uint64_t i = 0u; i < COMP_ROW_BYTES; ++i) {
            if (out.comp[e.off + i] != AC_POISON) {
                anyNonPoison = true;
                break;
            }
        }
        if (!anyNonPoison) {
            emptyRows++;
        }
    }
    const bool ok = (stray == 0u) && (emptyRows == 0u);
    if (count) {
        C.checks++;
        printf("[m15]   %-30s %s (期望行外非毒字节 %u 个%s；期望行为空的 %u/%zu 行)\n", tag, ok ? "PASS" : "FAIL",
               stray, stray ? "" : "", emptyRows, exps.size());
        if (stray != 0u) {
            printf("[m15]     首个越界非毒字节 @ off %llu got 0x%02x\n", (unsigned long long)firstStray,
                   out.comp[firstStray]);
        }
        if (!ok) {
            C.fails++;
        }
    }
    return ok;
}

}  // namespace AcH

// ============================================================
// 5. 段级入口（由 m15_attn_kv_host.h 的 H_RunKv 调用）
// ============================================================
namespace AcH {

static bool H_RunAttnCache(Ctx& C)
{
    using namespace M15Kv;
    using namespace M15AP;
    printf("\n[m15] ===== 验证 Ac：attention cache 填数学（M98：raw ring / compressed / packed / 主 KV 搬运）=====\n");
    printf("[m15]   权威：`m15_attn_kv.h`（M82 冻结的物理布局）+ 官方 `qwen4_exp`（池化/norm/rope 的语义）\n");
    printf("[m15]   输入溯源（docs/17 §1.3）：本段全用 ① 声明输入（host 生成：raw k 行/k_norm 权重/cos-sin 表/"
           "packed 行/环初值）；**无设备产物进期望值**\n");
    printf("[m15]   T3 的 ε：16×2^-24（池化 3 次 fp32 加 + Σx² 4 次 + ReduceSum 1 + rstd 3 次 + y 2 次 + "
           "rope 3 次 ≈ 16 次 fp32 舍入，每次 ≤0.5·2^-24 相对）\n");
    printf("[m15]   主 KV 的几何：本段按**官方**几何（页 %u B、token 步长 %u B、head 步长 %u B、V 与 K 同槽 "
           "+%u B）判定 —— 该几何已按**塔裁 A** 更正进 `m15_attn_kv.h`，本段直接引用其常量\n",
           KV_BLOCK_BYTES, KV_TOKEN_STRIDE, KV_HEAD_PLANE_BYTES, KV_HEAD_CONTENT_BYTES);

    const uint32_t checkBase = C.checks;
    const uint32_t guardBase = C.guards;
    bool ok = true;

    AcBufs B;
    if (!H_AcAlloc(C, B)) {
        H_AcFree(B);
        return false;
    }
    AcBuildCs(B.cs);
    H_AcBuildPackSeed(B);
    H_AcBuildInput(B, AC_CASE_M_START, AC_CASE_M_ROWS, AC_SALT_RAWK, AC_SALT_KV);
    H_AcBuildRingInit(B, AC_CASE_M_START, AC_CASE_M_ROWS, AC_SALT_RAWK);
    H_AcPushConst(C, B);

    // ---- ① 非空洞前置（guard）：比对面必须非零非退化 ----
    {
        std::vector<uint8_t> ringExp, mkvExp;
        std::vector<AcCompExpect> compExp;
        H_AcRingExpect(B, AC_CASE_M_START, AC_CASE_M_ROWS, ringExp);
        H_AcMkvExpect(B, AC_CASE_M_START, mkvExp);
        H_AcCompExpect(B, AC_CASE_M_START, AC_CASE_M_ROWS, compExp);
        // 非空洞的判据口径：逐段要求"与全毒值平面不同"（计数会被随机字节里恰好等于 0xCD 的字节干扰）
        bool ringSegsOk = true;
        for (uint32_t s = 0u; s < RING_ROWS_PER_BLOCK; ++s) {
            ringSegsOk = ringSegsOk &&
                         AcSegmentDiffersFromPoison(ringExp, static_cast<uint64_t>(s) * RING_ROW_BYTES, RING_ROW_BYTES);
        }
        // 主 KV：按**官方几何**算出 4 个槽的真实偏移（不是 0/512/1024/1536 —— 那是页内步长，不是槽偏移）
        bool mkvSegsOk = true;
        {
            const uint32_t physBlk = AC_CASE_M_START / KV_BLOCK_TOKENS;
            const uint32_t slot = AC_CASE_M_START % KV_BLOCK_TOKENS;
            for (uint32_t h = 0u; h < KV_HEADS; ++h) {
                for (uint32_t kv = 0u; kv < 2u; ++kv) {
                    const uint64_t off = M15KV_KV_BYTE_OFF_PHYS(physBlk, slot, h, kv, 0u);
                    mkvSegsOk = mkvSegsOk && AcSegmentDiffersFromPoison(mkvExp, off, KV_HEAD_DIM * ELEM_BYTES);
                }
            }
        }
        ok = H_Guard(C, ringSegsOk) && ok;
        ok = H_Guard(C, mkvSegsOk) && ok;
        ok = H_Guard(C, compExp.size() == 2u) && ok;
        bool rowsDiffer = false;
        if (compExp.size() == 2u) {
            for (uint32_t i = 0u; i < RING_KEY_DIM; ++i) {
                if (compExp[0].ref.out[i] != compExp[1].ref.out[i]) {
                    rowsDiffer = true;
                    break;
                }
            }
        }
        ok = H_Guard(C, rowsDiffer) && ok;
        uint32_t nonZeroRef = 0u;
        for (uint32_t i = 0u; i < RING_KEY_DIM && compExp.size() == 2u; ++i) {
            if (compExp[0].ref.out[i] != 0u) {
                nonZeroRef++;
            }
        }
        ok = H_Guard(C, nonZeroRef >= 96u) && ok;
        printf("[m15]   Ac.nonvac.expect            %s（环 4 行各自与全毒值平面不同=%d；主 KV 4 个 (head,K/V) 槽"
               "各自与毒值不同=%d；压缩期望行 2 条且两行互异=%d；行 0 非零元素 %u/128）\n",
               (ringSegsOk && mkvSegsOk && rowsDiffer && nonZeroRef >= 96u && compExp.size() == 2u) ? "PASS" : "FAIL",
               ringSegsOk ? 1 : 0, mkvSegsOk ? 1 : 0, rowsDiffer ? 1 : 0, nonZeroRef);
    }

    // ============================================================
    // 档 M：契约档（chunk 22..29）
    // ============================================================
    AcOut m;
    if (!H_AcRun(C, B, AC_CASE_M_START, AC_CASE_M_ROWS, MAC::AC_MODE_CONTRACT, m, "ac_case_m")) {
        H_AcFree(B);
        return false;
    }
    std::vector<uint8_t> ringExp, mkvExp;
    std::vector<AcCompExpect> compExp;
    H_AcRingExpect(B, AC_CASE_M_START, AC_CASE_M_ROWS, ringExp);
    H_AcMkvExpect(B, AC_CASE_M_START, mkvExp);
    H_AcCompExpect(B, AC_CASE_M_START, AC_CASE_M_ROWS, compExp);

    // **隔离判据**：设备写出的 pooled 行（bf16）与 host 的 `bf16(精确均值)` 逐字节 ——
    // 它把"池化/落 bf16"与"norm+rope"两段分开：这一段若逐字节相同，则下面压缩行的差异只能来自
    // norm/rope 段；若不同，则先修池化段。（pooled 是设备的中间 scratch，判据的期望来自 host 独立算）
    for (size_t i = 0u; i < compExp.size(); ++i) {
        std::vector<uint8_t> poolExp(RING_KEY_DIM * 2u, 0u);
        uint16_t* pe = reinterpret_cast<uint16_t*>(poolExp.data());
        for (uint32_t c = 0u; c < RING_KEY_DIM; ++c) {
            pe[c] = AcBf16Of(compExp[i].ref.pooled[c]);
        }
        char tag[64];
        snprintf(tag, sizeof(tag), "Ac.pool.row.p%u", compExp[i].p);
        ok = AcCheckBytes(C, tag, m.pooled.data() + i * RING_KEY_DIM * 2u, poolExp.data(), poolExp.size(), true) && ok;
    }
    ok = AcCheckBytes(C, "Ac.ring.row", m.ring.data(), ringExp.data(), ringExp.size(), true) && ok;
    ok = AcCheckBytes(C, "Ac.pack.row", m.pack.data(), B.packSeed.data(), B.packSeed.size(), true) && ok;
    ok = AcCheckBytes(C, "Ac.mainkv.kv", m.mkv.data(), mkvExp.data(), mkvExp.size(), true) && ok;
    for (size_t i = 0u; i < compExp.size(); ++i) {
        char tag[64];
        snprintf(tag, sizeof(tag), "Ac.comp.row.p%u", compExp[i].p);
        ok = AcJudgeCompRow(C, tag, m.comp.data() + compExp[i].off, compExp[i].ref, true) && ok;
    }
    ok = AcJudgeNoSpurious(C, "Ac.comp.nospurious", m, compExp, true) && ok;
    {
        const AcGate g = H_AcGateExpect(AC_CASE_M_START, AC_CASE_M_ROWS, MAC::AC_MODE_CONTRACT);
        ok = AcCheckEq(C, "Ac.gate.ring", static_cast<uint32_t>(m.flag[MAC::AC_FLAG_RING_WRITTEN]), g.ring) && ok;
        ok = AcCheckEq(C, "Ac.gate.comp", static_cast<uint32_t>(m.flag[MAC::AC_FLAG_COMP_WRITTEN]), g.comp) && ok;
        ok = AcCheckEq(C, "Ac.gate.read", static_cast<uint32_t>(m.flag[MAC::AC_FLAG_RING_READ]), g.read) && ok;
        ok = AcCheckEq(C, "Ac.gate.mask", static_cast<uint32_t>(m.flag[MAC::AC_FLAG_COMP_MASK]), g.mask) && ok;
        ok = H_Guard(C, m.flag[MAC::AC_FLAG_CHUNK_START] == static_cast<int32_t>(AC_CASE_M_START)) && ok;
        ok = H_Guard(C, m.flag[MAC::AC_FLAG_CHUNK_ROWS] == static_cast<int32_t>(AC_CASE_M_ROWS)) && ok;
        ok = H_Guard(C, m.flag[MAC::AC_FLAG_MODE] == static_cast<int32_t>(MAC::AC_MODE_CONTRACT)) && ok;
    }
    printf("[m15]   档 M 交代：chunk %u..%u；环只写末尾 %u 行 ⇒ 位置 %u..%u 落槽 %u,%u,%u,%u；"
           "压缩行 2 条落在页 1 的行 1/2（跨页边界用 OB 档）；跨 chunk 成员 %u 个从环读（先读后写）\n",
           AC_CASE_M_START, AC_CASE_M_START + AC_CASE_M_ROWS - 1u, RING_ROWS_PER_BLOCK, AC_CASE_M_START + 4u,
           AC_CASE_M_START + AC_CASE_M_ROWS - 1u, M15KV_RING_SLOT(AC_CASE_M_START + 4u),
           M15KV_RING_SLOT(AC_CASE_M_START + 5u), M15KV_RING_SLOT(AC_CASE_M_START + 6u),
           M15KV_RING_SLOT(AC_CASE_M_START + 7u),
           static_cast<uint32_t>(m.flag[MAC::AC_FLAG_RING_READ]));

    // ---- ④ 确定性：同档再跑一次，四个平面逐字节一致 ----
    {
        AcOut m2;
        if (H_AcRun(C, B, AC_CASE_M_START, AC_CASE_M_ROWS, MAC::AC_MODE_CONTRACT, m2, "ac_case_m_repeat")) {
            ok = AcCheckBytes(C, "Ac.det.repeat", m2.ring.data(), m.ring.data(), m.ring.size(), true) && ok;
            ok = AcCheckBytes(C, "Ac.det.repeat.comp", m2.comp.data(), m.comp.data(), m.comp.size(), true) && ok;
        } else {
            ok = false;
        }
    }

    // ---- ④ 敏感性：换输入（raw k 的盐）⇒ 环与压缩行都必须变 ----
    {
        AcBufs B2 = B;
        H_AcBuildInput(B2, AC_CASE_M_START, AC_CASE_M_ROWS, AC_SALT_RAWK + 1u, AC_SALT_KV);
        H_AcBuildRingInit(B2, AC_CASE_M_START, AC_CASE_M_ROWS, AC_SALT_RAWK + 1u);
        AcOut m3;
        if (H_AcRun(C, B2, AC_CASE_M_START, AC_CASE_M_ROWS, MAC::AC_MODE_CONTRACT, m3, "ac_case_m_sens")) {
            bool ringDiff = !AcBytesEqual(m3.ring.data(), m.ring.data(), m.ring.size());
            bool compDiff = !AcBytesEqual(m3.comp.data(), m.comp.data(), m.comp.size());
            C.checks++;
            printf("[m15]   %-30s %s (环变=%d、压缩平面变=%d)\n", "Ac.nonvac.sens", (ringDiff && compDiff) ? "PASS" : "FAIL",
                   ringDiff ? 1 : 0, compDiff ? 1 : 0);
            if (!(ringDiff && compDiff)) {
                C.fails++;
                ok = false;
            }
        } else {
            ok = false;
        }
    }

    // ============================================================
    // ⑤ 负向对照一：池化/norm/rope 的**方向级**错 + 跨 chunk 读环 + 两个"弄坏被测对象"档
    //    全部与**契约档的期望**比（`count=false` 的 silent 判据 ⇒ 只计"该判据是否变红"）
    // ============================================================
    struct Neg {
        uint32_t mode;
        const char* tag;
        uint32_t start;
        uint32_t rows;
        bool breakRing;     // true ⇒ 期望打破的是环判据；false ⇒ 打破压缩行判据
    };
    // ⚠ `AC_MODE_SUM_NOT_MEAN`（池化不除 4）**不在负向对照表里**：官方链在池化之后紧跟
    //   `y = x·rstd·(1+w)`、`rstd = 1/sqrt(mean(x²)+eps)` —— 对池化常数**尺度不变**（常数因子被 rstd
    //   精确抵消），只有 eps 项与 pooled 的 bf16 格点会留下痕迹 ⇒ 它**构造上就不是有判别力的方向级错误**。
    //   实跑读数见下面的 `Ac.rep.sumnomean`（报告项，不进判定项计数，docs/17 §2.1）。
    const Neg negs[] = {
        {MAC::AC_MODE_PLAIN_NORM, "Ac.neg.plaimnorm", AC_CASE_M_START, AC_CASE_M_ROWS, false},
        {MAC::AC_MODE_ROPE_LAST_POS, "Ac.neg.ropelast", AC_CASE_M_START, AC_CASE_M_ROWS, false},
        {MAC::AC_MODE_NO_RING_READ, "Ac.neg.noringread", AC_CASE_M_START, AC_CASE_M_ROWS, false},
        {MAC::AC_MODE_NO_COMP_STORE, "Ac.neg.nocompstor", AC_CASE_M_START, AC_CASE_M_ROWS, false},
        {MAC::AC_MODE_NO_RING_STORE, "Ac.neg.noringstor", AC_CASE_M_START, AC_CASE_M_ROWS, true},
    };
    for (const Neg& n : negs) {
        AcOut o;
        if (!H_AcRun(C, B, n.start, n.rows, n.mode, o, n.tag)) {
            ok = false;
            continue;
        }
        bool held;
        if (n.breakRing) {
            held = AcCheckBytes(C, n.tag, o.ring.data(), ringExp.data(), ringExp.size(), false);
        } else {
            held = AcJudgeCompRow(C, n.tag, o.comp.data() + compExp[0].off, compExp[0].ref, false);
        }
        C.checks++;
        const bool broken = !held;
        printf("[m15]   %-30s %s（注入档（mode=%u）下同一判据%s ⇒ %s）\n", n.tag, broken ? "PASS" : "FAIL", n.mode,
               held ? "仍成立（**判据无判别力**）" : "被打破", "判据对这条方向级错误有判别力");
        if (!broken) {
            C.fails++;
            ok = false;
        }
    }

    // ---- ⑤ 正对照：单行 280 B 拆成 256 B + 24 B 两次搬，结果必须与契约档逐字节相同 ----
    {
        AcOut o;
        if (H_AcRun(C, B, AC_CASE_M_START, AC_CASE_M_ROWS, MAC::AC_MODE_RING_SPLIT, o, "ac_ring_split")) {
            C.checks++;
            const bool same = AcBytesEqual(o.ring.data(), m.ring.data(), m.ring.size());
            printf("[m15]   %-30s %s（拆两段（256 B 块 + 24 B Pad）与一次 280 B Pad 的环平面逐字节相同=%d）\n",
                   "Ac.neg.ringsplit", same ? "PASS" : "FAIL", same ? 1 : 0);
            if (!same) {
                C.fails++;
                ok = false;
            }
        } else {
            ok = false;
        }
    }

    // ---- 报告项（**不计入判定项**）：池化常数「Σ 而非 Σ/4」在 RMSNorm 之后不可判别 ----
    {
        AcOut o;
        if (H_AcRun(C, B, AC_CASE_M_START, AC_CASE_M_ROWS, MAC::AC_MODE_SUM_NOT_MEAN, o, "ac_sum_nomean")) {
            const bool held = AcJudgeCompRow(C, "Ac.rep.sumnomean", o.comp.data() + compExp[0].off, compExp[0].ref,
                                             false);
            printf("[m15]   %-30s 报告项（**不算负向对照**）：注入档（池化不除 4）下压缩行判据%s —— 官方链"
                   "（pooled → RMSNorm）对池化常数尺度不变（常数被 rstd 精确抵消），只剩 eps 项与 pooled 的 "
                   "bf16 格点留下痕迹 ⇒ **对压缩行判据**这条错误不可判别；但它对本段的**池化逐字节判据** "
                   "`Ac.pool.row.*` 是**可判别**的（Σ 不除 4 ⇒ 设备写出 bf16(4·mean) ≠ 控制档的 "
                   "bf16(mean)）——本轮未把该档的 `Ac.pool.row` 判一次（见 README §4.3 / §5 第 8 条）\n",
                   "Ac.rep.sumnomean", held ? "仍成立" : "被打破");
        }
    }

    // ============================================================
    // 档 OB：off-by-one（chunk 4093..4096；组 1023 完成、4096 属开放组）
    // ============================================================
    {
        AcBufs Bo = B;
        H_AcBuildInput(Bo, AC_CASE_OB_START, AC_CASE_OB_ROWS, AC_SALT_RAWK, AC_SALT_KV);
        H_AcBuildRingInit(Bo, AC_CASE_OB_START, AC_CASE_OB_ROWS, AC_SALT_RAWK);
        AcOut obC;
        if (H_AcRun(C, Bo, AC_CASE_OB_START, AC_CASE_OB_ROWS, MAC::AC_MODE_CONTRACT, obC, "ac_ob_contract")) {
            std::vector<uint8_t> ringExpOb;
            std::vector<AcCompExpect> compExpOb;
            H_AcRingExpect(Bo, AC_CASE_OB_START, AC_CASE_OB_ROWS, ringExpOb);
            H_AcCompExpect(Bo, AC_CASE_OB_START, AC_CASE_OB_ROWS, compExpOb);
            ok = AcCheckBytes(C, "Ac.ob.ring", obC.ring.data(), ringExpOb.data(), ringExpOb.size(), true) && ok;
            ok = H_Guard(C, compExpOb.size() == 1u) && ok;
            if (compExpOb.size() == 1u) {
                ok = AcJudgeCompRow(C, "Ac.ob.comp", obC.comp.data() + compExpOb[0].off, compExpOb[0].ref, true) && ok;
                // 开放组（组 1024）的行必须仍是毒值；这就是 OFFBYONE 档要打的那条判据
                const uint64_t openOff = M15KV_COMP_ROW_OFF_PHYS(1024u / COMP_ROWS_PER_BLOCK, 1024u);
                bool openPoison = true;
                for (uint32_t i = 0u; i < COMP_ROW_BYTES; ++i) {
                    if (obC.comp[openOff + i] != AC_POISON) {
                        openPoison = false;
                        break;
                    }
                }
                C.checks++;
                printf("[m15]   %-30s %s（开放组 1024 的行 @ off %llu 仍为毒值（pos 4096 属开放组，不得写））\n",
                       "Ac.ob.open", openPoison ? "PASS" : "FAIL", (unsigned long long)openOff);
                if (!openPoison) {
                    C.fails++;
                    ok = false;
                }
                // 负向：OFFBYONE 档下**同一条**判据必须被打破
                AcOut obB;
                if (H_AcRun(C, Bo, AC_CASE_OB_START, AC_CASE_OB_ROWS, MAC::AC_MODE_OFFBYONE, obB,
                            "ac_ob_offbyone")) {
                    bool stillPoison = true;
                    for (uint32_t i = 0u; i < COMP_ROW_BYTES; ++i) {
                        if (obB.comp[openOff + i] != AC_POISON) {
                            stillPoison = false;
                            break;
                        }
                    }
                    const bool broken = !stillPoison;
                    C.checks++;
                    printf("[m15]   %-30s %s（注入档（边界判据写成 `pos %% 4 == 0`）下开放组的行%s ⇒ 判据对 "
                           "off-by-one 有判别力；同时 `Ac.ob.comp` 同档%s）\n",
                           "Ac.neg.offbyone", broken ? "PASS" : "FAIL", stillPoison ? "仍是毒值（**判据无判别力**）"
                                                                                 : "被写",
                           AcJudgeCompRow(C, "Ac.neg.offbyone.comp", obB.comp.data() + compExpOb[0].off,
                                          compExpOb[0].ref, false)
                               ? "仍成立"
                               : "也被打破");
                    if (!broken) {
                        C.fails++;
                        ok = false;
                    }
                } else {
                    ok = false;
                }
            }
        } else {
            ok = false;
        }
    }

    // ============================================================
    // ⑤ 不用 DataCopyPad 会怎样：`Block1(280)`（非 32 B 倍数 ⇒ `M15G::Block1` 直接 `Trap`）
    //    期望：launch **非正常结束**（这不是"必过"的判据的反面 —— 它是一条**难拿的 PASS**）
    // ============================================================
    {
        H_AcPushConst(C, B);
        H_AcPoison(C, B);
        MAC::m15_attn_cache_kernel<<<C.numBlocks, 0, C.stream>>>(
            reinterpret_cast<uint8_t*>(B.inDev), reinterpret_cast<uint8_t*>(B.csDev),
            reinterpret_cast<uint8_t*>(B.pooledDev), reinterpret_cast<uint8_t*>(B.ringDev),
            reinterpret_cast<uint8_t*>(B.compDev), reinterpret_cast<uint8_t*>(B.packDev),
            reinterpret_cast<uint8_t*>(B.packSeedDev), reinterpret_cast<uint8_t*>(B.mkvDev),
            reinterpret_cast<uint8_t*>(B.flagDev), AC_CASE_M_START, AC_CASE_M_ROWS, MAC::AC_MODE_RING_BLOCK1);
        const aclError e = aclrtSynchronizeStream(C.stream);
        C.checks++;
        const bool trapped = (e != ACL_SUCCESS);
        printf("[m15]   %-30s %s（mode=%u 走 `Block1(280)`：sync rc=%d%s；⇒ 单行 280 B **必须** DataCopyPad）\n",
               "Ac.dcpad.required", trapped ? "PASS" : "FAIL", MAC::AC_MODE_RING_BLOCK1, static_cast<int>(e),
               trapped ? "" : "（**未预期**：未 Trap ⇒ 该档的 Block1 防呆没生效）");
        if (!trapped) {
            C.fails++;
            ok = false;
        }
        // Trap 之后的流状态不可信：清一次错误并同步
        (void)aclGetRecentErrMsg();
    }

    H_AcFree(B);
    C.phaseKv = C.checks - checkBase;
    printf("[m15] 验证 Ac（cache 填数学）：本段判定项 %u 条 + guard %u 条（累计 checks %u / guards %u / fails %u）\n",
           C.phaseKv, C.guards - guardBase, C.checks, C.guards, C.fails);
    printf("[m15]   覆盖交代：本段**不覆盖** ① 压缩行的**选择语义**（打分/topk/expand）与 packed 行的内容来源"
           "（本段把 packed 行当①的声明输入，只判行距/传输/逐字节）；② 主 KV 的 paged 非恒等 block_table"
           "（归 M82 的 T-KV-PAGED）；③ 真实规模档（真实权重切片）**未做**（见 m15_layer_loop/evidence/"
           "attn_cache/README.md 的未完成项）。\n");
    return ok;
}

}  // namespace AcH

#endif  // M15_ATTN_CACHE_HOST_H
