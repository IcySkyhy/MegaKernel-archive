// ============================================================
// m15_moe_host.h —— M40 host 侧：MoE 段权重的装载 / 来源校验 / 验证 M
//
// 本文件只在 m15_layer_loop.asc（单 TU）里被 include，位置在 Ctx / H_* 工具之后、
// H_Alloc / H_LoadWeights / main 之前。它做三件事：
//   1. MoE 段权重的 host 侧期望值（真实 checkpoint 切片 or 合成）与槽内装配；
//   2. MoE 段权重的**独立来源判据**（不复用装载路径，按 manifest 自己 pread 再与设备槽位 memcmp）；
//   3. 验证 M：
//        M1「段间零串扰」——融合 kernel 的两个相位分别与**独立启动**的对照路逐字节一致；
//        M2「多 token（融合形态）」——3 个 token 连跑，非空洞性 + 与独立启动形态逐字节等价
//            + 交付形态（连发不 sync）可重复。
//
// 为什么 M1 是「融合」这件事的核心判据：两个相位共用 UB 段窗 / L1 / L0C / BufferID 编号
// （m15_layer_resources.h §2/§5 的叠放与复用），任何一处叠放或复用没被相位边界的
// PipeBarrier<PIPE_ALL> + mode-0 barrier 正确隔离，都会让**至少一个**相位的 ws 与独立启动
// 形态不同。逐字节比 ws（而不是只比 y）就是为了抓这种「值侥幸对了」的情况。
// ============================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ============================================================
// 0. MoE 段权重（host 侧一层）
// ============================================================

struct M15MoeW {
    std::vector<uint8_t> router;   // [E, HIDDEN] bf16
    std::vector<uint8_t> sgate;    // [HIDDEN] bf16
    std::vector<uint8_t> wGu;      // [E, GU_N, HIDDEN/2] u8（MXFP4 打包）
    std::vector<uint8_t> sGu;      // [E, GU_N, HIDDEN/32] u8（e8m0）
    std::vector<uint8_t> wDn;      // [E, HIDDEN, INTER/2]
    std::vector<uint8_t> sDn;      // [E, HIDDEN, INTER/32]
    std::vector<uint8_t> wGuShd;   // [GU_N, HIDDEN/2]（host 行拼接 gate||up）
    std::vector<uint8_t> sGuShd;
    std::vector<uint8_t> wDnShd;   // [HIDDEN, INTER/2]
    std::vector<uint8_t> sDnShd;
    std::vector<uint16_t> gamma1;  // [HIDDEN] bf16（合成；checkpoint 无层级 norm，同 GDN 占位口径）
    std::vector<uint16_t> gamma2;  // [HIDDEN] bf16
    uint32_t layer = 0;
    bool real = false;
};

static void H_MoeWAlloc(M15MoeW& W)
{
    W.router.resize(M15L::MW_ROUTER_BYTES);
    W.sgate.resize(M15L::MW_SGATE_BYTES);
    W.wGu.resize(M15L::MW_WGU_BYTES);
    W.sGu.resize(M15L::MW_SGU_BYTES);
    W.wDn.resize(M15L::MW_WDN_BYTES);
    W.sDn.resize(M15L::MW_SDN_BYTES);
    W.wGuShd.resize(M15L::MW_WGUSHD_BYTES);
    W.sGuShd.resize(M15L::MW_SGUSHD_BYTES);
    W.wDnShd.resize(M15L::MW_WDNSHD_BYTES);
    W.sDnShd.resize(M15L::MW_SDNSHD_BYTES);
    W.gamma1.resize(M15G::HIDDEN);
    W.gamma2.resize(M15G::HIDDEN);
}

// 合成档：确定性公式（无 rand 状态），只用于 M15_SYNTH=1 冒烟 —— 与 GDN 档同款口径
static void H_GenMoeW(M15MoeW& W, uint32_t layer)
{
    const uint32_t salt = 50000u + layer * 37u;
    auto fillU8 = [](std::vector<uint8_t>& v, uint32_t s) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<uint8_t>((s * 2654435761u + static_cast<uint32_t>(i) * 40503u) >> 13);
        }
    };
    fillU8(W.router, salt + 1u);
    fillU8(W.sgate, salt + 2u);
    fillU8(W.wGu, salt + 3u);
    fillU8(W.sGu, salt + 4u);
    fillU8(W.wDn, salt + 5u);
    fillU8(W.sDn, salt + 6u);
    fillU8(W.wGuShd, salt + 7u);
    fillU8(W.sGuShd, salt + 8u);
    fillU8(W.wDnShd, salt + 9u);
    fillU8(W.sDnShd, salt + 10u);
    H_GenBf16(W.gamma1, salt + 11u, 0.5f);
    H_GenBf16(W.gamma2, salt + 12u, 0.5f);
    W.layer = layer;
    W.real = false;
}

// 真实档：按 manifest 的 role 自己 pread。**本函数只做「取字节」**，不做任何数值换算
// （与 m15 的 GDN 档同款：kernel 直接消费 HF 原始 layout，不做离线转换）。
static bool H_LoadMoeW(const M15Manifest& M, uint32_t layer, M15MoeW& W, std::string& err)
{
    auto rd = [&](const char* role, std::vector<uint8_t>& dst) -> bool {
        const M15Tensor* T = H_FindTensor(M, layer, role);
        if (T == nullptr) {
            err = std::string("manifest 缺 role=") + role + " layer=" + std::to_string(layer);
            return false;
        }
        if (T->bytes != dst.size()) {
            char b[256];
            snprintf(b, sizeof(b), "role=%s layer=%u 字节数不符：manifest %llu，期望 %zu", role, layer,
                     static_cast<unsigned long long>(T->bytes), dst.size());
            err = b;
            return false;
        }
        return H_ReadTensor(M, *T, dst, err);
    };
    std::vector<uint8_t> gate, up, gateS, upS;   // 共享专家 gate/up 分开取，再行拼接
    gate.resize(M15L::MW_WGUSHD_BYTES / 2);
    up.resize(M15L::MW_WGUSHD_BYTES / 2);
    gateS.resize(M15L::MW_SGUSHD_BYTES / 2);
    upS.resize(M15L::MW_SGUSHD_BYTES / 2);
    if (!rd(M15L::MW_ROLE_ROUTER, W.router) || !rd(M15L::MW_ROLE_SGATE, W.sgate) ||
        !rd(M15L::MW_ROLE_WGU, W.wGu) || !rd(M15L::MW_ROLE_SGU, W.sGu) || !rd(M15L::MW_ROLE_WDN, W.wDn) ||
        !rd(M15L::MW_ROLE_SDN, W.sDn) || !rd("moe_shared_gate", gate) || !rd("moe_shared_up", up) ||
        !rd("moe_shared_gate_scale", gateS) || !rd("moe_shared_up_scale", upS) ||
        !rd(M15L::MW_ROLE_WDNSHD, W.wDnShd) || !rd(M15L::MW_ROLE_SDNSHD, W.sDnShd)) {
        return false;
    }
    // 唯一 host 侧组板：共享专家的 gate/up 行拼接（与 m13 口径一致，字节不变）
    __builtin_memcpy(W.wGuShd.data(), gate.data(), gate.size());
    __builtin_memcpy(W.wGuShd.data() + gate.size(), up.data(), up.size());
    __builtin_memcpy(W.sGuShd.data(), gateS.data(), gateS.size());
    __builtin_memcpy(W.sGuShd.data() + gateS.size(), upS.data(), upS.size());
    // 层级 norm：checkpoint 里没有 [HIDDEN] 形状的 MoE 段 norm（归一化在 *_hyper_connection，
    // 见 m15 README §4 占位点 ② 与 docs/14）→ 与 GDN 档同款：确定性合成，按层加盐。
    H_GenBf16(W.gamma1, 61000u + layer, 0.5f);
    H_GenBf16(W.gamma2, 62000u + layer, 0.5f);
    W.layer = layer;
    W.real = true;
    return true;
}

// 把一层 MoE 权重装进「与设备槽同布局」的 host 镜像
static void H_PutMoeSlot(std::vector<uint8_t>& arena, uint32_t layer, const M15MoeW& W)
{
    uint8_t* d = arena.data() + static_cast<size_t>(layer) * M15L::MOE_W_STRIDE;
    auto put = [d](uint32_t off, const void* src, size_t bytes) { __builtin_memcpy(d + off, src, bytes); };
    put(M15L::MW_ROUTER_OFF, W.router.data(), W.router.size());
    put(M15L::MW_SGATE_OFF, W.sgate.data(), W.sgate.size());
    put(M15L::MW_WGU_OFF, W.wGu.data(), W.wGu.size());
    put(M15L::MW_SGU_OFF, W.sGu.data(), W.sGu.size());
    put(M15L::MW_WDN_OFF, W.wDn.data(), W.wDn.size());
    put(M15L::MW_SDN_OFF, W.sDn.data(), W.sDn.size());
    put(M15L::MW_WGUSHD_OFF, W.wGuShd.data(), W.wGuShd.size());
    put(M15L::MW_SGUSHD_OFF, W.sGuShd.data(), W.sGuShd.size());
    put(M15L::MW_WDNSHD_OFF, W.wDnShd.data(), W.wDnShd.size());
    put(M15L::MW_SDNSHD_OFF, W.sDnShd.data(), W.sDnShd.size());
    put(M15L::MW_G1_OFF, W.gamma1.data(), W.gamma1.size() * 2);
    put(M15L::MW_G2_OFF, W.gamma2.data(), W.gamma2.size() * 2);
}

// ---- 从 MoE 权重槽算出 kernel 入口要的 10 个指针 ----
struct MoeWPtrs {
    uint8_t* router;
    uint8_t* sgate;
    uint8_t* wGu;
    uint8_t* sGu;
    uint8_t* wDn;
    uint8_t* sDn;
    uint8_t* wGuShd;
    uint8_t* sGuShd;
    uint8_t* wDnShd;
    uint8_t* sDnShd;
    uint8_t* g1;
    uint8_t* g2;
};

static MoeWPtrs H_MoeWPtrsOf(void* moeWBase, uint32_t layer)
{
    uint8_t* s = reinterpret_cast<uint8_t*>(moeWBase) + static_cast<size_t>(layer) * M15L::MOE_W_STRIDE;
    MoeWPtrs p;
    p.router = s + M15L::MW_ROUTER_OFF;
    p.sgate = s + M15L::MW_SGATE_OFF;
    p.wGu = s + M15L::MW_WGU_OFF;
    p.sGu = s + M15L::MW_SGU_OFF;
    p.wDn = s + M15L::MW_WDN_OFF;
    p.sDn = s + M15L::MW_SDN_OFF;
    p.wGuShd = s + M15L::MW_WGUSHD_OFF;
    p.sGuShd = s + M15L::MW_SGUSHD_OFF;
    p.wDnShd = s + M15L::MW_WDNSHD_OFF;
    p.sDnShd = s + M15L::MW_SDNSHD_OFF;
    p.g1 = s + M15L::MW_G1_OFF;
    p.g2 = s + M15L::MW_G2_OFF;
    return p;
}

// ============================================================
// 1. 启动器（全部在 M15Run::Ctx 已定义之后使用）
// ============================================================

// 融合：一层一次启动（GDN 段 + MoE 段）
// M136（Wave D）：`m` 改为**真传参**（原为函数内写死 `1u`）。本条路径的三个调用方都是"单 token"
// （H_M1 段间零串扰 / H_M2 多 token 逐 token 推进 / H_RunP 计时），故调用点显式传 `1u`；写死的
// 那一处被移除，m 不再由被调方替调用方决定。
static void H_LaunchFused(M15Run::Ctx& C, uint32_t layer, void* ws, void* xDev, void* yDev, void* csDev,
                          void* ssmDev, uint32_t m)
{
    const bool isGdn = (M15Loop::KindOf(layer) == M15Loop::KIND_GDN);
    const MoeWPtrs mp = H_MoeWPtrsOf(C.moeWDev, layer);
    const uint32_t topk = C.topk;
    if (isGdn) {
        M15L::m15_layer_kernel_gdn<<<C.numBlocks, 0, C.stream>>>(
            reinterpret_cast<uint8_t*>(ws), reinterpret_cast<uint8_t*>(xDev),
            reinterpret_cast<uint8_t*>(C.resZeroDev), reinterpret_cast<uint8_t*>(yDev),
            reinterpret_cast<uint8_t*>(C.resZeroDev), M15Run::H_W(C, layer, M15Loop::W_G1_OFF),
            M15Run::H_W(C, layer, M15Loop::W_G2_OFF), M15Run::H_W(C, layer, M15Loop::W_GG_OFF),
            M15Run::H_W(C, layer, M15Loop::W_IN_OFF), M15Run::H_W(C, layer, M15Loop::W_OUT_OFF),
            M15Run::H_W(C, layer, M15Loop::W_CONV_OFF), M15Run::H_W(C, layer, M15Loop::W_CONVB_OFF),
            M15Run::H_W(C, layer, M15Loop::W_ALOG_OFF), M15Run::H_W(C, layer, M15Loop::W_DTB_OFF),
            reinterpret_cast<uint8_t*>(csDev), reinterpret_cast<uint8_t*>(ssmDev), mp.router, mp.sgate, mp.wGu,
            mp.sGu, mp.wDn, mp.sDn, mp.wGuShd, mp.sGuShd, mp.wDnShd, mp.sDnShd, mp.g1, mp.g2, m, topk);
    } else {
        M15L::m15_layer_kernel_attn<<<C.numBlocks, 0, C.stream>>>(
            reinterpret_cast<uint8_t*>(ws), reinterpret_cast<uint8_t*>(xDev),
            reinterpret_cast<uint8_t*>(C.resZeroDev), reinterpret_cast<uint8_t*>(yDev),
            reinterpret_cast<uint8_t*>(C.resZeroDev), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, mp.router, mp.sgate, mp.wGu, mp.sGu, mp.wDn, mp.sDn, mp.wGuShd,
            mp.sGuShd, mp.wDnShd, mp.sDnShd, mp.g1, mp.g2, m, topk);
    }
}

// 独立启动形态（M25 的 GDN 层 kernel / attention 占位），供「段间零串扰」的对照路使用
static void H_LaunchGdnOnly(M15Run::Ctx& C, uint32_t layer, void* ws, void* xDev, void* yDev, void* csDev,
                           void* ssmDev)
{
    m15_gdn_layer_kernel<<<C.numBlocks, 0, C.stream>>>(
        reinterpret_cast<uint8_t*>(ws), reinterpret_cast<uint8_t*>(xDev),
        reinterpret_cast<uint8_t*>(C.resZeroDev), reinterpret_cast<uint8_t*>(yDev),
        M15Run::H_W(C, layer, M15Loop::W_G1_OFF), M15Run::H_W(C, layer, M15Loop::W_G2_OFF),
        M15Run::H_W(C, layer, M15Loop::W_GG_OFF), M15Run::H_W(C, layer, M15Loop::W_IN_OFF),
        M15Run::H_W(C, layer, M15Loop::W_OUT_OFF), M15Run::H_W(C, layer, M15Loop::W_CONV_OFF),
        M15Run::H_W(C, layer, M15Loop::W_CONVB_OFF), M15Run::H_W(C, layer, M15Loop::W_ALOG_OFF),
        M15Run::H_W(C, layer, M15Loop::W_DTB_OFF), reinterpret_cast<uint8_t*>(csDev),
        reinterpret_cast<uint8_t*>(ssmDev));
}

// 只跑 MoE 段（ws 传 MoE 段基址）
static void H_LaunchMoeOnly(M15Run::Ctx& C, uint32_t layer, void* moeWs, void* xDev, void* yDev)
{
    const MoeWPtrs mp = H_MoeWPtrsOf(C.moeWDev, layer);
    M15L::m15_moe_segment_kernel<<<C.numBlocks, 0, C.stream>>>(
        reinterpret_cast<uint8_t*>(moeWs), reinterpret_cast<uint8_t*>(xDev), reinterpret_cast<uint8_t*>(yDev),
        reinterpret_cast<uint8_t*>(C.resZeroDev), mp.g1, mp.g2, mp.router, mp.sgate, mp.wGu, mp.sGu, mp.wDn,
        mp.sDn, mp.wGuShd, mp.sGuShd, mp.wDnShd, mp.sDnShd, 1u, C.topk);
}

// ============================================================
// 1b. 「除 padding 车道外逐字节」比较（M1 用）
// ============================================================
// 为什么需要它（这是本 mission **实测发现**的一处既有 quirk，不是放宽容差）：
//   GDN 段 S3 的 g/β 出口是 **[48][8] fp32 的 stride-8 槽位**（m15_gdn_layer.h:941-943 明确
//   「仅 [h][0] 有效」），生产者每次写**整个 32B 槽**（:886 `DataCopy(gGm_[h*8], gL, cpSlot)`），
//   而 `gL` 的 8 个 float 里只有第 0 个是算出来的（其余 7 个来自 UB staging 的无关内容）；
//   消费者（S4 递推）**只读槽位首元素**（:1148「BRC 按槽位首元素广播」）。于是这 7 个 padding
//   float 的值取决于**同一 UB 区在本次启动前的残留内容**，而残留内容在「融合 kernel 的 GDN
//   相位」与「独立启动的 GDN kernel」两种上下文里本来就不同（融合路的相位 A 之前是上一次启动
//   的 MoE 相位，独立路之前是独立 kernel 启动）。
//   实测：两者**只在 g/β 数组的 padding 车道里**有差异（不变量；padding 外的任何差异都会让
//   M.gdnws FAIL），**其余 5.26MB 逐字节相同**；而所有被消费的张量（q/k/v/o、conv_state、
//   ssm_state、层输出）**位级一致**。差异字节数是报告项且随运行变化（padding = UB 残留），
//   本 commit 的 evidence/accept_run_m40.log 末尾报告项给出口径与合计值。
// 处置（不是忽略，而是把「哪些字节是被消费的」本身写成判据）：
//   · M.gdnws    = 逐字节，但**排除** g/β 的 [h][1..7] padding 车道（判定项）
//   · M.gb_lane0 = 只对 g/β 的 [h][0]（48 head × 2 数组 × 4B）逐字节（判定项，独立见证）
//   · padding 内的差异字节数作为**报告项**打印（不参与 PASS/FAIL），供 reviewer 复核。
struct SkipRange {
    uint32_t off;
    uint32_t len;
};

// g/β 的 [h][1..7] padding 车道（两个数组各 48 个 32B 槽位，每槽跳过 28B）
static void H_GbPaddingRanges(std::vector<SkipRange>& out)
{
    out.clear();
    const uint32_t slots[2] = {M15G::WS_G, M15G::WS_BETA};
    for (uint32_t s = 0; s < 2; ++s) {
        for (uint32_t h = 0; h < M15G::HEADS; ++h) {
            out.push_back({slots[s] + h * 32u + 4u, 28u});
        }
    }
}

static bool H_CmpBytesSkipping(M15Run::Ctx& C, const char* tag, const uint8_t* got, const uint8_t* exp, size_t n,
                               const std::vector<SkipRange>& skip, size_t* skippedDiffs)
{
    C.checks++;
    size_t bad = 0;
    size_t skipBad = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; ++i) {
        if (got[i] == exp[i]) {
            continue;
        }
        bool inSkip = false;
        for (const SkipRange& r : skip) {
            if (i >= r.off && i < static_cast<size_t>(r.off) + r.len) {
                inSkip = true;
                break;
            }
        }
        if (inSkip) {
            ++skipBad;
        } else {
            if (bad == 0) {
                first = i;
            }
            ++bad;
        }
    }
    if (skippedDiffs != nullptr) {
        *skippedDiffs = skipBad;
    }
    if (bad == 0) {
        printf("[m15]   %-30s PASS (除 g/β padding 外逐字节一致, n=%zu; padding 内差异 %zu B 见报告项)\n", tag, n, skipBad);
        return true;
    }
    printf("[m15]   %-30s FAIL (%zu/%zu 字节不同（padding 外）; first off %zu got 0x%02x exp 0x%02x)\n", tag, bad, n,
           first, got[first], exp[first]);
    C.fails++;
    return false;
}

// 只比 g/β 的 [h][0]（被消费的 48×2 个 fp32）
static bool H_CmpGbLane0(M15Run::Ctx& C, const char* tag, const uint8_t* got, const uint8_t* exp)
{
    const uint32_t slots[2] = {M15G::WS_G, M15G::WS_BETA};
    C.checks++;
    size_t bad = 0;
    size_t first = 0;
    for (uint32_t s = 0; s < 2; ++s) {
        for (uint32_t h = 0; h < M15G::HEADS; ++h) {
            const size_t o = static_cast<size_t>(slots[s]) + static_cast<size_t>(h) * 32u;
            for (size_t k = 0; k < 4; ++k) {
                if (got[o + k] != exp[o + k]) {
                    if (bad == 0) {
                        first = o + k;
                    }
                    ++bad;
                }
            }
        }
    }
    if (bad == 0) {
        printf("[m15]   %-30s PASS (g/β 槽位首元素 48×2 个 fp32 逐字节一致)\n", tag);
        return true;
    }
    printf("[m15]   %-30s FAIL (%zu 字节不同; first off %zu)\n", tag, bad, first);
    C.fails++;
    return false;
}


// ---- 非空洞性（docs/17 §4）：输出必须有限、且确实随输入变化 ----
static bool H_CmpFiniteBf16(M15Run::Ctx& C, const char* tag, const uint8_t* buf, size_t nbytes)
{
    C.checks++;
    const uint16_t* h = reinterpret_cast<const uint16_t*>(buf);
    const size_t n = nbytes / 2;
    size_t bad = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; ++i) {
        if ((h[i] & 0x7F80u) == 0x7F80u) {   // bf16: exp 全 1 = Inf/NaN
            if (bad == 0) {
                first = i;
            }
            ++bad;
        }
    }
    if (bad == 0) {
        printf("[m15]   %-30s PASS (bf16 有限：NaN/Inf 0/%zu)\n", tag, n);
        return true;
    }
    printf("[m15]   %-30s FAIL (%zu/%zu 个 bf16 是 NaN/Inf; first idx %zu bits 0x%04x)\n", tag, bad, n, first,
           h[first]);
    C.fails++;
    return false;
}

static bool H_CmpMustDiffer(M15Run::Ctx& C, const char* tag, const uint8_t* a, const uint8_t* b, size_t n)
{
    C.checks++;
    if (__builtin_memcmp(a, b, n) != 0) {
        printf("[m15]   %-30s PASS (与输入不同，非恒等)\n", tag);
        return true;
    }
    printf("[m15]   %-30s FAIL (输出与输入逐字节相同 → 该层恒等，非空洞性失败)\n", tag);
    C.fails++;
    return false;
}

// ============================================================
// 2b. dump（供 check_moe_ref.py 的独立 numpy 复算）
// ============================================================
// 形态：每层落一个 **整块 MoE 段 ws**（WS_MOE_BYTES）+ 层输入 + 层输出 + 段的输入锚点。
// 张量切分不写死在 python 里：随 dump 一起写一份 `moe_layout.txt`（**由 C++ 的 M15M 常量算出**），
// 每行 `tensor name=<n> mode=contig|expert_rows ws_off=<b> bytes=<n> stride=<b> count=<n> dtype=<d> shape=<s>`，
// python 侧按它切片 —— 这样「python 用的偏移」与「kernel 用的偏移」是**同一个来源**，
// 不存在两边常量抄错还能互相印证的可能。
struct MoeLayoutRow {
    const char* name;
    uint32_t    off;
    uint32_t    bytes;    // contig: 总字节；expert_rows: 每专家一行的字节
    uint32_t    stride;   // expert_rows: 相邻专家行距；contig: 0
    uint32_t    count;    // expert_rows: 专家数；contig: 1
    const char* dtype;
    const char* shape;
};

static uint32_t MoeLayoutRows(MoeLayoutRow* out)
{
        uint32_t k = 0;
    auto add = [&](const char* n, uint32_t o, uint32_t b, uint32_t st, uint32_t c, const char* d, const char* s) {
        out[k++] = MoeLayoutRow{n, o, b, st, c, d, s};
    };
    uint32_t row0 = 0;
    auto addRow0 = [&](const char* n, uint32_t o, uint32_t b, const char* d, const char* s) { add(n, o, b, 0, 1, d, s); };
    addRow0("x_norm", M15M::WS_XNORM, M15M::HIDDEN * 2, "BF16", "1x2560");
    addRow0("res1", M15M::WS_RES1, M15M::HIDDEN * 4, "F32", "1x2560");
    addRow0("router_logits", M15M::WS_LOGITS, M15M::NUM_EXPERTS * 4, "F32", "1x4");
    addRow0("topk_ids", M15M::WS_IDS, M15M::TOPK_MAX * 4, "I32", "1x4");
    addRow0("topk_weights", M15M::WS_W, M15M::TOPK_MAX * 4, "F32", "1x4");
    add("perm_src_token", M15M::WS_PERM_SRC, M15M::M_MAX * M15M::TOPK_MAX * 4, 0, 1, "I32", "256");
    add("perm_expert", M15M::WS_PERM_EXP, M15M::M_MAX * M15M::TOPK_MAX * 4, 0, 1, "I32", "256");
    add("expert_token_counts", M15M::WS_COUNTS, M15M::NUM_EXPERTS * 4, 0, 1, "I32", "4");
    add("expert_offsets", M15M::WS_OFFSETS, (M15M::NUM_EXPERTS + 1) * 4, 0, 1, "I32", "5");
    add("inv_slot", M15M::WS_INV, M15M::M_MAX * M15M::TOPK_MAX * 4, 0, 1, "I32", "256");
    add("w_tk_packed", M15M::WS_WTK, 16 * 4, 0, 1, "I32", "1x16");
    addRow0("sgate", M15M::WS_SGATE, 4, "F32", "1");
    addRow0("x_sorted", M15M::WS_XSORT, M15M::M_MAX * M15M::TOPK_MAX * M15M::HIDDEN * 2, "BF16", "256x2560");
    add("A_qx", M15M::WS_AQ + 0, (M15M::HIDDEN / 2), M15M::WS_OFFSETS, M15M::NUM_EXPERTS, "U8", "E x 1280");
    add("A_scale", M15M::WS_AS + 0, M15M::GU_SCALE_STRIDE, M15M::WS_OFFSETS, M15M::NUM_EXPERTS, "U8", "E x 80");
    addRow0("A_qx_shd", M15M::WS_AQ_SHD, M15M::HIDDEN / 2, "U8", "1x1280");
    addRow0("A_scale_shd", M15M::WS_AS_SHD, M15M::GU_SCALE_STRIDE, "U8", "1x80");
    add("GU", M15M::WS_GU + 0, M15M::GU_N * 2, M15M::WS_OFFSETS, M15M::NUM_EXPERTS, "BF16", "E x 1280");
    addRow0("GU_shd", M15M::WS_GU_SHD, M15M::GU_N * 2, "BF16", "1x1280");
    addRow0("h_swiglu", M15M::WS_H, M15M::M_MAX * M15M::TOPK_MAX * M15M::INTER * 2, "BF16", "256x640");
    addRow0("h_swiglu_shd", M15M::WS_H_SHD, M15M::INTER * 2, "BF16", "1x640");
    add("H_qx", M15M::WS_HQ + 0, M15M::INTER / 2, M15M::WS_OFFSETS, M15M::NUM_EXPERTS, "U8", "E x 320");
    add("H_scale", M15M::WS_HS + 0, M15M::DN_SCALE_STRIDE, M15M::WS_OFFSETS, M15M::NUM_EXPERTS, "U8", "E x 32");
    addRow0("H_qx_shd", M15M::WS_HQ_SHD, M15M::INTER / 2, "U8", "1x320");
    addRow0("H_scale_shd", M15M::WS_HS_SHD, M15M::DN_SCALE_STRIDE, "U8", "1x32");
    add("Y", M15M::WS_Y + 0, M15M::HIDDEN * 2, M15M::WS_OFFSETS, M15M::NUM_EXPERTS, "BF16", "E x 2560");
    addRow0("Y_shd", M15M::WS_Y_SHD, M15M::HIDDEN * 2, "BF16", "1x2560");
    addRow0("routed_output", M15M::WS_ROUTED, M15M::HIDDEN * 2, "BF16", "1x2560");
    addRow0("shared_output", M15M::WS_SHARED, M15M::HIDDEN * 2, "BF16", "1x2560");
    addRow0("moe_output", M15M::WS_MOE, M15M::HIDDEN * 2, "BF16", "1x2560");
    addRow0("y_final", M15M::WS_YFINAL, M15M::HIDDEN * 2, "BF16", "1x2560");
    addRow0("res2", M15M::WS_RES2, M15M::HIDDEN * 4, "F32", "1x2560");
    (void)row0;
    return k;
}

static void H_DumpMoeLayout(const char* path)
{
    std::vector<MoeLayoutRow> rows(40);
    const uint32_t n = MoeLayoutRows(rows.data());
    FILE* f = fopen(path, "wb");
    if (f == nullptr) {
        return;
    }
    fprintf(f, "# MoE 段 ws 张量布局（由 m15_layer_resources.h / m15_moe_resources.h 的常量算出）\n");
    fprintf(f, "# mode=contig     : 从 ws_off 起 bytes 字节连续\n");
    fprintf(f, "# mode=expert_rows: 第 e 个专家在 ws_off + e*stride 处取 bytes 字节（共 count 个）\n");
    for (uint32_t i = 0; i < n; ++i) {
        const MoeLayoutRow& r = rows[i];
        fprintf(f, "tensor name=%s mode=%s ws_off=%u bytes=%u stride=%u count=%u dtype=%s shape=%s\n", r.name,
                r.stride ? "expert_rows" : "contig", r.off, r.bytes, r.stride, r.count, r.dtype, r.shape);
    }
    fprintf(f, "meta ws_bytes=%u num_experts=%u topk_max=%u m_max=%u hidden=%u inter=%u\n",
            M15M::WS_BYTES, M15M::NUM_EXPERTS, M15M::TOPK_MAX, M15M::M_MAX, M15M::HIDDEN, M15M::INTER);
    fclose(f);
}

// 落一层：ws 整块 + 层输入 + 层输出（相位 B 的入口锚点 = 相位 A 出口，已在 ws 的 x_norm 里）
static void H_DumpMoeLayer(M15Run::Ctx& C, const char* prefix, const uint8_t* wsMoe, const uint8_t* xIn,
                           const uint8_t* yOut, bool isGdn)
{
    char nm[96];
    snprintf(nm, sizeof(nm), "%s_moe_ws", prefix);
    H_Dump(C, nm, wsMoe, M15L::WS_MOE_BYTES, "RAW", "MoE 段整块 ws（张量切分见 moe_layout.txt）");
    snprintf(nm, sizeof(nm), "%s_layer_in", prefix);
    H_Dump(C, nm, xIn, M15Loop::H_BYTES, "BF16", "1x2560", isGdn ? "role=GDN段出口" : "role=attn占位出口");
    snprintf(nm, sizeof(nm), "%s_layer_out", prefix);
    H_Dump(C, nm, yOut, M15Loop::H_BYTES, "BF16", "1x2560", "role=融合层出口(MoE S10)");
}

static bool H_InList2(const std::vector<uint32_t>& v, uint32_t x)
{
    for (uint32_t e : v) {
        if (e == x) {
            return true;
        }
    }
    return false;
}

// ============================================================
// 3. 验证 M1：段间零串扰（融合 vs 独立启动，逐字节）
// ============================================================
static bool H_RunM1(M15Run::Ctx& C)
{
    using namespace M15Run;
    const uint32_t n = C.O.nLayers;
    const uint32_t base = C.checks;
    printf("\n[m15] ===== 验证 M1：段间零串扰（融合 kernel vs 独立启动的两段，逐字节）=====\n");
    printf("[m15]   每层三路：融合(GDN|attn 段 + MoE 段) / 独立 GDN|attn / 独立 MoE(输入=独立段的出口)\n");

    H_ResetStates(C);
    H_ResetH(C, 0u);

    std::vector<uint8_t> wsGdnRef(M15L::WS_GDN_BYTES);
    std::vector<uint8_t> wsMoeRef(M15L::WS_MOE_BYTES);
    std::vector<uint8_t> tmpGDN(M15L::WS_GDN_BYTES);
    std::vector<uint8_t> tmpMOE(M15L::WS_MOE_BYTES);
    std::vector<uint8_t> hIn(M15Loop::H_BYTES);
    std::vector<uint8_t> yFused(M15Loop::H_BYTES);
    std::vector<uint8_t> ySub(M15Loop::H_BYTES);
    std::vector<uint8_t> yMoe(M15Loop::H_BYTES);
    std::vector<uint16_t> csBefore(M15Loop::CS_BYTES / 2);
    std::vector<uint8_t> csFused(M15Loop::CS_BYTES);
    std::vector<uint8_t> csSub(M15Loop::CS_BYTES);
    std::vector<float> ssmBefore(M15Loop::SSM_BYTES / 4);
    std::vector<uint8_t> ssmFused(M15Loop::SSM_BYTES);
    std::vector<uint8_t> ssmSub(M15Loop::SSM_BYTES);

    bool ok = true;
    __builtin_memcpy(hIn.data(), C.h0[0].data(), M15Loop::H_BYTES);

    for (uint32_t L = 0; L < n; ++L) {
        const bool isGdn = (M15Loop::KindOf(L) == M15Loop::KIND_GDN);
        char tag[64];

        // ---------- (a) 融合路 ----------
        H_H2D(C, C.xFusedDev, hIn.data(), M15Loop::H_BYTES, "x_fused");
        if (isGdn) {
            H_D2H(C, csBefore.data(), H_ST(C, L, 0), M15Loop::CS_BYTES, "cs_before");
            H_D2H(C, ssmBefore.data(), H_ST(C, L, M15Loop::SSM_OFF), M15Loop::SSM_BYTES, "ssm_before");
        }
        H_LaunchFused(C, L, C.wsFusedDev, C.xFusedDev, C.yFusedDev, isGdn ? H_ST(C, L, 0) : nullptr,
                      isGdn ? H_ST(C, L, M15Loop::SSM_OFF) : nullptr, 1u);
        if (!H_Sync(C, "fused layer")) {
            return false;
        }
        H_D2H(C, yFused.data(), C.yFusedDev, M15Loop::H_BYTES, "y_fused");

        // ---------- (b) 独立子层段 ----------
        if (isGdn) {
            H_H2D(C, H_ST(C, L, 0), csBefore.data(), M15Loop::CS_BYTES, "cs_restore");
            H_H2D(C, H_ST(C, L, M15Loop::SSM_OFF), ssmBefore.data(), M15Loop::SSM_BYTES, "ssm_restore");
            H_LaunchGdnOnly(C, L, C.wsGdnDev, C.xFusedDev, C.ySubDev, H_ST(C, L, 0), H_ST(C, L, M15Loop::SSM_OFF));
        } else {
            m15_attn_placeholder_kernel<<<C.numBlocks, 0, C.stream>>>(
                reinterpret_cast<uint8_t*>(C.xFusedDev), reinterpret_cast<uint8_t*>(C.ySubDev), M15Loop::H_BYTES);
        }
        if (!H_Sync(C, "sublayer only")) {
            return false;
        }
        H_D2H(C, ySub.data(), C.ySubDev, M15Loop::H_BYTES, "y_sub");

        // ---------- (c) 独立 MoE 段（输入 = 独立子层段的出口）----------
        H_H2D(C, C.xMoeDev, ySub.data(), M15Loop::H_BYTES, "x_moe");
        H_LaunchMoeOnly(C, L, C.wsMoeDev, C.xMoeDev, C.yMoeDev);
        if (!H_Sync(C, "moe only")) {
            return false;
        }
        H_D2H(C, yMoe.data(), C.yMoeDev, M15Loop::H_BYTES, "y_moe");

        // ---------- 判据 ----------
        // M.finite：融合层出口必须是有限 bf16（非空洞性的第一道：抓权重/量化器出 NaN）
        snprintf(tag, sizeof(tag), "M.finite.L%02u", L);
        ok &= H_CmpFiniteBf16(C, tag, yFused.data(), M15Loop::H_BYTES);
        // M.nontrivial：层输出 != 层输入（整层不是恒等映射；防「恒输出常数也能过」）
        snprintf(tag, sizeof(tag), "M.nontrivial.L%02u", L);
        ok &= H_CmpMustDiffer(C, tag, yFused.data(), hIn.data(), M15Loop::H_BYTES);
        // M.ym：融合层出口 == 独立 MoE 段出口（同一 MoE 段、同一输入）
        snprintf(tag, sizeof(tag), "M.ym.L%02u", L);
        ok &= H_CmpBytes(C, tag, yFused.data(), yMoe.data(), M15Loop::H_BYTES);
        // M.moews：两路的 MoE 段 ws 全量逐字节（这是「零串扰」的主要见证）
        H_D2H(C, tmpMOE.data(), reinterpret_cast<uint8_t*>(C.wsFusedDev) + M15L::WS_MOE_OFF, M15L::WS_MOE_BYTES,
              "ws_moe_fused");
        H_D2H(C, wsMoeRef.data(), C.wsMoeDev, M15L::WS_MOE_BYTES, "ws_moe_only");
        snprintf(tag, sizeof(tag), "M.moews.L%02u", L);
        ok &= H_CmpBytes(C, tag, tmpMOE.data(), wsMoeRef.data(), M15L::WS_MOE_BYTES);

        if (isGdn) {
            // M.gdnws：两路的 GDN 段 ws 全量逐字节
            H_D2H(C, tmpGDN.data(), C.wsFusedDev, M15L::WS_GDN_BYTES, "ws_gdn_fused");
            H_D2H(C, wsGdnRef.data(), C.wsGdnDev, M15L::WS_GDN_BYTES, "ws_gdn_only");
            snprintf(tag, sizeof(tag), "M.gdnws.L%02u", L);
            size_t gbPadDiffs = 0;
            static std::vector<SkipRange> gbSkip;
            if (gbSkip.empty()) {
                H_GbPaddingRanges(gbSkip);
            }
            ok &= H_CmpBytesSkipping(C, tag, tmpGDN.data(), wsGdnRef.data(), M15L::WS_GDN_BYTES, gbSkip, &gbPadDiffs);
            snprintf(tag, sizeof(tag), "M.gb_lane0.L%02u", L);
            ok &= H_CmpGbLane0(C, tag, tmpGDN.data(), wsGdnRef.data());
            C.gbPadDiffBytes += gbPadDiffs;
            C.gbPadDiffMax = (gbPadDiffs > C.gbPadDiffMax) ? gbPadDiffs : C.gbPadDiffMax;
            // 注：这里**没有**「融合路的相位 A 出口 == 独立 GDN 段出口」这条判据，因为它在本形态下
            // 不可观测也不成立：融合 kernel 的 `yLayer` 同时是相位 B 的**输入**与**输出**
            // （MoE S1 读它、S10 写回它，见 m15_layer_kernel.h 的段序），故启动结束后 `yLayer`
            // 里是 MoE 段的输出，相位 A 的出口已被覆盖。相位 A 的正确性由**更强的**三条见证：
            //   · M.gdnws / M.gb_lane0：GDN 段 ws 全量逐字节（含 WS_OPOUT 与 fp32 的 WS_RES2，
            //     而 y = f(WS_OPOUT, WS_RES1, gamma2) 是它们的确定函数）；
            //   · M.st_cs / M.st_ssm：状态逐字节；
            //   · M.ym：两侧喂同一输入给同一个 MoE 段（独立路的输入就是独立 GDN 段的出口），
            //     出口逐字节相同 ⇒ 相位 A 出口在两路里相同（MoE 段是确定函数）。
            // M.st：状态（融合路 vs 独立 GDN 段）。
            // (a) 先跑融合路，D2H 出状态；随后 (b) 把状态复原到同一初值再跑独立 GDN 段，
            // 故此刻 stDev 里的就是「独立路」的状态 —— 两路状态若不同，这条判据会红。
            H_D2H(C, csFused.data(), H_ST(C, L, 0), M15Loop::CS_BYTES, "cs_fused");
            H_D2H(C, ssmFused.data(), H_ST(C, L, M15Loop::SSM_OFF), M15Loop::SSM_BYTES, "ssm_fused");
            H_D2H(C, csSub.data(), H_ST(C, L, 0), M15Loop::CS_BYTES, "cs_sub");
            H_D2H(C, ssmSub.data(), H_ST(C, L, M15Loop::SSM_OFF), M15Loop::SSM_BYTES, "ssm_sub");
            snprintf(tag, sizeof(tag), "M.st_cs.L%02u", L);
            ok &= H_CmpBytes(C, tag, csFused.data(), csSub.data(), M15Loop::CS_BYTES);
            snprintf(tag, sizeof(tag), "M.st_ssm.L%02u", L);
            ok &= H_CmpBytes(C, tag, ssmFused.data(), ssmSub.data(), M15Loop::SSM_BYTES);
        } else {
            // M.attn：M25 的「占位直通」性质（独立路的入口 == 出口）
            snprintf(tag, sizeof(tag), "M.attn.L%02u", L);
            ok &= H_CmpBytes(C, tag, ySub.data(), hIn.data(), M15Loop::H_BYTES);
        }

        if (C.O.dump && H_InList2(C.O.moeDumpLayers, L)) {
            char pf[64];
            snprintf(pf, sizeof(pf), "L%02u", L);
            H_DumpMoeLayer(C, pf, tmpMOE.data(), ySub.data(), yFused.data(), isGdn);
        }

        __builtin_memcpy(hIn.data(), yFused.data(), M15Loop::H_BYTES);
    }
    C.phaseMm = C.checks - base;
    printf("[m15] 验证 M1：%u 层，本段判定项 %u 条，累计 FAIL %u\n", n, C.phaseMm, C.fails);
    printf("[m15]   [报告项·不计入判定] g/\u03b2 padding 车道差异：合计 %zu B（%u 层），单层最大 %u B；"
           "该区域被消费方（S4 递推）按契约不读（见本节 1b 说明）\n",
           C.gbPadDiffBytes, n, C.gbPadDiffMax);
    return ok;
}

// ============================================================
// 4. 验证 M2：多 token（融合形态）
// ============================================================
// 目的（mission task3「≥3 token 连续」+ docs/17 §4 非空洞性）：
//   ① 交付形态：3 个 token 连续跑融合 kernel（状态跨 token 常驻，层间只经 GM 残差流交接）；
//   ② 等价性：融合形态（**每层 1 次启动**）与**独立启动形态**（每层 2 次启动：GDN|attn 段 + MoE 段，
//      串在同一 stream 上，靠 kernel 边界交接）**逐字节等价** —— 后者正是 M25 已验证过的组合
//      （Ver A/B/C 对 host 参考链），因此这条判据把「融合形态」接到那条已验证的链上；
//   ③ 非空洞性：层输出确实随 token 变化（输入敏感性）、状态确实演化；
//   ④ 确定性前提：连发（不插 host sync，交付形态）与逐层 sync 两种下发形态逐字节一致，
//      且连发形态自身可重复 —— 「逐字节」类判据的前提（docs/17 §4 末条）。
static bool H_RunM2(Ctx& C)
{
    using namespace M15Run;
    const uint32_t n = C.O.nLayers;
    const uint32_t steps = C.O.steps;
    const uint32_t base = C.checks;
    printf("\n[m15] ===== 验证 M2：多 token（融合形态；%u token × %u 层）=====\n", steps, n);

    // fused=true  : 每层 1 次启动（融合 kernel）
    // fused=false : 每层 2 次启动（独立 GDN|attn 段 → 独立 MoE 段，同 stream 串行，kernel 边界交接）
    // 每 token 结束时快照**同一层**的 state（cs + ssm）——M2.state_evolved 的真正见证
    // （review r1：原先那条比的是「层 0 的 conv_state」与「末层 ssm_state」两块无关缓冲，
    //  永远不等 ⇒ 恒 PASS，是空洞判据）。
    std::vector<std::vector<uint8_t>> perTokenState;
    uint32_t evoLayer = 0;
    for (uint32_t L = 0; L < n; ++L) {
        if (M15Loop::KindOf(L) == M15Loop::KIND_GDN) {
            evoLayer = L;
            break;
        }
    }
    auto runLoop = [&](bool fused, bool perLayerSync, std::vector<uint8_t>& hLast, std::vector<uint8_t>& stOut,
                       std::vector<std::vector<uint8_t>>* perTokenOut) -> bool {
        std::vector<uint8_t> hOut(M15Loop::H_BYTES);
        H_ResetStates(C);
        for (uint32_t t = 0; t < steps; ++t) {
            H_ResetH(C, t);   // memset 两个 h 缓冲 + H2D h0[t] 到 h[0] 行 0
            for (uint32_t L = 0; L < n; ++L) {
                const bool isGdn = (M15Loop::KindOf(L) == M15Loop::KIND_GDN);
                void* xB = H_H(C, L);
                void* yB = H_H(C, L + 1);
                if (fused) {
                    H_LaunchFused(C, L, C.wsFusedDev, xB, yB, isGdn ? H_ST(C, L, 0) : nullptr,
                                  isGdn ? H_ST(C, L, M15Loop::SSM_OFF) : nullptr, 1u);
                } else if (isGdn) {
                    H_LaunchGdnOnly(C, L, C.wsGdnDev, xB, C.ySubDev, H_ST(C, L, 0), H_ST(C, L, M15Loop::SSM_OFF));
                    H_LaunchMoeOnly(C, L, C.wsMoeDev, C.ySubDev, yB);
                } else {
                    m15_attn_placeholder_kernel<<<C.numBlocks, 0, C.stream>>>(
                        reinterpret_cast<uint8_t*>(xB), reinterpret_cast<uint8_t*>(C.ySubDev), M15Loop::H_BYTES);
                    H_LaunchMoeOnly(C, L, C.wsMoeDev, C.ySubDev, yB);
                }
                if (perLayerSync) {
                    H_Sync(C, "m2 layer");
                }
            }
            if (!H_Sync(C, "m2 token")) {
                return false;
            }
            H_D2H(C, hOut.data(), H_H(C, n), M15Loop::H_BYTES, "m2 h_last");
            if (perTokenOut != nullptr) {
                std::vector<uint8_t> snap(M15Loop::CS_BYTES + M15Loop::SSM_BYTES);
                H_D2H(C, snap.data(), H_ST(C, evoLayer, 0), M15Loop::CS_BYTES, "m2 evo cs");
                H_D2H(C, snap.data() + M15Loop::CS_BYTES, H_ST(C, evoLayer, M15Loop::SSM_OFF),
                      M15Loop::SSM_BYTES, "m2 evo ssm");
                perTokenState.push_back(snap);
            }
            if (getenv("M15_DBG") != nullptr) {
                printf("[m15][dbg] %s t=%u h_last[0..7]=", fused ? "fused" : "comp ", t);
                for (int q = 0; q < 8; ++q) printf("%02x", hOut[q]);
                printf(" h[0]row0[0..7]=");
                H_D2H(C, hOut.data(), H_H(C, 0), M15Loop::H_BYTES, "dbg h0");
                for (int q = 0; q < 8; ++q) printf("%02x", hOut[q]);
                printf("\n");
            }
            if (perTokenOut != nullptr) {
                perTokenOut->push_back(hOut);
            }
        }
        hLast = hOut;
        stOut.clear();
        for (uint32_t L = 0; L < n; ++L) {
            if (M15Loop::KindOf(L) != M15Loop::KIND_GDN) {
                continue;
            }
            const size_t off = stOut.size();
            stOut.resize(off + M15Loop::CS_BYTES + M15Loop::SSM_BYTES);
            H_D2H(C, stOut.data() + off, H_ST(C, L, 0), M15Loop::CS_BYTES, "m2 cs");
            H_D2H(C, stOut.data() + off + M15Loop::CS_BYTES, H_ST(C, L, M15Loop::SSM_OFF), M15Loop::SSM_BYTES,
                  "m2 ssm");
        }
        return true;
    };

    std::vector<uint8_t> hFused, hComp, stFused, stComp;
    std::vector<std::vector<uint8_t>> yPerToken;
    if (!runLoop(true, true, hFused, stFused, &yPerToken)) {
        return false;
    }
    if (!runLoop(false, true, hComp, stComp, nullptr)) {
        return false;
    }

    bool ok = true;
    {
        char tag[64];
        snprintf(tag, sizeof(tag), "M2.fused_vs_composed_h");
        ok &= H_CmpBytes(C, tag, hFused.data(), hComp.data(), M15Loop::H_BYTES);
        snprintf(tag, sizeof(tag), "M2.fused_vs_composed_state");
        ok &= H_CmpBytes(C, tag, stFused.data(), stComp.data(), stFused.size());
    }
    // 非空洞性 ①（docs/17 T4）：**同一层**的 state 必须随 token 演化。
    // 比较对象 = 层 %u 在 token 0 结束时与 token %u 结束时的 (conv_state + ssm_state) 快照；
    // 两者逐字节相同即为空洞（状态没动）→ FAIL。这是真判据：它咬「状态不演化」，
    // 而不是像 review r1 指出的旧写法那样比两块不同层的无关缓冲。
    if (steps >= 2) {
        C.checks++;
        const std::vector<uint8_t>& a = perTokenState.front();
        const std::vector<uint8_t>& b = perTokenState.back();
        const bool evolved = (a.size() == b.size()) &&
                             (__builtin_memcmp(a.data(), b.data(), a.size()) != 0);
        size_t ndiff = 0;
        if (a.size() == b.size()) {
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i] != b[i]) {
                    ++ndiff;
                }
            }
        }
        if (evolved) {
            printf("[m15]   %-30s PASS（层 %u 的 cs+ssm：token 0 vs token %u 有 %zu/%zu 字节不同 → 状态在演化）\n",
                   "M2.state_evolved", evoLayer, steps - 1, ndiff, a.size());
        } else {
            printf("[m15]   %-30s FAIL（层 %u 的 cs+ssm 在 token 0 与 token %u 之间逐字节相同 → 状态未演化）\n",
                   "M2.state_evolved", evoLayer, steps - 1);
            C.fails++;
            ok = false;
        }
    } else {
        printf("[m15]   %-30s SKIP（M15_STEPS=%u < 2；验收档用 3）\n", "M2.state_evolved", steps);
    }
    // 非空洞性 ②：层输出随 token 变化（输入敏感性；steps ≥ 2 才有意义）
    if (steps >= 2) {
        C.checks++;
        const bool diff = (__builtin_memcmp(yPerToken.front().data(), yPerToken.back().data(), M15Loop::H_BYTES) != 0);
        if (diff) {
            printf("[m15]   %-30s PASS（末层输出 t0 != t%u，逐字节不同）\n", "M2.token_sensitive", steps - 1);
        } else {
            printf("[m15]   %-30s FAIL（末层输出不随 token 变化）\n", "M2.token_sensitive");
            C.fails++;
            ok = false;
        }
    } else {
        printf("[m15]   %-30s SKIP（M15_STEPS=%u < 2；验收档用 3）\n", "M2.token_sensitive", steps);
    }
    // 交付形态：连发（不插 sync） vs 逐层 sync，两次连发之间还要可比（确定性）
    {
        std::vector<uint8_t> hA, stA, hB, stB;
        if (!runLoop(true, false, hA, stA, nullptr)) {
            return false;
        }
        if (!runLoop(true, false, hB, stB, nullptr)) {
            return false;
        }
        char tag[64];
        snprintf(tag, sizeof(tag), "M2.nosync_vs_sync_h");
        ok &= H_CmpBytes(C, tag, hA.data(), hFused.data(), M15Loop::H_BYTES);
        snprintf(tag, sizeof(tag), "M2.nosync_vs_sync_state");
        ok &= H_CmpBytes(C, tag, stA.data(), stFused.data(), stA.size());
        snprintf(tag, sizeof(tag), "M2.nosync_repeat_h");
        ok &= H_CmpBytes(C, tag, hB.data(), hA.data(), M15Loop::H_BYTES);
        snprintf(tag, sizeof(tag), "M2.nosync_repeat_state");
        ok &= H_CmpBytes(C, tag, stB.data(), stA.data(), stB.size());
    }
    C.phaseM2 = C.checks - base;
    printf("[m15] 验证 M2：本段判定项 %u 条\n", C.phaseM2);
    return ok;
}

// ============================================================
// 5. 验证 P：计时/msprof 用的最小启动集（**无判据**，只发启动）
// ============================================================
// mission task4 要的是「段级 msprof 分解 = 设计输入」。段在 kernel 内，msprof 只能给到
// **每个 kernel 符号**的耗时/pipe 占比，所以这里发三种形态、让 msprof 按符号统计，再在
// README 里做差：
//   P1 融合：   每层 1 次 `m15_layer_kernel_gdn` / `_attn`（交付形态）
//   P2 子层段： 每层 1 次 `m15_gdn_layer_kernel`（M25 的层 kernel）
//   P3 MoE 段： 每层 1 次 `m15_moe_segment_kernel`
// 每形态内**层间不插 host sync**（连发），形态之间 sync 一次。于是：
//   MoE 段净增量/层 ≈ avg(P1) − avg(P2)（同为 1 次启动/层），并与 avg(P3) + 一次启动开销 交叉核对。
static bool H_RunP(M15Run::Ctx& C)
{
    using namespace M15Run;
    const uint32_t n = C.O.nLayers;
    const uint32_t reps = C.O.reps;
    printf("\n[m15] ===== 验证 P：计时/msprof 最小启动集（%u 层 × %u 轮 × 3 形态；无判据）=====\n", n, reps);

    // 需要每层的 x/y 指针：用一个 lambda 参数化的层循环
    auto launchLoop = [&](int which) {
        for (uint32_t L = 0; L < n; ++L) {
            const bool isGdn = (M15Loop::KindOf(L) == M15Loop::KIND_GDN);
            void* xB = H_H(C, L);
            void* yB = H_H(C, L + 1);
            if (which == 0) {
                H_LaunchFused(C, L, C.wsFusedDev, xB, yB, isGdn ? H_ST(C, L, 0) : nullptr,
                              isGdn ? H_ST(C, L, M15Loop::SSM_OFF) : nullptr, 1u);
            } else if (which == 1) {
                if (isGdn) {
                    H_LaunchGdnOnly(C, L, C.wsGdnDev, xB, yB, H_ST(C, L, 0), H_ST(C, L, M15Loop::SSM_OFF));
                } else {
                    m15_attn_placeholder_kernel<<<C.numBlocks, 0, C.stream>>>(
                        reinterpret_cast<uint8_t*>(xB), reinterpret_cast<uint8_t*>(yB), M15Loop::H_BYTES);
                }
            } else {
                H_LaunchMoeOnly(C, L, C.wsMoeDev, xB, C.ySubDev);
            }
        }
    };

    auto timedLoop = [&](const char* tag, int which) {
        H_ResetStates(C);
        H_ResetH(C, 0u);
        if (!H_Sync(C, "P warmup")) {
            return;
        }
        const double t0 = H_NowMs();
        for (uint32_t r = 0; r < reps; ++r) {
            H_ResetH(C, 0u);
            launchLoop(which);
        }
        if (!H_Sync(C, tag)) {
            return;
        }
        const double dt = H_NowMs() - t0;
        printf("[m15]   %-22s %u 层 × %u 轮 = %.3f ms（每层一次启动 %.4f ms，含 host 下发）\n", tag, n, reps, dt,
               dt / static_cast<double>(n) / static_cast<double>(reps));
    };
    timedLoop("P1 融合（1 次/层）", 0);
    timedLoop("P2 子层段（1 次/层）", 1);
    timedLoop("P3 MoE 段（1 次/层）", 2);
    printf("[m15]   说明：段在 kernel 内，msprof 只能按 kernel 符号统计；段级分解 = 上面三者的差\n");
    return true;
}
