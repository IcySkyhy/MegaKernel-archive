// ============================================================
// m15_hc_host.h —— M58 host 侧：hc 段权重的装载 / 来源校验 / 验证 H
//
// 本文件只在 m15_layer_loop.asc（单 TU）里被 include，位置在 m15_moe_host.h 之后（复用它的
// H_CmpBytes / H_CmpOk / H_Guard / H_CmpFiniteBf16 / H_CmpMustDiffer / H_Dump 等工具）。
// 它做四件事：
//   1. hc 段权重的 host 侧期望值（真实 checkpoint 切片 or 合成）与槽内装配
//      （**唯一 host 侧加工** = `block_inject_weight` 补到 16 行并把行 [4,16) 置零，见资源表 §1b）；
//   2. hc 段权重的**独立来源判据**（不复用装载路径，按 manifest 自己 pread 再与设备槽位 memcmp）；
//   3. **验证 H**：四相位形态（hc(attn) → 子层段 → hc(mlp) → MoE 段）的正确性 ——
//      **每层 7 条判定项**（计数 = 调用点自增 `C.checks`，与判据同源）：
//        H.finite        ：融合路层出口 H'' 有限（无 NaN/Inf）；
//        H.nontrivial    ：H'' ≠ 层输入 H（逐字节，整层不是恒等映射）；
//        H.h1ws          ：hc(attn) 那一块 ws 与**独立启动**的 hc 单边界 kernel 逐字节一致；
//        H.h2ws          ：hc(mlp) 那一块 ws 与独立启动逐字节一致（输入取独立跑出来的 H' 与
//                          injection，即「层内 handoff 不走 host」这条设计被逐字节见证）；
//        H.conly         ：**一条合并判定项、两个面** —— combine-only 档（第 4 档 mode）的 H'
//                          与 combine_and_mix 档逐字节相同 **且** BLK 区未被写出（该档只跑 W0+S1；
//                          「BLK 未写」是判别性的那一半，见函数内注释与 README M58-5）；
//        H.det_h1/_h2    ：同一二进制重复运行、两个边界 ws 逐字节相同（确定性）。
//      **没有 guard 参与计数**（review r1 的 P2-1：同一件事不得既进 checks 又进 guards）。
//   4. dump（`M15_DUMP=1`）：hc 的输入 / 权重 / 两个边界的 8 个中间张量 + 层出口，供
//      `check_hc_ref.py` 用 **m20 的独立参考**（`m20_hyperconn/check_ref.py` 的 `reference()`）复算。
//
// 为什么 H.h1ws/H.h2ws 是「融合」这件事的核心判据：hc 段与 GDN/MoE 段共用 UB 段窗 / L1 / L0C /
// BufferID 编号（m15_layer_resources.h §2/§5 的叠放与复用），任何一处叠放或复用没被相位边界的
// PipeBarrier<PIPE_ALL> + mode-0 barrier 正确隔离，都会让 hc 段的 ws 与独立启动不同。逐字节比
// ws（而不只比 BLK）就是为了抓「值侥幸对了」的情况。
// ============================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ============================================================
// 0. hc 段权重（host 侧一个**边界**的权重）
// ============================================================

struct M15HcW {
    std::vector<uint8_t> norm;   // [HYPER] bf16        ← hc_norm.weight
    std::vector<uint8_t> down;   // [LOWRANK, HYPER] bf16 ← input_mix_weight_down.weight
    std::vector<uint8_t> up;     // [UP_N, UP_K] bf16   ← input_mix_weight_up.weight
    std::vector<uint8_t> inj;    // [HC_WINJ_ROWS, HYPER] bf16 ← block_inject_weight.weight（行 [4,16) 置零）
    uint32_t layer = 0;
    bool real = false;
};

static void H_HcWAlloc(M15HcW& W)
{
    W.norm.resize(M15L::HW_NORM_BYTES);
    W.down.resize(M15L::HW_WDOWN_BYTES);
    W.up.resize(M15L::HW_WUP_BYTES);
    W.inj.resize(M15L::HW_WINJ_BYTES);
}

// 合成档：确定性公式（无 rand 状态），只用于 M15_SYNTH=1 冒烟
static void H_GenHcW(M15HcW& W, uint32_t layer, uint32_t boundarySalt)
{
    const uint32_t salt = 70000u + layer * 53u + boundarySalt * 977u;
    auto fill = [](std::vector<uint8_t>& v, uint32_t s) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<uint8_t>((s * 2654435761u + static_cast<uint32_t>(i) * 40503u) >> 13);
        }
    };
    for (size_t i = 0; i < W.norm.size(); i += 2) {          // bf16 小量级（~0.5）
        const uint16_t h = static_cast<uint16_t>(0x3F00u + ((salt + static_cast<uint32_t>(i / 2)) % 64u));
        W.norm[i] = static_cast<uint8_t>(h & 0xFFu);
        W.norm[i + 1] = static_cast<uint8_t>(h >> 8);
    }
    fill(W.down, salt + 1u);
    fill(W.up, salt + 2u);
    fill(W.inj, salt + 3u);
    // 行 [INJ_N, HC_WINJ_ROWS) 必须为零（device 契约）
    const uint32_t rowBytes = M15H::HYPER * 2u;
    __builtin_memset(W.inj.data() + M15H::INJ_N * rowBytes, 0, (M15L::HC_WINJ_ROWS - M15H::INJ_N) * rowBytes);
    W.layer = layer;
    W.real = false;
}

// 真实档：按 manifest 的 role 自己 pread。**本函数只做「取字节」+ 一处契约补零**，不做任何数值换算。
static bool H_LoadHcW(const M15Manifest& M, uint32_t layer, const char* prefix, M15HcW& W, std::string& err)
{
    char role[64];
    auto rd = [&](const char* suffix, std::vector<uint8_t>& dst, size_t want) -> bool {
        snprintf(role, sizeof(role), "%s_%s", prefix, suffix);
        const M15Tensor* T = H_FindTensor(M, layer, role);
        if (T == nullptr) {
            err = std::string("manifest 缺 role=") + role + " layer=" + std::to_string(layer);
            return false;
        }
        if (T->bytes != want) {
            char b[256];
            snprintf(b, sizeof(b), "role=%s layer=%u 字节数不符：manifest %llu，期望 %zu", role, layer,
                     static_cast<unsigned long long>(T->bytes), want);
            err = b;
            return false;
        }
        return H_ReadTensor(M, *T, dst, err);
    };
    if (!rd("norm", W.norm, M15L::HW_NORM_BYTES) || !rd("down", W.down, M15L::HW_WDOWN_BYTES) ||
        !rd("up", W.up, M15L::HW_WUP_BYTES)) {
        return false;
    }
    // inj：checkpoint 只有 INJ_N=4 行 ⇒ 先读到临时区，再落到 16 行槽的后半置零
    const uint32_t rowBytes = M15H::HYPER * 2u;
    std::vector<uint8_t> inj4(rowBytes * M15H::INJ_N);
    if (!rd("inj", inj4, inj4.size())) {
        return false;
    }
    __builtin_memset(W.inj.data(), 0, W.inj.size());
    __builtin_memcpy(W.inj.data(), inj4.data(), inj4.size());
    W.layer = layer;
    W.real = true;
    return true;
}

// 把一个边界的权重装进「与设备槽同布局」的 host 镜像
static void H_PutHcSlot(std::vector<uint8_t>& arena, uint32_t layer, uint32_t which, const M15HcW& W)
{
    uint8_t* d = arena.data() + static_cast<size_t>(layer) * M15L::HC_LAYER_W_STRIDE +
                 ((which == 0u) ? M15L::HC_ATTN_W_OFF : M15L::HC_MLP_W_OFF);
    auto put = [d](uint32_t off, const void* src, size_t bytes) { __builtin_memcpy(d + off, src, bytes); };
    put(M15L::HW_NORM_OFF, W.norm.data(), W.norm.size());
    put(M15L::HW_WDOWN_OFF, W.down.data(), W.down.size());
    put(M15L::HW_WINJ_OFF, W.inj.data(), W.inj.size());
    put(M15L::HW_WUP_OFF, W.up.data(), W.up.size());
}

// ---- 从 hc 权重区算出 kernel 入口要的 4+4 个指针 ----
struct HcWPtrs {
    uint8_t* attnNorm;
    uint8_t* attnDown;
    uint8_t* attnInj;
    uint8_t* attnUp;
    uint8_t* mlpNorm;
    uint8_t* mlpDown;
    uint8_t* mlpInj;
    uint8_t* mlpUp;
};

static HcWPtrs H_HcWPtrsOf(void* hcWBase, uint32_t layer)
{
    uint8_t* s = reinterpret_cast<uint8_t*>(hcWBase) + static_cast<size_t>(layer) * M15L::HC_LAYER_W_STRIDE;
    uint8_t* a = s + M15L::HC_ATTN_W_OFF;
    uint8_t* m = s + M15L::HC_MLP_W_OFF;
    HcWPtrs p;
    p.attnNorm = a + M15L::HW_NORM_OFF;
    p.attnDown = a + M15L::HW_WDOWN_OFF;
    p.attnInj = a + M15L::HW_WINJ_OFF;
    p.attnUp = a + M15L::HW_WUP_OFF;
    p.mlpNorm = m + M15L::HW_NORM_OFF;
    p.mlpDown = m + M15L::HW_WDOWN_OFF;
    p.mlpInj = m + M15L::HW_WINJ_OFF;
    p.mlpUp = m + M15L::HW_WUP_OFF;
    return p;
}

// ============================================================
// 1. 启动器
// ============================================================

// ---- M100：PLE 段的 17 个实参的组装 ----
// 语义：**只在该层走 PLE 打断点（pleBreak != 0）、接线开着（`M15_PLE_WIRE=1`）、PLE 装载路径
// 已跑过（`pleTableDev != nullptr`，见 `H_PleArgsOf` 的门控注释）且层号 = `M15_PLE_LAYER`
// （默认 1）** 时非空；否则整组为空 ⇒ kernel 内 `A.pleW == nullptr` ⇒ 挂载点逐字退化为 M65 的
// 空操作（`runs=all` 的读数因此不受影响）。
// `negMask` / `stageMask` 是负向对照与分档开关（见 `runs=plewire`）。
struct PleArgs {
    uint8_t* w = nullptr;
    uint8_t* scr = nullptr;
    uint8_t* table = nullptr;
    uint8_t* ids = nullptr;
    uint8_t* qsl = nullptr;
    uint8_t* ctx = nullptr;
    uint8_t* m = nullptr;
    uint8_t* sz = nullptr;
    uint8_t* of = nullptr;
    uint32_t tok = 0, nreq = 0, tableRows = 0, stageMask = 0, negMask = 0;
    uint32_t winBase = 0, winRows = 0, stateSlots = 0;
};

static PleArgs H_PleArgsOf(M15Run::Ctx& C, uint32_t layer, uint32_t pleBreak)
{
    PleArgs pa;
    if (pleBreak == 0u || C.O.pleWire == 0u) {
        return pa;                      // 空组 = 接线关（kernel 内空操作）
    }
    // M177：门控必须区分「平面**已分配**」与「装载**已完成**」。`H_Alloc` 无条件分配
    // pleWDev/pleScrDev（m15_layer_loop.asc:2056-2057），但只有 `H_PleWireRun` 会 H2D 真实权重
    // 并建 host-mapped 表窗口；`C.pleTableDev` 由 `H_PleWindowSetup` 在**权重 H2D 之后**才置位
    // （m15_layer_loop.asc:2830）⇒ 它非空 = 「整条装载路径已跑过」的见证。旧门控只看
    // `pleWDev != nullptr`（分配即放行），于是 `runs=chain` / `runs=all` 配 `M15_PLE_WIRE=1`
    // 会以「未初始化权重 + 空表基址」进设备。缺见证时整组置空（kernel 逐字退化为空操作），
    // 并在被判层打一条 WARN —— 是**响亮跳过**，不是静默送垃圾。
    if (C.pleWDev == nullptr || C.pleTableDev == nullptr) {
        if (layer == C.O.pleLayer) {
            printf("[m15][WARN] H_PleArgsOf：M15_PLE_WIRE=1 但 PLE 装载路径未跑过（pleWDev=%p "
                   "pleTableDev=%p）⇒ 层 %u 的 PLE 接线被门控关闭（退化为空操作）；"
                   "M15_PLE_WIRE 只应在 runs=plewire 档打开\n",
                   C.pleWDev, C.pleTableDev, layer);
        }
        return pa;
    }
    if (layer != C.O.pleLayer) {
        return pa;
    }
    pa.w = reinterpret_cast<uint8_t*>(C.pleWDev);
    pa.scr = reinterpret_cast<uint8_t*>(C.pleScrDev);
    pa.table = reinterpret_cast<uint8_t*>(C.pleTableDev);
    pa.ids = reinterpret_cast<uint8_t*>(C.pleIdsDev);
    pa.qsl = reinterpret_cast<uint8_t*>(C.pleQslDev);
    pa.ctx = reinterpret_cast<uint8_t*>(C.pleCtxDev);
    pa.m = reinterpret_cast<uint8_t*>(C.pleMDev);
    pa.sz = reinterpret_cast<uint8_t*>(C.pleSzDev);
    pa.of = reinterpret_cast<uint8_t*>(C.pleOfDev);
    pa.tok = C.O.pleTok;
    pa.nreq = C.O.pleNReq;
    pa.tableRows = C.O.pleWinRows;
    pa.stageMask = C.O.pleStageMask;
    pa.negMask = C.O.pleNegMask;
    pa.winBase = C.O.pleWinBase;
    pa.winRows = C.O.pleWinRows;
    pa.stateSlots = M15L::PLEW::T_MAX;
    // ---- 负向对照（「实参错位」）：故意把接线接错，判据必须变红 ----
    //   1 = 表基址接错（把 scratch slab 当注册窗口读 ⇒ ② 取到的行内容必错）
    //   2 = 行基址偏 1（`row = id - winBase` 整体错位 ⇒ ② 逐字节判据与设备侧行校验同时红）
    if (C.O.pleMiswire == 1u) {
        pa.table = pa.scr;
    } else if (C.O.pleMiswire == 2u) {
        pa.winBase = pa.winBase + 1u;
    }
    return pa;
}

#define M15_PLE_LAUNCH_ARGS(PA)                                                                     \
    (PA).w, (PA).scr, (PA).table, (PA).ids, (PA).qsl, (PA).ctx, (PA).m, (PA).sz, (PA).of, (PA).tok,  \
        (PA).nreq, (PA).tableRows, (PA).stageMask, (PA).negMask, (PA).winBase, (PA).winRows,         \
        (PA).stateSlots

// 四相位融合：一次启动跑 hc(attn) → 子层段 → hc(mlp) → MoE 段
//
// **M65 起，三条层界输入（H / pending BO / IJ）由调用方给**（不再固定读 C.hcHDev/…）：
//   · 验证 H（单层）与 48 层链用同一套启动器，区别只在指针；
//   · `pleBreak != 0` ⇒ 该层走 PLE 打断点（H1 拆成 combine-only → PLE 占位 → mix-only）。
static void H_LaunchLayerHc(M15Run::Ctx& C, uint32_t layer, void* ws, void* yLayerDev, void* csDev,
                            void* ssmDev, void* hInDev, void* boDev, void* ijDev, uint32_t ijStride,
                            uint32_t pleBreak, uint32_t attnMode, uint32_t mlpMode, uint32_t m,
                            void* apW = nullptr, void* apX = nullptr, void* apCs = nullptr,
                            void* apY0 = nullptr, void* apOut = nullptr, uint32_t apPos = 0u,
                            uint32_t apMode = 0u)
{
    const bool isGdn = (M15Loop::KindOf(layer) == M15Loop::KIND_GDN);
    const MoeWPtrs mp = H_MoeWPtrsOf(C.moeWDev, layer);
    const HcWPtrs hp = H_HcWPtrsOf(C.hcWDev, layer);
    uint8_t* hcws = reinterpret_cast<uint8_t*>(C.hcWsDev) + static_cast<size_t>(layer) * M15L::HC_WS_LAYER_STRIDE;
    uint8_t* ws0 = hcws + M15L::HC_WS0_IN_LAYER;
    uint8_t* ws1 = hcws + M15L::HC_WS1_IN_LAYER;
    __gm__ uint8_t* hcH = reinterpret_cast<uint8_t*>(hInDev);
    __gm__ uint8_t* hcBo = reinterpret_cast<uint8_t*>(boDev);
    __gm__ uint8_t* hcIj = reinterpret_cast<uint8_t*>(ijDev);
    __gm__ uint8_t* attnOut = reinterpret_cast<uint8_t*>(C.hcAttnOutDev);
    const PleArgs pa = H_PleArgsOf(C, layer, pleBreak);   // M100：PLE 段（接线关时整组为空）
    if (isGdn) {
        M15L::m15_layer_kernel_gdn_hc<<<C.numBlocks, 0, C.stream>>>(
            reinterpret_cast<uint8_t*>(ws), reinterpret_cast<uint8_t*>(C.xFusedDev),
            reinterpret_cast<uint8_t*>(C.resZeroDev), reinterpret_cast<uint8_t*>(yLayerDev),
            reinterpret_cast<uint8_t*>(C.resZeroDev), M15Run::H_W(C, layer, M15Loop::W_G1_OFF),
            M15Run::H_W(C, layer, M15Loop::W_G2_OFF), M15Run::H_W(C, layer, M15Loop::W_GG_OFF),
            M15Run::H_W(C, layer, M15Loop::W_IN_OFF), M15Run::H_W(C, layer, M15Loop::W_OUT_OFF),
            M15Run::H_W(C, layer, M15Loop::W_CONV_OFF), M15Run::H_W(C, layer, M15Loop::W_CONVB_OFF),
            M15Run::H_W(C, layer, M15Loop::W_ALOG_OFF), M15Run::H_W(C, layer, M15Loop::W_DTB_OFF),
            reinterpret_cast<uint8_t*>(csDev), reinterpret_cast<uint8_t*>(ssmDev), mp.router, mp.sgate, mp.wGu,
            mp.sGu, mp.wDn, mp.sDn, mp.wGuShd, mp.sGuShd, mp.wDnShd, mp.sDnShd, mp.g1, mp.g2, hcH, hcBo, hcIj,
            attnOut, ws0, ws1, hp.attnNorm, hp.attnDown, hp.attnInj, hp.attnUp, hp.mlpNorm, hp.mlpDown,
            hp.mlpInj, hp.mlpUp, ijStride, attnMode, mlpMode, pleBreak,
            // M97：GDN 入口的 attention 字段全部缺省（kernel 内 KIND==KIND_ATTN 才读它们）
            nullptr, nullptr, nullptr, nullptr, nullptr, 0u, 0u, layer, m, C.topk,
            M15_PLE_LAUNCH_ARGS(pa));
    } else {
        M15L::m15_layer_kernel_attn_hc<<<C.numBlocks, 0, C.stream>>>(
            reinterpret_cast<uint8_t*>(ws), reinterpret_cast<uint8_t*>(C.xFusedDev),
            reinterpret_cast<uint8_t*>(C.resZeroDev), reinterpret_cast<uint8_t*>(yLayerDev),
            reinterpret_cast<uint8_t*>(C.resZeroDev), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, mp.router, mp.sgate, mp.wGu, mp.sGu, mp.wDn, mp.sDn, mp.wGuShd,
            mp.sGuShd, mp.wDnShd, mp.sDnShd, mp.g1, mp.g2, hcH, hcBo, hcIj, attnOut, ws0, ws1, hp.attnNorm,
            hp.attnDown, hp.attnInj, hp.attnUp, hp.mlpNorm, hp.mlpDown, hp.mlpInj, hp.mlpUp,
            ijStride, attnMode, mlpMode, pleBreak,
            // M97：attention 相位 A 的 8 个实参（只有 KIND_ATTN 的入口读它们）
            reinterpret_cast<uint8_t*>(apW), reinterpret_cast<uint8_t*>(apX), reinterpret_cast<uint8_t*>(apCs),
            reinterpret_cast<uint8_t*>(apY0), reinterpret_cast<uint8_t*>(apOut), apPos, apMode, layer, m, C.topk,
            M15_PLE_LAUNCH_ARGS(pa));
    }
}

// 只跑一个 hc 边界（对照路 / combine-only 入口）
static void H_LaunchHcOnly(M15Run::Ctx& C, uint32_t layer, void* hcBoundaryWs, uint32_t which,
                           __gm__ uint8_t* hIn, __gm__ uint8_t* bo, __gm__ uint8_t* ij, uint32_t ijStride,
                           uint32_t mode, uint32_t m)
{
    const HcWPtrs hp = H_HcWPtrsOf(C.hcWDev, layer);
    if (which == 0u) {
        M15L::m15_hc_segment_kernel<<<C.numBlocks, 0, C.stream>>>(
            hIn, bo, ij, ijStride, hp.attnDown, hp.attnInj, hp.attnUp, hp.attnNorm,
            reinterpret_cast<uint8_t*>(hcBoundaryWs), m, mode);
    } else {
        M15L::m15_hc_segment_kernel<<<C.numBlocks, 0, C.stream>>>(
            hIn, bo, ij, ijStride, hp.mlpDown, hp.mlpInj, hp.mlpUp, hp.mlpNorm,
            reinterpret_cast<uint8_t*>(hcBoundaryWs), m, mode);
    }
}

// ============================================================
// 2. hc 段权重来源判据（独立 pread + 逐字节 memcmp）
// ============================================================
// 与 H_MoeWeightSourceCheck 同款：**不复用装载路径**（自己 H_FindTensor + H_ReadTensor），
// 与设备槽位回读逐字节比对并打印两侧 sha256。它同时兜住：
//   ① 层号 ↔ hc 权重槽错位（每个 (层, 边界) 的摘要互不相同）；
//   ② `block_inject_weight` 的**补零契约**方向（行 [4,16) 必须是零，而不是 checkpoint 的残值）；
//   ③ 权重字节没被任何中间转换改动（hc 权重不做任何离线转换）。
static bool H_HcWeightSourceCheck(M15Run::Ctx& C)
{
    std::string err;
    const uint32_t rowBytes = M15H::HYPER * 2u;
    std::vector<uint8_t> devSlot(M15L::HC_LAYER_W_STRIDE);
    // 独立期望：按**整块权重区**布局装一遍（H_PutHcSlot 按层号编址，故期望区也必须按层号定尺）
    std::vector<uint8_t> expAll(static_cast<size_t>(C.O.nLayers) * M15L::HC_LAYER_W_STRIDE, 0);
    uint32_t n = 0;
    for (uint32_t L = 0; L < C.O.nLayers; ++L) {
        for (uint32_t which = 0; which < 2u; ++which) {
            const char* prefix = (which == 0u) ? "attn_hc" : "mlp_hc";
            M15HcW W;
            H_HcWAlloc(W);
            if (C.real) {
                if (!H_LoadHcW(C.M, L, prefix, W, err)) {
                    printf("[m15][FAIL] hc 权重独立 pread 失败（层 %u %s）：%s\n", L, prefix, err.c_str());
                    return false;
                }
            } else {
                H_GenHcW(W, L, which);
            }
            H_PutHcSlot(expAll, L, which, W);
        }
        const uint8_t* exp = expAll.data() + static_cast<size_t>(L) * M15L::HC_LAYER_W_STRIDE;
        H_D2H(C, devSlot.data(),
              reinterpret_cast<uint8_t*>(C.hcWDev) + static_cast<size_t>(L) * M15L::HC_LAYER_W_STRIDE,
              devSlot.size(), "hc_w_source_backup");
        if (!H_Sync(C, "hc_w_source_backup")) {
            return false;
        }
        char tag[96];
        snprintf(tag, sizeof(tag), "Hc.ws.L%02u", L);
        bool ok = H_CmpBytes(C, tag, devSlot.data(), exp, devSlot.size());
        // 独立见证「补零契约」（只看设备字节，不看 manifest）：两个边界的注入行 [INJ_N,16) 全零
        bool zeroOk = true;
        for (uint32_t which = 0; which < 2u; ++which) {
            const uint8_t* injSlot = devSlot.data() +
                                     ((which == 0u) ? M15L::HC_ATTN_W_OFF : M15L::HC_MLP_W_OFF) +
                                     M15L::HW_WINJ_OFF;
            for (uint32_t r = M15H::INJ_N; r < M15L::HC_WINJ_ROWS && zeroOk; ++r) {
                for (uint32_t b = 0; b < rowBytes; ++b) {
                    if (injSlot[r * rowBytes + b] != 0u) {
                        zeroOk = false;
                        break;
                    }
                }
            }
        }
        snprintf(tag, sizeof(tag), "Hc.injZero.L%02u", L);
        // 注意（计数纪律）：这里只用 H_Guard（恒真前置断言，**不计入判定项**），不再套一层
        // H_CmpOk —— 否则同一件事会被计两次，并且因为 A/B/C 的计数是「C.checks 减已归属项」的
        // 增量式写法，多出来的条数会**污染 phaseA**（本 mission 实测踩过：+96 落到 A 段）。
        const bool zok = H_Guard(C, zeroOk);
        C.guardsHc += 1u;   // review r1 P2-1b：guard 按来源分类计数（汇总行据此写真实分解）
        if (ok && zok) {
            C.phaseWh += 1u;
        }
        ++n;
    }
    printf("[m15] hc 段权重来源判据：%u 层 × 2 边界，逐字节 memcmp（含注入行 [4,16) 全零见证）\n", n);
    return C.fails == 0;
}

// ============================================================
// 3. 验证 H（四相位融合形态）
// ============================================================

static std::string H_ListStr(const std::vector<uint32_t>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i != 0) {
            s += ",";
        }
        s += std::to_string(v[i]);
    }
    return s;
}

struct HcDumpNames {
    const char* hcp;
    const char* xn;
    const char* rstd;
    const char* injw;
    const char* oh;
    const char* ls;
    const char* gate;
    const char* blk;
};

static void H_DumpHcBoundary(M15Run::Ctx& C, const char* tag, const uint8_t* wsBase, uint32_t m)
{
    if (!C.O.dump) {
        return;
    }
    char nm[96];
    struct Row {
        const char* suffix;
        uint32_t    off;
        uint32_t    bytes;   // 行 0..m-1 的字节数（按张量行距算）
        const char* dtype;
        const char* shape;
    };
    const uint32_t rows = m;
    const Row rows_[] = {
        {"hcp", M15H::WS_HCP, rows * M15H::HYPER * 2u, "BF16", "[m,10240]"},
        {"xn", M15H::WS_XN, rows * M15H::HYPER * 2u, "BF16", "[m,10240]"},
        {"rstd", M15H::WS_RSTD, rows * M15H::HC * 4u, "F32", "[m,4]"},
        {"injw", M15H::WS_INJW, rows * M15H::HC * M15H::INJW_SLOT * 4u, "F32", "[m,4,8]"},
        {"oh", M15H::WS_OH, rows * M15H::OH_W * 2u, "BF16", "[m,336]"},
        {"ls", M15H::WS_LS, rows * M15H::LOWRANK * 2u, "BF16", "[m,320]"},
        {"gate", M15H::WS_GATE, rows * M15H::UP_N * 2u, "BF16", "[m,10240]"},
        {"blk", M15H::WS_BLK, rows * M15H::HID * 2u, "BF16", "[m,2560]"},
    };
    for (const Row& r : rows_) {
        snprintf(nm, sizeof(nm), "%s_%s", tag, r.suffix);
        H_Dump(C, nm, wsBase + r.off, r.bytes, r.dtype, r.shape);
    }
}

static bool H_RunH(M15Run::Ctx& C)
{
    using namespace M15G;
    const uint32_t m = C.O.hcM;
    const uint32_t hyb = m * M15H::HYPER * 2u;
    const uint32_t hbytes = m * HIDDEN * 2u;

    printf("\n[m15] ===== 验证 H：四相位融合形态（hc(attn) → 子层段 → hc(mlp) → MoE 段；m=%u）=====\n", m);

    // ---- 输入（确定性）：H（4 路残差流）、BO、IJ ----
    std::vector<uint16_t> hin(static_cast<size_t>(M_MAX) * M15H::HYPER);
    std::vector<uint16_t> bo(static_cast<size_t>(M_MAX) * HIDDEN);
    std::vector<uint16_t> ij(static_cast<size_t>(M_MAX) * M15H::IJ_STRIDE);
    H_GenBf16(hin, 90001u, 1.0f);
    H_GenBf16(bo, 90002u, 0.8f);
    H_GenBf16(ij, 90003u, 1.5f);
    H_H2D(C, C.hcHDev, hin.data(), hin.size() * 2, "hc_hin");
    H_H2D(C, C.hcBoDev, bo.data(), bo.size() * 2, "hc_bo");
    H_H2D(C, C.hcIjDev, ij.data(), ij.size() * 2, "hc_ij");

    // ---- 逐层 ----
    const uint32_t checkBase = C.checks;   // 判定项计数口径：本段 = checks 的增量（与 M1 同款）
    uint32_t nL = 0;
    for (uint32_t idx = 0; idx < C.O.hcLayers.size(); ++idx) {
        const uint32_t L = C.O.hcLayers[idx];
        if (L >= C.O.nLayers) {
            continue;
        }
        ++nL;
        const uint32_t attnMode = (L == 0u) ? M15H::MODE_MIX : M15H::MODE_COMBINE_MIX;
        const uint32_t mlpMode = M15H::MODE_COMBINE_MIX;

        // 该层的 hc ws / 三块 scratch（0/1 = 单边界对照；2 = combine-only）
        uint8_t* hcw = reinterpret_cast<uint8_t*>(C.hcWsDev) + static_cast<size_t>(L) * M15L::HC_WS_LAYER_STRIDE;
        uint8_t* sc0 = reinterpret_cast<uint8_t*>(C.hcWsDev) +
                       static_cast<size_t>(M15Loop::NL + 0u) * M15L::HC_WS_LAYER_STRIDE;
        uint8_t* sc1 = reinterpret_cast<uint8_t*>(C.hcWsDev) +
                       static_cast<size_t>(M15Loop::NL + 1u) * M15L::HC_WS_LAYER_STRIDE;
        uint8_t* sc2 = reinterpret_cast<uint8_t*>(C.hcWsDev) +
                       static_cast<size_t>(M15Loop::NL + 2u) * M15L::HC_WS_LAYER_STRIDE;

        // ---- (1) 融合路：一次启动跑四个相位 ----
        aclrtMemset(hcw, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        aclrtMemset(C.hcAttnOutDev, static_cast<size_t>(M_MAX) * HIDDEN * 2u, 0xCD,
                    static_cast<size_t>(M_MAX) * HIDDEN * 2u);
        aclrtMemset(C.yFusedDev, M15Loop::H_ROWS_BYTES, 0xCD, M15Loop::H_ROWS_BYTES);
        if (M15Loop::KindOf(L) == M15Loop::KIND_GDN) {
            H_H2D(C, H_ST(C, L, 0), C.csInit.data(), M15Loop::CS_BYTES, "hc_cs_init");
            H_H2D(C, H_ST(C, L, M15Loop::SSM_OFF), C.ssmInit.data(), M15Loop::SSM_BYTES, "hc_ssm_init");
        }
        H_LaunchLayerHc(C, L, C.wsFusedDev, C.yFusedDev,
                        (M15Loop::KindOf(L) == M15Loop::KIND_GDN) ? H_ST(C, L, 0) : nullptr,
                        (M15Loop::KindOf(L) == M15Loop::KIND_GDN) ? H_ST(C, L, M15Loop::SSM_OFF) : nullptr,
                        C.hcHDev, C.hcBoDev, C.hcIjDev, M15H::HC_IJ_STRIDE_PLANE, 0u,
                        attnMode, mlpMode, m);
        if (!H_Sync(C, "hc_fused")) {
            return false;
        }
        std::vector<uint8_t> fusedH1(M15H::WS_BYTES), fusedH2(M15H::WS_BYTES);
        H_D2H(C, fusedH1.data(), hcw + M15L::HC_WS0_IN_LAYER, fusedH1.size(), "hc_fused_h1ws");
        H_D2H(C, fusedH2.data(), hcw + M15L::HC_WS1_IN_LAYER, fusedH2.size(), "hc_fused_h2ws");
        std::vector<uint8_t> hOut(hyb), yOut(hbytes), attnOut(hbytes);
        H_D2H(C, hOut.data(), hcw + M15L::HC_WS1_IN_LAYER + M15H::WS_HCP, hyb, "hc_hout");
        H_D2H(C, yOut.data(), C.yFusedDev, hbytes, "hc_yout");
        H_D2H(C, attnOut.data(), C.hcAttnOutDev, hbytes, "hc_attnout");
        if (!H_Sync(C, "hc_fused_d2h")) {
            return false;
        }

        char tag[96];
        snprintf(tag, sizeof(tag), "H.finite.L%02u", L);
        bool ok = H_CmpFiniteBf16(C, tag, hOut.data(), hyb);
        snprintf(tag, sizeof(tag), "H.nontrivial.L%02u", L);
        ok = H_CmpMustDiffer(C, tag, hOut.data(), reinterpret_cast<const uint8_t*>(hin.data()), hyb) && ok;

        // ---- (2) 独立启动 hc 边界 #1（同一组输入）----
        aclrtMemset(sc0, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        H_LaunchHcOnly(C, L, sc0, 0u, reinterpret_cast<uint8_t*>(C.hcHDev),
                       reinterpret_cast<uint8_t*>(C.hcBoDev), reinterpret_cast<uint8_t*>(C.hcIjDev),
                       M15H::HC_IJ_STRIDE_PLANE, attnMode, m);
        if (!H_Sync(C, "hc_only1")) {
            return false;
        }
        std::vector<uint8_t> ref1(M15H::WS_BYTES);
        H_D2H(C, ref1.data(), sc0, ref1.size(), "hc_only1_ws");
        if (!H_Sync(C, "hc_only1_d2h")) {
            return false;
        }
        snprintf(tag, sizeof(tag), "H.h1ws.L%02u", L);
        ok = H_CmpBytes(C, tag, fusedH1.data(), ref1.data(), fusedH1.size()) && ok;

        // ---- (3) 独立启动 hc 边界 #2：输入与融合路**同源** ----
        // 层内 handoff（hIn / ij）按融合 kernel 的同一算式取：
        //   hIn = (attnMode == MODE_MIX) ? 层输入 H : sc0 的 WS_HCP
        //   ij  = sc0 的 OH+320*2B（行距 OH_W）
        // 前者是**必要条件**：mode 0 的边界不写 H'（其 H' ≡ H），融合路这一步取的就是层输入。
        // bo = 子层段出口（同一个设备缓冲，未被修改 ⇒ 两侧同源）。
        aclrtMemset(sc1, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        H_LaunchHcOnly(C, L, sc1, 1u,
                       (attnMode == M15H::MODE_MIX) ? reinterpret_cast<uint8_t*>(C.hcHDev)
                                                    : (sc0 + M15H::WS_HCP),
                       reinterpret_cast<uint8_t*>(C.hcAttnOutDev), sc0 + M15H::WS_OH + M15H::OH_INJ * 2u,
                       M15H::OH_W, mlpMode, m);
        if (!H_Sync(C, "hc_only2")) {
            return false;
        }
        std::vector<uint8_t> ref2(M15H::WS_BYTES);
        H_D2H(C, ref2.data(), sc1, ref2.size(), "hc_only2_ws");
        if (!H_Sync(C, "hc_only2_d2h")) {
            return false;
        }
        snprintf(tag, sizeof(tag), "H.h2ws.L%02u", L);
        ok = H_CmpBytes(C, tag, fusedH2.data(), ref2.data(), fusedH2.size()) && ok;

        // ---- (4) combine-only 档（第 4 档 mode）----
        // 判据：① 它的 H' 与 combine_and_mix 档的 H' 逐字节相同（同一个 S1 公式、同一组输入）；
        //       ② 它**不写 BLK**（该档只跑 W0+S1，BLK 区保持预填的 0xCD）。
        aclrtMemset(sc2, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        H_LaunchHcOnly(C, L, sc2, 0u, reinterpret_cast<uint8_t*>(C.hcHDev),
                       reinterpret_cast<uint8_t*>(C.hcBoDev), reinterpret_cast<uint8_t*>(C.hcIjDev),
                       M15H::HC_IJ_STRIDE_PLANE, M15H::MODE_COMBINE_ONLY, m);
        if (!H_Sync(C, "hc_conly")) {
            return false;
        }
        std::vector<uint8_t> conly(M15H::WS_BYTES);
        H_D2H(C, conly.data(), sc2, conly.size(), "hc_conly_ws");
        if (!H_Sync(C, "hc_conly_d2h")) {
            return false;
        }
        // 参考 = combine_and_mix 档的同一组输入跑出来的 H'：
        //   · attn 边界本身就是 combine_and_mix ⇒ 直接取**融合路** H1 的 H'（fusedH1 是 host 副本）；
        //   · 层 0 的 attn 边界是 MODE_MIX（不写 H'）⇒ 为该层单独跑一次 MODE_COMBINE_MIX 到 sc1，
        //     再回读到 host（**注意：必须回读，不能拿设备指针做 memcmp**）。
        uint32_t refMode = attnMode;
        std::vector<uint8_t> refHcp(hyb);
        if (refMode == M15H::MODE_MIX) {
            aclrtMemset(sc1, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
            H_LaunchHcOnly(C, L, sc1, 0u, reinterpret_cast<uint8_t*>(C.hcHDev),
                           reinterpret_cast<uint8_t*>(C.hcBoDev), reinterpret_cast<uint8_t*>(C.hcIjDev),
                           M15H::HC_IJ_STRIDE_PLANE, M15H::MODE_COMBINE_MIX, m);
            if (!H_Sync(C, "hc_conly_ref")) {
                return false;
            }
            H_D2H(C, refHcp.data(), sc1 + M15H::WS_HCP, hyb, "hc_conly_ref_hcp");
            if (!H_Sync(C, "hc_conly_ref_d2h")) {
                return false;
            }
        } else {
            __builtin_memcpy(refHcp.data(), fusedH1.data() + M15H::WS_HCP, hyb);
        }
        // ---- 合并成**一条**判定项（review r1 的 P2-1）：这是同一个论断的两个面 ----
        //   ① combine-only 档的 `H'` 与 combine_and_mix 档**逐字节相同**（同一组输入下的 S1 公式）；
        //   ② 它**没有多跑阶段** —— `BLK` 区保持预填的 `0xCD`（该档只跑 W0+S1，AIC 不跑 GEMM）。
        // ② 是**判别性**的那一半：若 mode 3 没被兑现（进程走了完整链），`H'` 仍会与 ① 相同
        // （S1 一样），只有「BLK 是否被写出」能区分 ⇒ 两件事必须一起判、也必须**只计一次**。
        // 旧写法把 ② 写成 `H_Guard` 又套 `H_CmpOk`，同一件事既进 guards 又进 checks（计数虚高）。
        snprintf(tag, sizeof(tag), "H.conly.L%02u", L);
        {
            const auto* blk = conly.data() + M15H::WS_BLK;
            size_t hcpBad = 0;
            for (uint32_t b = 0; b < hyb; ++b) {
                if (conly[b + M15H::WS_HCP] != refHcp[b]) {
                    ++hcpBad;
                }
            }
            size_t blkBad = 0;
            for (uint32_t b = 0; b < hbytes; ++b) {
                if (blk[b] != 0xCDu) {
                    ++blkBad;
                }
            }
            C.checks += 1u;
            if (hcpBad == 0u && blkBad == 0u) {
                printf("[m15]   %-30s PASS (H' 与 combine_and_mix 档逐字节一致 n=%u；BLK 未被写出 "
                       "0xCD 全场保持)\n", tag, hyb);
            } else {
                printf("[m15]   %-30s FAIL (H' 差 %zu/%u 字节；BLK 被写 %zu/%u 字节)\n", tag, hcpBad, hyb,
                       blkBad, hbytes);
                C.fails++;
                ok = false;
            }
        }

        // ---- (5) 交付形态确定性：同一二进制重复一次，两个边界 ws 逐字节相同 ----
        // **必须把 GDN 状态复位到与第一次相同**：conv_state/ssm_state 是 in-place RMW，状态变了
        // 子层段出口就变，H2 的输入跟着变 —— 那不是「不确定性」，是输入本来就不同。
        aclrtMemset(hcw, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        aclrtMemset(C.hcAttnOutDev, static_cast<size_t>(M_MAX) * HIDDEN * 2u, 0xCD,
                    static_cast<size_t>(M_MAX) * HIDDEN * 2u);
        aclrtMemset(C.yFusedDev, M15Loop::H_ROWS_BYTES, 0xCD, M15Loop::H_ROWS_BYTES);
        if (M15Loop::KindOf(L) == M15Loop::KIND_GDN) {
            H_H2D(C, H_ST(C, L, 0), C.csInit.data(), M15Loop::CS_BYTES, "hc_cs_init_rep");
            H_H2D(C, H_ST(C, L, M15Loop::SSM_OFF), C.ssmInit.data(), M15Loop::SSM_BYTES, "hc_ssm_init_rep");
        }
        H_LaunchLayerHc(C, L, C.wsFusedDev, C.yFusedDev,
                        (M15Loop::KindOf(L) == M15Loop::KIND_GDN) ? H_ST(C, L, 0) : nullptr,
                        (M15Loop::KindOf(L) == M15Loop::KIND_GDN) ? H_ST(C, L, M15Loop::SSM_OFF) : nullptr,
                        C.hcHDev, C.hcBoDev, C.hcIjDev, M15H::HC_IJ_STRIDE_PLANE, 0u,
                        attnMode, mlpMode, m);
        if (!H_Sync(C, "hc_fused_rep")) {
            return false;
        }
        std::vector<uint8_t> rep1(M15H::WS_BYTES), rep2(M15H::WS_BYTES);
        H_D2H(C, rep1.data(), hcw + M15L::HC_WS0_IN_LAYER, rep1.size(), "hc_rep_h1ws");
        H_D2H(C, rep2.data(), hcw + M15L::HC_WS1_IN_LAYER, rep2.size(), "hc_rep_h2ws");
        if (!H_Sync(C, "hc_rep_d2h")) {
            return false;
        }
        snprintf(tag, sizeof(tag), "H.det_h1.L%02u", L);
        ok = H_CmpBytes(C, tag, fusedH1.data(), rep1.data(), fusedH1.size()) && ok;
        snprintf(tag, sizeof(tag), "H.det_h2.L%02u", L);
        ok = H_CmpBytes(C, tag, fusedH2.data(), rep2.data(), fusedH2.size()) && ok;

        // ---- dump（供 check_hc_ref.py 用 m20 的独立参考复算）----
        if (C.O.dump) {
            snprintf(tag, sizeof(tag), "hc_L%02u", L);
            H_Dump(C, std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_hin",
                   reinterpret_cast<const uint8_t*>(hin.data()), hyb, "BF16", "[m,10240]");
            H_Dump(C, std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_bo",
                   reinterpret_cast<const uint8_t*>(bo.data()), hbytes, "BF16", "[m,2560]");
            H_Dump(C, std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_ij",
                   reinterpret_cast<const uint8_t*>(ij.data()), m * M15H::IJ_STRIDE * 2u, "BF16", "[m,16]");
            H_Dump(C, std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_hout", hOut.data(),
                   hyb, "BF16", "[m,10240]");
            H_Dump(C, std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_y", yOut.data(), hbytes,
                   "BF16", "[m,2560]");
            H_Dump(C, std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_attnout", attnOut.data(),
                   hbytes, "BF16", "[m,2560]");
            H_Dump(C, std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_conly_hcp",
                   conly.data() + M15H::WS_HCP, hyb, "BF16", "[m,10240]");
            {
                const HcWPtrs hp = H_HcWPtrsOf(C.hcWDev, L);
                const char* pfx[2] = {"attn", "mlp"};
                uint8_t* wp[2][4] = {{hp.attnNorm, hp.attnDown, hp.attnInj, hp.attnUp},
                                     {hp.mlpNorm, hp.mlpDown, hp.mlpInj, hp.mlpUp}};
                const char* sfx[4] = {"norm", "wdown", "winj", "wup"};
                const uint32_t sz[4] = {M15L::HW_NORM_BYTES, M15L::HW_WDOWN_BYTES, M15L::HW_WINJ_BYTES,
                                        M15L::HW_WUP_BYTES};
                const char* shp[4] = {"[10240]", "[320,10240]", "[16,10240]", "[10240,320]"};
                for (uint32_t b = 0; b < 2u; ++b) {
                    for (uint32_t t = 0; t < 4u; ++t) {
                        std::string nm = std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_" +
                                         pfx[b] + "_" + sfx[t];
                        // 权重在设备上：先回读到 host 再落盘（每层 ~13.5MB，仅 M15_DUMP 档）
                        std::vector<uint8_t> tmp(sz[t]);
                        H_D2H(C, tmp.data(), wp[b][t], sz[t], nm.c_str());
                        H_Dump(C, nm, tmp.data(), sz[t], "BF16", shp[t]);
                    }
                }
                if (!H_Sync(C, "hc_dump_w")) {
                    return false;
                }
            }
            // 两个边界的 8 个中间张量（各按行距切片；dump 时用**落盘的 ws 副本**，不走设备）
            H_DumpHcBoundary(C, (std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_b1").c_str(),
                             fusedH1.data(), m);
            H_DumpHcBoundary(C, (std::string("hc_L") + (L < 10 ? "0" : "") + std::to_string(L) + "_b2").c_str(),
                             fusedH2.data(), m);
        }
        printf("[m15]   层 %02u（%s，attn mode=%u / mlp mode=%u）：H.fused/H.h1ws/H.h2ws/H.conly/H.det 全部比对完成\n", L,
               (M15Loop::KindOf(L) == M15Loop::KIND_GDN) ? "GDN" : "ATTN", attnMode, mlpMode);
    }
    C.phaseMh = C.checks - checkBase;   // 判定项计数与 checks 同源（H_CmpBytes/H_CmpFinite 等各计 1 条）
    printf("[m15] 验证 H：%u 层（层表 M15_HC_LAYERS=%s，m=%u），本段判定项累计 %u，累计 FAIL %u\n", nL,
           H_ListStr(C.O.hcLayers).c_str(), m, C.phaseMh, C.fails);
    return C.fails == 0;
}

// hc 段的布局表（供 check_hc_ref.py 按同一套偏移切片）
static void H_DumpHcLayout(const char* path)
{
    FILE* f = fopen(path, "wb");
    if (f == nullptr) {
        return;
    }
    fprintf(f, "# m15 hc 段 ws 布局（由 m15_hc_resources.h 的常量算出；与 kernel 同一套偏移）\n");
    fprintf(f, "hyper=%u hid=%u lowrank=%u hc=%u inj_n=%u inj_stride=%u injw_slot=%u oh_w=%u oh_inj=%u "
               "up_n=%u up_k=%u m_max=%u\n",
            M15H::HYPER, M15H::HID, M15H::LOWRANK, M15H::HC, M15H::INJ_N, M15H::IJ_STRIDE, M15H::INJW_SLOT,
            M15H::OH_W, M15H::OH_INJ, M15H::UP_N, M15H::UP_K, M15H::M_MAX);
    const struct { const char* n; uint32_t off; } t[] = {
        {"hcp", M15H::WS_HCP}, {"xn", M15H::WS_XN},   {"rstd", M15H::WS_RSTD}, {"injw", M15H::WS_INJW},
        {"oh", M15H::WS_OH},   {"ls", M15H::WS_LS},   {"gate", M15H::WS_GATE}, {"blk", M15H::WS_BLK},
    };
    for (const auto& e : t) {
        fprintf(f, "off_%s=%u\n", e.n, e.off);
    }
    fprintf(f, "ws_bytes=%u\n", M15H::WS_BYTES);
    fprintf(f, "layer_w_stride=%u attn_w_off=%u mlp_w_off=%u\n", M15L::HC_LAYER_W_STRIDE, M15L::HC_ATTN_W_OFF,
            M15L::HC_MLP_W_OFF);
    fclose(f);
}
