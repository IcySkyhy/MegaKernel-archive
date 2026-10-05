// ============================================================
// m15_chain_host.h —— M65 host 侧：**48 层链切到四相位入口** + PLE 打断点 + 末层全局 mixer
//
// 本文件只在 m15_layer_loop.asc（单 TU）里被 include，位置在 m15_hc_host.h 之后（复用它的
// H_LaunchLayerHc / H_LaunchHcOnly / H_HcWPtrsOf / H_GenHcW / H_PutHcSlot 与 m15_moe_host.h 的
// H_LaunchGdnOnly / H_LaunchMoeOnly / H_CmpBytes / H_CmpMustDiffer / H_CmpFiniteBf16）。
//
// 它做四件事（对应 mission 的 task 1/2/3/4）：
//
//  1. **层链**：每层一次 `m15_layer_kernel_{gdn,attn}_hc` 启动（四相位），层间只搬三条长寿状态
//     （`H [m,10240]` bf16 多流残差、pending `BO [m,2560]` bf16 上一层 mlp_out、
//     `IJ [m,4]` bf16 上一层 injection logits），**全部零拷贝**：
//       layer L 的 hcH  = `hcWs[L-1] + HC_WS1_IN_LAYER + WS_HCP`（= L-1 层的 H''）
//       layer L 的 hcIj = `hcWs[L-1] + HC_WS1_IN_LAYER + WS_OH + OH_INJ*2`，行距 `OH_W`
//       layer L 的 hcBo = `chainBo[(L-1) % 2]`（双缓冲）
//     层 0 的 hcH = 入口平面（`embed_tokens` 后的 4 路复制），层 0 的 attn 边界是 `MODE_MIX`。
//     **pending BO 为什么必须独立平面**：它由下一层的 hc(attn) combine 直接消费，而 GDN/MoE 的
//     单块 ws 每层复用 ⇒ 落不下（见 m15_layer_resources.h §3 的同一论证）。
//
//  2. **PLE 打断点**（层 1）：附加 `pleBreak=1` ⇒ kernel 内把 H1 拆成 combine-only（物化 bf16
//     `H'`）→ PLE 挂载点（**未实现**）→ mix-only。语义 = `V-N:model.py:290-301`。
//
//  3. **末层全局 mixer**：层 47 之后一次 `m15_final_mixer_kernel`（`MODE_FINAL_MIX`），产出
//     `multi_hidden`（ws 的 `WS_HCP`）与 `sample_hidden`（ws 的 `WS_BLK`）。
//
//  4. **判据 ①②③⑤**（逐条见 `H_RunChain` 的分组注释）+ 为 ④（与 M39 官方单层参考对拍）落盘。
//     计数由调用点自增 `C.checks`/`C.guards`，本段差分记入 `C.phaseCh`/`C.guardsCh`。
//
// 覆盖范围交代（统计工具必须交代自己检查了什么、什么会被漏掉）：
//   * 覆盖 = `M15_LAYERS` 层（默认 48）× {非恒等、MoE 非恒等、两边界互异、有限、指纹互异、
//     确定性、手工四段对拍}；层 1 另有 PLE 打断点的 3 条；末层 mixer 5 条；全局 mixer 权源 1 条。
//   * **不覆盖**：① 子层段（GDN/attention）与 MoE 段的数值正确性 —— M25/M40 的 `check_ref.py` /
//     `check_moe_ref.py`；② hc 段的数值正确性 —— Ver H + `check_hc_ref.py`（m20 参考）与
//     `check_chain_ref.py`（**M39 官方单层参考**，torch）；③ PLE 本体 —— **未实现**（显式披露）；
//     ④ attention 段的真实 QSA 语义 —— kernel 内仍是占位直通（M25 起的已知项）。
// ============================================================

// ============================================================
// 0. 层链的常量与地址（全部由 m15_layer_resources.h 的登记常量算出，不手写偏移）
// ============================================================

// 紧凑 OH 布局的元素数（lora 段 + injection 段），读回 / dump 与 python 判据共用
constexpr uint32_t CH_OH_PACK_ELEMS = M15H::LOWRANK + M15H::INJ_N;   // 324

// pending BO 的双缓冲平面数：层 L 的 mlp_out 只被层 L+1 读，且层间是同一 stream 上的严格串行，
// 故 2 块足够（对应 docs/14 §9.2 第 3 条「3 张量 × 双缓冲」的规划）。
constexpr uint32_t CH_BO_PLANES = 2;

static_assert(M15L::HC_GMIX_LAYER_SLOT == M15Loop::NL,
              "全局 mixer 的权重槽号必须正好排在 48 个层槽之后（M15Loop::NL == 48）");
static_assert(M15L::HC_WS_GMIX_SLOT == M15Loop::NL + 3,
              "全局 mixer 的 ws 槽必须紧接 M58 的三个验证 H scratch 槽（48,49,50 ⇒ 51）");
static_assert(M15L::HC_WS_SLOTS == M15Loop::NL + 4, "hc ws 槽数 = 48 层 + 3 scratch + 1 全局 mixer");

static uint8_t* H_ChWs(Ctx& C, uint32_t slot)
{
    return reinterpret_cast<uint8_t*>(C.hcWsDev) + static_cast<size_t>(slot) * M15L::HC_WS_LAYER_STRIDE;
}

// 层 L 的两个 hc 边界的 ws
static uint8_t* H_ChWs0(Ctx& C, uint32_t L) { return H_ChWs(C, L) + M15L::HC_WS0_IN_LAYER; }
static uint8_t* H_ChWs1(Ctx& C, uint32_t L) { return H_ChWs(C, L) + M15L::HC_WS1_IN_LAYER; }

// 层 L 的**三态出口**在 GM 上的位置（下一层的三条输入直接指这里 ⇒ 零拷贝 handoff）
static uint8_t* H_ChHOut(Ctx& C, uint32_t L) { return H_ChWs1(C, L) + M15H::WS_HCP; }
static uint8_t* H_ChIjOut(Ctx& C, uint32_t L) { return H_ChWs1(C, L) + M15H::WS_OH + M15H::OH_INJ * 2u; }
static uint8_t* H_ChBoPlane(Ctx& C, uint32_t idx)
{
    return reinterpret_cast<uint8_t*>(C.chainBoDev) + static_cast<size_t>(idx % CH_BO_PLANES) * M15Loop::H_ROWS_BYTES;
}

// ---- 接线正确性的**结构性** guard（跑之前先断言一遍；把「接线错」与「数值错」分开）----
static bool H_ChWiredCheck(Ctx& C)
{
    bool ok = true;
    for (uint32_t a = 0; a < M15Loop::NL && ok; ++a) {          // (i) 48 层的 H'' 平面两两不同
        for (uint32_t b = a + 1; b < M15Loop::NL; ++b) {
            if (H_ChHOut(C, a) == H_ChHOut(C, b)) {
                ok = false;
                break;
            }
        }
    }
    for (uint32_t L = 0; L < M15Loop::NL; ++L) {                // (ii) H'' 不与自己的 H1 输入别名
        if (H_ChHOut(C, L) == H_ChWs0(C, L) + M15H::WS_HCP) {
            ok = false;
        }
        if (H_ChIjOut(C, L) == H_ChWs1(C, L) + M15H::WS_HCP) {  // (iii) IJ'' 不与 H'' 重叠
            ok = false;
        }
    }
    if (H_ChBoPlane(C, 0) == H_ChBoPlane(C, 1)) {               // (iv) 双缓冲两个平面不相交
        ok = false;
    }
    if ((M15H::OH_INJ * 2u + M15H::INJ_N * 2u) > M15H::OH_W * 2u ||
        (M15H::OH_INJ * 2u) % 32u != 0u) {                      // (v) IJ 源按行直读的前提
        ok = false;
    }
    return H_Guard(C, ok);
}

// ============================================================
// 1. 末层全局 mixer 的权重（host 侧）
// ============================================================
// 3 个张量（checkpoint 的 `model.language_model.hyper_connection_mixer.*`），**没有**
// `block_inject_weight`（`V-N:model.py:612` 显式丢弃）⇒ 注入区 16 行全零。
// 角色名与 manifest 的 `layer=<NL> role=gmixer_*` 行一一对应。
static bool H_LoadGmW(const M15Manifest& M, M15HcW& W, std::string& err)
{
    char role[64];
    auto rd = [&](const char* suffix, std::vector<uint8_t>& dst, size_t want) -> bool {
        snprintf(role, sizeof(role), "gmixer_%s", suffix);
        const M15Tensor* T = H_FindTensor(M, M15L::HC_GMIX_LAYER_SLOT, role);
        if (T == nullptr) {
            err = std::string("manifest 缺 role=") + role + " layer=" + std::to_string(M15L::HC_GMIX_LAYER_SLOT);
            return false;
        }
        if (T->bytes != want) {
            char b[256];
            snprintf(b, sizeof(b), "role=%s 字节数不符：manifest %llu，期望 %zu", role,
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
    __builtin_memset(W.inj.data(), 0, W.inj.size());
    W.layer = M15L::HC_GMIX_LAYER_SLOT;
    W.real = true;
    return true;
}

static void H_GenGmW(M15HcW& W)
{
    H_GenHcW(W, M15L::HC_GMIX_LAYER_SLOT, 0u);
    __builtin_memset(W.inj.data(), 0, W.inj.size());
}

// 把一个边界的 4 件权重装进**单槽** arena（`H_PutHcSlot` 是按层号在**整个**权重区里编址的，
// 直接拿它往单槽上写会越界 —— 判据里的 arena 是单槽，故这里另写一个同布局的 put）。
static void H_PutGmSlot(std::vector<uint8_t>& arena, const M15HcW& W)
{
    auto put = [&arena](uint32_t off, const void* src, size_t bytes) {
        __builtin_memcpy(arena.data() + off, src, bytes);
    };
    put(M15L::HW_NORM_OFF, W.norm.data(), W.norm.size());
    put(M15L::HW_WDOWN_OFF, W.down.data(), W.down.size());
    put(M15L::HW_WINJ_OFF, W.inj.data(), W.inj.size());
    put(M15L::HW_WUP_OFF, W.up.data(), W.up.size());
}

// 全局 mixer 的权重来源判据（独立 pread + 逐字节 memcmp，与 H_HcWeightSourceCheck 同款；
// 额外见证「注入区 16 行全零」这条 host 契约）
static bool H_GmWeightSourceCheck(Ctx& C)
{
    std::string err;
    M15HcW W;
    H_HcWAlloc(W);
    if (C.real) {
        if (!H_LoadGmW(C.M, W, err)) {
            printf("[m15][FAIL] 全局 mixer 权重独立 pread 失败：%s\n", err.c_str());
            return false;
        }
    } else {
        H_GenGmW(W);
    }
    std::vector<uint8_t> exp(M15L::HC_LAYER_W_STRIDE, 0);
    H_PutGmSlot(exp, W);
    std::vector<uint8_t> dev(M15L::HC_LAYER_W_STRIDE);
    H_D2H(C, dev.data(), reinterpret_cast<uint8_t*>(C.hcWDev) +
                             static_cast<size_t>(M15L::HC_GMIX_LAYER_SLOT) * M15L::HC_LAYER_W_STRIDE,
          dev.size(), "gm_w_source_backup");
    if (!H_Sync(C, "gm_w_source_backup")) {
        return false;
    }
    bool ok = H_CmpBytes(C, "Ch.wsrc.gm", dev.data(), exp.data(), dev.size());

    bool zeroOk = true;
    const uint32_t rowBytes = M15H::HYPER * 2u;
    const uint8_t* inj = dev.data() + M15L::HC_ATTN_W_OFF + M15L::HW_WINJ_OFF;
    for (uint32_t r = 0; r < M15L::HC_WINJ_ROWS && zeroOk; ++r) {
        for (uint32_t b = 0; b < rowBytes; ++b) {
            if (inj[r * rowBytes + b] != 0u) {
                zeroOk = false;
                break;
            }
        }
    }
    const bool zok = H_Guard(C, zeroOk);
    C.guardsCh += 1u;
    printf("[m15]   全局 mixer 权重来源：3 个张量（norm/down/up）+ 注入区 16 行全零见证，"
           "逐字节 memcmp（%u 字节）\n",
           M15L::HC_LAYER_W_STRIDE);
    return ok && zok;
}

// ============================================================
// 2. 启动器与读回
// ============================================================

// 链上的一层：一次四相位启动（PLE 层由 pleBreak 走五段）
// M136（Wave D）：`m` 改为**真传参**（原为写死 `1u`）—— 链的名义 m 一直是 `C.O.chainM`
// （上界 `M15H::M_MAX`），但启动器把它丢弃、改传 1u ⇒ 这里修回，调用方（H_ChRunOnce）传链的 m。
static void H_LaunchChainLayer(Ctx& C, uint32_t L, void* hIn, void* bo, void* ij, uint32_t ijStride,
                               void* yLayer, uint32_t attnMode, uint32_t mlpMode, uint32_t pleBreak, uint32_t m)
{
    const bool isGdn = (M15Loop::KindOf(L) == M15Loop::KIND_GDN);
    H_LaunchLayerHc(C, L, C.wsFusedDev, yLayer, isGdn ? H_ST(C, L, 0) : nullptr,
                    isGdn ? H_ST(C, L, M15Loop::SSM_OFF) : nullptr, hIn, bo, ij, ijStride, pleBreak,
                    attnMode, mlpMode, m);
}

static void H_LaunchFinalMix(Ctx& C, void* hIn, void* bo, void* ij, uint32_t ijStride, uint32_t m)
{
    const HcWPtrs gp = H_HcWPtrsOf(C.hcWDev, M15L::HC_GMIX_LAYER_SLOT);
    M15L::m15_final_mixer_kernel<<<C.numBlocks, 0, C.stream>>>(
        reinterpret_cast<uint8_t*>(hIn), reinterpret_cast<uint8_t*>(bo), reinterpret_cast<uint8_t*>(ij),
        ijStride, gp.attnDown, gp.attnInj, gp.attnUp, gp.attnNorm, H_ChWs(C, M15L::HC_WS_GMIX_SLOT), m);
}

// 手工路径用的「只跑一层的一个 hc 边界」（复用 M58 的对照入口）
static void H_LaunchHcSeg(Ctx& C, uint32_t L, uint32_t which, void* scratchWs, void* hIn, void* bo, void* ij,
                          uint32_t ijStride, uint32_t mode, uint32_t m)
{
    H_LaunchHcOnly(C, L, scratchWs, which, reinterpret_cast<uint8_t*>(hIn), reinterpret_cast<uint8_t*>(bo),
                   reinterpret_cast<uint8_t*>(ij), ijStride, mode, m);
}

// 一个 hc 边界 ws 里 m 行的 8 个张量（`H_DumpHcBoundary` 的读侧孪生；偏移同源）
struct ChWsRow {
    std::vector<uint8_t> hcp, xn, rstd, injw, oh, ls, gate, blk;
};

static void H_ChReadWs(Ctx& C, const char* tag, const uint8_t* wsBase, uint32_t m, ChWsRow& out)
{
    auto rd = [&](std::vector<uint8_t>& dst, uint32_t off, uint32_t bytes) {
        dst.resize(bytes);
        H_D2H(C, dst.data(), wsBase + off, bytes, tag);
    };
    rd(out.hcp, M15H::WS_HCP, m * M15H::HYPER * 2u);
    rd(out.xn, M15H::WS_XN, m * M15H::HYPER * 2u);
    rd(out.rstd, M15H::WS_RSTD, m * M15H::HC * 4u);
    rd(out.injw, M15H::WS_INJW, m * M15H::HC * M15H::INJW_SLOT * 4u);
    rd(out.ls, M15H::WS_LS, m * M15H::LOWRANK * 2u);
    rd(out.gate, M15H::WS_GATE, m * M15H::UP_N * 2u);
    rd(out.blk, M15H::WS_BLK, m * M15H::HID * 2u);
    std::vector<uint8_t> ohRaw(m * M15H::OH_W * 2u);
    H_D2H(C, ohRaw.data(), wsBase + M15H::WS_OH, ohRaw.size(), tag);
    // 紧凑布局 [m, LOWRANK + INJ_N]：lora 段（列 [0,LOWRANK)）+ injection 段（列 [OH_INJ, OH_INJ+INJ_N)）
    out.oh.resize(m * CH_OH_PACK_ELEMS * 2u);
    for (uint32_t r = 0; r < m; ++r) {
        const uint8_t* src = ohRaw.data() + static_cast<size_t>(r) * M15H::OH_W * 2u;
        uint8_t* dst = out.oh.data() + static_cast<size_t>(r) * CH_OH_PACK_ELEMS * 2u;
        __builtin_memcpy(dst, src, M15H::LOWRANK * 2u);
        __builtin_memcpy(dst + M15H::LOWRANK * 2u, src + M15H::OH_INJ * 2u, M15H::INJ_N * 2u);
    }
    if (!H_Sync(C, tag)) {
        C.fails++;
    }
}

// 从紧凑 OH（`[m, LOWRANK + INJ_N]`）里抽出 injection 段（`[m, INJ_N]`）
static void H_ChExtractInj(const std::vector<uint8_t>& packedOh, uint32_t m, std::vector<uint8_t>& out)
{
    out.resize(m * M15H::INJ_N * 2u);
    for (uint32_t r = 0; r < m; ++r) {
        __builtin_memcpy(out.data() + static_cast<size_t>(r) * M15H::INJ_N * 2u,
                         packedOh.data() + (static_cast<size_t>(r) * CH_OH_PACK_ELEMS + M15H::LOWRANK) * 2u,
                         M15H::INJ_N * 2u);
    }
}

static void H_ChReadIj(Ctx& C, const char* tag, const uint8_t* devIj, uint32_t strideElems, uint32_t m,
                       std::vector<uint8_t>& out)
{
    out.resize(m * M15H::INJ_N * 2u);
    if (strideElems == M15H::INJ_N) {
        H_D2H(C, out.data(), devIj, out.size(), tag);
        H_Sync(C, tag);
        return;
    }
    std::vector<uint8_t> raw(m * strideElems * 2u);
    H_D2H(C, raw.data(), devIj, raw.size(), tag);
    for (uint32_t r = 0; r < m; ++r) {
        __builtin_memcpy(out.data() + static_cast<size_t>(r) * M15H::INJ_N * 2u,
                         raw.data() + static_cast<size_t>(r) * strideElems * 2u, M15H::INJ_N * 2u);
    }
    H_Sync(C, tag);
}

// ============================================================
// 3. 验证 Ch：48 层链（四相位入口）+ PLE 打断点 + 末层全局 mixer
// ============================================================
//
// 判据分组（`Ch.` 前缀，与 `C.phaseCh` 同源）：
//   ① 非空洞性：`Ch.nonid.L*`（H'' ≠ 层输入 H）、`Ch.moe.L*`（mlp_out ≠ BLK_mlp）、`Ch.finite.L*`、
//      `Ch.gm.multi` / `Ch.gm.sample` / `Ch.gm.sample_nonconst`、`Ch.pert.last` / `Ch.pert.gm`
//      （入口 H ×1.5 ⇒ 末层三态与 mixer 输出必须变）+ guard `Ch.pert.in`
//   ② 跨层可区分：`Ch.blks.L*`（同层两边界 BLK 互异）、`Ch.fp.L*`（48 层指纹两两互异）
//   ③ 确定性：`Ch.detH/BO/IJ.L*`（同一二进制重跑链，三条长寿状态逐字节相同）
//   ④ 与官方参考对拍：`check_chain_ref.py`（M39 的 torch 官方单层参考）在本段落的 dump 上做；
//      本段只落盘 + 打印覆盖交代（**与 ①②③ 的计数分开**：那段有自己的计数与退出码）
//   ⑤ 与两相位形态回归：`Ch.h1*.L*` / `Ch.sub.L*` / `Ch.h2*.L*` / `Ch.bo.L*` =
//      「四段独立启动、host 手工串」与「一次启动的四相位链」逐字节一致（同输入同权重）；
//      另有 `runs=all` 的既有读数（A/B/C/M1/M2/H = 216/506/3/348/8/21）作为形态回归，不得倒退
//
// **PLE 打断点的舍入点判据**（task 2 的核心）：`Ch.ple.bf16.L01` —— 链上层 1 的 H1 规范化输出
// `XN` 必须与**单次 combine_and_mix**（`m15_hc_segment_kernel(mode=MODE_COMBINE_MIX)`、同输入
// 同权重）**逐字节相同**，且 `Ch.ple.single.L01` 再见证 BLK 也逐字节相同。
//   为什么能判「combine → RMSNorm 之间的 bf16 舍入点」：
//     · 单次 combine_and_mix 的 `CombineStage` 把 `H'` 以 bf16 写进 GM（`WS_HCP`），`NormStage`
//       再从 GM 读回 —— 就是 docs/14 §3.2 的 `V-N:ops/hc.py:325-327` 那次早舍入；
//     · 分段路径（combine-only → 同一块 GM 的 bf16 `H'` → mix-only）读的是同一批字节；
//     · 若实现把 combine 的 fp32 结果**不物化**直接喂给规范化，XN 就会与这条参考不同 ——
//       差多少由 `check_chain_ref.py` 的负向对照在**真实层 1 数据**上当场量（M39 B4 同族数据
//       实测 14029/40960 = 34.25% 元素受影响）。
//   档位理由：**逐字节**（不是 ulp 门限）。两侧是同一条指令序列作用在同一批 bf16 字节上，
//   唯一可能的分歧就是「有没有那次 bf16 物化」⇒ 不存在需要用误差吸收的中间态（docs/17 §1.1：
//   良态元素应当逐位一致）。
struct ChLayerRec {
    std::vector<uint8_t> hIn, boIn, ijIn;      // 该层的三条输入（host 副本）
    std::vector<uint8_t> hOut, boOut, ijOut;   // 该层的三条出口
    std::vector<uint8_t> blkA, blkM, attnOut;  // ws0 的 BLK / ws1 的 BLK / 子层段出口
    ChWsRow b1, b2;                            // 两个边界的 8 个中间张量（b1.xn 是舍入点判据的对象）
    std::vector<uint8_t> fp;                   // 跨层指纹（② 用）= hOut|boOut|ijOut 的**原始字节**（不做哈希）
};

// 跑一遍 48 层链 + 末层 mixer。full=true 时额外读回两个边界的 ws 与子层段出口（手工路径要用）。
static bool H_ChRunOnce(Ctx& C, const std::vector<uint8_t>& hin, uint32_t m, bool full,
                        std::vector<ChLayerRec>& rec, std::vector<uint8_t>& multiHidden,
                        std::vector<uint8_t>& sampleHidden)
{
    using namespace M15G;
    const uint32_t nL = C.O.nLayers;
    const uint32_t hyb = m * M15H::HYPER * 2u;
    const uint32_t hbytes = m * HIDDEN * 2u;
    const uint32_t ijbytes = m * M15H::INJ_N * 2u;

    H_ResetStates(C);
    H_H2D(C, C.hcHDev, hin.data(), hin.size(), "ch_hin");
    for (uint32_t p = 0; p < CH_BO_PLANES; ++p) {
        aclrtMemset(H_ChBoPlane(C, p), M15Loop::H_ROWS_BYTES, 0, M15Loop::H_ROWS_BYTES);
    }
    rec.assign(nL, ChLayerRec());
    std::vector<uint8_t> curH(hin.begin(), hin.begin() + hyb), curBo(hbytes, 0), curIj(ijbytes, 0);
    for (uint32_t L = 0; L < nL; ++L) {
        ChLayerRec& R = rec[L];
        R.hIn = curH;
        R.boIn = curBo;
        R.ijIn = curIj;
        void* hInDev = (L == 0u) ? static_cast<void*>(C.hcHDev) : static_cast<void*>(H_ChHOut(C, L - 1u));
        void* ijDev = (L == 0u) ? static_cast<void*>(C.hcIjDev) : static_cast<void*>(H_ChIjOut(C, L - 1u));
        void* boDev = (L == 0u) ? static_cast<void*>(H_ChBoPlane(C, 1u))
                                : static_cast<void*>(H_ChBoPlane(C, (L - 1u) % CH_BO_PLANES));
        void* yDev = H_ChBoPlane(C, L % CH_BO_PLANES);
        const uint32_t ijStride = (L == 0u) ? M15H::HC_IJ_STRIDE_PLANE : M15H::OH_W;
        const uint32_t attnMode = (L == 0u) ? M15H::MODE_MIX : M15H::MODE_COMBINE_MIX;
        // `M15_CHAIN_PLE=0` 把打断点关掉（层 1 走普通的单次 combine_and_mix）——**只用于
        // msprof 的 A/B**（见 README 的段级表）：因为分段路径与单次路径**按设计逐字节等价**，
        // 所以「打断点是否真的接上」只能靠耗时差 + 代码结构见证，不能靠输出（见 README 的口径说明）。
        const uint32_t pleBreak = (L == 1u) ? C.O.chainPle : 0u;
        H_LaunchChainLayer(C, L, hInDev, boDev, ijDev, ijStride, yDev, attnMode, M15H::MODE_COMBINE_MIX,
                           pleBreak, m);
        if (!H_Sync(C, "ch_layer")) {
            return false;
        }
        R.hOut.resize(hyb);
        H_D2H(C, R.hOut.data(), H_ChHOut(C, L), hyb, "ch_hout");
        R.boOut.resize(hbytes);
        H_D2H(C, R.boOut.data(), yDev, hbytes, "ch_bo");
        H_ChReadIj(C, "ch_ijout", H_ChIjOut(C, L), M15H::OH_W, m, R.ijOut);
        if (full) {
            R.attnOut.resize(hbytes);
            H_D2H(C, R.attnOut.data(), C.hcAttnOutDev, hbytes, "ch_attnout");
            H_ChReadWs(C, "ch_ws1", H_ChWs1(C, L), m, R.b2);
            H_ChReadWs(C, "ch_ws0", H_ChWs0(C, L), m, R.b1);
            R.blkM = R.b2.blk;
            R.blkA = R.b1.blk;
        }
        if (!H_Sync(C, "ch_layer_d2h")) {
            return false;
        }
        curH.assign(R.hOut.begin(), R.hOut.end());
        curH.resize(hin.size(), 0);
        curBo.assign(R.boOut.begin(), R.boOut.end());
        curBo.resize(M15H::M_MAX * HIDDEN * 2u, 0);
        curIj = R.ijOut;
    }

    // 末层全局 mixer（MODE_FINAL_MIX ⇒ multi_hidden / sample_hidden）
    uint8_t* gmWs = H_ChWs(C, M15L::HC_WS_GMIX_SLOT);
    aclrtMemset(gmWs, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
    H_LaunchFinalMix(C, H_ChHOut(C, nL - 1u), H_ChBoPlane(C, (nL - 1u) % CH_BO_PLANES), H_ChIjOut(C, nL - 1u),
                     M15H::OH_W, m);
    if (!H_Sync(C, "ch_gm")) {
        return false;
    }
    multiHidden.resize(hyb);
    sampleHidden.resize(hbytes);
    H_D2H(C, multiHidden.data(), gmWs + M15H::WS_HCP, hyb, "ch_gm_multi");
    H_D2H(C, sampleHidden.data(), gmWs + M15H::WS_BLK, hbytes, "ch_gm_sample");
    if (!H_Sync(C, "ch_gm_d2h")) {
        return false;
    }
    return true;
}

// 一条统一格式的「逐字节相同」判据（与 H_CmpBytes 同源计数，只是便于批量调用）
static bool H_ChSame(Ctx& C, const char* tag, const std::vector<uint8_t>& got,
                     const std::vector<uint8_t>& exp)
{
    if (got.size() != exp.size()) {
        printf("[m15]   %-30s FAIL (字节数 %zu != %zu)\n", tag, got.size(), exp.size());
        C.checks++;
        C.fails++;
        return false;
    }
    return H_CmpBytes(C, tag, got.data(), exp.data(), got.size());
}

// 单层的手工路径：四段独立启动（hc#1 → 子层段 → hc#2 → MoE），与链上的该层逐字节对拍
static bool H_ChManualLayer(Ctx& C, uint32_t L, const ChLayerRec& R, uint32_t m)
{
    using namespace M15G;
    const bool isGdn = (M15Loop::KindOf(L) == M15Loop::KIND_GDN);
    const uint32_t hyb = m * M15H::HYPER * 2u;
    const uint32_t hbytes = m * HIDDEN * 2u;
    uint8_t* sc0 = H_ChWs(C, M15Loop::NL + 0u);
    uint8_t* sc1 = H_ChWs(C, M15Loop::NL + 1u);
    uint8_t* sc2 = H_ChWs(C, M15Loop::NL + 2u);
    uint8_t* moeWs = reinterpret_cast<uint8_t*>(C.wsFusedDev) + M15L::WS_MOE_OFF;
    char tag[96];
    bool ok = true;

    // 该层状态复位 + 三条输入 H2D（手工路径必须自己造出链上那一刻的输入）
    if (isGdn) {
        H_H2D(C, H_ST(C, L, 0), C.csInit.data(), M15Loop::CS_BYTES, "ch_m_cs");
        H_H2D(C, H_ST(C, L, M15Loop::SSM_OFF), C.ssmInit.data(), M15Loop::SSM_BYTES, "ch_m_ssm");
    }
    H_H2D(C, C.hcHDev, R.hIn.data(), hyb, "ch_m_hin");
    H_H2D(C, H_ChBoPlane(C, 0u), R.boIn.data(), hbytes, "ch_m_bo");
    H_H2D(C, C.hcIjDev, R.ijIn.data(), R.ijIn.size(), "ch_m_ij");

    // ---- (a) hc 边界 #1 ----
    ChWsRow man1;
    std::vector<uint8_t> subIn(hbytes);   // 子层段输入（手工路径的 BLK_attn）
    if (L == 0u) {
        aclrtMemset(sc0, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        H_LaunchHcSeg(C, L, 0u, sc0, C.hcHDev, H_ChBoPlane(C, 0u), C.hcIjDev, M15H::HC_IJ_STRIDE_PLANE,
                      M15H::MODE_MIX, m);
        if (!H_Sync(C, "ch_m1")) { return false; }
        H_ChReadWs(C, "ch_m1", sc0, m, man1);
        snprintf(tag, sizeof(tag), "Ch.h1blk.L%02u", L);
        ok = H_ChSame(C, tag, R.blkA, man1.blk) && ok;
        snprintf(tag, sizeof(tag), "Ch.h1inj.L%02u", L);
        ok = H_ChSame(C, tag, R.b1.oh, man1.oh) && ok;
    } else if (L == 1u) {
        // PLE 打断点：combine-only（物化 bf16 H'）→ mix-only；另跑一次**单次 combine_and_mix** 参考
        aclrtMemset(sc2, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        H_LaunchHcSeg(C, L, 0u, sc2, C.hcHDev, H_ChBoPlane(C, 0u), C.hcIjDev, M15H::HC_IJ_STRIDE_PLANE,
                      M15H::MODE_COMBINE_ONLY, m);
        if (!H_Sync(C, "ch_ple_conly")) { return false; }
        std::vector<uint8_t> conlyHcp(hyb);
        H_D2H(C, conlyHcp.data(), sc2 + M15H::WS_HCP, hyb, "ch_ple_conly_hcp");
        if (!H_Sync(C, "ch_ple_conly_d2h")) { return false; }
        snprintf(tag, sizeof(tag), "Ch.h1hcp.L%02u", L);
        ok = H_ChSame(C, tag, R.b1.hcp, conlyHcp) && ok;
        // 先把原输入灌回去（单次 combine_and_mix 参考要与链同输入）
        H_H2D(C, C.hcHDev, R.hIn.data(), hyb, "ch_m_hin2");
        aclrtMemset(sc1, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        H_LaunchHcSeg(C, L, 0u, sc1, C.hcHDev, H_ChBoPlane(C, 0u), C.hcIjDev, M15H::HC_IJ_STRIDE_PLANE,
                      M15H::MODE_COMBINE_MIX, m);
        if (!H_Sync(C, "ch_ple_single")) { return false; }
        ChWsRow single;
        H_ChReadWs(C, "ch_ple_single", sc1, m, single);
        // **舍入点判据**（逐字节）：分段路径的 XN 必须等于单次 combine_and_mix 的 XN
        snprintf(tag, sizeof(tag), "Ch.ple.bf16.L%02u", L);
        ok = H_ChSame(C, tag, R.b1.xn, single.xn) && ok;
        snprintf(tag, sizeof(tag), "Ch.ple.single.L%02u", L);
        ok = H_ChSame(C, tag, R.blkA, single.blk) && ok;
        // mix-only（读 combine 物化好的 H'）——手工路径的子层段输入由此得到
        {
            std::vector<uint8_t> hcpDev(M15H::M_MAX * M15H::HYPER * 2u, 0);
            __builtin_memcpy(hcpDev.data(), conlyHcp.data(), hyb);
            H_H2D(C, C.hcHDev, hcpDev.data(), hcpDev.size(), "ch_ple_hcp_seed");
        }
        aclrtMemset(sc0, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        H_LaunchHcSeg(C, L, 0u, sc0, C.hcHDev, H_ChBoPlane(C, 0u), C.hcIjDev, M15H::HC_IJ_STRIDE_PLANE,
                      M15H::MODE_MIX, m);
        if (!H_Sync(C, "ch_ple_mix")) { return false; }
        H_ChReadWs(C, "ch_ple_mix", sc0, m, man1);
        snprintf(tag, sizeof(tag), "Ch.h1blk.L%02u", L);
        ok = H_ChSame(C, tag, R.blkA, man1.blk) && ok;
        snprintf(tag, sizeof(tag), "Ch.h1inj.L%02u", L);
        ok = H_ChSame(C, tag, R.b1.oh, man1.oh) && ok;
    } else {
        aclrtMemset(sc0, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
        H_LaunchHcSeg(C, L, 0u, sc0, C.hcHDev, H_ChBoPlane(C, 0u), C.hcIjDev, M15H::HC_IJ_STRIDE_PLANE,
                      M15H::MODE_COMBINE_MIX, m);
        if (!H_Sync(C, "ch_m1")) { return false; }
        H_ChReadWs(C, "ch_m1", sc0, m, man1);
        snprintf(tag, sizeof(tag), "Ch.h1hcp.L%02u", L);
        ok = H_ChSame(C, tag, R.b1.hcp, man1.hcp) && ok;
        snprintf(tag, sizeof(tag), "Ch.h1blk.L%02u", L);
        ok = H_ChSame(C, tag, R.blkA, man1.blk) && ok;
        snprintf(tag, sizeof(tag), "Ch.h1inj.L%02u", L);
        ok = H_ChSame(C, tag, R.b1.oh, man1.oh) && ok;
    }
    __builtin_memcpy(subIn.data(), man1.blk.data(), hbytes);

    // ---- (b) 子层段（输入 = 手工路径的 BLK_attn）----
    H_H2D(C, C.xFusedDev, subIn.data(), hyb < hbytes ? hyb : hbytes, "ch_m_subin");
    std::vector<uint8_t> manAttn(hbytes);
    if (isGdn) {
        uint8_t* subWs = reinterpret_cast<uint8_t*>(C.wsGdnDev);
        m15_gdn_layer_kernel<<<C.numBlocks, 0, C.stream>>>(
            subWs, reinterpret_cast<uint8_t*>(C.xFusedDev), reinterpret_cast<uint8_t*>(C.resZeroDev),
            reinterpret_cast<uint8_t*>(C.ySubDev), H_W(C, L, M15Loop::W_G1_OFF), H_W(C, L, M15Loop::W_G2_OFF),
            H_W(C, L, M15Loop::W_GG_OFF), H_W(C, L, M15Loop::W_IN_OFF), H_W(C, L, M15Loop::W_OUT_OFF),
            H_W(C, L, M15Loop::W_CONV_OFF), H_W(C, L, M15Loop::W_CONVB_OFF),
            H_W(C, L, M15Loop::W_ALOG_OFF), H_W(C, L, M15Loop::W_DTB_OFF), H_ST(C, L, 0),
            H_ST(C, L, M15Loop::SSM_OFF));
    } else {
        m15_attn_placeholder_kernel<<<C.numBlocks, 0, C.stream>>>(reinterpret_cast<uint8_t*>(C.xFusedDev),
                                                                  reinterpret_cast<uint8_t*>(C.ySubDev),
                                                                  M15Loop::H_BYTES);
    }
    if (!H_Sync(C, "ch_m_sub")) { return false; }
    H_D2H(C, manAttn.data(), C.ySubDev, hbytes, "ch_m_attnout");
    if (!H_Sync(C, "ch_m_attnout_d2h")) { return false; }
    snprintf(tag, sizeof(tag), "Ch.sub.L%02u", L);
    ok = H_ChSame(C, tag, R.attnOut, manAttn) && ok;

    // ---- (c) hc 边界 #2（输入 = 手工路径的 H' / attn_out / 手工 H1 的 OH[:,320:324)）----
    // 手工 H1 的 IJ 源：`sc0 + WS_OH + OH_INJ*2`，行距 OH_W（与链上同一算式）
    aclrtMemset(sc1, M15L::HC_WS_LAYER_STRIDE, 0xCD, M15L::HC_WS_LAYER_STRIDE);
    {
        // boundary #2 的 hIn（三个形态各不同，必须与 kernel 内 `M15L_FillHcPtrs` 的算式逐字对齐）：
        //   层 0  = 层输入 H（mode MIX 不写 H' ⇒ H' ≡ H）；
        //   层 1  = combine-only 物化的 H'（`sc2+WS_HCP`；**mix-only 那一遍不写 HCP**，故不能取 sc0）；
        //   其余  = 手工 H1（combine_and_mix）写出的 H'（`sc0+WS_HCP`）。
        uint8_t* h2In = (L == 0u) ? reinterpret_cast<uint8_t*>(C.hcHDev)
                                  : ((L == 1u) ? (sc2 + M15H::WS_HCP) : (sc0 + M15H::WS_HCP));
        if (L == 0u) {
            H_H2D(C, C.hcHDev, R.hIn.data(), hyb, "ch_m_hin3");
        }
        H_H2D(C, C.hcAttnOutDev, manAttn.data(), hbytes, "ch_m_attnout_h2d");
        // boundary #2 的 IJ 源：**永远**是本层 H1 的 `OH[:,320:324)`（行距 OH_W）——层 0 的
        // H1 是 mode MIX，但它照样产出 injection（`use_combine` 只影响 combine，不影响 S3）。
        H_LaunchHcSeg(C, L, 1u, sc1, h2In, C.hcAttnOutDev, sc0 + M15H::WS_OH + M15H::OH_INJ * 2u,
                      M15H::OH_W, M15H::MODE_COMBINE_MIX, m);
    }
    if (!H_Sync(C, "ch_m2")) { return false; }
    ChWsRow man2;
    H_ChReadWs(C, "ch_m2", sc1, m, man2);
    snprintf(tag, sizeof(tag), "Ch.h2hcp.L%02u", L);
    ok = H_ChSame(C, tag, R.hOut, man2.hcp) && ok;
    snprintf(tag, sizeof(tag), "Ch.h2blk.L%02u", L);
    ok = H_ChSame(C, tag, R.blkM, man2.blk) && ok;
    snprintf(tag, sizeof(tag), "Ch.h2inj.L%02u", L);
    {
        std::vector<uint8_t> man2Inj;
        H_ChExtractInj(man2.oh, m, man2Inj);
        ok = H_ChSame(C, tag, R.ijOut, man2Inj) && ok;
    }

    // ---- (d) MoE 段（输入 = 手工路径的 BLK_mlp）----
    H_H2D(C, C.xMoeDev, man2.blk.data(), hbytes, "ch_m_moein");
    H_LaunchMoeOnly(C, L, moeWs, C.xMoeDev, C.yMoeDev);
    if (!H_Sync(C, "ch_m_moe")) { return false; }
    std::vector<uint8_t> manBo(hbytes);
    H_D2H(C, manBo.data(), C.yMoeDev, hbytes, "ch_m_moeout");
    if (!H_Sync(C, "ch_m_moeout_d2h")) { return false; }
    snprintf(tag, sizeof(tag), "Ch.bo.L%02u", L);
    ok = H_ChSame(C, tag, R.boOut, manBo) && ok;
    return ok;
}

// M39 对拍所需的 dump（只对 `M15_HC_LAYERS` 列出的层；脚本 `check_chain_ref.py` 消费）
static void H_ChDump(Ctx& C, uint32_t L, const ChLayerRec& R, uint32_t m, const std::vector<uint8_t>& multiHidden,
                     const std::vector<uint8_t>& sampleHidden)
{
    if (!C.O.dump) {
        return;
    }
    char nm[96];
    auto tag = [&](const char* suffix) {
        snprintf(nm, sizeof(nm), "ch_L%02u_%s", L, suffix);
        return std::string(nm);
    };
    // 只落盘**前 m 行**（host 副本按 [M_MAX, …] 定尺，故一律按 m 截断 —— 判据侧按同一形状 reshape）
    const uint32_t hyb = m * M15H::HYPER * 2u;
    const uint32_t hbytes = m * HIDDEN * 2u;
    const uint32_t ijbytes = m * M15H::INJ_N * 2u;
    char shape[64];
    snprintf(shape, sizeof(shape), "[%u,%u]", m, M15H::HYPER);
    H_Dump(C, tag("hin"), R.hIn.data(), hyb, "BF16", shape);
    snprintf(shape, sizeof(shape), "[%u,%u]", m, HIDDEN);
    H_Dump(C, tag("bo"), R.boIn.data(), hbytes, "BF16", shape);
    snprintf(shape, sizeof(shape), "[%u,%u]", m, M15H::INJ_N);
    H_Dump(C, tag("ij"), R.ijIn.data(), ijbytes, "BF16", shape);
    snprintf(shape, sizeof(shape), "[%u,%u]", m, M15H::HYPER);
    H_Dump(C, tag("hout"), R.hOut.data(), hyb, "BF16", shape);
    H_Dump(C, tag("attnout"), R.attnOut.data(), hbytes, "BF16",
           (snprintf(shape, sizeof(shape), "[%u,%u]", m, HIDDEN), shape));
    H_Dump(C, tag("y"), R.boOut.data(), hbytes, "BF16",
           (snprintf(shape, sizeof(shape), "[%u,%u]", m, HIDDEN), shape));
    snprintf(shape, sizeof(shape), "[%u,%u]", m, M15H::INJ_N);
    H_Dump(C, tag("ijout"), R.ijOut.data(), ijbytes, "BF16", shape);
    // 两个边界的 8 个张量（与 Ver H 的 `hc_L*_b{1,2}_*` 同口径；python 侧按同一偏移切片）
    const struct {
        const char* sfx;
        const std::vector<uint8_t>* v;
    } b1[] = {{"hcp", &R.b1.hcp}, {"xn", &R.b1.xn},   {"rstd", &R.b1.rstd}, {"injw", &R.b1.injw},
              {"oh", &R.b1.oh},   {"ls", &R.b1.ls},   {"gate", &R.b1.gate}, {"blk", &R.b1.blk}};
    for (const auto& e : b1) {
        std::string n2 = tag("b1_") + e.sfx;
        H_Dump(C, n2, e.v->data(), e.v->size(), "BF16", shape);
    }
    const struct {
        const char* sfx;
        const std::vector<uint8_t>* v;
    } b2[] = {{"hcp", &R.b2.hcp}, {"xn", &R.b2.xn},   {"rstd", &R.b2.rstd}, {"injw", &R.b2.injw},
              {"oh", &R.b2.oh},   {"ls", &R.b2.ls},   {"gate", &R.b2.gate}, {"blk", &R.b2.blk}};
    for (const auto& e : b2) {
        std::string n2 = tag("b2_") + e.sfx;
        H_Dump(C, n2, e.v->data(), e.v->size(), "BF16", shape);
    }
    char gshape[64];
    snprintf(gshape, sizeof(gshape), "[%u,%u]", m, M15H::HYPER);
    H_Dump(C, "ch_gm_multi", multiHidden.data(), hyb, "BF16", gshape);
    snprintf(gshape, sizeof(gshape), "[%u,%u]", m, HIDDEN);
    H_Dump(C, "ch_gm_sample", sampleHidden.data(), hbytes, "BF16", gshape);
}

// 末层全局 mixer 的 dump：输入（末层的三态）+ 输出 ws 的 8 个张量 + 两个出口
static void H_ChDumpGm(Ctx& C, const ChLayerRec& last, uint32_t m, const std::vector<uint8_t>& multiHidden,
                       const std::vector<uint8_t>& sampleHidden)
{
    if (!C.O.dump) {
        return;
    }
    ChWsRow gw;
    H_ChReadWs(C, "ch_gm_ws", H_ChWs(C, M15L::HC_WS_GMIX_SLOT), m, gw);
    char shape[64];
    snprintf(shape, sizeof(shape), "[%u,%u]", m, M15H::HYPER);
    H_Dump(C, "ch_gm_hin", last.hOut.data(), last.hOut.size(), "BF16", shape);
    snprintf(shape, sizeof(shape), "[%u,%u]", m, HIDDEN);
    H_Dump(C, "ch_gm_bo", last.boOut.data(), last.boOut.size(), "BF16", shape);
    snprintf(shape, sizeof(shape), "[%u,%u]", m, M15H::INJ_N);
    H_Dump(C, "ch_gm_ij", last.ijOut.data(), last.ijOut.size(), "BF16", shape);

    const struct {
        const char* sfx;
        const std::vector<uint8_t>* v;
    } g[] = {{"hcp", &gw.hcp}, {"xn", &gw.xn},   {"rstd", &gw.rstd}, {"injw", &gw.injw},
             {"oh", &gw.oh},   {"ls", &gw.ls},   {"gate", &gw.gate}, {"blk", &gw.blk}};
    for (const auto& e : g) {
        std::string n2 = std::string("ch_gm_b1_") + e.sfx;
        H_Dump(C, n2, e.v->data(), e.v->size(), "BF16", shape);
    }
    snprintf(shape, sizeof(shape), "[%u,%u]", m, M15H::HYPER);
    H_Dump(C, "ch_gm_multi", multiHidden.data(), multiHidden.size(), "BF16", shape);
    snprintf(shape, sizeof(shape), "[%u,%u]", m, HIDDEN);
    H_Dump(C, "ch_gm_sample", sampleHidden.data(), sampleHidden.size(), "BF16", shape);
}

static bool H_RunChain(Ctx& C)
{
    using namespace M15G;
    const uint32_t m = C.O.chainM;
    const uint32_t nL = C.O.nLayers;
    const uint32_t hyb = m * M15H::HYPER * 2u;
    const uint32_t hbytes = m * HIDDEN * 2u;
    const uint32_t ijbytes = m * M15H::INJ_N * 2u;

    printf("\n[m15] ===== 验证 Ch：48 层链走四相位入口（每层一次启动；m=%u，层数 %u）=====\n", m, nL);
    printf("[m15]   段序：hc(attn) → 子层段 → hc(mlp) → MoE（层 1 另把 H1 拆成 "
           "combine-only → PLE 占位 → mix-only）；层间零拷贝搬 H/BO/IJ 三条长寿状态\n");
    printf("[m15]   PLE 打断点 = %s（M15_CHAIN_PLE=%u；关掉只用于 msprof 的 A/B）\n",
           (C.O.chainPle != 0u) ? "开" : "**关**（层 1 走单次 combine_and_mix）", C.O.chainPle);
    const uint32_t checkBase = C.checks;
    const uint32_t guardBase = C.guards;
    bool ok = H_ChWiredCheck(C);
    ok = H_GmWeightSourceCheck(C) && ok;

    // ---- 入口 H 平面 = `embed_tokens(...).repeat(1, hc_count)`（`V-N:model.py:506`）----
    std::vector<uint8_t> hin(M15H::M_MAX * M15H::HYPER * 2u, 0);
    {
        const uint16_t* h0 = C.h0[0].data();
        for (uint32_t r = 0; r < m; ++r) {
            for (uint32_t s = 0; s < M15H::HC; ++s) {
                __builtin_memcpy(hin.data() + (static_cast<size_t>(r) * M15H::HYPER + s * HIDDEN) * 2u,
                                 h0 + static_cast<size_t>(r) * HIDDEN, HIDDEN * 2u);
            }
        }
    }
    std::vector<uint8_t> hinAlt(hin.size());
    for (size_t i = 0; i < hin.size() / 2; ++i) {
        const uint16_t v = reinterpret_cast<const uint16_t*>(hin.data())[i];
        reinterpret_cast<uint16_t*>(hinAlt.data())[i] = H_F32ToBf16(H_Bf16ToF32(v) * 1.5f);
    }

    // ---- pass 1：跑链（full 记录）+ pass 2：确定性重跑 ----
    std::vector<ChLayerRec> rec1, rec2, rec3;
    std::vector<uint8_t> mh1(hyb), sh1(hbytes), mh2(hyb), sh2(hbytes), mh3(hyb), sh3(hbytes);
    if (!H_ChRunOnce(C, hin, m, true, rec1, mh1, sh1)) {
        return false;
    }
    printf("[m15]   pass 1：%u 层链跑通（一次启动/层；%s）\n", nL,
           (nL > 1u && C.O.chainPle != 0u) ? "层 1 走 PLE 打断点（H1 拆成五段）"
                                           : "**未**走 PLE 打断点（层数 < 2 或 M15_CHAIN_PLE=0）");
    printf("[m15][DISCLOSE] 层 1 的 PLE 段**未实现**（空操作，见 m15_layer_kernel.h §3b）："
           "本段验证的是打断点的结构与它的舍入点判据，**不是** PLE 的数值。%s\n",
           (nL > 1u && C.O.chainPle != 0u) ? "" : "（本次链未启用打断点，该披露对本 run 不适用）");
    const uint32_t lastL = nL - 1u;

    // ---- ⑤/链接线：手工路径（逐层四段独立启动）----
    const bool manual = (getenv("M15_CHAIN_MANUAL") == nullptr) || (atoi(getenv("M15_CHAIN_MANUAL")) != 0);
    if (manual) {
        printf("[m15]   手工路径：逐层 4 段独立启动（hc#1 → 子层段 → hc#2 → MoE）与链逐字节对拍"
               "（PLE 层另加 combine-only 与单次 combine_and_mix 参考各一次 ⇒ 该层 6 次启动）\n");
        for (uint32_t L = 0; L < nL; ++L) {
            ok = H_ChManualLayer(C, L, rec1[L], m) && ok;
        }
    } else {
        printf("[m15][WARN] M15_CHAIN_MANUAL=0：手工路径对拍被显式跳过（覆盖下降，见 README 覆盖交代）\n");
    }

    // ---- ③ 确定性：pass 2 重跑，三条长寿状态逐字节相同 ----
    if (!H_ChRunOnce(C, hin, m, false, rec2, mh2, sh2)) {
        return false;
    }
    for (uint32_t L = 0; L < nL; ++L) {
        char tag[96];
        snprintf(tag, sizeof(tag), "Ch.detH.L%02u", L);
        ok = H_ChSame(C, tag, rec1[L].hOut, rec2[L].hOut) && ok;
        snprintf(tag, sizeof(tag), "Ch.detBO.L%02u", L);
        ok = H_ChSame(C, tag, rec1[L].boOut, rec2[L].boOut) && ok;
        snprintf(tag, sizeof(tag), "Ch.detIJ.L%02u", L);
        ok = H_ChSame(C, tag, rec1[L].ijOut, rec2[L].ijOut) && ok;
    }
    ok = H_ChSame(C, "Ch.det.gm_multi", mh1, mh2) && ok;
    ok = H_ChSame(C, "Ch.det.gm_sample", sh1, sh2) && ok;

    // ---- ① 非空洞性（整链）：入口 H ×1.5 ⇒ 末层三态与 mixer 输出必须变 ----
    ok = H_Guard(C, __builtin_memcmp(hin.data(), hinAlt.data(), hin.size()) != 0) &&
         ok;   // 扰动确实改了输入（否则下面两条是空洞的）
    C.guardsCh += 1u;
    if (!H_ChRunOnce(C, hinAlt, m, false, rec3, mh3, sh3)) {
        return false;
    }
    ok = H_CmpMustDiffer(C, "Ch.pert.lastH", rec3[lastL].hOut.data(), rec1[lastL].hOut.data(), hyb) && ok;
    ok = H_CmpMustDiffer(C, "Ch.pert.gm", sh3.data(), sh1.data(), hbytes) && ok;
    ok = H_CmpMustDiffer(C, "Ch.pert.gmmerge", mh3.data(), mh1.data(), hyb) && ok;

    // ---- ①/② 逐层：非恒等、MoE 非恒等、两边界互异、有限、指纹 ----
    for (uint32_t L = 0; L < nL; ++L) {
        char tag[96];
        snprintf(tag, sizeof(tag), "Ch.nonid.L%02u", L);
        ok = H_CmpMustDiffer(C, tag, rec1[L].hOut.data(), rec1[L].hIn.data(), hyb) && ok;
        snprintf(tag, sizeof(tag), "Ch.moe.L%02u", L);
        ok = H_CmpMustDiffer(C, tag, rec1[L].boOut.data(), rec1[L].blkM.data(), hbytes) && ok;
        snprintf(tag, sizeof(tag), "Ch.blks.L%02u", L);
        ok = H_CmpMustDiffer(C, tag, rec1[L].blkA.data(), rec1[L].blkM.data(), hbytes) && ok;
        snprintf(tag, sizeof(tag), "Ch.finite.L%02u", L);
        ok = H_CmpFiniteBf16(C, tag, rec1[L].hOut.data(), hyb) && ok;
    }
    // mixer 的两条输出：multi_hidden 必须 ≠ 末层 H''（combine 真的发生过）、sample_hidden 非常量且有限
    ok = H_CmpMustDiffer(C, "Ch.gm.multi", mh1.data(), rec1[lastL].hOut.data(), hyb) && ok;
    ok = H_CmpFiniteBf16(C, "Ch.gm.sample", sh1.data(), hbytes) && ok;
    {
        C.checks++;
        const uint16_t* s = reinterpret_cast<const uint16_t*>(sh1.data());
        bool nonconst = false;
        for (size_t i = 1; i < sh1.size() / 2; ++i) {
            if (s[i] != s[0]) {
                nonconst = true;
                break;
            }
        }
        if (nonconst) {
            printf("[m15]   %-30s PASS (非常量)\n", "Ch.gm.sample_nonconst");
        } else {
            printf("[m15]   %-30s FAIL (sample_hidden 全常量)\n", "Ch.gm.sample_nonconst");
            C.fails++;
            ok = false;
        }
    }

    // ---- ② 跨层指纹互异（层号 ↔ 槽位不错位）----
    for (uint32_t L = 0; L < nL; ++L) {
        std::vector<uint8_t> blob;
        blob.insert(blob.end(), rec1[L].hOut.begin(), rec1[L].hOut.end());
        blob.insert(blob.end(), rec1[L].boOut.begin(), rec1[L].boOut.end());
        blob.insert(blob.end(), rec1[L].ijOut.begin(), rec1[L].ijOut.end());
        rec1[L].fp = blob;
    }
    // 指纹 = 三条长寿状态的**原始字节**（不用哈希 ⇒ 判据是精确的，不存在碰撞假说）
    for (uint32_t L = 0; L < nL; ++L) {
        char tag[96];
        bool distinct = true;
        uint32_t hit = 0;
        for (uint32_t K = 0; K < nL; ++K) {
            if (K != L && rec1[K].fp.size() == rec1[L].fp.size() &&
                __builtin_memcmp(rec1[K].fp.data(), rec1[L].fp.data(), rec1[L].fp.size()) == 0) {
                distinct = false;
                hit = K;
                break;
            }
        }
        snprintf(tag, sizeof(tag), "Ch.fp.L%02u", L);
        C.checks++;
        if (distinct) {
            printf("[m15]   %-30s PASS (跨层三态指纹互异，n=%zu 字节)\n", tag, rec1[L].fp.size());
        } else {
            printf("[m15]   %-30s FAIL (与层 %02u 的三态指纹逐字节相同 ⇒ 槽位/层号可能错位)\n", tag, hit);
            C.fails++;
            ok = false;
        }
    }
    (void)ijbytes;

    // ---- ④ 与 M39 官方单层参考对拍所需的 dump（脚本 check_chain_ref.py 消费）----
    // 覆盖 = `M15_HC_LAYERS`（默认 0=GDN 层 / 1=PLE 层 / 3=attention 层）× 该层的输入与两边界输出
    if (C.O.dump) {
        uint32_t nDump = 0;
        for (uint32_t L : C.O.hcLayers) {
            if (L >= nL) {
                continue;
            }
            H_ChDump(C, L, rec1[L], m, mh1, sh1);
            ++nDump;
        }
        H_ChDumpGm(C, rec1[lastL], m, mh1, sh1);   // 末层 mixer 的输入/输出（④ 的对拍对象）
        char detail[160];
        snprintf(detail, sizeof(detail), "M39 对拍 dump 已落盘：%u 层（M15_HC_LAYERS）+ 末层 mixer", nDump);
        const bool gd = H_Guard(C, nDump > 0);
        C.guardsCh += 1u;
        printf("[m15]   %-30s %s\n", "Ch.m39.dump", detail);
        if (!gd) {
            ok = false;
        }
        printf("[m15]   ④ 与 M39 官方单层参考对拍：请在 dump 目录跑 "
               "`check_chain_ref.py`（独立的三态退出码与计数；本段只保证输入/输出已落盘）\n");
    } else {
        printf("[m15]   ④ 与 M39 官方单层参考对拍需要 M15_DUMP=1（本次未落盘 ⇒ 该条为 SKIPPED，"
               "不计入判定项）\n");
    }

    C.phaseCh = C.checks - checkBase;
    C.guardsCh = C.guards - guardBase;
    printf("[m15] 验证 Ch：本段判定项 %u 条 + guard %u 条（累计 checks %u / guards %u / fails %u）\n",
           C.phaseCh, C.guardsCh, C.checks, C.guards, C.fails);
    return ok && C.fails == 0;
}
