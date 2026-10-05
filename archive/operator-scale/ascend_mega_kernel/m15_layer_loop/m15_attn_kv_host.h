// ============================================================
// m15_attn_kv_host.h —— M82 host 侧：attention 三套 cache 的**布局判据 + 容量自检**（验证 Kv）
//
// 只被 m15_layer_loop.asc 在 m15_chain_host.h 之后 include（用到那里的 H_CmpBytes / Ctx / 工具）。
//
// 本段做四件事：
//   ① **容量自检**：从规范里的**原始数字**（16 / 2 / 256 / 4 / 140 / 2052 / 4097）**独立复算**
//      每层/每平面的字节数，与 `m15_attn_kv.h` 的登记常量逐条核对（guard），并打印算式→数值；
//      再把每个平面**最后一层槽的最后一个字节**写标记读回（见 ③）——证明分配真的覆盖了 12 层。
//   ② **对齐约束判据**：哪些 stride 是 32 B 对齐、哪几个不是（raw ring 单行 280 B ≡ 24 (mod 32)、
//      packed 单行 8,208 B ≡ 16 (mod 32)）⇒ 单行搬运**必须** DataCopyPad。这不是注释：它是判据，
//      而且 ③ 的实跑正是这些非 32 B 倍数传输成功完成的正对照。
//   ③ **device 探针的逐字节判据**（`m15_attn_kv_probe.h`）：对一组 `pos` 各启动一次探针，比
//      · T-KV-PAGED：paged 地址写 → 连续地址读，**逐字节相同**；
//      · raw ring 行、compressed 行、packed 两行：逐字节相同；
//      · 门控标志（写没写压缩行）与 `CompRowWritten()` 一致；
//      · 层槽/平面边界标记 round-trip（12 个层槽互不别名）。
//   ④ **两条负向对照**（docs/17 §4 硬要求）：
//      · off-by-one：注入档（`MODE_BROKEN_OFFBYONE`，pos 4096 也写压缩行）下，**同一条判据必须
//        FAIL**（= 判据有判别力）；
//      · paged：给**非恒等** block_table（交换物理页 1↔2）⇒
//        ① host 算式 `KvPagedContractHolds()` 必须**不成立**；**且** ② 用交换表**真启动一次探针**，
//        host 按连续地址读回的内容必须**不等于** seed（`Kv.neg.paged`，`H_CmpMustDiffer`）。
//        两条缺一不可：① 只证明算式不同、② 才证明 device 真读了表。
//
// 验收基准交代（docs/17 §7 裁决 ①）：本段判的是**布局与门控契约**，不是数值精度。首期数值基准 =
// 自建稠密 causal 参考，**它不是官方行为**（官方 QSA 每 token 只 attend 约一半历史，docs/17:281）。
// 本文件**不**出现任何「对齐官方输出」的表述。
// ============================================================

// 单请求的 block_table 长度：prefill 需要的逻辑页数（257）+ 8 个余量（paged 契约的 stride 用同一值）
constexpr uint32_t H_KvTblElems() { return M15Kv::PREFILL_BLOCKS + 8u; }   // 265
constexpr size_t H_KvTblBytes() { return static_cast<size_t>(H_KvTblElems()) * sizeof(int32_t); }

// packed 行的期望内容（host 生成 ⇒ 逐字节判据在 host 侧闭合）
static int32_t H_KvPackValue(uint32_t row, uint32_t col)
{
    using namespace M15Kv;
    if (col < 12u) {
        return static_cast<int32_t>(row * 1000u + col);
    }
    if (col >= PACK_TAIL_COL && col < PACK_TAIL_COL + 2u) {
        return static_cast<int32_t>(row * 1000u + 2000u + (col - PACK_TAIL_COL));
    }
    if (col == PACK_COUNT_COL) {
        return 14;
    }
    return static_cast<int32_t>(PACK_FILL);
}

static void H_KvLaunchProbe(Ctx& C, uint32_t pos, uint32_t mode)
{
    M15KvProbe::m15_attn_kv_probe_kernel<<<C.numBlocks, 0, C.stream>>>(
        reinterpret_cast<uint8_t*>(C.attnKvDev), reinterpret_cast<uint8_t*>(C.attnRingDev),
        reinterpret_cast<uint8_t*>(C.attnCompDev), reinterpret_cast<uint8_t*>(C.attnPackDev),
        reinterpret_cast<uint8_t*>(C.attnSeedDev), reinterpret_cast<uint8_t*>(C.attnPackSeedDev),
        reinterpret_cast<uint8_t*>(C.attnProbeOutDev), reinterpret_cast<int32_t*>(C.attnTblDev),
        H_KvTblElems(), H_KvTblElems(), pos, mode);
}

// host 侧输入面（三段各自成盐生成 ⇒ 判据能分辨"写到了哪一段"）
static bool H_KvPrep(Ctx& C)
{
    using namespace M15Kv;
    using namespace M15KvProbe;
    C.attnSeed.assign(SEED_BYTES, 0u);
    struct Seg {
        uint32_t off;
        uint32_t n;
        uint32_t salt;
    };
    const Seg segs[] = {{SEED_KV_ELEM_OFF, KV_TOKEN_ELEMS, 9001u},
                        {SEED_RING_ELEM_OFF, RING_HEAD_SIZE, 9002u},
                        {SEED_COMP_ELEM_OFF, COMP_HEAD_DIM, 9003u}};
    for (const Seg& s : segs) {
        std::vector<uint16_t> t(s.n, 0u);
        H_GenBf16(t, s.salt, 1.0f);
        __builtin_memcpy(C.attnSeed.data() + static_cast<size_t>(s.off) * ELEM_BYTES, t.data(),
                         static_cast<size_t>(s.n) * ELEM_BYTES);
    }
    C.attnPackSeed.assign(static_cast<size_t>(OUT_PACK_ROWS) * PACK_ROW_BYTES, 0u);
    int32_t* pr = reinterpret_cast<int32_t*>(C.attnPackSeed.data());
    for (uint32_t r = 0; r < OUT_PACK_ROWS; ++r) {
        for (uint32_t c = 0; c < PACK_COLS; ++c) {
            pr[static_cast<size_t>(r) * PACK_COLS + c] = H_KvPackValue(r, c);
        }
    }
    C.attnProbeOut.assign(OUT_BYTES, 0u);
    H_H2D(C, C.attnSeedDev, C.attnSeed.data(), C.attnSeed.size(), "kv_seed");
    H_H2D(C, C.attnPackSeedDev, C.attnPackSeed.data(), C.attnPackSeed.size(), "kv_packseed");
    return C.fails == 0u;
}

// block_table 两档：恒等（契约档）/ 交换物理页 1↔2（负向对照）
static void H_KvTblPrep(Ctx& C, bool identity)
{
    C.attnTbl.assign(H_KvTblElems(), 0);
    for (uint32_t b = 0; b < H_KvTblElems(); ++b) {
        C.attnTbl[b] = static_cast<int32_t>(b);
    }
    if (!identity) {
        const int32_t t = C.attnTbl[1];
        C.attnTbl[1] = C.attnTbl[2];
        C.attnTbl[2] = t;
    }
    H_H2D(C, C.attnTblDev, C.attnTbl.data(), H_KvTblBytes(), "kv_tbl");
}

// ============================================================
// ① 容量自检：**独立复算**（不复用 m15_attn_kv.h 的中间常量，只用规范里的原始数字）
// ============================================================
static bool H_KvCapacityCheck(Ctx& C)
{
    using namespace M15Kv;
    // 规范原始数字（docs/14:616-627、docs/11:25、docs/17:286）
    const uint64_t blkTokens = 16u;      // block_size
    const uint64_t kvHeads = 2u;         // N2
    const uint64_t headDim = 256u;       // head_size（K）
    // ⚠ M98 更正（塔裁 A）：官方主 KV 的 content 维 = K ‖ V，`head_size_v = head_dim`
    //   `qwen4_exp/nvidia/qsa.py:434-442` 的 `FullAttentionSpec(..., head_size_v=self.head_dim)`；
    //   页字节算式 `page_size_bytes = num_heads*num_states*state_content_size_bytes`
    //   （`vllm/v1/kv_cache_interface.py:506-528`）⇒ 2 × 16 × ((256+256)*2) = 32,768
    const uint64_t headDimV = 256u;      // head_size_v
    const uint64_t elem = 2u;            // bf16
    const uint64_t keyDim = 128u;        // raw k / compressed head_dim
    const uint64_t tailBytes = 24u;      // 3 × int64 MRoPE 位置
    const uint64_t ratio = 4u;           // indexer_compress_ratio
    const uint64_t packCols = 2052u;     // NVIDIA 契约
    const uint64_t prefillM = 4097u;

    const uint64_t pageBytes = blkTokens * kvHeads * (headDim + headDimV) * elem;   // = 32,768（官方算式）
    const uint64_t blocks = (prefillM + blkTokens - 1u) / blkTokens;
    const uint64_t mainLayer = blocks * pageBytes;
    const uint64_t compRows = blocks * blkTokens / ratio;
    const uint64_t compRowBytes = keyDim * elem;
    const uint64_t compLayer = compRows * compRowBytes;
    const uint64_t ringRowBytes = keyDim * elem + tailBytes;
    const uint64_t ringLayer = 4u * ringRowBytes;   // 4*ceil((4+num_spec)/4)，num_spec=0
    const uint64_t packPlane = prefillM * packCols * 4u;
    const uint64_t compWritten = (prefillM - 1u) / ratio;   // 3,7,…,4095

    printf("[m15]   Kv.cap 独立复算（只用规范原始数字 16/2/256/256/4/140/2052/4097；主 KV 的 content = K‖V）：\n");
    printf("[m15]     主 KV   : ceil(4097/16)=%llu 页 × %llu B/页 = %llu B/层；×%u 层 = %llu B\n",
           (unsigned long long)blocks, (unsigned long long)pageBytes, (unsigned long long)mainLayer,
           N_ATTN, (unsigned long long)(mainLayer * N_ATTN));
    printf("[m15]     compressed: %llu 行 × %llu B/行 = %llu B/层（写入 %llu 行 = 位置 3,7,…,4095）；"
           "×%u 层 = %llu B\n",
           (unsigned long long)compRows, (unsigned long long)compRowBytes, (unsigned long long)compLayer,
           (unsigned long long)compWritten, N_ATTN, (unsigned long long)(compLayer * N_ATTN));
    printf("[m15]     raw ring : 128×2 + 24 = %llu B/行 × 4 行 = %llu B/层；×%u 层 = %llu B\n",
           (unsigned long long)ringRowBytes, (unsigned long long)ringLayer, N_ATTN,
           (unsigned long long)(ringLayer * N_ATTN));
    printf("[m15]     packed   : %llu 行 × %llu 列 × 4 B = %llu B（**层间复用一块**）\n",
           (unsigned long long)prefillM, (unsigned long long)packCols, (unsigned long long)packPlane);
    // decode 档落点：**从权威 `DECODE_CTX` 派生**（M184）。历史上这里按字面 4096 独立复算 ⇒
    // 打印 256 页，比 4097-token context（位置 0..4096）所需的 257 页少 1 页，与实际口径脱节。
    // 本行只作信息打印，不参与上面的 guard（那些仍只用规范原始数字独立复算）。
    const uint64_t decodeCtx = DECODE_CTX;
    const uint64_t decodeBlocks = (decodeCtx + blkTokens - 1u) / blkTokens;
    const uint64_t decodeCompRows = decodeBlocks * blkTokens / ratio;
    printf("[m15]     decode(1,ctx=%llu)：主 KV %llu 页 = %llu B/层；compressed %llu 行 = %llu B/层\n",
           (unsigned long long)decodeCtx,
           (unsigned long long)decodeBlocks, (unsigned long long)(decodeBlocks * pageBytes),
           (unsigned long long)decodeCompRows, (unsigned long long)(decodeCompRows * compRowBytes));

    bool ok = true;
    ok = H_Guard(C, mainLayer == KV_LAYER_STRIDE) && ok;
    ok = H_Guard(C, compLayer == COMP_LAYER_STRIDE) && ok;
    ok = H_Guard(C, ringLayer == RING_LAYER_STRIDE) && ok;
    ok = H_Guard(C, packPlane == PACK_PLANE_BYTES) && ok;
    ok = H_Guard(C, mainLayer * N_ATTN == KV_PLANE_BYTES) && ok;
    ok = H_Guard(C, compLayer * N_ATTN == COMP_PLANE_BYTES) && ok;
    ok = H_Guard(C, ringLayer * N_ATTN == RING_PLANE_BYTES) && ok;
    ok = H_Guard(C, compWritten == PREFILL_COMP_ROWS_WRITTEN) && ok;
    ok = H_Guard(C, compWritten < compRows) && ok;
    return ok;
}

// ② 对齐约束判据（+ 层槽编址）
static bool H_KvAlignAndSlotCheck(Ctx& C)
{
    using namespace M15Kv;
    bool ok = true;
    // 32 B 对齐的（可直接 DataCopy/Block1）
    ok = H_Guard(C, KV_BLOCK_BYTES % 32u == 0u) && ok;
    ok = H_Guard(C, KV_TOKEN_STRIDE % 32u == 0u) && ok;
    ok = H_Guard(C, KV_HEAD_PLANE_BYTES % 32u == 0u) && ok;
    ok = H_Guard(C, COMP_ROW_BYTES % 32u == 0u) && ok;
    ok = H_Guard(C, KV_LAYER_STRIDE % 32u == 0u) && ok;
    ok = H_Guard(C, COMP_LAYER_STRIDE % 32u == 0u) && ok;
    ok = H_Guard(C, RING_LAYER_BYTES % 32u == 0u) && ok;
    // **非** 32 B 整数倍的两处（⇒ 单行搬运必须 DataCopyPad）
    ok = H_Guard(C, RING_ROW_BYTES % 32u == 24u) && ok;
    ok = H_Guard(C, RING_ROW_BYTES % 8u == 0u) && ok;
    ok = H_Guard(C, PACK_ROW_BYTES % 32u == 16u) && ok;
    ok = H_Guard(C, PACK_ROW_BYTES % 16u == 0u) && ok;
    for (uint32_t s = 1u; s < RING_ROWS_PER_BLOCK; ++s) {   // ring 第 1/2/3 行的起点都不是 32 B 对齐
        ok = H_Guard(C, (RingRowOffset(0u, s) % 32u) != 0u) && ok;
    }
    ok = H_Guard(C, (PackRowOffset(1u) % 32u) == 16u) && ok;
    // 数字一律由宏打印（M98 更正后不得再出现硬编码的旧geometry文字）
    printf("[m15]   Kv.align 判据：主 KV 页 %u / 页内 token 步长 %u / head 平面 %u / head 槽内 K‖V = %u|%u / "
           "compressed 行 %u —— 均 32 B 整数倍；**raw ring 单行 %u B ≡ %u (mod 32)**、**packed 单行 %u B ≡ "
           "%u (mod 32)** ⇒ 单行传输只能走 DataCopyPad（实跑见证见 Kv.ring.row.* / Kv.pack.rows.*）\n",
           KV_BLOCK_BYTES, KV_TOKEN_STRIDE, KV_HEAD_PLANE_BYTES, KV_HEAD_CONTENT_BYTES, KV_HEAD_CONTENT_BYTES,
           COMP_ROW_BYTES, RING_ROW_BYTES, RING_ROW_BYTES % 32u, PACK_ROW_BYTES, PACK_ROW_BYTES % 32u);
    // head 平面 ≠ 页内 token 步长 ⇒ 主 KV 一个 token 的 2,048 B（2 head × 1,024 B）**不是**连续区间
    ok = H_Guard(C, KV_HEAD_PLANE_BYTES != KV_TOKEN_STRIDE) && ok;
    // 层槽编址：12 个 attention 层 → k = 0..11，且各自基址两两不同、末层槽不越界
    uint32_t seen = 0u;
    for (uint32_t L = 0; L < NL; ++L) {
        const uint32_t k = AttnSlot(L);
        if (KindOf(L) != KIND_ATTN) {
            ok = H_Guard(C, k == NO_SLOT && LayerSlot(L) != NO_SLOT) && ok;
            continue;
        }
        ok = H_Guard(C, k < N_ATTN) && ok;
        ok = H_Guard(C, (L % ATTN_INTERVAL) == (ATTN_INTERVAL - 1u)) && ok;
        seen |= (1u << k);
        ok = H_Guard(C, KvLayerOffset(k) + KV_LAYER_STRIDE <= KV_PLANE_BYTES) && ok;
        ok = H_Guard(C, CompLayerOffset(k) + COMP_LAYER_STRIDE <= COMP_PLANE_BYTES) && ok;
        ok = H_Guard(C, RingLayerOffset(k) + RING_LAYER_STRIDE <= RING_PLANE_BYTES) && ok;
    }
    ok = H_Guard(C, seen == ((1u << N_ATTN) - 1u)) && ok;   // 12 个 k 恰好各出现一次
    ok = H_Guard(C, AttnSlot(3u) == 0u && AttnSlot(7u) == 1u && AttnSlot(47u) == 11u) && ok;
    ok = H_Guard(C, LayerSlot(3u) == NO_SLOT) && ok;   // attention 层不占 GDN 槽（M25 语义未变）
    return ok;
}

// ③ 层槽/平面边界标记 round-trip：每个平面「第 k 层槽的第一个字节 + 最后一个字节」写标记读回，
//    写满**全部**标记后才读 ⇒ 同时验证「可寻址」与「写第 k 层不踩第 k+1 层」。
//    每个平面一条判定项（`Kv.cap.rt.*`，逐字节）。
static bool H_KvBoundaryRoundTrip(Ctx& C)
{
    using namespace M15Kv;
    struct Plane {
        uint8_t* base;
        uint64_t stride;
        uint32_t slots;
        const char* tag;
    };
    const Plane planes[4] = {
        {reinterpret_cast<uint8_t*>(C.attnKvDev), KV_LAYER_STRIDE, N_ATTN, "Kv.cap.rt.main"},
        {reinterpret_cast<uint8_t*>(C.attnCompDev), COMP_LAYER_STRIDE, N_ATTN, "Kv.cap.rt.comp"},
        {reinterpret_cast<uint8_t*>(C.attnRingDev), RING_LAYER_STRIDE, N_ATTN, "Kv.cap.rt.ring"},
        {reinterpret_cast<uint8_t*>(C.attnPackDev), PACK_PLANE_BYTES, 1u, "Kv.cap.rt.pack"},
    };
    bool ok = true;
    size_t nProbe = 0;
    for (uint32_t p = 0; p < 4u; ++p) {
        const Plane& pl = planes[p];
        std::vector<uint64_t> offs;
        std::vector<uint64_t> want;
        for (uint32_t s = 0; s < pl.slots; ++s) {
            const uint64_t base = static_cast<uint64_t>(s) * pl.stride;
            const uint64_t last = (pl.slots == 1u) ? (PACK_PLANE_BYTES - 8u) : (base + pl.stride - 8u);
            offs.push_back(base);
            offs.push_back(last);
            want.push_back(0x5A5A000000000000ull | (static_cast<uint64_t>(p) << 40) |
                           (static_cast<uint64_t>(s) << 8) | 0x11u);
            want.push_back(0x5A5A000000000000ull | (static_cast<uint64_t>(p) << 40) |
                           (static_cast<uint64_t>(s) << 8) | 0x22u);
        }
        for (size_t i = 0; i < offs.size(); ++i) {
            H_H2D(C, pl.base + offs[i], &want[i], sizeof(uint64_t), "kv_rt_w");
        }
        std::vector<uint8_t> got(offs.size() * sizeof(uint64_t), 0u);
        for (size_t i = 0; i < offs.size(); ++i) {
            H_D2H(C, got.data() + i * sizeof(uint64_t), pl.base + offs[i], sizeof(uint64_t), "kv_rt_r");
        }
        ok = H_CmpBytes(C, pl.tag, got.data(), want.data(), got.size()) && ok;
        nProbe += offs.size();
    }
    printf("[m15]   Kv.cap.rt 标记 round-trip：%zu 个位置（%u 层主 KV 的槽首/槽尾 24 + compressed 24 + "
           "ring 24 + packed 2）= 「第 k 槽可寻址且不被第 k+1 槽的写踩掉」\n",
           nProbe, N_ATTN);
    return ok;
}

// ④ 负向对照之二：非恒等 block_table ⇒ paged 写与连续读**必须不同**
//
// 两条一起给（**缺任一条都不够**）：
//   (a) **host 算式**：`KvPagedContractHolds()` 在恒等表上成立、在交换表上**不成立**（2 条 guard）；
//   (b) **device 字节**：用交换表**真启动一次探针** —— 探针按 paged 把 head0/head1 写到
//       `block_table[1] = 2` 那一页，host 再按**连续地址**（页 1）读回 ⇒ 必须**不等于** seed
//       （判据 `Kv.neg.paged`，`H_CmpMustDiffer`）。这条同时见证「表真的被 device 读了」。
// 启动前后都复位恒等表，保证本函数的副作用不外溢（依赖闭包：`Kv.lay.headsplit` 用 `KvPhysBlock`
// 取页号，必须在恒等表上跑）。
static bool H_KvNegPaged(Ctx& C)
{
    using namespace M15Kv;
    using namespace M15KvProbe;
    const uint32_t pos = 17u;   // 逻辑页 1、页内槽 1（非平凡位置）
    bool ok = true;
    ok = H_Guard(C, KvPagedContractHolds(C.attnTbl.data(), 0u, H_KvTblElems(), pos, 0u, KV_LANE_K, KV_HEAD_DIM)) && ok;
    ok = H_Guard(C, KvPagedContractHolds(C.attnTbl.data(), 0u, H_KvTblElems(), pos, 0u, KV_LANE_V, KV_HEAD_DIM)) && ok;
    H_KvTblPrep(C, false);
    ok = H_Guard(C, !KvPagedContractHolds(C.attnTbl.data(), 0u, H_KvTblElems(), pos, 0u, KV_LANE_K, KV_HEAD_DIM)) &&
         ok;
    // (b) device 见证：主 KV 平面清零后再启动（paged 写落在页 2、连续读读页 1 ⇒ 读到全零）
    aclrtMemset(C.attnKvDev, KV_PLANE_BYTES, 0, KV_PLANE_BYTES);
    H_KvLaunchProbe(C, pos, MODE_CONTRACT);
    if (!H_Sync(C, "kv_neg_paged")) {
        H_KvTblPrep(C, true);
        return false;
    }
    H_D2H(C, C.attnProbeOut.data(), C.attnProbeOutDev, OUT_BYTES, "kv_neg_paged");
    if (!H_Sync(C, "kv_neg_paged")) {
        H_KvTblPrep(C, true);
        return false;
    }
    ok = H_CmpMustDiffer(C, "Kv.neg.paged", C.attnProbeOut.data() + OUT_KV_OFF,
                         C.attnSeed.data() + SEED_KV_ELEM_OFF * ELEM_BYTES, KV_TOKEN_ALL_BYTES) && ok;
    H_KvTblPrep(C, true);   // 复位恒等表（本函数的副作用不外溢）
    return ok;
}

// ③/④ 主体：一组 pos 各跑一次探针 + 逐字节判据
static bool H_KvProbeSweep(Ctx& C)
{
    using namespace M15Kv;
    using namespace M15KvProbe;
    const uint32_t poses[] = {0u, 1u, 2u, 3u, 4u, 7u, 17u, 300u, 4095u, 4096u};
    bool ok = true;
    std::vector<uint8_t> zeros(COMP_ROW_BYTES, 0u);
    for (uint32_t pos : poses) {
        char tag[96];
        // 每次启动前把 compressed 平面清零 ⇒ "没写"这一档的期望值是无歧义的全零
        aclrtMemset(C.attnCompDev, COMP_PLANE_BYTES, 0, COMP_PLANE_BYTES);
        H_KvLaunchProbe(C, pos, MODE_CONTRACT);
        snprintf(tag, sizeof(tag), "kv_probe_pos%u", pos);
        if (!H_Sync(C, tag)) {
            return false;
        }
        H_D2H(C, C.attnProbeOut.data(), C.attnProbeOutDev, OUT_BYTES, tag);
        if (!H_Sync(C, tag)) {
            return false;
        }
        const uint8_t* kvGot = C.attnProbeOut.data() + OUT_KV_OFF;
        const uint8_t* ringGot = C.attnProbeOut.data() + OUT_RING_OFF;
        const uint8_t* compGot = C.attnProbeOut.data() + OUT_COMP_OFF;
        const int32_t* flag = reinterpret_cast<const int32_t*>(C.attnProbeOut.data() + OUT_FLAG_OFF);
        const uint8_t* packGot = C.attnProbeOut.data() + OUT_PACK_OFF;
        const bool expectWrite = CompRowWritten(pos);

        // (a) T-KV-PAGED：paged 写（两个 head 分别 512 B）→ 连续读，逐字节
        snprintf(tag, sizeof(tag), "Kv.kv.paged.L%u", pos);
        ok = H_CmpBytes(C, tag, kvGot, C.attnSeed.data() + SEED_KV_ELEM_OFF * ELEM_BYTES, KV_TOKEN_ALL_BYTES) && ok;
        // (b) raw ring 行（280 B，非 32 B 倍数 ⇒ 实跑见证 DataCopyPad 路径）
        snprintf(tag, sizeof(tag), "Kv.ring.row.L%u", pos);
        ok = H_CmpBytes(C, tag, ringGot, C.attnSeed.data() + SEED_RING_ELEM_OFF * ELEM_BYTES,
                        RING_ROW_BYTES) && ok;
        // (c) compressed 门控（off-by-one 契约）
        snprintf(tag, sizeof(tag), "Kv.comp.L%u", pos);
        ok = H_CmpBytes(C, tag, compGot,
                        expectWrite ? (C.attnSeed.data() + SEED_COMP_ELEM_OFF * ELEM_BYTES) : zeros.data(),
                        COMP_ROW_BYTES) && ok;
        // (d) packed 两行（8,208 B 行距，非 32 B 倍数）
        snprintf(tag, sizeof(tag), "Kv.pack.rows.L%u", pos);
        ok = H_CmpBytes(C, tag, packGot, C.attnPackSeed.data(), OUT_PACK_ROWS * PACK_ROW_BYTES) && ok;
        // (e) 门控标志（host 不靠猜）
        {
            C.checks++;
            const uint32_t g = CompGroupOf(pos);
            const bool flagOk = (flag[0] == (expectWrite ? 1 : 0)) && (flag[1] == static_cast<int32_t>(g)) &&
                                (flag[2] == static_cast<int32_t>(RingSlot(pos))) &&
                                (flag[3] == static_cast<int32_t>(MODE_CONTRACT));
            snprintf(tag, sizeof(tag), "Kv.flag.L%u", pos);
            if (flagOk) {
                printf("[m15]   %-30s PASS (gate=%u g=%u slot=%u mode=%u)\n", tag, (uint32_t)flag[0], g,
                       RingSlot(pos), MODE_CONTRACT);
            } else {
                printf("[m15]   %-30s FAIL (gate=%d g=%d slot=%d mode=%d；期望 gate=%u g=%u slot=%u)\n", tag,
                       flag[0], flag[1], flag[2], flag[3], expectWrite ? 1u : 0u, g, RingSlot(pos));
                C.fails++;
                ok = false;
            }
        }
    }
    return ok;
}

// ④ 负向对照之一：off-by-one 注入档下，**同一条判据必须 FAIL**
static bool H_KvNegOffByOne(Ctx& C)
{
    using namespace M15Kv;
    using namespace M15KvProbe;
    const uint32_t pos = 4096u;   // (4096+1)%4 = 1 ⇒ 开放组 ⇒ 契约档**不写**压缩行
    bool ok = true;
    ok = H_Guard(C, !CompRowWritten(pos)) && ok;
    ok = H_Guard(C, CompGroupOf(pos) == 1024u) && ok;
    const auto critHolds = [&](uint32_t mode, bool expectWrite) -> bool {
        aclrtMemset(C.attnCompDev, COMP_PLANE_BYTES, 0, COMP_PLANE_BYTES);
        H_KvLaunchProbe(C, pos, mode);
        if (!H_Sync(C, "kv_neg_offbyone")) {
            return false;
        }
        H_D2H(C, C.attnProbeOut.data(), C.attnProbeOutDev, OUT_BYTES, "kv_neg_offbyone");
        if (!H_Sync(C, "kv_neg_offbyone")) {
            return false;
        }
        const uint8_t* got = C.attnProbeOut.data() + OUT_COMP_OFF;
        const uint8_t* exp = expectWrite ? (C.attnSeed.data() + SEED_COMP_ELEM_OFF * ELEM_BYTES) : nullptr;
        if (expectWrite) {
            return __builtin_memcmp(got, exp, COMP_ROW_BYTES) == 0;
        }
        for (uint32_t i = 0; i < COMP_ROW_BYTES; ++i) {
            if (got[i] != 0u) {
                return false;
            }
        }
        return true;
    };
    const bool contractHolds = critHolds(MODE_CONTRACT, false);
    const bool brokenHolds = critHolds(MODE_BROKEN_OFFBYONE, false);
    ok = H_Guard(C, contractHolds) && ok;      // 契约档：pos 4096 不写压缩行
    ok = H_Guard(C, !brokenHolds) && ok;       // 注入档：同一条判据**必须**被打破
    printf("[m15]   Kv.neg.offbyone.L4096        %s（契约档判据 %s；注入档（pos 4096 也写压缩行）同一"
           "判据 %s ⇒ 判据对 off-by-one 有判别力）\n",
           (contractHolds && !brokenHolds) ? "PASS" : "FAIL", contractHolds ? "成立" : "不成立",
           brokenHolds ? "仍成立（**判据无判别力**）" : "被打破");
    // 「head 步长 16,384 ≠ 同 head 内 token 步长 1,024」的 device 见证：naive 地以为"head1 K 紧跟在
    // head0 K 后面 1,024 B"会落到**同一 head 的下一 token 的 K+V 槽**，而不是 head1 的 K
    // ⇒ 该处必须**不等于** seed 的 head1-K（官方几何下 head 步长 = N*C*2 = 16,384 B）。
    {
        const uint32_t pos2 = 17u;
        const uint32_t physBlk = KvPhysBlock(C.attnTbl.data(), 0u, H_KvTblElems(), KvBlockOf(pos2));
        const uint64_t naive = static_cast<uint64_t>(physBlk) * KV_BLOCK_BYTES +
                               KvInBlockOffset(KvSlotInBlock(pos2), 0u, KV_LANE_K, 0u) + KV_TOKEN_STRIDE;
        std::vector<uint8_t> naiveBuf(KV_HEAD_CONTENT_BYTES);
        H_D2H(C, naiveBuf.data(), reinterpret_cast<uint8_t*>(C.attnKvDev) + naive, KV_HEAD_CONTENT_BYTES,
              "kv_headsplit");
        H_Sync(C, "kv_headsplit");
        // seed 里 head1 的 K 在元素偏移 KV_HEAD_DIM（= 1 个 head 槽的 K 部分）
        ok = H_CmpMustDiffer(C, "Kv.lay.headsplit", naiveBuf.data(),
                             C.attnSeed.data() + (SEED_KV_ELEM_OFF + KV_HEAD_DIM) * ELEM_BYTES,
                             KV_HEAD_CONTENT_BYTES) && ok;
        // K 与 V 在同一 (head, slot) 槽内相距 512 B：这条距离本身也是判据（结构性 guard）
        ok = H_Guard(C, KvByteOffsetPhys(physBlk, KvSlotInBlock(pos2), 0u, KV_LANE_V, 0u) -
                             KvByteOffsetPhys(physBlk, KvSlotInBlock(pos2), 0u, KV_LANE_K, 0u) ==
                         KV_HEAD_CONTENT_BYTES) && ok;
    }
    return ok;
}

// ④ 负向对照之三（**M98 更正后的新判据**）：M82 的**旧主 KV 几何**必须被新判据否定
//
// 两条一起给：
//   (a) **host 算式**：旧式 `KvOldGeomByteOffset()` 与新式 `KvByteOffsetPhys()` 在同一批
//       (pos, head, kv, dim) 采样上**必须不相等**（若相等 ⇒ 说明"更正"没生效，判据变红）；
//   (b) **device 字节**：`MODE_BROKEN_OLDGEOM` 档用旧几何落页 ⇒ host 按**新几何**连续地址读回的
//       内容必须**不等于** seed（`Kv.neg.oldgeom.dev`）。这条同时见证"表/寻址真的走到官方几何上"。
static bool H_KvNegOldGeom(Ctx& C)
{
    using namespace M15Kv;
    using namespace M15KvProbe;
    const uint32_t pos = 17u;
    bool ok = true;
    uint32_t nDiff = 0u;
    for (uint32_t kv = 0u; kv < KV_LANES; ++kv) {
        for (uint32_t h = 0u; h < KV_HEADS; ++h) {
            for (uint32_t d = 0u; d < KV_HEAD_DIM; d += 37u) {
                const uint64_t a = KvByteOffsetContig(pos, h, kv, d);
                const uint64_t b = KvOldGeomByteOffset(pos, h, kv, d);
                if (a != b) {
                    nDiff++;
                }
            }
        }
    }
    ok = H_Guard(C, nDiff > 0u) && ok;
    printf("[m15]   Kv.neg.oldgeom              %s（旧式与新式在 %u 个采样点上不同；旧 = 页 %u B / "
           "token 步长 %u B / head 平面 %u B / 无 V，新 = 页 %u B / token 步长 %u B / head 步长 %u B / "
           "K‖V 同槽 +%u B）\n",
           (nDiff > 0u) ? "PASS" : "FAIL", nDiff, KV_OLD_BLOCK_BYTES, KV_OLD_TOKEN_STRIDE,
           KV_OLD_HEAD_PLANE_BYTES, KV_BLOCK_BYTES, KV_TOKEN_STRIDE, KV_HEAD_PLANE_BYTES, KV_HEAD_CONTENT_BYTES);
    if (nDiff == 0u) {
        C.fails++;
    }
    // (b) device 见证：旧几何落页 ⇒ 按新几何读回必须 ≠ seed
    aclrtMemset(C.attnKvDev, KV_PLANE_BYTES, 0, KV_PLANE_BYTES);
    H_KvLaunchProbe(C, pos, MODE_BROKEN_OLDGEOM);
    if (!H_Sync(C, "kv_neg_oldgeom")) {
        return false;
    }
    H_D2H(C, C.attnProbeOut.data(), C.attnProbeOutDev, OUT_BYTES, "kv_neg_oldgeom");
    if (!H_Sync(C, "kv_neg_oldgeom")) {
        return false;
    }
    ok = H_CmpMustDiffer(C, "Kv.neg.oldgeom.dev", C.attnProbeOut.data() + OUT_KV_OFF,
                         C.attnSeed.data() + SEED_KV_ELEM_OFF * ELEM_BYTES, KV_TOKEN_ALL_BYTES) && ok;
    return ok;
}

// ============================================================
// 验证 Kv 的总入口
// ============================================================
// M98：cache 填数学的段级判据（定义在本文件**末尾** include 的 `m15_attn_cache_host.h`）
namespace AcH {
static bool H_RunAttnCache(Ctx& C);
}

static bool H_RunKv(Ctx& C)
{
    using namespace M15Kv;
    printf("\n[m15] ===== 验证 Kv：attention 三套 cache + packed indices 的布局冻结（M82）=====\n");
    printf("[m15]   权威来源：docs/14:616-627（三套 cache）、docs/15:528-531（paged 契约现在就定死）、"
           "docs/17:286（pos 4096 属开放组）、m19_qsa_indexer/README.md:90-123（行为）\n");
    printf("[m15]   **基准交代**：本段判的是布局/寻址/门控契约；首期数值基准 = 自建稠密 causal 参考，"
           "**它不是官方行为**（官方 QSA 每 token 只 attend 约一半历史，docs/17:281）\n");
    const uint32_t checkBase = C.checks;
    const uint32_t guardBase = C.guards;
    bool ok = true;

    ok = H_KvPrep(C) && ok;
    H_KvTblPrep(C, true);
    ok = H_KvCapacityCheck(C) && ok;
    ok = H_KvAlignAndSlotCheck(C) && ok;
    ok = H_KvBoundaryRoundTrip(C) && ok;
    ok = H_KvProbeSweep(C) && ok;
    ok = H_KvNegOffByOne(C) && ok;
    ok = H_KvNegPaged(C) && ok;
    ok = H_KvNegOldGeom(C) && ok;
    // ---- M98：cache 填数学（本 mission 的主体交付；判据/负向对照见 m15_attn_cache_host.h）----
    ok = AcH::H_RunAttnCache(C) && ok;

    // ⚠ r1 复审 F4e：这段「覆盖交代」在 base（`9603c65`，M88 已合入）之后就是过时的 ——
    //   ① 已由 **M88** 实现（prolog），② 的**落页**已由 **M98** 的 Ac 段覆盖；这里按 tip 重写。
    printf("[m15]   覆盖交代（r2 重写）：**本段不覆盖** ① 主 KV 的 KV **向量来源**（`v_proj` 的输出由 "
           "prolog 段给：M88 已实现 prolog、M98 的 `Ac.mainkv.kv` 只判它的**落页字节**）；"
           "② 打分/topk/expand 与 packed 的**选择语义**（本段只判 packed 行的字节布局/行距）；"
           "③ 压缩行与主 KV 的 **paged 非恒等 block_table**（归 `Kv.neg.paged` / `T-KV-PAGED`）。\n");
    printf("[m15]   未决交代：raw ring 行宽 140 是**带时点的设计决定**（默认按规范宽，见 "
           "m15_attn_kv.h 的 D1）；「serving 栈是否逐字节要求 3 个 MRoPE 位置尾」未取证。\n");
    C.phaseKv = C.checks - checkBase;
    printf("[m15] 验证 Kv：本段判定项 %u 条 + guard %u 条（累计 checks %u / guards %u / fails %u）\n",
           C.phaseKv, C.guards - guardBase, C.checks, C.guards, C.fails);
    return ok;
}

// ============================================================
// M98：cache 填数学的段级判据 + 独立参考 + 负向对照（实现在 m15_attn_cache_host.h）
// 放在**末尾** include：该文件用到本文件上方已定义的 Ctx / H_Guard / H_CmpBytes / H_CmpMustDiffer 等工具。
// ============================================================
#include "m15_attn_cache_host.h"
