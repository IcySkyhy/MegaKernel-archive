// ============================================================
// m15_attn_kv_probe.h —— M82：**布局 / paged 契约 / off-by-one 门控的 device 探针**
//
// 它不是交付形态的 kernel，而是「把 `m15_attn_kv.h` 冻死的物理布局**在真机上跑一遍**」的最小
// 可执行判据源。每次启动针对一个 `pos`，做四件事：
//
//   1) 主 KV：按 **paged 地址**（`block_table[pos/16]` → 物理页）写 token 的 **2 head × (K‖V) = 4 个
//      512 B lane**（**四次独立传输**），再按**连续地址**读回 2,048 B 到 `out`。恒等表下两侧必须逐字节
//      相同（判据 T-KV-PAGED）；给非恒等表则必须不同（负向对照）；给 `MODE_BROKEN_OLDGEOM`
//      （M82 旧几何：页 16,384 / token 步长 512 / head 外层）则**必须不同**（`Kv.neg.oldgeom.dev`）。
//      **为什么必须逐 lane 传输**：官方页内 head 步长 16,384 B、同 head 内 token 步长 1,024 B，
//      而一个 (head, slot) 槽里的 K 与 V 相距 512 B（`flash_attn.py:1525-1526` 的
//      `(B,H,N,2*D) -> (B,N,H,D)` + K‖V）⇒ 一次 2,048 B 连续写会跨到别的 token/别的 head。这条由
//      host 的 `Kv.lay.headsplit` 见证（`H_CmpMustDiffer`：naive 落点必须 ≠ seed 的 head1-K）；另有**一条
//      未打标签的** `H_Guard` 断言同一 (head, slot) 槽内 V − K == `KV_HEAD_CONTENT_BYTES`（512 B）。
//      （r1 复审 F4a：旧注释里的 `Kv.lay.klane` 这个名字在本分支并不存在，已按实际实现改写。）
//   2) raw key ring：把 seed 的 140 元素写到第 `pos % 4` 行（**280 B，必须 DataCopyPad**）。
//   3) compressed 门控：`M15KV_COMP_ROW_WRITTEN(pos)` 为真才写第 `pos/4` 组的行（256 B）；
//      `mode = MODE_BROKEN_OFFBYONE` 时**无条件写**（负向对照；host 判据必须 FAIL）。
//      门控结果落 `out` 的标志区 —— host 不靠"猜"。
//   4) packed indices：写第 0/1 行（`[m,2052] int32`，行距 **8,208 B = 32 B 余 16** ⇒ 只能
//      DataCopyPad）。**内容由 host 给**（`packSeed`）⇒ 逐字节判据在 host 侧闭合；两行互异
//      ⇒ 行 stride 写错（例如误用 8,192）会当场红。
//
// **device 侧只用 `m15_attn_kv.h` §1b 的宏**（`__aicore__` 函数不能跨空间调用，见该节说明）。
//
// 为什么 UB 窗取 128 KB：本探针是**独立启动**（不与融合 kernel 共存），UB 用量独立记账；融合
// kernel 的段窗峰值在 64 KB 以内（`m15_layer_resources.h` §2），[128 KB, 138 KB) 与它不重叠且
// 在 248 KB 硬限内。
//
// 交付形态的边界（如实交代）：本探针**不做** mean/norm/RoPE —— 压缩行的**内容**（4 个 raw k 的
// 均值 → GemmaRMSNorm → RoPE@组首）属 attention prolog，本 mission 未实现。本探针冻结并实跑的是
// **寻址、传输、门控（off-by-one）、对齐约束**这四件与布局强耦合的事。
// ============================================================
#ifndef M15_ATTN_KV_PROBE_H
#define M15_ATTN_KV_PROBE_H

#include "m15_attn_kv.h"

namespace M15KvProbe {

using namespace M15Kv;

// ---- UB 独立窗（各段起点 32 B 对齐）----
constexpr uint32_t UB_PROBE = 131072u;                 // 128 KB
constexpr uint32_t UB_KV = UB_PROBE;                   // 2,048 B（2 head × (K 512 B + V 512 B)）
constexpr uint32_t UB_RING = UB_PROBE + 2048u;         // 288 B（140 bf16 = 280 B）
constexpr uint32_t UB_COMP = UB_PROBE + 2368u;         // 256 B
constexpr uint32_t UB_PACK = UB_PROBE + 2624u;         // 8,208 B
constexpr uint32_t UB_FLAG = UB_PROBE + 10848u;        // 16 B（4 × int32）
constexpr uint32_t UB_PROBE_BYTES = 10864u;
constexpr uint32_t UB_PROBE_LIMIT = 248u * 1024u;
static_assert(UB_PROBE + UB_PROBE_BYTES <= UB_PROBE_LIMIT, "探针 UB 窗必须落在 248 KB 硬限内");
static_assert(UB_KV % 32u == 0u && UB_RING % 32u == 0u && UB_COMP % 32u == 0u && UB_PACK % 32u == 0u &&
                  UB_FLAG % 32u == 0u,
              "UB 各段起点按 32 B 对齐（DataCopyPad 的 UB 侧要求）");

// ---- out 缓冲区段（host 按同一套偏移切片）----
constexpr uint32_t OUT_KV_OFF = 0;                     // 2,048 B：连续地址读回（2 head × (K512+V512)）
constexpr uint32_t OUT_RING_OFF = 2048;                // 280 B：ring 行整行读回
constexpr uint32_t OUT_COMP_OFF = 2368;                // 256 B：compressed 行读回
constexpr uint32_t OUT_FLAG_OFF = 2624;                // 16 B：{gateWritten, group, ringSlot, mode}
constexpr uint32_t OUT_PACK_OFF = 2656;                // 行 0/1 各 8,208 B
constexpr uint32_t OUT_PACK_ROWS = 2;
constexpr uint32_t OUT_BYTES = OUT_PACK_OFF + OUT_PACK_ROWS * PACK_ROW_BYTES;   // 19,072 B
static_assert(OUT_KV_OFF % 32u == 0u && OUT_RING_OFF % 32u == 0u && OUT_COMP_OFF % 32u == 0u &&
                  OUT_FLAG_OFF % 32u == 0u && OUT_PACK_OFF % 32u == 0u,
              "out 各区起点按 32 B 对齐");

// ---- seed 平面（输入；各区起点 32 B 对齐）----
constexpr uint32_t SEED_KV_ELEM_OFF = 0;               // 1,024 elems：主 KV 的 2 head × (K256+V256)
constexpr uint32_t SEED_RING_ELEM_OFF = 1040;          // 140 elems：raw key ring 一整行
constexpr uint32_t SEED_COMP_ELEM_OFF = 1184;          // 128 elems：compressed 行内容
constexpr uint32_t SEED_ELEMS = 1312;
constexpr uint32_t SEED_BYTES = SEED_ELEMS * ELEM_BYTES;   // 1,600 B
static_assert(SEED_RING_ELEM_OFF * ELEM_BYTES % 32u == 0u && SEED_COMP_ELEM_OFF * ELEM_BYTES % 32u == 0u,
              "seed 的 ring/comp 段起点按 32 B 对齐");

// ---- 档 ----
constexpr uint32_t MODE_CONTRACT = 0u;             // 契约档（正确）
constexpr uint32_t MODE_BROKEN_OFFBYONE = 1u;      // 负向对照：压缩行**无条件**写
constexpr uint32_t MODE_BROKEN_OLDGEOM = 2u;       // 负向对照：主 KV 用 **M82 旧几何**落页
                                                   //（页 16,384 / token 步长 512 / head 外层 / 无 V）
// 旧几何（**只作负向对照**；任何分配/编址都不使用）
constexpr uint32_t OLD_KV_PAGE_BYTES = 16384u;
constexpr uint32_t OLD_KV_HEAD_PLANE = 8192u;
constexpr uint32_t OLD_KV_TOKEN_STRIDE = 512u;
static_assert(OLD_KV_PAGE_BYTES != KV_BLOCK_BYTES, "旧几何与权威不同（这正是它作对照的用途）");

__aicore__ inline void m15_attn_kv_probe_body(__gm__ uint8_t* kv, __gm__ uint8_t* ring,
                                              __gm__ uint8_t* comp, __gm__ uint8_t* pack,
                                              __gm__ uint8_t* seed, __gm__ uint8_t* packSeed,
                                              __gm__ uint8_t* out, __gm__ int32_t* tbl, uint32_t tblElems,
                                              uint32_t tblStride, uint32_t pos, uint32_t mode)
{
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() != 0u) {
            return;   // 单核探针：AIV0 干活
        }
        const uint32_t logicBlk = M15KV_KV_BLOCK_OF(pos);
        const uint32_t slot = M15KV_KV_SLOT_OF(pos);
        const uint32_t g = M15KV_COMP_GROUP_OF(pos);

        AscendC::GlobalTensor<int32_t> tblGm;
        tblGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tbl), tblElems);
        AscendC::GlobalTensor<bfloat16_t> kvGm;
        kvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(kv), KV_LAYER_STRIDE / ELEM_BYTES);
        AscendC::GlobalTensor<bfloat16_t> ringGm;
        ringGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ring), RING_LAYER_BYTES / ELEM_BYTES);
        AscendC::GlobalTensor<bfloat16_t> compGm;
        compGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(comp), COMP_LAYER_STRIDE / ELEM_BYTES);
        AscendC::GlobalTensor<int32_t> packGm;
        packGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pack),
                               static_cast<uint64_t>(OUT_PACK_ROWS) * PACK_COLS);
        AscendC::GlobalTensor<bfloat16_t> seedKvGm;
        seedKvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed) + SEED_KV_ELEM_OFF, KV_TOKEN_ALL_ELEMS);
        AscendC::GlobalTensor<bfloat16_t> seedRingGm;
        seedRingGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed) + SEED_RING_ELEM_OFF,
                                   RING_HEAD_SIZE);
        AscendC::GlobalTensor<bfloat16_t> seedCompGm;
        seedCompGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed) + SEED_COMP_ELEM_OFF,
                                   COMP_HEAD_DIM);
        AscendC::GlobalTensor<int32_t> packSeedGm;
        packSeedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packSeed),
                                   static_cast<uint64_t>(OUT_PACK_ROWS) * PACK_COLS);
        AscendC::GlobalTensor<bfloat16_t> outKvGm;
        outKvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out + OUT_KV_OFF), KV_TOKEN_ALL_ELEMS);
        AscendC::GlobalTensor<bfloat16_t> outRingGm;
        outRingGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out + OUT_RING_OFF), RING_HEAD_SIZE);
        AscendC::GlobalTensor<bfloat16_t> outCompGm;
        outCompGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out + OUT_COMP_OFF), COMP_HEAD_DIM);
        AscendC::GlobalTensor<int32_t> outFlagGm;
        outFlagGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(out + OUT_FLAG_OFF), 4u);
        AscendC::GlobalTensor<int32_t> outPackGm;
        outPackGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(out + OUT_PACK_OFF),
                                  static_cast<uint64_t>(OUT_PACK_ROWS) * PACK_COLS);

        AscendC::LocalTensor<bfloat16_t> kvL(TPosition::VECCALC, UB_KV, KV_TOKEN_ALL_ELEMS);
        AscendC::LocalTensor<bfloat16_t> ringL(TPosition::VECCALC, UB_RING, RING_HEAD_SIZE);
        AscendC::LocalTensor<bfloat16_t> compL(TPosition::VECCALC, UB_COMP, COMP_HEAD_DIM);
        AscendC::LocalTensor<int32_t> packL(TPosition::VECCALC, UB_PACK, PACK_COLS);
        AscendC::LocalTensor<int32_t> flagL(TPosition::VECCALC, UB_FLAG, 4u);

        const AscendC::DataCopyPadExtParams<bfloat16_t> padB{false, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
        // 物理页号：**device 侧**读表项（下标算式来自 m15_attn_kv.h 的宏）
        const uint32_t physKv = static_cast<uint32_t>(tblGm.GetValue(M15KV_TBL_IDX(0u, tblStride, logicBlk)));
        const uint32_t physCmp = static_cast<uint32_t>(
            tblGm.GetValue(M15KV_TBL_IDX(0u, tblStride, g / COMP_ROWS_PER_BLOCK)));

        // ---- (1) seed → UB（主 KV 1,024 B、ring 行 280 B、comp 行 256 B）----
        BufAcquire<PIPE_MTE2>(BUF_NORM_ROW);
        DataCopyPad(kvL, seedKvGm[0], AscendC::DataCopyExtParams{1, KV_TOKEN_ALL_BYTES, 0, 0, 0}, padB);
        DataCopyPad(ringL, seedRingGm[0], AscendC::DataCopyExtParams{1, RING_ROW_BYTES, 0, 0, 0}, padB);
        DataCopyPad(compL, seedCompGm[0], AscendC::DataCopyExtParams{1, COMP_ROW_BYTES, 0, 0, 0}, padB);
        BufRelease<PIPE_MTE2>(BUF_NORM_ROW);   // drain：MTE2 排空后 UB 才归 MTE3

        // ---- (2) 写：主 KV 两个 head（paged 地址，两次独立传输）+ ring 行 ----
        BufAcquire<PIPE_MTE3>(BUF_NORM_ROW);
        for (uint32_t h = 0; h < KV_HEADS; ++h) {
            for (uint32_t kv = 0u; kv < KV_LANES; ++kv) {
                const uint64_t off = (mode == MODE_BROKEN_OLDGEOM)
                                         ? (static_cast<uint64_t>(physKv) * OLD_KV_PAGE_BYTES +
                                            static_cast<uint64_t>(h) * OLD_KV_HEAD_PLANE +
                                            static_cast<uint64_t>(slot) * OLD_KV_TOKEN_STRIDE +
                                            static_cast<uint64_t>(kv) * KV_HEAD_CONTENT_BYTES)
                                         : M15KV_KV_BYTE_OFF_PHYS(physKv, slot, h, kv, 0u);
                DataCopyPad(kvGm[static_cast<uint64_t>(off / ELEM_BYTES)],
                            kvL[(h * KV_LANES + kv) * KV_HEAD_DIM], M15H::ExtBlock1(KV_HEAD_CONTENT_BYTES));
            }
        }
        DataCopyPad(ringGm[static_cast<uint64_t>(M15KV_RING_ROW_OFF(0u, pos) / ELEM_BYTES)], ringL,
                    M15H::ExtBlock1(RING_ROW_BYTES));
        BufRelease<PIPE_MTE3>(BUF_NORM_ROW);

        // ---- (3) 读回：**连续地址**（恒等 block_table 时物理页 == 逻辑页）----
        BufAcquire<PIPE_MTE2>(BUF_NORM_ROW);
        for (uint32_t h = 0; h < KV_HEADS; ++h) {
            for (uint32_t kv = 0u; kv < KV_LANES; ++kv) {
                DataCopyPad(kvL[(h * KV_LANES + kv) * KV_HEAD_DIM],
                            kvGm[static_cast<uint64_t>(M15KV_KV_BYTE_OFF_CONTIG(pos, h, kv, 0u) / ELEM_BYTES)],
                            AscendC::DataCopyExtParams{1, KV_HEAD_CONTENT_BYTES, 0, 0, 0}, padB);
            }
        }
        DataCopyPad(ringL, ringGm[static_cast<uint64_t>(M15KV_RING_ROW_OFF(0u, pos) / ELEM_BYTES)],
                    AscendC::DataCopyExtParams{1, RING_ROW_BYTES, 0, 0, 0}, padB);
        BufRelease<PIPE_MTE2>(BUF_NORM_ROW);
        BufAcquire<PIPE_MTE3>(BUF_NORM_ROW);
        DataCopyPad(outKvGm[0], kvL, M15H::ExtBlock1(KV_TOKEN_ALL_BYTES));
        DataCopyPad(outRingGm[0], ringL, M15H::ExtBlock1(RING_ROW_BYTES));
        BufRelease<PIPE_MTE3>(BUF_NORM_ROW);

        // ---- (4) compressed：门控（off-by-one）→ 写 → 读回 ----
        const bool contractWrites = M15KV_COMP_ROW_WRITTEN(pos);
        const bool doWrite = (mode == MODE_BROKEN_OFFBYONE) ? true : contractWrites;
        const uint64_t compOff = M15KV_COMP_ROW_OFF_PHYS(physCmp, g);
        BufAcquire<PIPE_MTE3>(BUF_NORM_ROW);
        if (doWrite) {
            DataCopyPad(compGm[static_cast<uint64_t>(compOff / ELEM_BYTES)], compL,
                        M15H::ExtBlock1(COMP_ROW_BYTES));
        }
        BufRelease<PIPE_MTE3>(BUF_NORM_ROW);
        BufAcquire<PIPE_MTE2>(BUF_NORM_ROW);
        DataCopyPad(compL, compGm[static_cast<uint64_t>(compOff / ELEM_BYTES)],
                    AscendC::DataCopyExtParams{1, COMP_ROW_BYTES, 0, 0, 0}, padB);
        BufRelease<PIPE_MTE2>(BUF_NORM_ROW);
        BufAcquire<PIPE_MTE3>(BUF_NORM_ROW);
        DataCopyPad(outCompGm[0], compL, M15H::ExtBlock1(COMP_ROW_BYTES));
        BufRelease<PIPE_MTE3>(BUF_NORM_ROW);

        // ---- (5) packed indices：两行（行 stride = 8,208 B，非 32 B 整数倍）----
        for (uint32_t row = 0; row < OUT_PACK_ROWS; ++row) {
            BufAcquire<PIPE_MTE2>(BUF_NORM_ROW);
            DataCopyPad(packL, packSeedGm[static_cast<uint64_t>(row) * PACK_COLS],
                        AscendC::DataCopyExtParams{1, PACK_ROW_BYTES, 0, 0, 0}, padI);
            BufRelease<PIPE_MTE2>(BUF_NORM_ROW);
            BufAcquire<PIPE_MTE3>(BUF_NORM_ROW);
            DataCopyPad(packGm[static_cast<uint64_t>(row) * PACK_COLS], packL, M15H::ExtBlock1(PACK_ROW_BYTES));
            BufRelease<PIPE_MTE3>(BUF_NORM_ROW);
        }
        for (uint32_t row = 0; row < OUT_PACK_ROWS; ++row) {
            BufAcquire<PIPE_MTE2>(BUF_NORM_ROW);
            DataCopyPad(packL, packGm[static_cast<uint64_t>(row) * PACK_COLS],
                        AscendC::DataCopyExtParams{1, PACK_ROW_BYTES, 0, 0, 0}, padI);
            BufRelease<PIPE_MTE2>(BUF_NORM_ROW);
            BufAcquire<PIPE_MTE3>(BUF_NORM_ROW);
            DataCopyPad(outPackGm[static_cast<uint64_t>(row) * PACK_COLS], packL,
                        M15H::ExtBlock1(PACK_ROW_BYTES));
            BufRelease<PIPE_MTE3>(BUF_NORM_ROW);
        }

        // ---- (6) 门控标志（host 读它，不靠猜）----
        flagL.SetValue(0, doWrite ? 1 : 0);
        flagL.SetValue(1, static_cast<int32_t>(g));
        flagL.SetValue(2, static_cast<int32_t>(M15KV_RING_SLOT(pos)));
        flagL.SetValue(3, static_cast<int32_t>(mode));
        AscendC::PipeBarrier<PIPE_ALL>();   // 标量写 UB → MTE3 读：核内 drain（不是跨核同步）
        BufAcquire<PIPE_MTE3>(BUF_NORM_ROW);
        DataCopyPad(outFlagGm[0], flagL, M15H::ExtBlock1(4u * 4u));
        BufRelease<PIPE_MTE3>(BUF_NORM_ROW);
    }
}

__global__ __mix__(1, 2) void m15_attn_kv_probe_kernel(__gm__ uint8_t* kv, __gm__ uint8_t* ring,
                                                       __gm__ uint8_t* comp, __gm__ uint8_t* pack,
                                                       __gm__ uint8_t* seed, __gm__ uint8_t* packSeed,
                                                       __gm__ uint8_t* out, __gm__ int32_t* tbl,
                                                       uint32_t tblElems, uint32_t tblStride, uint32_t pos,
                                                       uint32_t mode)
{
    AscendC::InitSocState();
    m15_attn_kv_probe_body(kv, ring, comp, pack, seed, packSeed, out, tbl, tblElems, tblStride, pos, mode);
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace M15KvProbe

#endif  // M15_ATTN_KV_PROBE_H
