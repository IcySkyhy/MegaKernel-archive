// ============================================================
// m15_attn_layer.h —— full attention 层**占位直通**（待 QSA 替换）
//
// 背景（mission Context，用户原话）：层类型 3:1 交替（每 4 层中前 3 层 GDN、第 4 层 full
// attention = QSA）。QSA 内核（M24）未通 → 本 mission 用「占位直通」把层链先打通：
//   层输出 y = 层输入 x（逐位），即 attention 分支贡献 0，不引入任何状态。
//
// **替换点（改按符号/结构引用，不再引 README 的行号或章节号）**：
//   · 交付形态里本函数**只有一个调用点** = `m15_layer_kernel.h` 的 `M15L_FusedBody`（`KIND_ATTN`
//     分支的相位 A）；本文件末尾的 `__global__` 入口是**独立启动形态**，只用于 M25 基线与对拍。
//   · **⚠ 现状接口不足（M76 证伪，M82 复核）**：README 曾经写"真 QSA 接入时只替换本函数、
//     层循环与 MoE 段零改动" —— **该声明不成立**。现签名 `(xIn, yOut, bytes)` 与 `LayerArgs`
//     （`m15_layer_kernel.h`）**不带**层号、qkv/o/gate/indexer 权重指针、KV/raw ring/compressed/
//     packed 指针、`pos`/`slot_mapping`；且 **AIC 侧对 attention 完全缺席**（`M15L_FusedBody` 的
//     AIC 分支里没有任何 `KIND_ATTN` 路径）。真实接入要改的接口面清单见 README
//     **Part D · M82** 的 M82-8 与 M82-6。
//   · 布局（三套 cache + packed）已由 `m15_attn_kv.h` 冻结（M82）；本占位函数**不碰**那些 cache。
//
// 为什么核配置与 GDN 层一致（__mix__(1,2) + blockDim = AIC 数）：
//   Ver C 的基线是「48 次启动的 host 下发 vs 设备执行」，若占位层的核配置与真层不同（如
//   单 AIV 启动），基线的启动开销就不代表最终形态。故占位层与 GDN 层同款 mix(1,2) 启动。
//
// 数据面：AIV 侧把 [bytes] 从 GM 拷到 GM（GM→UB→GM），按 32B 块条带划分：
//   每 AIV 一个 512B 连续块（16 块），共 10 个 AIV 有活（H_BYTES=5120）。**每个有活 AIV
//   只有一次 MTE2 + 一次 MTE3**（无循环 → 无同 pipe 缓冲复用 → 不需要 PipeBarrier），
//   唯一需要的同步是 MTE2→MTE3 的跨 pipe 数据可见性 → drain BufferID（docs/05 §2 规则）。
//   每个 AIV 写的是互不相交的 GM 区间 → **不需要 CrossCore**（AIC 侧无活，直接返回）。
// ============================================================
#ifndef M15_ATTN_LAYER_H
#define M15_ATTN_LAYER_H

namespace M15Attn {
// UB 缓冲：与 GDN 层段窗同址（UB_M6）——两者是**不同的 kernel 启动**，UB 生命周期不共存，
// 故复用同一偏移（资源表 §4 的段窗起点，32KB 对齐）。
constexpr uint32_t UB_BUF = UB_M6;
constexpr uint32_t CHUNK_BLOCKS = 16;                  // 512B / AIV
constexpr uint32_t BYTES_PER_BLOCK = 32;               // 单次 DataCopy 粒度（Block1 的 32B 块）
constexpr uint32_t ELEMS_PER_BLOCK = BYTES_PER_BLOCK / 2;   // bf16 元素数
constexpr AscendC::MutexID BUF_ATTN = BUF_NORM_ROW;    // 单 token：MTE2 → MTE3
}  // namespace M15Attn

// 占位直通的**主体**（M40 起被融合 kernel 的 attention 相位内联调用；见 m15_layer_kernel.h）。
// 相对 M25 的唯一重构：把 kernel 主体抽成一个 inline 函数，`__global__` 入口只是它的薄壳 ——
// **数据面 / 同步 / BufferID 用法一字未动**（M25 的 Ver A/B/C 判据在融合前后可直接对照）。
__aicore__ inline void m15_attn_passthrough_body(__gm__ uint8_t* xIn, __gm__ uint8_t* yOut, uint32_t bytes)
{
    if ASCEND_IS_AIV {
        using namespace M15Attn;
        if ((bytes % BYTES_PER_BLOCK) != 0u) {
            AscendC::Trap();   // 非 32B 整数倍必须走 DataCopyPad，这里防呆
        }
        const uint32_t bid = AscendC::GetBlockIdx();        // 0 .. 2*numBlocks-1
        const uint32_t nAiv = AscendC::GetBlockNum() * 2;   // mix(1,2)：AIV 数 = 2×AIC 数
        (void)nAiv;
        const uint32_t nBlk = bytes / BYTES_PER_BLOCK;
        const uint32_t start = bid * CHUNK_BLOCKS;
        if (start < nBlk) {
            const uint32_t n = (nBlk - start < CHUNK_BLOCKS) ? (nBlk - start) : CHUNK_BLOCKS;
            AscendC::GlobalTensor<bfloat16_t> srcGm;
            AscendC::GlobalTensor<bfloat16_t> dstGm;
            srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xIn), nBlk * ELEMS_PER_BLOCK);
            dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(yOut), nBlk * ELEMS_PER_BLOCK);
            AscendC::LocalTensor<bfloat16_t> buf(TPosition::VECCALC, UB_BUF, n * ELEMS_PER_BLOCK);
            const uint64_t off = static_cast<uint64_t>(start) * ELEMS_PER_BLOCK;

            BufAcquire<PIPE_MTE2>(BUF_ATTN);
            DataCopy(buf, srcGm[off], Block1(n * BYTES_PER_BLOCK));
            BufRelease<PIPE_MTE2>(BUF_ATTN);    // drain：MTE2 排空后 token 才归 MTE3

            BufAcquire<PIPE_MTE3>(BUF_ATTN);
            DataCopy(dstGm[off], buf, Block1(n * BYTES_PER_BLOCK));
            BufRelease<PIPE_MTE3>(BUF_ATTN);
        }
    }
    // AIC 侧无活：真 attention 层的 QK^T / PV 落在这里（当前直接返回）
}

// 独立启动形态（仅供 m15 的「段间零串扰」对照与 M25 基线复跑；交付形态是融合 kernel）
__global__ __mix__(1, 2) void m15_attn_placeholder_kernel(__gm__ uint8_t* xIn, __gm__ uint8_t* yOut,
                                                          uint32_t bytes)
{
    AscendC::InitSocState();
    m15_attn_passthrough_body(xIn, yOut, bytes);
    AscendC::PipeBarrier<PIPE_ALL>();
}

#endif  // M15_ATTN_LAYER_H
