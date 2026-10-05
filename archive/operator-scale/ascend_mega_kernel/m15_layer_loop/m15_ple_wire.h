// ============================================================
// m15_ple_wire.h —— **机械生成物，勿手改**（M100）
// ============================================================
// 由 `m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py` 从 `m15_ple.asc` 逐字抽出
// `namespace M85P { … }`（M85 的 PLE device 段：①`IdsOneToken` / ②`PleGather` / ③`PleGemv` /
// ④`PleGateItem` / ⑤`PleConvItem` 与它们的常量、UB 布局、flag id）。
//
// **为什么是抽取而不是 `#include "m15_ple.asc"`**（实测，见 evidence/ple_wire/WITNESS.md §6）：
// 直接把 `m15_ple.asc` 借进层循环 TU（它自带 `main()`/匿名 namespace `Ctx`/system 头）会让
// `runs=all` 出现 36 条 `M.moews.*` FAIL（MoE 段 `WS_OFFSETS+32` 的 4 字节）；只抽 device 段
// 则 `runs=all` 干净。两份一致性由 `--check` 复跑保证。
//
// **包含前提**：本文件**不自带任何 include**（不引入 system 头、不引入 AscendC 头），
// 依赖 include 它的 TU 已经包含 `m15_hc_layer.h`（提供 `M15H::NormDonor` / `M15H::BarrierAiv` /
// `M15H::Block1` / cast traits / `M15H::CHUNK`）与 AscendC 头。层循环 TU 里的包含顺序是
// `m15_gdn_layer.h` → `m15_hc_layer.h` → `m15_ple_wire.h` → … 见该 TU 的 include 块。

namespace M85P {

using namespace AscendC;
using namespace AscendC::Reg;
// 复用既有件（人类裁决"优先从现有代码库抄改"）：m15_hc_layer.h 的向量 helper 与搬运参数
using M15H::Block1;
using M15H::ExtBlock1;
using M15H::CHUNK;
// M111：落盘通路的 buffer id（= `Mutex::Lock/Unlock` 的同一套 token，见 m15_hc_layer.h:58-67）
using M15H::BufAcquire;
using M15H::BufRelease;
namespace NormDonor = M15H::NormDonor;

// ============================================================
// §1 常量（语义权威 PLE_SPEC.md §4.1；checkpoint/config 实测）
// ============================================================
constexpr uint32_t HID = 2560;                  // hidden_size
constexpr uint32_t HC = 4;                      // hc_count
constexpr uint32_t HYPER = HID * HC;            // 10240 = 多流态宽度
constexpr uint32_t HE = 2560;                   // ple_embed_dim
constexpr uint32_t P = 8;                       // heads_per_ngram
constexpr uint32_t NGR = 3;                     // ngram_size
constexpr uint32_t NG = (NGR - 1) * P;          // 16 = 每 token 的 n-gram head 数
constexpr uint32_t HDIM = HE / NG;              // 160 = 每 head 的 embedding 维度
constexpr uint32_t NC = 2;                      // ngram_context_len = NGR-1
constexpr uint32_t KVW = HYPER + HID;           // 12800 = kv 投影输出宽
constexpr uint32_t KCONV = 4;                   // ple_conv_kernel_size
constexpr uint32_t DIL = NGR;                   // 3
constexpr uint32_t STLEN = (KCONV - 1) * DIL;   // 9 = short-conv 状态长度
constexpr float EPS = 1e-6f;                    // rms_norm_eps
constexpr int32_t EOS = 248044;
constexpr uint32_t FAIL_SLOTS = 128;            // 设备侧判据计数器的每核槽位数（host 侧按核求和）
constexpr uint32_t VL = 64;                     // 256B/4B：fp32 SIMD lane 数
constexpr uint32_t NCH_H = HID / VL;            // 40 = 一路 2560 的 chunk 数

// §FLAG：本 kernel 自己的 cross-core flag id（与 m15 主体无关：独立启动，互不并发）
// **M124 起 ②→③ 与 ③→④ 是**跨核型**的交接（② 在 AIV、③ 在 AIC、④ 回 AIV）⇒ 按
//   `docs/05` §6.1 / §2 的硬规矩，跨类型 all-to-all **不能**用 mode 0（"mode 0 仅同类型"），
//   标准组合是「AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）→ AIC→AIVs（mode 2）」。
//   ⇒ 这两条从「全体 AIV 的 mode-0 自封 barrier」改成 **mode-2 成对**（每个 AIC 只等/只通知
//   与它配对的那 2 条 AIV）。形态照 `m15_attn_prolog_probe.h:507-510` 与
//   `m15_gdn_layer.h:1758-1760` 的既定先例（`AP_AIC_M0_OUT → AP_A2V_GEMM`、`FLAG_AIC_SEG0B →
//   FLAG_BOUND_INPROJ`）。id 沿用 12/13 —— 层路径里 AIV 侧 mode-2 的相邻序是
//   hc(H1a) 的 8/9/10/11 → **12 → 13** → hc(H1b) 的 8/9/10/11，两两不同（m15_layer_kernel.h 有断言）。
constexpr uint16_t FLAG_B2 = 12;   // ② → ③（AIV→AIC，mode 2）
constexpr uint16_t FLAG_B3 = 13;   // ③ → ④（AIC→AIV，mode 2）
constexpr uint16_t FLAG_B4 = 14;   // ④ → ⑤（AIV 全体，mode 0，原样）
// AIC 侧 mode-0 的两条对齐点（③ 的内部）：进 mmad 前、Fixpipe 落 GM 后。
//   set/wait 的 pipe 只取 `IsSplitCubePipe = {S,MTE1,MTE2,FIX,M}` 里的（`docs/05` §6.2 的硬规则：
//   AIC 上挂 `PIPE_MTE3` 是**静默空操作** ⇒ 对侧永久挂死）；**set 侧还不能挂 `PIPE_S`**
//   （编译期实测：`ffts_cross_core_sync` 的 builtin 只收 `pipe ∈ [2,5] ∪ {10}`，见 `PleGemv` 的注释）。
//   **id 取 14/15**（不是 4/5）：`docs/05 §6.1` 明写「每核 16 个 flagId 是 mode0/mode2 **共享**的池」
//   ⇒ 挑号的判据是「在**本层**的 AIC 上，这个号不被另一 mode 占用」：4–7 是 GDN 的 AIC mode2、
//   0–3 是 hc 的 AIC mode0、8–11 是 hc 的 AIC mode2 与 MoE 的 AIC mode0、12–15 是 GDN 的 AIC mode0。
//   取 14/15 的代价是**同 mode 的先后复用**（GDN 的 phase A 在 PLE 之后用同一对号）—— 这正是
//   仓内既有的复用形态（hc 的 `FLAG_AC0..AC3` 在 H1/H2 两段各用一次），且每一段都 set/wait 配平
//   ⇒ 计数在段间回到 0。两种核型的相邻性逐对钉在 `m15_layer_kernel.h` 的 static_assert 里。
constexpr uint16_t FLAG_CUBE_IN = 14;   // ③ 的入口对齐：全体 AIC 都等到配对的 AIV 交完 ②
constexpr uint16_t FLAG_CUBE_OUT = 15;  // ③ 的出口对齐：Fixpipe 写 GM 排空 + 全体 AIC 到齐

// §BUF：本段自己的 buffer id（**核内** pipeline 的 token，不是 flag）。
//   CANN 9.1.0 的 `Mutex::Lock/Unlock<pipe>(id)` 就是 `GetBufInternal/RlsBufInternal<pipe,0>(id)`
//   ⇒ `MutexLock<PIPE_S>(X)` 与 `BufAcquire<PIPE_MTE3>(X)` 是**同一 id 上的同一套 token**
//   （`kernel_common.h:137-160`）。本段用它做「标量写 UB → MTE3 读」的归属交接。
//   id 取 20/21：避开 hc 段的 0–16、attn 段的 10–12 与 `MAX_MUTEXID = 27`（`kernel_event.h:695`）。
constexpr AscendC::MutexID BUF_PLE_IDS = static_cast<AscendC::MutexID>(20);   // ① ids 落盘暂存
constexpr AscendC::MutexID BUF_PLE_SINK = static_cast<AscendC::MutexID>(21);  // 设备侧计数落盘暂存
//
// ⚠ **成对纪律（血的教训，M111 r1 复审实测）**：写侧的 `BufAcquire<PIPE_S>(id)` 与
//   `BufRelease<PIPE_S>(id)` **必须成对**。只把 `PipeBarrier<PIPE_ALL>` 换成
//   `BufRelease<PIPE_S>(id)` 而**不配**写侧 `BufAcquire<PIPE_S>(id)` ⇒ **挂死**
//   （复审实测 `timeout 60` rc=124；同条件对照 8 s rc=0）。原因：token 是**跨 pipe 共享**的
//   一套计数，只 release 不 acquire 会把它推成不平衡状态，下一次 acquire 永远等不到。
//   ⇒ 本文件三处落盘一律是「写侧 acquire → 写 → 写侧 release」的完整三元组。

// ============================================================
// §CUBE：③ kv 投影的 cube 侧编译期资源（**M124**）
// ============================================================
// 人类口径（逐字）：「凡是涉及矩阵乘法的操作都需要用 mmad 实现，不管 M=1 或者是多大」+
//   「矩阵乘法 ⇒ 必须 cube」。③ 是 `kv = emb @ wcatᵀ`（M=n_tok、K=2560、N=12800，**固定权重矩阵**
//   `wcat`）⇒ 必须走 cube `Mmad`。
// **抄改自仓内既有件（不新发明）**：`m15_gdn_layer.h:Bf16Gemm`（M19/M22 的 bf16 GEMM）与
//   `m15_hc_layer.h:HcBf16Gemm`（同形，M36/M58）。两者的数学形态逐字是
//   **`C[m, N] = A[m, K] · B[N, K]ᵀ`，且 B 直接按权重的**原始 `[N, K]` 行主序**读** ——
//   与 `G.wcat` 的既有排布 `[12800, 2560]`（`GlobalTensor<bfloat16_t> wcat; // [12800,2560]`，
//   PLEW::W_CAT 的行主序 (out,in)）**逐字同形** ⇒ **不做任何在线权重布局转换 / 预转置**
//   （人类口径：尽量少做在线权重格式转换、尊重权重现状）。B 侧的 Nd2Nz 一次搬 BASE_N 行的
//   `[kb*BASE_K, kb*BASE_K+BASE_K)` 列段，行距 = HE —— 就是 donor 的 `bGMOri[kBlock*BASE_K + nBlock*BASE_N*K]`。
// **M 维**：一次 tile 吃下 n_tok ≤ BASE_M 行（decode 档 n_tok=1，独立 kernel 档 n_tok=2），
//   不按 token 拆 tile —— 一个 A 大包 + 一个 K 循环就出全部 token 的 kv 行。
// **3510 契约**（donor 注释逐字）：「Nd2Nz 行数为 1 时退化为 1D 拷贝、m=1 数据错位」⇒ 计算侧
//   统一抬到 ≥2 行（`calcM = max(curM, 2)`），只在 Fixpipe 用 `curM` mask。多读的那 1 行由 host
//   保证可读（独立 kernel 的 `emb` 平面按 +1 行分配；层路径的 `PLEW::S_EMB` 是 T_MAX=64 行的平面，
//   n_tok=1 时第 2 行落在自己平面内）。
// **舍入**：L0C 内 fp32 累加 → Fixpipe `QuantMode_t::F322BF16`（RNE）落 bf16，与改前
//   `Cast<bfloat16_t,float,NormDonor::castTraitB322B16>`（`CAST_RINT`）同方向；累加次序由
//   「40 个 chunk 的 Mul+Add + 64 lane 归约」变成「K=2560 的 mmad 内累加」⇒ 判据按 `docs/17` §1.1
//   的第 ③ 条触发条件（**含 mmad / cube 累加**）走 T3，界见 `m15_ple_check.py` 的 `EPS_MMAD` 推导。
constexpr uint32_t CUBE_BLOCK = 16;    // cube 分形边长（bf16: 16 元素 = 32B）
constexpr uint32_t BASE_M = 64;        // M 方向单 tile（= M15H::M_MAX；n_tok ≤ 64 由 host 契约保证）
constexpr uint32_t BASE_K = 64;        // K 方向 tile（HE = 40 × 64）
constexpr uint32_t BASE_N = 160;       // N 方向 tile（KVW = 80 × 160；继承 m11/m14/hc 的 BASE_N）
constexpr uint32_t KLOOP = HE / BASE_K;       // 40
constexpr uint32_t NTILES = KVW / BASE_N;     // 80
static_assert(HE % BASE_K == 0u, "HE must be divisible by BASE_K");
static_assert(KVW % BASE_N == 0u, "KVW must be divisible by BASE_N");
// ---- L1 静态布局（字节偏移；A 区 [0,256KB)、B 区 [256KB,512KB)，与 m11/m14/hc 同构）----
constexpr uint32_t L1_A_ELEMS = BASE_M * BASE_K;   // 4096 元素 = 8 KB
constexpr uint32_t L1_B_ELEMS = BASE_N * BASE_K;   // 10240 元素 = 20 KB
constexpr uint32_t L1_OFF_A0 = 0;
constexpr uint32_t L1_OFF_A1 = L1_OFF_A0 + L1_A_ELEMS * 2;
constexpr uint32_t L1_B_REGION = 256u * 1024u;
constexpr uint32_t L1_OFF_B0 = L1_B_REGION;
constexpr uint32_t L1_OFF_B1 = L1_B_REGION + L1_B_ELEMS * 2;
constexpr uint32_t L1_BYTES_TOTAL = 512u * 1024u;
static_assert(L1_B_REGION + 2u * L1_B_ELEMS * 2u <= L1_BYTES_TOTAL,
              "A/B ping-pong L1 footprint exceeds 512KB");
// ---- L0 静态布局（L0A/L0B 各 64KB，ping/pong 各 32KB；L0C 一 tile 40KB）----
constexpr uint32_t L0_PP_BYTES = 32u * 1024u;
constexpr uint32_t L0_OFF_0 = 0;
constexpr uint32_t L0_OFF_1 = L0_PP_BYTES;
static_assert(BASE_M * BASE_K * 2u <= L0_PP_BYTES, "A2 tile exceeds L0A half");
static_assert(BASE_K * BASE_N * 2u <= L0_PP_BYTES, "B2 tile exceeds L0B half");
static_assert(BASE_M * BASE_N * 4u <= 256u * 1024u, "L0C tile exceeds 256KB");
// ---- AIC BufferID：**复用 hc 段登记的那一套**（0–6）----
// 为什么可以复用：三个 cube 段（hc / GDN / PLE）在同一个核上**执行相位互斥**，且每段的
//   `Mutex::Lock/Unlock` 都成对配平 ⇒ 同一套 token 的计数在段间回到原值。层路径里
//   hc(H1a) 是 combine-only（AIC 不跑 GEMM）、PLE 段在它之后、hc(H1b)／GDN 在更后 ⇒ 无重叠。
//   （这也是 GDN 段自己的做法：`m15_gdn_layer.h` 把 `BUF_A0..BUF_L0C` 直接别名到 `M15G::BUF_AIC_A0..`，
//   而 hc 段用的就是同一组 0–6。）
using M15H::BUF_AIC_A0;
using M15H::BUF_AIC_A1;
using M15H::BUF_AIC_B0;
using M15H::BUF_AIC_B1;
using M15H::BUF_AIC_L00;
using M15H::BUF_AIC_L01;
using M15H::BUF_AIC_L0C;
using M15H::MutexLock;
using M15H::MutexUnlock;
using M15H::BarrierAic;

// 词表总行数（① 的值域判据、② 的输入值域守卫共用同一个常量；由 ple_sz 前缀和给出）
constexpr int64_t VOCAB_ROWS = 320001536LL;

// §UB 布局：**编译期静态分配**（本文件自己管理，不用 AscendC 资源管理）
//   M124 起 ③ 走 cube（L1/L0，见 §CUBE）⇒ 原来只有 ③ 用的 5 个 UB 槽
//   （`UB_EMB` / `UB_WR` / `UB_ACC` / `UB_KVF` / `UB_KVB`）已随之删除；②/④/⑤ 的槽一字未动。
constexpr uint32_t UB_ROW = 0;                        // bf16 [HDIM=160]      = 320 B   （② 行暂存）
constexpr uint32_t UB_KB = 0;                         // bf16 [HID]  ④ k 切片
constexpr uint32_t UB_QB = 5120;                      // bf16 [HID]  ④ q（= 多流态切片）
constexpr uint32_t UB_NKB = 10240;                    // bf16 [HID]
constexpr uint32_t UB_NQB = 15360;                    // bf16 [HID]
constexpr uint32_t UB_NCB = 20480;                    // bf16 [HID]
constexpr uint32_t UB_KNB = 25600;                    // bf16 [HID]  k_n
constexpr uint32_t UB_QNB = 30720;                    // bf16 [HID]  q_n
constexpr uint32_t UB_VB = 35840;                     // bf16 [HID]  v 广播源
constexpr uint32_t UB_GB = 40960;                     // bf16 [HID]  gated
constexpr uint32_t UB_NRB = 46080;                     // bf16 [HID]  normed
constexpr uint32_t UB_RS2 = 51200;                    // fp32 [8]    归约落点
constexpr uint32_t UB_CW = 0;                         // bf16 [4*VL]  ⑤ w 四 tap（tap-major）
constexpr uint32_t UB_CST = 512;                      // bf16 [9*VL]  ⑤ 旧状态 9 行
constexpr uint32_t UB_CIN = 1664;                     // bf16 [VL]    ⑤ 卷积输入（= normed 切片）
constexpr uint32_t UB_CG = 1792;                      // bf16 [VL]    ⑤ gated 切片
constexpr uint32_t UB_CX = 1920;                      // bf16 [VL]    ⑤ 多流态切片 / out 落点
constexpr uint32_t UB_NST = 2048;                     // bf16 [9*VL]  ⑤ 新状态落点
// M92 host-mapped ② 的设备侧校验（与上面各段不重叠；② 段与 ③④⑤ 段之间有 PipeBarrier 分隔）
constexpr uint32_t UB_HMCHK = 52224;                  // bf16 [VL]    窗口刚取到的行首 64 元素
constexpr uint32_t UB_HMEXP = 52352;                  // bf16 [VL]    host 直读分片文件得到的期望行首 64 元素
constexpr uint32_t UB_HMACCL = 52480;                 // fp32 [8]     设备侧行校验累计器（元素 0 = 累计错行数）
// M111：**落盘通路**的两块暂存（与上面各段不重叠；①/② 各自在一次调用里用完即走）
//   为什么必须有它：GM 标量 `SetValue` 走的不是 MTE3 ⇒ ① 与 ② 之间的相位边界（set 挂 PIPE_MTE3）
//   覆盖不到它；实测「只有写 ids 的那个核读到自己的值」（evidence/ple_wire/WITNESS.md §4）。
//   改成 UB → MTE3 `DataCopy/DataCopyPad` 后，落盘与其它输出同一条 DMA 通路；**核内次序**按
//   `docs/05 §6.1 ⓔ` 的标量条款：写侧 `BufAcquire<PIPE_S> → 写 → BufRelease<PIPE_S>`（mode=false）
//   → 搬侧 `BufAcquire<PIPE_MTE3> → DataCopy → BufRelease<PIPE_MTE3>`（见 §BUF）。
constexpr uint32_t UB_IDS = 52512;                    // int64 [NG=16] = 128 B  ① ids 落盘暂存
constexpr uint32_t UB_SINK = 52640;                   // uint32 [8]   = 32 B    设备侧计数落盘暂存
constexpr uint32_t UB_END = 52672;                    // 峰值占用（< 248 KB，见 m15_hc_resources.h:159）

// ============================================================
// §2 ① n-gram id（int64 位运算 + floor-mod + 偏移）
// ============================================================
// int64 取模：移位-减法实现（**回避设备端 int64 除法**；64 次迭代/元素）
__aicore__ inline uint64_t UMod64(uint64_t x, uint64_t d)
{
    uint64_t r = 0;
    for (int32_t i = 63; i >= 0; --i) {
        r = (r << 1) | ((x >> static_cast<uint32_t>(i)) & 1ULL);
        if (r >= d) {
            r -= d;
        }
    }
    return r;
}

// M171：`(商, 余)` 一次求出（移位-减法，同一 64 次迭代）—— 多槽窗口要按 `rows_per_shard`
//   把全局 id 拆成 `(shard, local)` 才能**跨分片**选槽（槽可横跨 2500012 的边界）。与 `UMod64`
//   同一套算法、只是顺带记商的每一位，故不需要第二遍 64 次循环。
__aicore__ inline void UDivMod64(uint64_t x, uint64_t d, uint64_t* q, uint64_t* r)
{
    const uint64_t dd = (d != 0ULL) ? d : 1ULL;
    uint64_t qq = 0, rr = 0;
    for (int32_t i = 63; i >= 0; --i) {
        rr = (rr << 1) | ((x >> static_cast<uint32_t>(i)) & 1ULL);
        if (rr >= dd) {
            rr -= dd;
            qq |= (1ULL << static_cast<uint32_t>(i));
        }
    }
    *q = qq;
    *r = rr;
}

// floor-mod（torch.remainder 语义：结果与除数同号 ⇒ 对正除数恒非负）
__aicore__ inline uint64_t FloorMod64(int64_t x, int64_t d)
{
    const uint64_t ux = static_cast<uint64_t>(x);
    const uint64_t mag = (x < 0) ? (~ux + 1ULL) : ux;
    uint64_t r = UMod64(mag, static_cast<uint64_t>(d));
    if ((x < 0) && (r != 0ULL)) {
        r = static_cast<uint64_t>(d) - r;
    }
    return r;
}

// 请求归属：qsl 单调，线性扫描（R 很小）
__aicore__ inline uint32_t ReqOf(int32_t t, GlobalTensor<int32_t>& qsl, uint32_t nReq)
{
    uint32_t r = 0;
    for (uint32_t i = 0; i + 1 <= nReq; ++i) {
        if (qsl.GetValue(i) <= t) {
            r = i;
        }
    }
    return r;
}

// 一个 token 的 16 个 head id 写入 outG[?]；idBuf 为 16 个 int64 的落点
//
// **M111：落盘通路 = UB → MTE3 `DataCopy`（不再用 GM 标量 `SetValue`）。**
//   为什么这是它该有的形态：① 的消费者是 **② 的别的核**（已接线路径上 ① 与 ② 在**同一次启动**
//   里，只隔一条相位边界 `M15L_PhaseBoundaryAiv`；边界 set 挂 `PIPE_MTE3`）。GM 标量走 scalar
//   pipe，**不在 MTE3 的 drain 覆盖内** ⇒ 除「自己写自己读」的那个核，其余核读到旧值
//   （逐 item 实测：g=0 对、g=1..15 全是同一个旧值，见 evidence/ple_wire/WITNESS.md §4）。
//
//   **落盘次序 = `docs/05 §6.1 ⓔ` 的标量条款形态（M111 r2 按复审 P1-1 改）**：
//     `BufAcquire<PIPE_S>(id)`（写侧成对，S pipe 取得槽）→ **标量写 UB** →
//     `BufRelease<PIPE_S>(id)`（= `RlsBufInternal<PIPE_S,false>`，mode=false = CANN `ASC_LOCK_BLOCK`
//     默认；M181 起与 acquire 侧同模式，此前为 `true`/drain）→ `BufAcquire<PIPE_MTE3>(id)`（搬侧等写侧的 release）→
//     `DataCopy` → `BufRelease<PIPE_MTE3>(id)`。
//   本仓**正对照**（塔已升格为舰队级）= `m15_hc_layer.h` 的 `HyperConnOp::CombineStage`
//   （纯 BufferID 的 V→MTE3 握手，每一步 release 都是 mode=false）。⚠ **不得**把
//   `m15_attn_cache.h` / `m15_attn_kv_probe.h` 里「写 UB → PipeBarrier<PIPE_ALL> → MTE3 搬」
//   当正确形态引用 —— `docs/05 §6.1 ⓔ` 明写这两处「既不是反例也不是先例」；同一条还逐字写着
//   「`PipeBarrier` 单独不足以建立跨 pipe 依赖」（`docs/06-m0-bringup.md` §5.2/§5.3 判它「不可靠」）。
//   ⇒ 此处**没有** `PipeBarrier<PIPE_ALL>` 承担可见性。
//   **循环里复用同一块 UB 的次序也由同一套 token 承担**：`BufAcquire<PIPE_S>` 等上一轮
//   `BufRelease<PIPE_MTE3>`（mode=false）⇒ 下一轮的标量写不会压到上一轮还没读完的内容。
__aicore__ inline uint32_t IdsOneToken(uint32_t t, GlobalTensor<int32_t>& idsG, GlobalTensor<int32_t>& qslG,
                                       GlobalTensor<int32_t>& ctxG, GlobalTensor<int64_t>& mG,
                                       GlobalTensor<int64_t>& szG, GlobalTensor<int64_t>& ofG,
                                       GlobalTensor<int64_t>& outG, uint32_t nReq, uint32_t negMask,
                                       uint32_t* rangeFail)
{
    LocalTensor<int64_t> idsL(TPosition::VECCALC, UB_IDS, NG);
    BufAcquire<PIPE_S>(BUF_PLE_IDS);   // 写侧成对（缺这一句会挂死，见文件头 §BUF 的陷阱说明）
    const uint32_t r = ReqOf(static_cast<int32_t>(t), qslG, nReq);
    const int32_t c = static_cast<int32_t>(t) - qslG.GetValue(r);
    const int64_t cur = static_cast<int64_t>(idsG.GetValue(t));

    // prev1 / prev2：lag = shift 的候选。chunk 内取 input_ids[t-shift]，跨 chunk 取 ngram_context
    // 的列 `ctx_col = NC - shift + c`（= 上游 triton `ops/ple.py:81`；ctx 列序 = [computed-2, computed-1]）。
    // ★ 注意 `ctx_col` **依赖 c**：c=1 时 lag-2 的候选是 ctx[1]（= computed-1），不是 ctx[0]。见 PLE_SPEC.md D3。
    const int32_t col1 = static_cast<int32_t>(NC) - 1 + c;   // shift=1（仅 c=0 时用）
    const int32_t col2 = static_cast<int32_t>(NC) - 2 + c;   // shift=2（c∈{0,1} 时用）
    // 负向对照 bit1：把 ctx 列退回「与 c 无关」的简化（= D3 的错法）
    const int32_t col1m = ((negMask & 2u) != 0u) ? static_cast<int32_t>(NC) - 1 : col1;
    const int32_t col2m = ((negMask & 2u) != 0u) ? static_cast<int32_t>(NC) - 2 : col2;
    const int64_t p1 = (c >= 1) ? static_cast<int64_t>(idsG.GetValue(t - 1))
                                : static_cast<int64_t>(ctxG.GetValue(r * NC + col1m));
    const int64_t p2raw = (c >= 2) ? static_cast<int64_t>(idsG.GetValue(t - 2))
                                   : static_cast<int64_t>(ctxG.GetValue(r * NC + col2m));
    // EOS 回退：从新到旧，一旦遇到 EOS，更老的候选全部换成 EOS
    const bool crossed1 = (p1 == static_cast<int64_t>(EOS));
    const int64_t p2 = crossed1 ? static_cast<int64_t>(EOS) : p2raw;

    const int64_t m0 = mG.GetValue(0);
    const int64_t m1 = mG.GetValue(1);
    const int64_t m2 = mG.GetValue(2);
    const int64_t base = cur * m0;                          // int64 回绕（与 triton 一致）
    const int64_t t1v = p1 * m1;
    const int64_t t2v = p2 * m2;

    uint32_t bad = 0;
    for (uint32_t g = 0; g < NG; ++g) {
        const uint32_t order = g / P + 2;                   // g<8 → bigram；g>=8 → trigram
        int64_t mixed = base;
        if (order > 1) {
            mixed ^= t1v;
        }
        if (order > 2) {
            mixed ^= t2v;
        }
        const int64_t size = szG.GetValue(g);
        const uint64_t r0 = FloorMod64(mixed, size);
        const int64_t off = ofG.GetValue(g);
        int64_t id;
        if ((negMask & 1024u) != 0u) {
            // 变异 bit10：**漏掉取模与偏移**（`id = mixed`）⇒ id 远超 [0,320001536)
            //   ⇒ 判据 B1.ids.A.range / B1.ids.B.range 必须 FAIL
            //   （注：`mixed < 0` 在本模型参数范围内不可达 —— 三项乘积都 < 2^63 ⇒ XOR 符号位恒 0，
            //     数值见 ple/README.md §6；所以值域判据的**上界**能这样演示，**下界**不能）
            id = mixed;
        } else {
            // 变异 bit0：把取模方向反过来（ceil-mod）。正常运行 negMask=0
            const uint64_t rr = ((negMask & 1u) != 0u)
                                    ? ((static_cast<uint64_t>(size) - r0) % static_cast<uint64_t>(size))
                                    : r0;
            id = static_cast<int64_t>(rr) + off;
        }
        idsL.SetValue(g, id);
        if ((id < 0) || (id >= VOCAB_ROWS)) {
            bad++;
        }
    }
    *rangeFail += bad;
    // ⓔ 的写侧 release（mode=false）（**不是** PipeBarrier：那一句不足以建立跨 pipe 可见性）
    BufRelease<PIPE_S>(BUF_PLE_IDS);
    BufAcquire<PIPE_MTE3>(BUF_PLE_IDS);   // 搬侧等写侧的 release
    AscendC::DataCopy(outG[static_cast<uint64_t>(t) * NG], idsL, Block1(NG * 8));
    BufRelease<PIPE_MTE3>(BUF_PLE_IDS);   // mode=false（跨 pipe 交接）
    return bad;
}

// ============================================================
// §3 ②③④⑤：PLE body（② 行 gather → ③ kv 投影 → ④ 门控 → ⑤ 卷积+残差+状态）
// ============================================================
struct BodyGm {
    GlobalTensor<int64_t> ids;
    GlobalTensor<bfloat16_t> table;
    GlobalTensor<bfloat16_t> wcat;    // [12800,2560] = [key ; value]（装载期融合，见 PLE_SPEC.md D1）
    GlobalTensor<bfloat16_t> wtap;    // [4,10240] conv 权重（tap-major，同 GDN 的 w[j][c]）
    GlobalTensor<bfloat16_t> nk;
    GlobalTensor<bfloat16_t> nq;
    GlobalTensor<bfloat16_t> ncw;
    GlobalTensor<bfloat16_t> hid;
    GlobalTensor<bfloat16_t> stIn;    // [slot][9][10240]
    GlobalTensor<bfloat16_t> stOut;
    GlobalTensor<int32_t> sidx;
    GlobalTensor<bfloat16_t> emb;
    GlobalTensor<bfloat16_t> kv;
    GlobalTensor<bfloat16_t> gated;
    GlobalTensor<bfloat16_t> normed;
    GlobalTensor<bfloat16_t> out;
    GlobalTensor<uint32_t> fail;
    // ---- M92：host-mapped 窗口模式（`winRows == 0` 时以下字段不使用，走旧的平坦 GM 表）----
    GlobalTensor<bfloat16_t> exp64;   // [nItems*VL] 每个 item 期望的行首 64 bf16（host 直读分片文件）
    GlobalTensor<float> hmBad;        // [FAIL_SLOTS*8] 设备侧计数（每核 1 槽：0=错行数 1=越窗行数 2=核号+1 3=winRows）
    uint32_t winBase = 0;             // 窗口第 0 行对应的全局行 id
    uint32_t winRows = 0;             // 窗口行数（0 = 关闭窗口模式）
    // ---- M171：多槽窗口池（`nSlots != 0` 时优先于 winBase/winRows）----
    //   每槽 4×u32：`shard / local_base / rows / row_off`；槽是**连续全局行区间**，可横跨分片边界。
    GlobalTensor<uint32_t> slotMeta;
    uint32_t nSlots = 0;
    uint32_t rowsPerShard = 0;        // 2500012：跨分片的 (shard, local) 分解用
};

// ---- ② 表行 gather（T1：行索引 + 行内容逐字节）----
// M92 扩展：`G.winRows != 0` 时 `G.table` 是 host-mapped 的注册窗口（平坦 `[winRows,160]` bf16），
// 行号 = `id - winBase`，越窗即计入设备侧计数；并对刚取到的行做**设备侧**校验（前 64 元素 vs 期望）。
//
// M171：`G.nSlots != 0` 时窗口是**多槽池**（`G.table` = `[pool_rows,160]`）。设备侧**槽选择**：
//   ① `UDivMod64(id, rowsPerShard)` 把全局 id 拆成 `(shard, local)` —— 这是**跨分片**的关键，
//      槽 0 可以横跨 `rowsPerShard` 的边界（shard 0 尾 → shard 1 头）；
//   ② 对每个槽 s（`shard_s/local_base_s/rows_s/row_off_s`）算
//      `delta = (shard - shard_s)*rowsPerShard + local - local_base_s`，`0 ≤ delta < rows_s` 即命中；
//      命中行 = `row_off_s + delta`；
//   ③ 一个槽都没命中 ⇒ 计 `miss`（真实行也可能落在池外）。
//   与 legacy 单窗口的区别：跨分片时 `row = id - winBase` 不再成立（winBase 只能锚在一个分片），
//   必须按 `(shard, local)` 分解，这就是 U-B 缺的那块算术。
__aicore__ inline void PleGather(uint32_t bid, uint32_t nblk, BodyGm& G, uint32_t nTok, uint32_t negMask)
{
    LocalTensor<bfloat16_t> rowL(TPosition::VECCALC, UB_ROW, HDIM);
    LocalTensor<bfloat16_t> chkL(TPosition::VECCALC, UB_HMCHK, VL);
    LocalTensor<bfloat16_t> expL(TPosition::VECCALC, UB_HMEXP, VL);
    LocalTensor<float> accL(TPosition::VECCALC, UB_HMACCL, 8);
    __ubuf__ bfloat16_t* chkUb = reinterpret_cast<__ubuf__ bfloat16_t*>(chkL.GetPhyAddr());
    __ubuf__ bfloat16_t* expUb = reinterpret_cast<__ubuf__ bfloat16_t*>(expL.GetPhyAddr());
    __ubuf__ float* accUb = reinterpret_cast<__ubuf__ float*>(accL.GetPhyAddr());
    const bool pool = (G.nSlots != 0u);           // M171 多槽池（优先于 legacy 单窗口）
    const bool win = (G.winRows != 0u) || pool;   // 两种窗口模式共用设备侧校验/计数
    const uint64_t winBase = static_cast<uint64_t>(G.winBase);
    const uint64_t winEnd = winBase + static_cast<uint64_t>(G.winRows);
    const uint64_t n = static_cast<uint64_t>(nTok) * NG;
    uint32_t miss = 0;
    uint32_t oob = 0;   // M111：② 看到的**词表外 id** 数（设备侧计数；0 是「真的核过」的读数）
    if (win) {
        // 累计器清零（设备侧判据的落点，最后整体走 DataCopy 落 GM —— **不用 SetValue**：
        // 实测 SetValue 的落点只对部分核可见，故设备侧计数一律走与其它输出同一条 DataCopy 通路）
        __VEC_SCOPE__
        {
            RegTensor<float> z, b, w;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(z, 0.0f, maskAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accUb + 0, z, maskOne);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accUb + 1, z, maskOne);
            Duplicate(b, static_cast<float>(bid) + 1.0f, maskAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accUb + 2, b, maskOne);
            Duplicate(w, static_cast<float>(G.winRows), maskAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accUb + 3, w, maskOne);
        }
        PipeBarrier<PIPE_ALL>();
    }
    for (uint64_t idx = bid; idx < n; idx += nblk) {
        const uint32_t t = static_cast<uint32_t>(idx / NG);
        const uint32_t g = static_cast<uint32_t>(idx - static_cast<uint64_t>(t) * NG);
        int64_t id = G.ids.GetValue(idx);
        // 变异 bit15：把 id 推离词表值域（`id += VOCAB_ROWS`）—— 只触发下面那条值域守卫并跳过
        //   该行（**不越界读表**）⇒ 判据 B_dev.fail 必须 FAIL（② 的设备侧计数是「真的在数」）
        if ((negMask & 32768u) != 0u) {
            id += VOCAB_ROWS;
        }
        // M111：② 自己的**输入值域守卫**（与 ① 的 `bad` 同口径、同常量）—— 词表外的 id 不拿去
        //   当行号（否则就是越界读表），而是计数。这个计数以前**根本不存在**（`PleGateItem`/
        //   `PleConvItem` 里只有 `(void)bad;` ⇒ `dev_fail` 恒 0、判据永不可能失败，见
        //   ple/REAL_TABLE.md §6.4 末段）；现在它是真的，落盘走下面同一条 UB → MTE3 通路。
        if ((id < 0) || (id >= VOCAB_ROWS)) {
            oob++;
            continue;
        }
        uint64_t row = 0;
        if (pool) {
            // ---- M171：多槽 + 跨分片选择。`rps` 是设备侧看到的 rows_per_shard ----
            // 变异 bit16（65536）：把 rows_per_shard 改错 ⇒ (shard, local) 分解错 ⇒ 对
            //   `shard_s != 0` 的槽 `delta` 整体偏移 ⇒ 取错行（Hd.row_fail / H2.emb 必红）。
            uint64_t rps = static_cast<uint64_t>(G.rowsPerShard);
            if ((negMask & 65536u) != 0u) {
                rps += 1ULL;
            }
            uint64_t shard = 0, local = 0;
            UDivMod64(static_cast<uint64_t>(id), rps, &shard, &local);
            bool found = false;
            for (uint32_t s = 0; s < G.nSlots; ++s) {
                const uint64_t ss = G.slotMeta.GetValue(static_cast<uint64_t>(s) * 4u + 0u);
                const uint64_t lb = G.slotMeta.GetValue(static_cast<uint64_t>(s) * 4u + 1u);
                const uint64_t rr = G.slotMeta.GetValue(static_cast<uint64_t>(s) * 4u + 2u);
                const uint64_t ro = G.slotMeta.GetValue(static_cast<uint64_t>(s) * 4u + 3u);
                if (shard < ss) {           // id 在该槽起点之前
                    continue;
                }
                const uint64_t base = (shard - ss) * rps + local;
                if (base < lb) {            // 同分片但在 local_base 之前
                    continue;
                }
                const uint64_t delta = base - lb;
                if (delta < rr) {           // 命中：可跨分片（rr 可越过一行 rows_per_shard 边界）
                    row = ro + delta;
                    found = true;
                    break;
                }
            }
            if (!found) {
                miss++;                     // 池外行计数（真实 id 也可能落在池外）
                continue;
            }
        } else if (win) {
            if ((id < 0) || (static_cast<uint64_t>(id) < winBase) || (static_cast<uint64_t>(id) >= winEnd)) {
                miss++;                       // 设备侧越窗行计数（真实 id 也可能落在窗口外）
                continue;
            }
            row = static_cast<uint64_t>(id) - winBase;
            // 变异 bit14：把「id → 窗口行号」的换算按**方向**改错（行号反转）
            //   ⇒ 设备侧行校验计数 hmBad > 0，且 host 的 T1 判据 H2.emb 必 FAIL
            if ((negMask & 16384u) != 0u) {
                row = static_cast<uint64_t>(G.winRows) - 1u - row;
            }
        } else {
            row = static_cast<uint64_t>(id);
        }
        // 变异 bit6：把 head 落点顺序反过来（行 id 与 head 槽位不匹配）
        const uint32_t gUse = ((negMask & 64u) != 0u) ? (NG - 1u - g) : g;
        AscendC::DataCopy(rowL, G.table[row * HDIM], Block1(HDIM * 2));
        PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(G.emb[static_cast<uint64_t>(t) * HE + static_cast<uint64_t>(gUse) * HDIM], rowL,
                          Block1(HDIM * 2));
        PipeBarrier<PIPE_ALL>();
        if (win) {
            // ★ **设备侧判据**（覆盖 M85 §7 U6）：kernel 自己把「刚取到的窗口行的前 64 个 bf16」与
            //   host 直读该分片文件得到的期望值逐元素比（期望值走独立 GM 缓冲，不经窗口）；
            //   不等的 item 记 1，累计器留在 UB，段末整体写 GM 由 host 读。
            AscendC::DataCopy(chkL, G.table[row * HDIM], Block1(VL * 2));
            AscendC::DataCopy(expL, G.exp64[idx * VL], Block1(VL * 2));
            PipeBarrier<PIPE_ALL>();
            __VEC_SCOPE__
            {
                RegTensor<float> a, b, d, sq, red, cnt, pf, one, zero;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
                MaskReg mGT;
                NormDonor::LoadRegForDtype<bfloat16_t>(chkUb, a, maskAll, 0);
                NormDonor::LoadRegForDtype<bfloat16_t>(expUb, b, maskAll, 0);
                Muls(b, b, -1.0f, maskAll);
                Add(d, a, b, maskAll);
                Mul(sq, d, d, maskAll);
                Reduce<ReduceType::SUM>(red, sq, maskAll);
                Duplicate(one, 1.0f, maskAll);
                Duplicate(zero, 0.0f, maskAll);
                Compares<float, CMPMODE::GT>(mGT, red, 0.0f, maskOne);
                Select(pf, one, zero, mGT);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(cnt, accUb + 0);
                Add(cnt, cnt, pf, maskOne);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accUb + 0, cnt, maskOne);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
    if (win) {
        // 设备侧计数一律经 UB → GM 的 DataCopy 落盘（与其它输出同一条通路）
        __VEC_SCOPE__
        {
            RegTensor<float> m;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(m, static_cast<float>(miss), maskAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accUb + 1, m, maskOne);
        }
        PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(G.hmBad[static_cast<uint64_t>(bid) * 8u], accL, Block1(32));
        PipeBarrier<PIPE_ALL>();
    }
    // M111：② 的值域坏 id 计数落盘 = `G.fail[bid]`（与 ① 的 `failG[bid]` 同形，但走 UB → MTE3
    //   `DataCopyPad`）。**无条件写**（0 也写）：「0」是设备真的核过一遍的读数，不是「没写」的默认值。
    //   ⓔ 的写侧成对（M111 r2 按复审 P1-1 改）：`BufAcquire<PIPE_S>` → 标量写 → **release**。
    {
        LocalTensor<uint32_t> sinkL(TPosition::VECCALC, UB_SINK, 8);
        BufAcquire<PIPE_S>(BUF_PLE_SINK);
        sinkL.SetValue(0, oob);
        BufRelease<PIPE_S>(BUF_PLE_SINK);     // release（mode=false）：与写侧 acquire 成对
        BufAcquire<PIPE_MTE3>(BUF_PLE_SINK);
        AscendC::DataCopyPad(G.fail[static_cast<uint64_t>(bid)], sinkL, ExtBlock1(4));
        BufRelease<PIPE_MTE3>(BUF_PLE_SINK);
    }
}

// ---- ③ 的 AIC 段：cube bf16 mmad（**M124**）----
//   `C[n_tok, KVW] = emb[n_tok, HE] · wcat[KVW, HE]ᵀ`；N 方向按 BASE_N 的 tile 跨 AIC 轮转条带
//   （`nt = aBid; nt < NTILES; nt += nAic`，与 m13/MoE/attn 的 N-tile 条带同款）。
//   函数体逐字抄改自 `m15_gdn_layer.h:Bf16Gemm::RunTile`（Nd2Nz / LoadData2D / Mmad / Fixpipe
//   四组参数与 ping-pong 的 BufferID 形态一致；只换 K→HE、N→KVW，并把 mBlock 折掉 —— 本段 M ≤ BASE_M）。
__aicore__ inline void PleCubeGemv(uint32_t aBid, uint32_t nAic, BodyGm& G, uint32_t nTok, uint32_t negMask)
{
    if (nTok == 0u) {
        return;   // 退化输入（无 token）：不写 kv 平面，也不下发 mmad/Fixpipe（`mSize=0` 无定义语义）
    }
    const uint32_t curM = (nTok < BASE_M) ? nTok : BASE_M;   // host 契约：n_tok ≤ BASE_M（层路径 T_MAX=64）
    // 3510 契约（同 donor）：Nd2Nz 行数 = 1 时退化为 1D 拷贝、m=1 数据错位 ⇒ 计算侧抬到 ≥2 行，
    //   只在 Fixpipe 用 curM mask 写出。多读的那 1 行必须**可读**（见 §CUBE 的说明）。
    const uint32_t calcM = (curM < 2u) ? 2u : curM;
    const uint32_t calcMAlign = M15H::AlignUp(calcM, CUBE_BLOCK);

    AscendC::LocalTensor<bfloat16_t> a1Ping(AscendC::TPosition::A1, L1_OFF_A0, L1_A_ELEMS);
    AscendC::LocalTensor<bfloat16_t> a1Pong(AscendC::TPosition::A1, L1_OFF_A1, L1_A_ELEMS);
    AscendC::LocalTensor<bfloat16_t> b1Ping(AscendC::TPosition::B1, L1_OFF_B0, L1_B_ELEMS);
    AscendC::LocalTensor<bfloat16_t> b1Pong(AscendC::TPosition::B1, L1_OFF_B1, L1_B_ELEMS);
    AscendC::LocalTensor<bfloat16_t> a2Ping(AscendC::TPosition::A2, L0_OFF_0, BASE_M * BASE_K);
    AscendC::LocalTensor<bfloat16_t> a2Pong(AscendC::TPosition::A2, L0_OFF_1, BASE_M * BASE_K);
    AscendC::LocalTensor<bfloat16_t> b2Ping(AscendC::TPosition::B2, L0_OFF_0, BASE_K * BASE_N);
    AscendC::LocalTensor<bfloat16_t> b2Pong(AscendC::TPosition::B2, L0_OFF_1, BASE_K * BASE_N);
    AscendC::LocalTensor<float> cL0C(AscendC::TPosition::CO1, 0, BASE_M * BASE_N);

    for (uint32_t nt = aBid; nt < NTILES; nt += nAic) {
        // 变异 bit7：key/value 两个输出块对调（= D1 的顺序反）。HID/BASE_N = 16 整除
        //   （HYPER = 10240 = 64 × 160、KVW = 12800 = 80 × 160）⇒ 一个 tile 不跨 key/value 边界，
        //   整块平移 ±16 个 tile 即可，与改前「逐列读对侧权重行、落到本列」等价。
        const uint32_t ntRead = ((negMask & 128u) != 0u)
                                    ? ((nt < (HYPER / BASE_N)) ? (nt + (HID / BASE_N))
                                                               : (nt - (HID / BASE_N)))
                                    : nt;
        // M 先取 L0C 所有权：挡住上一 tile 的 Fixpipe 读
        MutexLock<PIPE_M>(BUF_AIC_L0C);
        for (uint32_t kb = 0; kb < KLOOP; ++kb) {
            const uint32_t p = kb & 1u;
            const AscendC::MutexID bufA = p ? BUF_AIC_A1 : BUF_AIC_A0;
            const AscendC::MutexID bufB = p ? BUF_AIC_B1 : BUF_AIC_B0;
            const AscendC::MutexID bufL0 = p ? BUF_AIC_L01 : BUF_AIC_L00;
            AscendC::LocalTensor<bfloat16_t> a1 = p ? a1Pong : a1Ping;
            AscendC::LocalTensor<bfloat16_t> b1 = p ? b1Pong : b1Ping;
            AscendC::LocalTensor<bfloat16_t> a2 = p ? a2Pong : a2Ping;
            AscendC::LocalTensor<bfloat16_t> b2 = p ? b2Pong : b2Ping;

            // ---- MTE2：GM → L1（Nd2Nz），大包生产 ----
            MutexLock<PIPE_MTE2>(bufA);
            {
                AscendC::Nd2NzParams par = {};   // 全字段 NSDMI 零初始化后逐一赋值（M16 加固）
                par.ndNum = 1;
                par.nValue = calcM;              // 行数（≥2，见上）
                par.dValue = BASE_K;             // 每行元素数（bf16 按元素计）
                par.srcNdMatrixStride = 0;
                par.srcDValue = HE;              // emb 平面的行距 = HE
                par.dstNzC0Stride = BASE_M;      // L1 NZ 布局按满 tile 行距（与 LoadData 的 srcStride 配套）
                par.dstNzNStride = 1;
                par.dstNzMatrixStride = 0;
                AscendC::DataCopy(a1, G.emb[kb * BASE_K], par);
            }
            MutexUnlock<PIPE_MTE2>(bufA);
            MutexLock<PIPE_MTE2>(bufB);
            {
                AscendC::Nd2NzParams par = {};
                par.ndNum = 1;
                par.nValue = BASE_N;             // N 编译期整除 BASE_N，无尾块
                par.dValue = BASE_K;
                par.srcNdMatrixStride = 0;
                par.srcDValue = HE;              // wcat 的 [KVW, HE] 行主序：行距 = HE
                par.dstNzC0Stride = BASE_N;
                par.dstNzNStride = 1;
                par.dstNzMatrixStride = 0;
                AscendC::DataCopy(b1, G.wcat[ntRead * BASE_N * HE + kb * BASE_K], par);
            }
            MutexUnlock<PIPE_MTE2>(bufB);

            // ---- MTE1：L1 → L0（LoadData），消费 L1、生产 L0 ----
            MutexLock<PIPE_MTE1>(bufL0);
            MutexLock<PIPE_MTE1>(bufA);
            MutexLock<PIPE_MTE1>(bufB);
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
            MutexUnlock<PIPE_MTE1>(bufA);
            MutexUnlock<PIPE_MTE1>(bufB);
            MutexUnlock<PIPE_MTE1>(bufL0);

            // ---- M：Mmad 累加（L0C fp32），消费 L0 ----
            MutexLock<PIPE_M>(bufL0);
            AscendC::MmadParams mp = {};
            mp.m = calcM;
            mp.n = BASE_N;
            mp.k = BASE_K;
            mp.cmatrixInitVal = (kb == 0);
            AscendC::Mmad(cL0C, a2, b2, mp);
            MutexUnlock<PIPE_M>(bufL0);
        }
        MutexUnlock<PIPE_M>(BUF_AIC_L0C);

        // ---- FIXP：L0C → GM（F322BF16 = RNE，与改前 Cast 的 CAST_RINT 同方向）----
        MutexLock<PIPE_FIX>(BUF_AIC_L0C);
        {
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp = {};
            // 该结构体仅 reluScalar/vectorRelu/deqScalar 三个成员无 NSDMI，均显式清零（M16 加固）
            fp.nSize = BASE_N;
            fp.mSize = curM;                 // 只写出真有 token 的行（多读的行不落 GM）
            fp.srcStride = calcMAlign;
            fp.dstStride = KVW;              // kv 平面的行距
            fp.quantPre = QuantMode_t::F322BF16;
            fp.reluScalar = 0;
            fp.vectorRelu = 0;
            fp.deqScalar = 0;
            AscendC::Fixpipe(G.kv[static_cast<uint64_t>(nt) * BASE_N], cL0C, fp);
        }
        MutexUnlock<PIPE_FIX>(BUF_AIC_L0C);
    }
}

// ---- ③ kv 投影：kv = emb @ wcatᵀ（**cube mmad**，M124；改前是 AIV 逐列 GEMV）----
// 同一个函数被**两种核型**调用，各自只做自己那一半：
//   · **AIV**：不做任何计算 —— 只做核间握手（② 的 emb 落 GM 已完成 → 通知配对的 AIC；
//     再等配对 AIC 的 kv 落 GM）。② 的落盘是 MTE3 `DataCopy`，故 set 挂 `PIPE_MTE3`
//     （= 该 pipe 排空后放行，`docs/05` §6.1 ⓔ 的次序）；③ 的产物是 AIC 的 Fixpipe 写 GM，
//     故本侧的 wait 只用 mode-2 等配对 AIC、并挂 `PIPE_MTE2`（挡住随后的 ④ 的 DataCopy 读）。
//   · **AIC**：等配对 AIV 交完 ② → **全体 AIC 对齐**（这一步才是 all-to-all 的关键：
//     每个 AIC 都要读**全部** token 的 emb 行，而它只等到自己配对的 2 条 AIV）→ cube mmad →
//     **Fixpipe 排空 + 全体 AIC 对齐** → 通知配对 AIV。
// 形态先例：`m15_attn_prolog_probe.h:507-510`（`AP_AIC_M0_OUT` → `AP_A2V_GEMM`）与
//   `m15_gdn_layer.h:1758-1760`（`FLAG_AIC_SEG0B` → `FLAG_BOUND_INPROJ`）；`docs/05` §2 逐字：
//   「mode 0 仅同类型（'全部 AIC 之间'或'全部 AIV 之间'二选一，不支持 AIC 组 set、AIV 组 wait
//   的跨类型用法）；跨类型 all-to-all 的标准组合：AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）
//   → AIC→AIVs（mode 2）」。
__aicore__ inline void PleGemv(uint32_t bid, uint32_t nblk, BodyGm& G, uint32_t nTok, uint32_t negMask)
{
    if ASCEND_IS_AIV {
        // ② 已在本核落盘（`PleGather` 的 MTE3 DataCopy）⇒ set 挂 MTE3：排空后才放行配对的 AIC
        AscendC::CrossCoreSetFlag<M15H::CC_MODE2, PIPE_MTE3>(FLAG_B2);
        // 等配对 AIC 的 ③：wait 挂 MTE2 ⇒ 挡住 ④ 的第一步 DataCopy（读 G.kv）
        AscendC::CrossCoreWaitFlag<M15H::CC_MODE2, PIPE_MTE2>(FLAG_B3);
        // AIV 侧不再使用这两个形参（③ 的 N 条带改由 AIC 自己按 `GetBlockNum()` 轮转）
        (void)bid;
        (void)nblk;
        return;
    }
    if ASCEND_IS_AIC {
        // AIC 侧的 (bid, nblk) 由核自己取：同一个函数在 AIV 侧被调用时传的是 **AIV 的** bid/nblk
        //   （层路径的调用点就是这样），取形参会在两种核型上语义混淆。
        const uint32_t aBid = AscendC::GetBlockIdx();
        const uint32_t nAic = AscendC::GetBlockNum();
        // wait 后紧接 set（下面那条 mode-0 barrier）⇒ wait 挂 PIPE_S（docs/05 §2 的链式口径）
        AscendC::CrossCoreWaitFlag<M15H::CC_MODE2, PIPE_S>(FLAG_B2);
        // 全体 AIC 到齐 ⇒ 任一 AIC 都可以读任一条 AIV 写的 emb 行。
        // ⚠ **set 不能挂 PIPE_S**（编译期实测，M124）：`ffts_cross_core_sync` 的编译器 builtin
        //   （`__builtin_cce_ffts_cross_core_sync` → `kernel_operator_sync_impl.h` 的
        //   `NotifyEventImpl`）对**第 1 个实参（pipe）**的合法范围是 **`[2,5] ∪ {10}`**
        //   = {`PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_FIX`}
        //   （`cce_aicore_intrinsics.h` 的 `pipe_t`：`PIPE_S=0`/`PIPE_V=1`/`PIPE_M=2`/`PIPE_MTE1=3`/
        //   `PIPE_MTE2=4`/`PIPE_MTE3=5`/`PIPE_FIX=10`）—— 挂 `PIPE_S` 直接**编译不过**
        //   （报错逐字：`the ranges of 1st parameter must be [2, 5], [10, 10]`）。
        //   **wait 侧**没有这条限制（`wait_flag_dev` 那条路，GDN/hc 段大量 `CrossCoreWaitFlag<…, PIPE_S>`）。
        //   取 PIPE_MTE2 作 set：与 GDN 的「S2 in_proj **前**对齐」（`CrossCoreSetFlag<CC_MODE0,
        //   PIPE_MTE2>(FLAG_AIC_SEG0A)`，`m15_gdn_layer.h:1751`）同形；这一步只是对齐、本核没有
        //   需要排空的写。（`PIPE_MTE3` 虽在 builtin 的合法域内，但在 AIC 上是 `IsSplitCubePipe`
        //   之外的**静默空操作** ⇒ **绝不可用**，对侧会永久挂死 —— `docs/05` §6.2。）
        BarrierAic<PIPE_S, PIPE_MTE2, FLAG_CUBE_IN>();
        PleCubeGemv(aBid, nAic, G, nTok, negMask);
        // Fixpipe 写 GM 排空 + 全体 AIC 对齐（set 挂 PIPE_FIX = 排空本核的 FIXP 写；
        //   与 `m15_gdn_layer.h:1758` / `m15_attn_prolog_probe.h:507` 同形）
        BarrierAic<PIPE_S, PIPE_FIX, FLAG_CUBE_OUT>();
        AscendC::CrossCoreSetFlag<M15H::CC_MODE2, PIPE_MTE2>(FLAG_B3);   // 通知本核配对的 2 条 AIV
    }
}

// ---- ④ 门控 + 分组归一（逐 (token, stream)）----
__aicore__ inline void PleGateItem(uint32_t t, uint32_t s, BodyGm& G, uint32_t* bad, uint32_t negMask)
{
    LocalTensor<bfloat16_t> kL(TPosition::VECCALC, UB_KB, HID);
    LocalTensor<bfloat16_t> qL(TPosition::VECCALC, UB_QB, HID);
    LocalTensor<bfloat16_t> nkL(TPosition::VECCALC, UB_NKB, HID);
    LocalTensor<bfloat16_t> nqL(TPosition::VECCALC, UB_NQB, HID);
    LocalTensor<bfloat16_t> ncL(TPosition::VECCALC, UB_NCB, HID);
    LocalTensor<bfloat16_t> knL(TPosition::VECCALC, UB_KNB, HID);
    LocalTensor<bfloat16_t> qnL(TPosition::VECCALC, UB_QNB, HID);
    LocalTensor<bfloat16_t> vL(TPosition::VECCALC, UB_VB, HID);
    LocalTensor<bfloat16_t> gL(TPosition::VECCALC, UB_GB, HID);
    LocalTensor<bfloat16_t> nrL(TPosition::VECCALC, UB_NRB, HID);
    LocalTensor<float> rsL(TPosition::VECCALC, UB_RS2, 8);
    __ubuf__ bfloat16_t* kUb = reinterpret_cast<__ubuf__ bfloat16_t*>(kL.GetPhyAddr());
    __ubuf__ bfloat16_t* qUb = reinterpret_cast<__ubuf__ bfloat16_t*>(qL.GetPhyAddr());
    __ubuf__ bfloat16_t* nkUb = reinterpret_cast<__ubuf__ bfloat16_t*>(nkL.GetPhyAddr());
    __ubuf__ bfloat16_t* nqUb = reinterpret_cast<__ubuf__ bfloat16_t*>(nqL.GetPhyAddr());
    __ubuf__ bfloat16_t* ncUb = reinterpret_cast<__ubuf__ bfloat16_t*>(ncL.GetPhyAddr());
    __ubuf__ bfloat16_t* knUb = reinterpret_cast<__ubuf__ bfloat16_t*>(knL.GetPhyAddr());
    __ubuf__ bfloat16_t* qnUb = reinterpret_cast<__ubuf__ bfloat16_t*>(qnL.GetPhyAddr());
    __ubuf__ bfloat16_t* vUb = reinterpret_cast<__ubuf__ bfloat16_t*>(vL.GetPhyAddr());
    __ubuf__ bfloat16_t* gUb = reinterpret_cast<__ubuf__ bfloat16_t*>(gL.GetPhyAddr());
    __ubuf__ bfloat16_t* nrUb = reinterpret_cast<__ubuf__ bfloat16_t*>(nrL.GetPhyAddr());
    __ubuf__ float* rsUb = reinterpret_cast<__ubuf__ float*>(rsL.GetPhyAddr());

    const uint64_t kvBase = static_cast<uint64_t>(t) * KVW + static_cast<uint64_t>(s) * HID;
    const uint64_t hBase = static_cast<uint64_t>(t) * HYPER + static_cast<uint64_t>(s) * HID;
    AscendC::DataCopy(kL, G.kv[kvBase], Block1(HID * 2));                        // key
    AscendC::DataCopy(qL, G.hid[hBase], Block1(HID * 2));                        // query = 多流态
    // value 的 4 个 stream 共享同一段 [t, 0:2560]；变异 bit5 改成「各 stream 用自己那段 key」
    const uint64_t vOff = ((negMask & 32u) != 0u) ? kvBase
                                                  : (static_cast<uint64_t>(t) * KVW + HYPER);
    AscendC::DataCopy(vL, G.kv[vOff], Block1(HID * 2));
    AscendC::DataCopy(nkL, G.nk[s * HID], Block1(HID * 2));
    AscendC::DataCopy(nqL, G.nq[s * HID], Block1(HID * 2));
    AscendC::DataCopy(ncL, G.ncw[s * HID], Block1(HID * 2));
    PipeBarrier<PIPE_ALL>();

    // pass A：k / q 的平方和（bf16 值域内，fp32 累加）
    __VEC_SCOPE__
    {
        RegTensor<float> a1, a2, xr, t1;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> r1, r2;
        Duplicate(a1, 0.0f, maskAll);
        Duplicate(a2, 0.0f, maskAll);
        for (uint16_t c = 0; c < NCH_H; ++c) {
            NormDonor::LoadRegForDtype<bfloat16_t>(kUb, xr, maskAll, c * CHUNK);
            Mul(t1, xr, xr, maskAll);
            Add(a1, a1, t1, maskAll);
            NormDonor::LoadRegForDtype<bfloat16_t>(qUb, xr, maskAll, c * CHUNK);
            Mul(t1, xr, xr, maskAll);
            Add(a2, a2, t1, maskAll);
        }
        Reduce<ReduceType::SUM>(r1, a1, maskAll);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb + 0, r1, maskOne);
        Reduce<ReduceType::SUM>(r2, a2, maskAll);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb + 1, r2, maskOne);
    }
    PipeBarrier<PIPE_ALL>();
    // pass A2：rstd = NR rsqrt(mean + eps)
    __VEC_SCOPE__
    {
        RegTensor<float> v, rstd;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(v, rsUb + 0);
        Muls(v, v, 1.0f / static_cast<float>(HID), maskAll);
        NormDonor::ComputeRstdNewtonRaphsonReg(v, rstd, maskAll, EPS);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb + 2, rstd, maskOne);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(v, rsUb + 1);
        Muls(v, v, 1.0f / static_cast<float>(HID), maskAll);
        NormDonor::ComputeRstdNewtonRaphsonReg(v, rstd, maskAll, EPS);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb + 3, rstd, maskOne);
    }
    PipeBarrier<PIPE_ALL>();
    // pass B：k_n / q_n = bf16(x·rstd·(1+w))
    __VEC_SCOPE__
    {
        RegTensor<float> rk, rq, xr, wr, pr;
        RegTensor<bfloat16_t> yb;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(rk, rsUb + 2);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(rq, rsUb + 3);
        for (uint16_t c = 0; c < NCH_H; ++c) {
            NormDonor::LoadRegForDtype<bfloat16_t>(kUb, xr, maskAll, c * CHUNK);
            NormDonor::LoadRegForDtype<bfloat16_t>(nkUb, wr, maskAll, c * CHUNK);
            Mul(pr, xr, rk, maskAll);
            Mul(xr, pr, wr, maskAll);
            Add(pr, pr, xr, maskAll);
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, pr, maskAll);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(knUb + c * CHUNK, yb, maskAll);

            NormDonor::LoadRegForDtype<bfloat16_t>(qUb, xr, maskAll, c * CHUNK);
            NormDonor::LoadRegForDtype<bfloat16_t>(nqUb, wr, maskAll, c * CHUNK);
            Mul(pr, xr, rq, maskAll);
            Mul(xr, pr, wr, maskAll);
            Add(pr, pr, xr, maskAll);
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, pr, maskAll);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(qnUb + c * CHUNK, yb, maskAll);
        }
    }
    PipeBarrier<PIPE_ALL>();
    // pass C：dot = fp32( bf16( Σ bf16(k_n·q_n) ) )（**和先物化到 bf16 再进 pass D 的除法**，
    // 见 PLE_SPEC.md §4.1 ① / 上游 `ops/ple.py:218-219` 的 `tl.sum(...).to(dtype).to(tl.float32)`）
    __VEC_SCOPE__
    {
        RegTensor<float> acc, a, b, p, red;
        RegTensor<bfloat16_t> pb;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
        Duplicate(acc, 0.0f, maskAll);
        for (uint16_t c = 0; c < NCH_H; ++c) {
            NormDonor::LoadRegForDtype<bfloat16_t>(knUb, a, maskAll, c * CHUNK);
            NormDonor::LoadRegForDtype<bfloat16_t>(qnUb, b, maskAll, c * CHUNK);
            Mul(p, a, b, maskAll);
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(pb, p, maskAll);
            Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(p, pb, maskAll);
            Add(acc, acc, p, maskAll);
        }
        Reduce<ReduceType::SUM>(red, acc, maskAll);
        // ★ 和的 bf16 物化（变异 bit2 = 跳过这一取整 = M85 r1 的 P2 缺陷）
        if ((negMask & 4u) == 0u) {
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(pb, red, maskAll);
            Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(red, pb, maskAll);
        }
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb + 4, red, maskOne);
    }
    PipeBarrier<PIPE_ALL>();
    // pass D：门控标量 g = bf16(sigmoid(sign(d)·bf16(sqrt(max(|d|,1e-6)))))
    __VEC_SCOPE__
    {
        RegTensor<float> d, ab, ng, one, neg1, mag, sgn;
        RegTensor<bfloat16_t> db;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg mNeg, mPos;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(d, rsUb + 4);
        Muls(d, d, 1.0f / 50.59644256269407f, maskAll);   // 1/sqrt(2560)
        Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(db, d, maskAll);
        Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(d, db, maskAll);
        Compares<float, CMPMODE::LT>(mNeg, d, 0.0f, maskAll);
        Compares<float, CMPMODE::GT>(mPos, d, 0.0f, maskAll);
        Muls(ng, d, -1.0f, maskAll);
        Select(ab, ng, d, mNeg);                          // |d|
        Maxs(ab, ab, 1e-6f, maskAll);
        Sqrt(mag, ab, maskAll);
        Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(db, mag, maskAll);
        Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(mag, db, maskAll);
        Duplicate(sgn, 0.0f, maskAll);
        Duplicate(neg1, -1.0f, maskAll);
        Duplicate(one, 1.0f, maskAll);
        Select(sgn, neg1, sgn, mNeg);
        Select(sgn, one, sgn, mPos);
        Mul(d, sgn, mag, maskAll);
        Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(db, d, maskAll);
        Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(d, db, maskAll);
        NormDonor::SigmoidReg(d, d, one, maskAll);
        Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(db, d, maskAll);
        Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(d, db, maskAll);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb + 6, d, maskOne);
    }
    PipeBarrier<PIPE_ALL>();
    // pass E：gated = bf16(g·v)
    __VEC_SCOPE__
    {
        RegTensor<float> gg, vr, p;
        RegTensor<bfloat16_t> pb;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(gg, rsUb + 6);
        for (uint16_t c = 0; c < NCH_H; ++c) {
            NormDonor::LoadRegForDtype<bfloat16_t>(vUb, vr, maskAll, c * CHUNK);
            Mul(p, gg, vr, maskAll);
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(pb, p, maskAll);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(gUb + c * CHUNK, pb, maskAll);
        }
    }
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(G.gated[hBase], gL, Block1(HID * 2));
    PipeBarrier<PIPE_ALL>();
    // pass F：gated 的平方和 → rstd → normed = bf16(gated·rstd·(1+ncw))
    __VEC_SCOPE__
    {
        RegTensor<float> acc, xr, t1, red;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
        Duplicate(acc, 0.0f, maskAll);
        for (uint16_t c = 0; c < NCH_H; ++c) {
            NormDonor::LoadRegForDtype<bfloat16_t>(gUb, xr, maskAll, c * CHUNK);
            Mul(t1, xr, xr, maskAll);
            Add(acc, acc, t1, maskAll);
        }
        Reduce<ReduceType::SUM>(red, acc, maskAll);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb + 7, red, maskOne);
    }
    PipeBarrier<PIPE_ALL>();
    __VEC_SCOPE__
    {
        RegTensor<float> v, rstd;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(v, rsUb + 7);
        Muls(v, v, 1.0f / static_cast<float>(HID), maskAll);
        NormDonor::ComputeRstdNewtonRaphsonReg(v, rstd, maskAll, EPS);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rsUb + 5, rstd, maskOne);
    }
    PipeBarrier<PIPE_ALL>();
    __VEC_SCOPE__
    {
        RegTensor<float> rstd, xr, wr, pr;
        RegTensor<bfloat16_t> yb;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd, rsUb + 5);
        for (uint16_t c = 0; c < NCH_H; ++c) {
            NormDonor::LoadRegForDtype<bfloat16_t>(gUb, xr, maskAll, c * CHUNK);
            NormDonor::LoadRegForDtype<bfloat16_t>(ncUb, wr, maskAll, c * CHUNK);
            Mul(pr, xr, rstd, maskAll);
            Mul(xr, pr, wr, maskAll);
            Add(pr, pr, xr, maskAll);
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, pr, maskAll);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(nrUb + c * CHUNK, yb, maskAll);
        }
    }
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(G.normed[hBase], nrL, Block1(HID * 2));
    PipeBarrier<PIPE_ALL>();
    (void)bad;
}

// ---- ⑤ 膨胀深度卷积（lag 9/6/3/0）+ SiLU + 残差 + 状态移位 ----
__aicore__ inline void PleConvItem(uint32_t t, BodyGm& G, uint32_t* bad, uint32_t negMask)
{
    LocalTensor<bfloat16_t> wL(TPosition::VECCALC, UB_CW, KCONV * VL);
    LocalTensor<bfloat16_t> stL(TPosition::VECCALC, UB_CST, STLEN * VL);
    LocalTensor<bfloat16_t> cinL(TPosition::VECCALC, UB_CIN, VL);
    LocalTensor<bfloat16_t> gdL(TPosition::VECCALC, UB_CG, VL);
    LocalTensor<bfloat16_t> xL(TPosition::VECCALC, UB_CX, VL);
    LocalTensor<bfloat16_t> nbL(TPosition::VECCALC, UB_NST, STLEN * VL);
    __ubuf__ bfloat16_t* wUb = reinterpret_cast<__ubuf__ bfloat16_t*>(wL.GetPhyAddr());
    __ubuf__ bfloat16_t* stUb = reinterpret_cast<__ubuf__ bfloat16_t*>(stL.GetPhyAddr());
    __ubuf__ bfloat16_t* cinUb = reinterpret_cast<__ubuf__ bfloat16_t*>(cinL.GetPhyAddr());
    __ubuf__ bfloat16_t* gdUb = reinterpret_cast<__ubuf__ bfloat16_t*>(gdL.GetPhyAddr());
    __ubuf__ bfloat16_t* xUb = reinterpret_cast<__ubuf__ bfloat16_t*>(xL.GetPhyAddr());
    __ubuf__ bfloat16_t* nbUb = reinterpret_cast<__ubuf__ bfloat16_t*>(nbL.GetPhyAddr());
    (void)nbUb;

    const int32_t slot = G.sidx.GetValue(t);
    const bool nullSlot = (slot < 0);      // 非活跃行（上游 `ops/ple.py:20,391-405` 的 NULL_STATE_ID）
    const uint32_t bid = AscendC::GetBlockIdx();
    const uint32_t nblk = AscendC::GetBlockNum() * 2u;
    if (nullSlot) {
        // 上游：null 槽位 `out_ok=false` ⇒ conv_output = 0，但**仍然写** out
        // = bf16(outer_residual + bf16(residual + 0))，且**不写**状态行（state_ok=false）。
        // 变异 bit9 = 整行跳过（M85 r1 的旧行为，判据 B5.null 必须抓住）
        if ((negMask & 512u) != 0u) {
            return;
        }
        for (uint32_t c0 = bid * VL; c0 < HYPER; c0 += nblk * VL) {
            AscendC::DataCopy(gdL, G.gated[static_cast<uint64_t>(t) * HYPER + c0], Block1(VL * 2));
            AscendC::DataCopy(xL, G.hid[static_cast<uint64_t>(t) * HYPER + c0], Block1(VL * 2));
            PipeBarrier<PIPE_ALL>();
            __VEC_SCOPE__
            {
                RegTensor<float> gr, hr, pr;
                RegTensor<bfloat16_t> yb;
                MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
                NormDonor::LoadRegForDtype<bfloat16_t>(gdUb, gr, maskAll, 0);
                NormDonor::LoadRegForDtype<bfloat16_t>(xUb, hr, maskAll, 0);
                Add(pr, hr, gr, maskAll);                    // out = bf16(hidden + gated)
                Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, pr, maskAll);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(xUb, yb, maskAll);
            }
            PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(G.out[static_cast<uint64_t>(t) * HYPER + c0], xL, Block1(VL * 2));
            PipeBarrier<PIPE_ALL>();
            // 变异 bit11：null 行**也写状态**（把它当活跃行、写进 slot 0 的行 8）
            //   ⇒ 判据 B5.null.state 必须 FAIL
            if ((negMask & 2048u) != 0u) {
                AscendC::DataCopy(cinL, G.normed[static_cast<uint64_t>(t) * HYPER + c0], Block1(VL * 2));
                PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(G.stOut[static_cast<uint64_t>(STLEN - 1) * HYPER + c0], cinL,
                                  Block1(VL * 2));
                PipeBarrier<PIPE_ALL>();
            }
        }
        return;
    }
    for (uint32_t c0 = bid * VL; c0 < HYPER; c0 += nblk * VL) {
        const uint64_t stBase = static_cast<uint64_t>(slot) * STLEN * HYPER;
        for (uint32_t h = 0; h < STLEN; ++h) {
            AscendC::DataCopy(
                LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_CST + h * VL * 2, VL),
                G.stIn[stBase + static_cast<uint64_t>(h) * HYPER + c0], Block1(VL * 2));
        }
        AscendC::DataCopy(cinL, G.normed[static_cast<uint64_t>(t) * HYPER + c0], Block1(VL * 2));
        AscendC::DataCopy(gdL, G.gated[static_cast<uint64_t>(t) * HYPER + c0], Block1(VL * 2));
        AscendC::DataCopy(xL, G.hid[static_cast<uint64_t>(t) * HYPER + c0], Block1(VL * 2));
        for (uint32_t k = 0; k < KCONV; ++k) {
            AscendC::DataCopy(LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_CW + k * VL * 2, VL),
                              G.wtap[static_cast<uint64_t>(k) * HYPER + c0], Block1(VL * 2));
        }
        PipeBarrier<PIPE_ALL>();
        __VEC_SCOPE__
        {
            RegTensor<float> acc, wr, sr, pr;
            RegTensor<bfloat16_t> yb;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            // taps：h = 3k（k=0,1,2 读状态行 0/3/6；k=3 读当前卷积输入）
            // 变异 bit3：tap 归属反转（kt = KCONV-1-k），w 仍按 k
            for (uint16_t k = 0; k < KCONV; ++k) {
                const uint16_t kt = ((negMask & 8u) != 0u) ? static_cast<uint16_t>(KCONV - 1 - k) : k;
                NormDonor::LoadRegForDtype<bfloat16_t>(wUb, wr, maskAll, k * VL);
                if (kt < KCONV - 1) {
                    NormDonor::LoadRegForDtype<bfloat16_t>(stUb, sr, maskAll, kt * 3 * VL);
                } else {
                    NormDonor::LoadRegForDtype<bfloat16_t>(cinUb, sr, maskAll, 0);
                }
                Mul(pr, wr, sr, maskAll);
                if (k == 0) {
                    Duplicate(acc, 0.0f, maskAll);
                }
                Add(acc, acc, pr, maskAll);
            }
            // conv = bf16(acc)；y = conv·sigmoid(conv)（SiLU 在 bf16 取整后的值上算）
            RegTensor<float> one, cvt;
            Duplicate(one, 1.0f, maskAll);
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, acc, maskAll);
            Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(cvt, yb, maskAll);
            NormDonor::SigmoidReg(pr, cvt, one, maskAll);
            Mul(cvt, cvt, pr, maskAll);
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, cvt, maskAll);   // conv_output = bf16(y)
            // ple_output = bf16(gated + conv_output)  →  out = bf16(hidden + ple_output)
            // 变异 bit4：残差加的分组反过来（bf16(bf16(hidden+gated) + conv_output)）
            RegTensor<float> gr, cr, hr, qr;
            NormDonor::LoadRegForDtype<bfloat16_t>(gdUb, gr, maskAll, 0);
            // 变异 bit13：**跳过 conv_output 的 bf16 物化**（= D2 那个取整点本身）
            if ((negMask & 8192u) != 0u) {
                Adds(cr, cvt, 0.0f, maskAll);
            } else {
                Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(cr, yb, maskAll);
            }
            NormDonor::LoadRegForDtype<bfloat16_t>(xUb, hr, maskAll, 0);
            if ((negMask & 16u) != 0u) {
                Add(pr, hr, gr, maskAll);
                Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, pr, maskAll);
                Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(qr, yb, maskAll);
                Add(pr, qr, cr, maskAll);
            } else {
                Add(pr, gr, cr, maskAll);
                Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, pr, maskAll);
                Cast<float, bfloat16_t, NormDonor::castTraitB162B32>(qr, yb, maskAll);
                Add(pr, hr, qr, maskAll);
            }
            Cast<bfloat16_t, float, NormDonor::castTraitB322B16>(yb, pr, maskAll);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(xUb, yb, maskAll);
        }
        PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(G.out[static_cast<uint64_t>(t) * HYPER + c0], xL, Block1(VL * 2));
        PipeBarrier<PIPE_ALL>();
        // 状态移位：new[i] = old[i+1] (i<8)；new[8] = 当前卷积输入
        // 变异 bit8：移位方向反过来（new[i] = old[i-1]；new[0] = 当前卷积输入）
        // 变异 bit12：活跃行**完全不写回**状态 ⇒ 判据 B5.state_evolve 必须 FAIL
        if ((negMask & 4096u) == 0u) {
        if ((negMask & 256u) != 0u) {
            AscendC::DataCopy(LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_NST, VL), cinL, Block1(VL * 2));
            for (uint32_t i = 1; i < STLEN; ++i) {
                AscendC::DataCopy(LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_NST + i * VL * 2, VL),
                                  LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_CST + (i - 1) * VL * 2, VL),
                                  Block1(VL * 2));
            }
        } else {
            for (uint32_t i = 0; i < STLEN - 1; ++i) {
                AscendC::DataCopy(LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_NST + i * VL * 2, VL),
                                  LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_CST + (i + 1) * VL * 2, VL),
                                  Block1(VL * 2));
            }
            AscendC::DataCopy(LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_NST + (STLEN - 1) * VL * 2, VL),
                              cinL, Block1(VL * 2));
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t h = 0; h < STLEN; ++h) {
            AscendC::DataCopy(G.stOut[stBase + static_cast<uint64_t>(h) * HYPER + c0],
                              LocalTensor<bfloat16_t>(TPosition::VECCALC, UB_NST + h * VL * 2, VL),
                              Block1(VL * 2));
        }
        PipeBarrier<PIPE_ALL>();
        }
    }
    (void)bad;
}

}  // namespace M85P
