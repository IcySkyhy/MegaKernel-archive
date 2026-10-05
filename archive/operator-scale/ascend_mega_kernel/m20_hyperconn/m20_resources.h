/**
 * m20_resources.h —— M36 hyper-connection 融合 kernel 的**全局静态资源表**
 * （约束：docs/05-megakernel-design.md §5.2「内存不模块化，单一全局编译期资源表」
 *         §6.1「BufferID 每核 0-27 / cross core flagId 0-15 / 全部编译期静态分配」）
 *
 * 单位约定：**所有偏移一律字节**；UB/L1 偏移均已 32B 对齐（docs/05 §6.1 UB 对齐要求）。
 * 命名空间 M20 单一来源；kernel 与 host 共享本表，不出现第二份形状常量。
 *
 * 规格来源（官方 vLLM 实现，逐字对读，见 README §1 数学对照表）：
 *   /workspace/vllm/vllm/models/qwen4_exp/common/hyperconnection.py   （GatedResidual 数学）
 *   /workspace/vllm/vllm/models/qwen4_exp/nvidia/ops/hc.py            （实际运行时的 triton 核）
 *   /workspace/vllm/vllm/models/qwen4_exp/nvidia/model.py             （调用点 / 层序）
 *   checkpoint: /workspace/Qwen3.8-Flash-Next-MXFP4（实测张量形状见 README §1.2）
 */

#ifndef M20_RESOURCES_H
#define M20_RESOURCES_H

#include <cstdint>

namespace M20 {

// ============================================================
// §0.0 资源声明自查（口径 + 可复算命令；M69 加入）
//   口径：**每个资源/缓冲/槽/常量声明至少被 kernel 引用一次** —— 判据是在本目录下按
//         **整词**匹配计数：`grep -o "\b<符号>\b" m20_hyperconn.asc | wc -l` ≥ 1。
//         kernel 内 0 引用的声明必须在其声明处以「⚠ 未使用」标出并写明缘由，
//         **不得让它的注释读起来像一个在跑的设计**（M56 就是被这类注释误导过一次，
//         见 README §3 与 §7.3 第①条）。
//   复算（在 m20_hyperconn/ 下跑；本文件共 134 个 constexpr 常量 + 1 个 constexpr 函数 `WS_AlignUp`
//   ⇒ 下面这条命令按名字遍历 135 个（= 134 常量 + 1 函数），措辞与命令指向同一个集合）：
//     for s in $(grep -oP 'constexpr\s+\w+\s+\K\w+' m20_resources.h); do
//       printf '%-18s %s\n' "$s" "$(grep -o "\b$s\b" m20_hyperconn.asc | wc -l)"
//     done | awk '$2 == 0'
//   实测（M69 基线 `57f896c`，2026-09-26）：上面这条命令列出 36 行 —— 35 个 constexpr
//   常量 + 1 个 constexpr 函数 `WS_AlignUp`（`grep -v '^WS_AlignUp$'` 过滤后 35；
//   README §3 报的 36 就是含它的那一版）。其中「窗/缓冲/槽/常量」类 17 个 =
//     NGROUPS / MODE_COUNT / DN_KLOOP / UP_KLOOP                                   （4）
//     UB_PERSIST / UB_NORMF32 / UB_PERSIST_END / UB_PC_STAGE / UB_PC_END           （5，保留窗）
//     UB_PW_IJ / UB_PW_SIG / UB_S1_IW / UB_S4_ONE / UB_S6_AB                        （5，窗内空槽）
//     BUF_AIV_IW / BUF_AIV_GJ / BUF_AIV_AB                                          （3）
//   其余 19 个是窗边界/尺寸/偏移辅助（UB_W0_END..UB_W5_END、SZ_*、WS_ALIGN、
//   L1_BYTES_TOTAL / L0_PP_BYTES / UB_BYTES_TOTAL、WS_AlignUp）—— 用途就是在本文件内
//   算出下一个偏移。README §3 另有一栏口径 (b)「除自己声明行外是否还有引用」（M56 时点、当时那版
//   工具口径记 9 个；M69 加「⚠ 未使用」注释后旧口径报 2 个，M71 `f0cbc7d` 起剥注释后只数代码引用、
//   报 10 个 —— 只能用来看「有没有人引用过」）。两口径不矛盾：(a) 问 kernel 用没用，(b) 问文件内有没有人引用。
//   **正对照（先做，免得把恒 0 的哑扫描当成发现）**：同一命令形态对活符号会报非 0 ——
//   `UB_S2_WB` 1、`BUF_AIV_IJ` 4、`UB_IW_TAB` 2；凭空名字 `UB_NOT_EXIST` 报 0。
//   ⚠ `m15_layer_loop/m15_hc_resources.h` 是本表 §2 的一份**拷贝**（属 m15 的 scope），
//     本文件的改动不会传播过去。
// ============================================================

// ============================================================
// §0 形状常量（模型规格，改动会级联到全部 tile 常量）
// ============================================================
constexpr uint32_t HC = 4;        // hc_count（text_config.hc_count）
constexpr uint32_t HID = 2560;    // hidden_size（text_config.hidden_size）
constexpr uint32_t HYPER = HC * HID;   // 10240 = 多流残差流宽度 [..., HC*HS]（HS 内层）
constexpr uint32_t LOWRANK = 320;      // hc_lowrank
constexpr uint32_t M_MAX = 64;         // M 方向只做一个 tile（BASE_M=64）；m ∈ [1,64]

constexpr uint32_t INJ_N = 4;     // block_inject_weight 行数 = hc_count
// IJ 的物理行距：16 bf16 = 32 B（逻辑上只用前 4 列；32B 对齐便于 DataCopyPad）
constexpr uint32_t IJ_STRIDE = 16;
// injW 槽：每个 (token,stream) 一个 32B 槽（8 fp32，有效值在槽首）——BRC 广播要求地址 32B 对齐
constexpr uint32_t INJW_SLOT = 8;
constexpr uint32_t UB_IWTAB_SLOTS = M_MAX * HC;

constexpr float EPS = 1e-6f;      // rms_norm_eps（config.json / text_config.rms_norm_eps）

// ---- down(+inject) GEMM 形状：C[M,324] = xn[M,10240] · [Wdown;Winj]^T ----
constexpr uint32_t DN_K = HYPER;              // 10240
constexpr uint32_t DN_N = LOWRANK + INJ_N;    // 324（有效列）
// C（OH）的物理行距取 32B 对齐的最小上界：324 bf16 = 648 B 不是 32B 倍数，
// 故行距取 336 元素 = 672 B = 21×32B。列语义：
//   [0,320)   lora（input_mix_weight_up 的输入）
//   [320,324) injection logits（combine 用；vLLM 侧 shape [M,4]）
//   [324,336) padding（padding 列由硬件写入但无意义，判定时不比较）
constexpr uint32_t OH_W = 336;
constexpr uint32_t OH_LORA = 0;          // lora 列起点
constexpr uint32_t OH_INJ = LOWRANK;     // injection 列起点 = 320

// ---- up GEMM 形状：gate[M,10240] = lora_s[M,320] · Wup[10240,320]^T ----
constexpr uint32_t UP_K = LOWRANK;       // 320
constexpr uint32_t UP_N = HYPER;         // 10240

// ---- 标量派生量 ----
constexpr uint32_t VL_F32 = 64;                       // 256B / 4B = 64 lane（SIMD VL 固定 256B）
constexpr uint32_t CHUNK = VL_F32;                    // AIV 每个 item 处理 64 元素（= 1 VL）
constexpr uint32_t NCH_H = HID / CHUNK;               // 40：一路残差流内的 chunk 数
constexpr uint32_t NCH_D = HYPER / CHUNK;             // 160：整行 chunk 数（S1 的 item 网格）
constexpr uint32_t NCH_R = LOWRANK / CHUNK;           // 5：lowrank 行 chunk 数（S4）
constexpr uint32_t NGROUPS = HC;                      // 每个 token 的 norm 组数 = hc_count
static_assert(HYPER % CHUNK == 0, "HYPER must be a multiple of CHUNK");
static_assert(HID % CHUNK == 0, "HID must be a multiple of CHUNK");
static_assert(LOWRANK % CHUNK == 0, "LOWRANK must be a multiple of CHUNK");
static_assert(OH_W % 16 == 0, "OH row stride must be a multiple of 16 elements");
static_assert((OH_W * 2) % 32 == 0, "OH row stride must be 32B aligned");
static_assert((LOWRANK * 2) % 32 == 0, "LS row stride must be 32B aligned");
static_assert((HYPER * 2) % 32 == 0, "HYPER row stride must be 32B aligned");
static_assert((HID * 2) % 32 == 0, "HID row stride must be 32B aligned");
static_assert(((LOWRANK + INJ_N + 11) / 12) <= 32, "inject tile rows sanity");

// kernel 运行档（runtime 标量，非模板：mix kernel 内为统一分支，不产生设备端发散）
constexpr uint32_t MODE_MIX = 0;          // mix-only：第 0 层 attn（无 pending combine）
constexpr uint32_t MODE_COMBINE_MIX = 1;  // combine_and_mix：层内 attn/mlp（use_combine=true）
constexpr uint32_t MODE_FINAL_MIX = 2;    // 全局 hyper_connection_mixer（use_combine=false）
constexpr uint32_t MODE_COUNT = 3;

// ============================================================
// §1 AIC 侧：tile 常量 / L1 / L0 / BufferID
// ============================================================
constexpr uint32_t CUBE_BLOCK = 16;   // cube 分形边长（bf16: 16 元素 = 32B）
constexpr uint32_t BASE_M = M_MAX;    // 64：M 方向单 tile（m ≤ 64，尾块用 curM mask）
constexpr uint32_t BASE_K = 64;       // K 方向 tile；DN_K=10240=160×64、UP_K=320=5×64 均整除
constexpr uint32_t BASE_N = 160;      // N 方向 tile（继承 m11/m14；DN 3 tile / UP 64 tile）

constexpr uint32_t DN_KLOOP = DN_K / BASE_K;                       // 160
constexpr uint32_t DN_NTILES = (OH_W + BASE_N - 1) / BASE_N;       // 3（覆盖列 0..479）
constexpr uint32_t DN_LAST_NSIZE = OH_W - (DN_NTILES - 1) * BASE_N; // 16（32B 对齐写）
constexpr uint32_t UP_KLOOP = UP_K / BASE_K;                       // 5
constexpr uint32_t UP_NTILES = UP_N / BASE_N;                      // 64
static_assert(DN_K % BASE_K == 0, "DN_K must be divisible by BASE_K");
static_assert(UP_K % BASE_K == 0, "UP_K must be divisible by BASE_K");
static_assert(UP_N % BASE_N == 0, "UP_N must be divisible by BASE_N");
static_assert(DN_LAST_NSIZE % 16 == 0 && (DN_LAST_NSIZE * 2) % 32 == 0,
              "last down tile nSize must be 32B aligned");

// ---- L1 静态布局（字节偏移；A 区 [0,256KB)，B 区 [256KB,512KB)，与 m11/m14 同构）----
constexpr uint32_t L1_A_ELEMS = BASE_M * BASE_K;   // 4096 元素 = 8KB
constexpr uint32_t L1_B_ELEMS = BASE_N * BASE_K;   // 10240 元素 = 20KB
constexpr uint32_t L1_OFF_A0 = 0;
constexpr uint32_t L1_OFF_A1 = L1_OFF_A0 + L1_A_ELEMS * 2;
constexpr uint32_t L1_B_REGION = 256 * 1024;
constexpr uint32_t L1_OFF_B0 = L1_B_REGION;
constexpr uint32_t L1_OFF_B1 = L1_B_REGION + L1_B_ELEMS * 2;
constexpr uint32_t L1_BYTES_TOTAL = 512 * 1024;
static_assert(L1_B_REGION + 2 * L1_B_ELEMS * 2 <= L1_BYTES_TOTAL,
              "A/B ping-pong L1 footprint exceeds 512KB");

// ---- L0 静态布局（字节偏移；L0A/L0B 各 64KB，ping/pong 各 32KB）----
constexpr uint32_t L0_PP_BYTES = 32 * 1024;
constexpr uint32_t L0_OFF_0 = 0;
constexpr uint32_t L0_OFF_1 = L0_PP_BYTES;
static_assert(BASE_M * BASE_K * 2 <= L0_PP_BYTES, "A2 tile exceeds L0A half");   // 8KB
static_assert(BASE_K * BASE_N * 2 <= L0_PP_BYTES, "B2 tile exceeds L0B half");   // 20KB
static_assert(BASE_M * BASE_N * 4 <= 256 * 1024, "L0C tile exceeds 256KB");      // 40KB

// ---- AIC BufferID（MutexID 有效范围 0..27，编译期静态分配；.asc 侧 cast 成 MutexID）----
constexpr uint32_t BUF_AIC_A0 = 0;   // A1 ping: MTE2 -> MTE1
constexpr uint32_t BUF_AIC_A1 = 1;   // A1 pong
constexpr uint32_t BUF_AIC_B0 = 2;   // B1 ping: MTE2 -> MTE1
constexpr uint32_t BUF_AIC_B1 = 3;   // B1 pong
constexpr uint32_t BUF_AIC_L00 = 4;  // L0A/L0B ping: MTE1 -> M
constexpr uint32_t BUF_AIC_L01 = 5;  // L0A/L0B pong
constexpr uint32_t BUF_AIC_L0C = 6;  // L0C: M -> FIXP

// ============================================================
// §2 AIV 侧：UB 静态布局（248KB 可用）+ BufferID
//    分段窗**并列叠放**（不做跨窗覆盖），每窗自带 static_assert
//    ⚠ 区间口径（M69 标注；**本文件的值一个都没改**，只加注释，见下「保留窗」）：
//      [UB_PERSIST, UB_PC_END)     = [0, 61440)      保留窗，从不访问       61440 B
//      [UB_PC_END,  UB_W0_END)     = [61440, 61728)  旧 W0 staging，从不访问   288 B
//      [UB_W0_END,  UB_BYTES_USED) = [61728, 89472)  窗 W1..W5 跨度          27744 B
//                                                  └ 其中 320 B 是无引用槽
//      ⇒ 全文三个数都由此得出、互不矛盾：声明足迹 89472；窗口径（README §3）
//        89472−61440 = 28032；kernel 实际建 LocalTensor 覆盖 27744−320 = 27424。
// ============================================================
constexpr uint32_t UB_BYTES_TOTAL = 248 * 1024;

// ---- 保留窗 [0, 61440)：**声明保留、kernel 从不访问**（M69 标注，值未改动）----
// 这里是「hc_norm 整核一次性预转 fp32 常驻」那版设计的地址空间：UB_NORMF32 40960 B 的
// fp32 预转表 + UB_PC_STAGE 20480 B 的 bf16 原始行 staging。**那一版没有实现** —— 实际实现是
// NormStage 每 (token,stream) 组 DataCopy 读 bf16 5120 B 到 UB_S2_WB、再在 __VEC_SCOPE__ 内
// Cast（与 docs/14 §9.1「直接读 bf16」一致；口径见 README §3 UB 行与 §7.3 第①条）。
// 为什么保留而**不删**：删掉这五个声明就得把窗 W0 的基准从 UB_PC_END 改成 0，W0 之后所有窗的
// 偏移会整体 −61440 ⇒ **一个已验收 kernel 的静态布局会变**，而收益只是可读性（对正确性/性能
// 无影响）。按 tower 裁定「注释优先、不移动活地址」，此处只改注释。
constexpr uint32_t UB_PERSIST = 0;                           // ⚠ 未使用（保留窗起点）
constexpr uint32_t UB_NORMF32 = UB_PERSIST;                  // ⚠ 未使用（声明 40960 B：bf16→fp32 [HYPER]）
constexpr uint32_t UB_PERSIST_END = UB_NORMF32 + HYPER * 4;  // ⚠ 未使用（= 40960）
static_assert(UB_PERSIST_END % 32 == 0, "persist region must be 32B aligned");

constexpr uint32_t UB_PC_STAGE = UB_PERSIST_END;             // ⚠ 未使用（声明 20480 B：bf16 [HYPER] staging）
constexpr uint32_t UB_PC_END = UB_PC_STAGE + HYPER * 2;      // = 61440：保留窗末端 = 活区起点（窗 W0 的唯一基准）

// ---- 旧 W0 staging [UB_PC_END, UB_W0_END)：**kernel 从不访问**（M69 标注）----
// 这两槽属于「IJ 前导段落 UB、再由 MTE3 写 GM INJW」那版 W0；现实现把 W0 段（InjwStage：
// injW 全表由 V 自己算出并落盘）的缓冲放在**窗 W5**（UB_IW_IJ / UB_IW_TAB），本段空置。
// UB_W0_END 仍是**活锚点**：它是窗 W1 的起点。
constexpr uint32_t UB_PW_IJ = UB_PC_END;                     // ⚠ 未使用（旧 staging：16 bf16 = 32 B）
constexpr uint32_t UB_PW_SIG = UB_PW_IJ + 32;                // ⚠ 未使用（旧 staging：64 fp32 = 256 B）
constexpr uint32_t UB_W0_END = UB_PW_SIG + 256;              // 活：窗 W1 起点

// ---- 窗 W1：S1 combine（逐 chunk 元素级）----
constexpr uint32_t UB_S1_HB = UB_W0_END;                     // 64 bf16 = 128 B（h chunk）
constexpr uint32_t UB_S1_BOB = UB_S1_HB + 128;               // 64 bf16 = 128 B（bo chunk）
constexpr uint32_t UB_S1_OB = UB_S1_BOB + 128;               // 64 bf16 = 128 B（out chunk）
constexpr uint32_t UB_S1_IW = UB_S1_OB + 128;                // ⚠ 未使用（32 B）：已废弃路径的 injW BRC 单值槽
constexpr uint32_t UB_W1_END = UB_S1_IW + 32;

// ---- 窗 W2：S2 分组 GemmaRMSNorm（一个 (token,stream) 组常驻 UB）----
constexpr uint32_t UB_S2_XB = UB_W1_END;                     // H' 组 bf16 [HID] = 5120 B
constexpr uint32_t UB_S2_WB = UB_S2_XB + HID * 2;            // hc_norm 组 bf16 [HID] = 5120 B
constexpr uint32_t UB_S2_YB = UB_S2_WB + HID * 2;            // XN 组 bf16 [HID] = 5120 B
constexpr uint32_t UB_S2_RS = UB_S2_YB + HID * 2;            // 32 B（rstd 单值落点）
constexpr uint32_t UB_W2_END = UB_S2_RS + 32;

// ---- 窗 W3：S4 silu(lora/HC) ----
constexpr uint32_t UB_S4_LB = UB_W2_END;                     // 64 bf16 = 128 B（lora chunk）
constexpr uint32_t UB_S4_SB = UB_S4_LB + 128;                // 64 bf16 = 128 B（silu out）
constexpr uint32_t UB_S4_ONE = UB_S4_SB + 128;               // ⚠ 未使用（32 B）：1.0 在 __VEC_SCOPE__ 内 Duplicate
constexpr uint32_t UB_W3_END = UB_S4_ONE + 32;

// ---- 窗 W4：S6 gate mix ----
constexpr uint32_t UB_S6_GB = UB_W3_END;                     // 4×64 bf16 = 512 B（gate chunk，4 流并列）
constexpr uint32_t UB_S6_XB = UB_S6_GB + 512;                // 4×64 bf16 = 512 B（xn chunk，4 流并列）
constexpr uint32_t UB_S6_AB = UB_S6_XB + 512;                // ⚠ 未使用（256 B）：累加器是 V 内私有 RegTensor
constexpr uint32_t UB_S6_OB = UB_S6_AB + 256;                // 64 bf16 = 128 B（blk out）
constexpr uint32_t UB_W4_END = UB_S6_OB + 128;

// ---- 窗 W5：per-AIV 私有的 injW 全表 + IJ 全行 staging ----
// 设计说明（bring-up 实测教训）：S1 里若按 item 直接从 GM 取 injW 再 BRC 广播，
// 该 4B/32B 搬运与 V 侧 BRC 读之间的可见性在 3510 上不稳定（实测非确定性错读 → 部分 chunk
// 退化为 out = h）。改为「每 AIV 在 W0 段用 V 自己写出全表（V→V，S2 的 passA→passA2 已实证可靠），
// S1 再从 32B 对齐槽 BRC 广播」。**W0 段（InjwStage）用的就是本节这两个槽**，不是窗 W0。
constexpr uint32_t UB_IW_IJ = UB_W4_END;                     // IJ 全行 staging: M_MAX×IJ_STRIDE bf16
constexpr uint32_t UB_IW_TAB = UB_IW_IJ + M_MAX * IJ_STRIDE * 2;
constexpr uint32_t UB_W5_END = UB_IW_TAB + UB_IWTAB_SLOTS * INJW_SLOT * 4;
static_assert((M_MAX * IJ_STRIDE * 2) % 32 == 0, "IJ staging must be 32B aligned");
static_assert((UB_IWTAB_SLOTS * INJW_SLOT * 4) % 32 == 0, "injW table must be 32B aligned");

constexpr uint32_t UB_BYTES_USED = UB_W5_END;                // 89472 = **声明足迹**（含 61440 B 保留窗）
// ⚠ 口径：UB_BYTES_USED 是「声明到的最高地址」，**不是**工作集。要报工作集请用下面两个明确的量
// （都只由本文件的常量算出，README §3 引用的是第一个）：
//   窗口径      UB_BYTES_USED − UB_PC_END = 89472 − 61440 = 28032 B
//   实际引用口径 UB_BYTES_USED − UB_W0_END − 320（UB_S1_IW/UB_S4_ONE/UB_S6_AB）= 27424 B
static_assert(UB_W0_END <= UB_S1_HB, "W0/W1 overlap");
static_assert(UB_W1_END <= UB_S2_XB, "W1/W2 overlap");
static_assert(UB_W2_END <= UB_S4_LB, "W2/W3 overlap");
static_assert(UB_W3_END <= UB_S6_GB, "W3/W4 overlap");
static_assert(UB_W4_END <= UB_IW_IJ, "W4/W5 overlap");
static_assert(UB_BYTES_USED <= UB_BYTES_TOTAL, "UB footprint exceeds 248KB");
static_assert(UB_BYTES_USED % 32 == 0, "UB footprint must be 32B aligned");

// ---- AIV BufferID（每核核内 token，编译期静态分配；AIC 与 AIV 空间独立）----
// 说明：本 kernel 的 AIV 各段之间用 CrossCore mode-0 barrier 分隔（不用 BufferID 跨段交接），
// 段内 MTE2 → V → MTE3 的单缓冲区乒乓用 BufferID 表达「同核内 buffer 生命周期」。
// ⚠ M69 标注：3 / 8 / 13 三个 id **未被 kernel 引用**（逐符号 `grep -o "\bBUF_AIV_xx\b" |
//   wc -l` 为 0，复算见 §0.0），保留不复用（编号在本表里是一次性分配的）。
constexpr uint32_t BUF_AIV_H = 0;     // S1: h chunk MTE2 -> V
constexpr uint32_t BUF_AIV_BO = 1;    // S1: bo chunk MTE2 -> V
constexpr uint32_t BUF_AIV_OB = 2;    // S1: out chunk V -> MTE3
constexpr uint32_t BUF_AIV_IW = 3;    // ⚠ 未使用：README §4.1(b) 那条已废弃路径（S1 从 GM 取 injW 再 BRC）的 id
constexpr uint32_t BUF_AIV_IJ = 4;    // W0: IJ 32B MTE2 -> V
constexpr uint32_t BUF_AIV_SG = 5;    // W0: sigmoid 4 lane V -> MTE3
constexpr uint32_t BUF_AIV_X = 6;     // S2: H' 组 bf16 MTE2 -> V
constexpr uint32_t BUF_AIV_Y = 7;     // S2: XN 组 bf16 V -> MTE3
constexpr uint32_t BUF_AIV_WB = 15;   // S2: hc_norm 组 bf16 MTE2 -> V
constexpr uint32_t BUF_AIV_GJ = 8;    // ⚠ 未使用：那版未实现的「hc_norm 整核预转」的 id
constexpr uint32_t BUF_AIV_LB = 9;    // S4: lora chunk MTE2 -> V
constexpr uint32_t BUF_AIV_SB = 10;   // S4: silu chunk V -> MTE3
constexpr uint32_t BUF_AIV_GB = 11;   // S6: gate chunk MTE2 -> V
constexpr uint32_t BUF_AIV_XB = 12;   // S6: xn chunk MTE2 -> V
constexpr uint32_t BUF_AIV_AB = 13;   // ⚠ 未使用：S6 累加器是 V 内私有 RegTensor、无跨 pipe 交接 ⇒ 本就不需 BufferID
constexpr uint32_t BUF_AIV_OB2 = 14;  // S6: blk chunk V -> MTE3
constexpr uint32_t BUF_AIV_RS = 16;   // S2: rstd 4B V -> MTE3

// ============================================================
// §3 CrossCore flagId（0-15 全可用：本项目不用 SyncAll、不用 matmul 高阶 API）
//    规则：相邻同步点 flagId 必不同（本表全部一次性分配，不复用）；
//          set 挂生产 pipe（AIV: MTE3 / AIC: MTE2、FIX），wait 用尽可能窄的 pipe；
//          **pipe 类必须与核型匹配**（AIV: S/V/MTE2/MTE3；AIC: S/MTE1/MTE2/FIX/M）。
// ============================================================
constexpr uint8_t CC_MODE0 = 0x0;   // 同类型 all-to-all（全体 AIV 或全体 AIC）
constexpr uint8_t CC_MODE2 = 0x2;   // 一个 AIC ↔ 其配对 2 个 AIV

// AIV 段内 barrier（mode 0）
constexpr uint16_t FLAG_AV0 = 0;    // S1 → S2（H' 全量落盘）
constexpr uint16_t FLAG_AV1 = 1;    // S2 → AIC（XN 全量落盘）
constexpr uint16_t FLAG_AV2 = 2;    // S4 → AIC（LS 全量落盘）
// AIC 段内 barrier（mode 0）
constexpr uint16_t FLAG_AC0 = 4;    // down GEMM 前对齐
constexpr uint16_t FLAG_AC1 = 5;    // down GEMM 后（FIXP 写 GM 排空）
constexpr uint16_t FLAG_AC2 = 6;    // up GEMM 前对齐
constexpr uint16_t FLAG_AC3 = 7;    // up GEMM 后（FIXP 写 GM 排空）
// AIC ↔ AIV 交接（mode 2）
constexpr uint16_t FLAG_A2C_XN = 8;    // AIV→AIC：XN 就绪（set PIPE_MTE3 / wait PIPE_S）
constexpr uint16_t FLAG_C2A_OH = 9;    // AIC→AIV：OH（lora+inj）就绪（set PIPE_MTE2 / wait PIPE_MTE2）
constexpr uint16_t FLAG_A2C_LS = 10;   // AIV→AIC：LS 就绪
constexpr uint16_t FLAG_C2A_GATE = 11; // AIC→AIV：GATE 就绪

// ============================================================
// §4 GM workspace（外部由 host 分配并传入；偏移/尺寸全编译期常量）
//    每个张量的生命周期以注释标出（产 → 消费）
// ============================================================
constexpr uint32_t WS_ALIGN = 32;
constexpr uint32_t WS_AlignUp(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

// H' : S1 产 → S2 消费（同时是 M36 的输出「多流残差流」）
constexpr uint32_t SZ_HCP = M_MAX * HYPER * 2;
// XN : S2 产 → S3 GEMM(A) 消费 + S6 gate mix 消费
constexpr uint32_t SZ_XN = M_MAX * HYPER * 2;
// RSTD : S2 产（证据张量）
constexpr uint32_t SZ_RSTD = M_MAX * HC * 4;
// INJW : W0 产（证据张量，每 (token,stream) 一个 32B 槽）→ S1 在 UB 内直接用
constexpr uint32_t SZ_INJW = UB_IWTAB_SLOTS * INJW_SLOT * 4;
// OH : S3 产 → S4 消费（lora 段）+ 输出（injection 段）
constexpr uint32_t SZ_OH = M_MAX * OH_W * 2;
// LS : S4 产 → S5 GEMM(A) 消费
constexpr uint32_t SZ_LS = M_MAX * LOWRANK * 2;
// GATE : S5 产 → S6 消费
constexpr uint32_t SZ_GATE = M_MAX * UP_N * 2;
// BLK : S6 产（M36 的输出「block input」）
constexpr uint32_t SZ_BLK = M_MAX * HID * 2;

constexpr uint32_t WS_HCP = 0;
constexpr uint32_t WS_XN = WS_HCP + WS_AlignUp(SZ_HCP, WS_ALIGN);
constexpr uint32_t WS_RSTD = WS_XN + WS_AlignUp(SZ_XN, WS_ALIGN);
constexpr uint32_t WS_INJW = WS_RSTD + WS_AlignUp(SZ_RSTD, WS_ALIGN);
constexpr uint32_t WS_OH = WS_INJW + WS_AlignUp(SZ_INJW, WS_ALIGN);
constexpr uint32_t WS_LS = WS_OH + WS_AlignUp(SZ_OH, WS_ALIGN);
constexpr uint32_t WS_GATE = WS_LS + WS_AlignUp(SZ_LS, WS_ALIGN);
constexpr uint32_t WS_BLK = WS_GATE + WS_AlignUp(SZ_GATE, WS_ALIGN);
constexpr uint32_t WS_BYTES = WS_BLK + WS_AlignUp(SZ_BLK, WS_ALIGN);

}  // namespace M20

#endif  // M20_RESOURCES_H
