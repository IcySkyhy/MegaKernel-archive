# M101 · attention 核心段抽取（`m15_attn_core.h`）+ o_proj / ×sigmoid(gate) 段（`m15_attn_oproj.h`）

**状态（r3，2026-10-04）：复审 r1 的 5 条 p2 逐条闭环（§8）+ 塔裁的 B 方案落地（§2.4）+
复审 r2 的 3 条 p2 逐条闭环（§9）。第一～四步的读数在 r3 二进制（`ab12e2e4…`）上重跑，
见 §3.2/§3.3/§4.3。本 mission 只做「段」，接进层 kernel 交 M102。**
本文件记录：盘点（第一步）、抽取形态与生成器（第二步）、独立设备验证路与读数（第三步）、
o_proj 段的判据与读数（第四步）、**显式未完成项**（第五步）、复现命令（§6）、纪律自查（§7）、
以及 **r1 复审 5 条 p2 的逐条对账（§8）**。

纪律口径（本文件全程遵守，用户的硬约束逐字）：**不得写「已全部 / 无残留 / 0 命中」类绝对断言**；
每条写进仓库的 grep 计数/读数都在本提交的 tip 上可复跑，命令与输出列在 §6/§7。

---

## 1. 第一步：盘点（先于写码）

### 1.1 M10 的 device 段：内容锚点 / 常量 / UB 偏移 / flagId 分配

| 项 | `文件:符号` |
|---|---|
| donor 文件 | `m10_attn_decode/m10_attn_decode.asc`（只读；**一字未动**），sha256 `e9ddfbd85e0f0540…`（全长见 `m15_attn_core.h` 头部） |
| 抽取的内容锚点 | 顶格 `namespace m10 {`（asc 行 58）… 顶格 `}  // namespace m10`（asc 行 1005）。其后是 host 侧 `main()` 与数据集装载，**不抽取** |
| donor 的 kernel 入口（**不抽取**） | `m10_attn_decode.asc` → `m10_attn_decode_kernel`（`__global__ __mix__(1,2)`），函数体只有 `if ASCEND_IS_AIV { m10::M10Aiv op; op.Init(...); op.Run(); }` 与 AIC 同构 + `PipeBarrier<PIPE_ALL>` ⇒ 本项目把这段"分核调度"搬进生成物的 `M15AC::AttnCoreBody` |
| 形状/schedule 常量 | 段内 `GQA_N2=2`、`GQA_G=12`、`M_PAD=16`、`HD=256`、`S2T=256`、`SPLITS=14`、`AIV_ROWS=8`、`SCALE=0.0625`、`MASKV` |
| L1 静态布局 | 段内 `L1_OFF_P0/P1/P2`、`L1_OFF_Q`、`L1_OFF_KV0/KV1`、`L1_KV_BYTES`（+ `static_assert(L1_OFF_KV1 + L1_KV_BYTES <= 512*1024)`） |
| L0 静态布局 | 段内 `L0A_OFF_Q`、`L0A_OFF_P`、`L0B_BYTES`、`L0C_SLOT`（+ `static_assert(4*L0C_SLOT <= L0C_BYTES)`） |
| AIV UB 静态布局 | 段内 `UB_MM0/UB_MM1/UB_PC/UB_ACC/UB_MASK/UB_MB/UB_MASKZ/UB_MVEC/UB_SVEC/UB_TVEC/UB_WMAT/UB_MMAT/UB_SMAT/UB_ACCR/UB_TMPR/UB_NUMR/UB_OUTR/UB_CVF_M/S/W`、`UB_TOTAL2` |
| AIC BufferID | 段内 `B_Q(0) / B_KV0(1) / B_KV1(2) / B_L0(3) / B_C0..B_C3(4..7) / B_PL1(8)` |
| AIV BufferID | 段内 `B_PC(0) / B_ACC(1) / B_MSK(2) / B_CIN(3) / B_COUT(4) / B_ACCIN(5) / B_DBG(6)` |
| cross-core flagId | 段内 `CC_MM0(0) / CC_MM1(1)`（mode 4）、`CC_P0(5)/CC_P1(6)/CC_P2(7)`（mode 4）、`CC_BAR(8)`（mode 0）、`CC_AIVDONE(9)/CC_RDY(10)/CC_ALLDONE(11)`（mode 2）、`AIV_CH(16)` |
| 核内同步封装 | 段内 `template <pipe_t p> Acq(MutexID)` = `GetBuffImpl<p,false>`、`Rls(MutexID)` = `ReleaseBuffImpl<p,false>`（**阻塞释放 = CANN `ASC_LOCK_BLOCK` 默认，不用 set/wait flag**） |
| 段内类与 VF | 段内 `M10Aic`（→ `AttnCoreAic`）、`M10Aiv`（→ `AttnCoreAiv`）；`__simd_vf__` 函数 **5 个**：`SoftmaxTileVf / AccUpdateTileVf / CombineWeightsVf / CastRowToBf16Vf / CopySTileVf`，每个内部一个 `__VEC_SCOPE__` |
| 装载/FIXP/mmad 原语 | 段内 `GmToL1Nz`、`LoadL0_2D`、`LoadL0B_Transpose`、`FixpToUb`、`FixpToGmDbg`、`MmBf16` |

**纪律面（抽出的头 vs donor，读数见 §7①）**：`CrossCore(Set|Wait)Flag` 35 处、`SetFlag|WaitFlag` 35 处
（⇒ 两者相等 ⇒ 这一段里**没有核内** set/wait flag）；`__VEC_SCOPE__` 6 处（**5 处在 `__simd_vf__` 函数体内，
第 6 处在 donor 自己的说明注释里** —— donor README 记的"6 处"就是这个数，本文件沿用同一数法并注明构成）；
经典 memory-based 向量 API 0 处。
（另：`Duplicate(` 是 UB 常量填充、不在计算链上，donor README §7 已有存量说明。）

### 1.2 M88 / M97 已合入的 prolog 产物接口

| 项 | `文件:符号` |
|---|---|
| prolog 的段布局常量 | `m15_attn_prolog.h` → `M15AP::OUT_Q`(0..6144)、`M15AP::OUT_GATE`(6144..12288)、`M15AP::AP_OUT_K`(12288..12800)、`M15AP::OUT_V`(12800..13312)、`M15AP::OUT_QIDX`、`M15AP::OUT_KRAW` |
| 形状 | `m15_attn_prolog.h` → `M15AP::NH=24 / HD=256 / NKV=2 / QW=6144 / QG_W=12288`（+ `static_assert(QG_W + NKV * HD + NKV * HD == 13312u)`） |
| prolog 的入口 | `m15_attn_prolog_probe_body(...)`（`m15_layer_kernel.h` 的 `KIND_ATTN` 相位 A 调用点，AIV 分支与 AIC 分支各一处） |
| 接线机械清单 | `evidence/attn_prolog/README.md` §5b 的 **W1–W9**（塔 2026-09-27 裁决要求原样保留） |
| 与本段的关系 | prolog 把 gate 去交织成 **按头排布**的连续 [6144] 平面（`OUT_GATE`），与 attention 输出的内存序同序（§1.4）⇒ 门控可逐元素相乘，无需重排 |

### 1.3 M98 正在更正的主 KV 几何 —— 本段与它**无耦合**

| 项 | `文件:符号` |
|---|---|
| 唯一权威（主 KV 几何的宏） | `m15_attn_kv.h` → `M15KV_KV_BLOCK_OF / M15KV_KV_SLOT_OF / M15KV_KV_IN_BLOCK_OFF / M15KV_KV_BYTE_OFF_PHYS / M15KV_KV_BYTE_OFF_CONTIG` |
| 当前（更正前）页几何 | `m15_attn_kv.h` → `KV_BLOCK_BYTES = KV_HEADS(2) * KV_BLOCK_TOKENS(16) * KV_TOKEN_STRIDE(512) = 16,384` |
| M98 的状态 | 分支 `feat/m98-attention-kv-cache-fill-math`（未合）；把页几何更正为 32,768 B 并补 V 通道 |
| **本段的接口** | 保持 M10 的 `q/k/v` **裸指针**形态；核里**不出现任何 cache 偏移**。实测：本 mission 新增/改的 5 个源码文件里 `M15KV_KV_` 出现 0 次（命令与读数见 §7②）。⇒ M98 的更正**不需要**动本段 |

### 1.4 o_proj 的输入/输出形状与「gate 在哪一侧」

| 项 | `文件:符号` | 读出 |
|---|---|---|
| o_proj 形状 | `docs/11-attn-analysis.md:39` | `o_proj [2560, 6144]`：24×256 → 2560 |
| q_proj 宽度与 gate | `docs/11-attn-analysis.md:37` | `q_proj [12288, 2560]` = 24 头 × **2**（256 q + 256 gate）；gate 不 norm 不旋转 |
| 计算式 | `docs/11-attn-analysis.md:43` | `out = o_proj( attn(q,k,v) · sigmoid(gate) )` |
| 箭头序（第二处独立的同义表述） | `docs/11-attn-analysis.md:22` | `… sparse GQA ► sigmoid(gate) ► o_proj` |
| 权重形状（manifest） | `slice_layer_manifest.py:78` | `("attn_o_proj", "self_attn.o_proj.weight", [2560, 6144], 2560, 6144)` |
| 可抄的 donor | `m11_bf16_gemm.asc` → `template <uint32_t K, uint32_t N> class Bf16Gemm`（`Process/CopyInA/CopyInB/LoadA/LoadB/CopyOut`）；入口 `bf16_gemm_kernel<K, N>`；shapes 表里已有 `out_proj (K=6144 N=2560)` | 单 AIC bf16 GEMM，K/N 编译期模板 |
| **头序**（为什么不需要重排） | `m10_attn_decode.asc` 的 `Combine()`：`outGM[((uint64_t)n2 * GQA_G + row) * S2T]` | 内存序 = 头序 `h = n2*12 + row`（GQA 相邻配对 `num_queries_per_kv = 24/2 = 12`）⇒ 与 prolog 的 `OUT_Q`/`OUT_GATE` **同序** |
| 子层出口 | `m15_layer_kernel.h` 的 `subOut = A.hcAttnOut`（HC 形态） | 本段输出 `y[2560]` 就是 `subOut` 的生产者（M97 §5 第 8 项的收口前置） |

**「×sigmoid(gate) 在 o_proj 之前」的两条依据**：① `docs/11` 的两处表述（:43 的括号、:22 的箭头序）；
② **宽度自洽**：gate 的宽度 = 6144 = attention 输出宽度（不是 2560）⇒ 逐元素相乘只能落在 `o_proj` **之前**。
若将来发现次序相反，gate 宽度应为 2560 ⇒ `m15_attn_oproj.h` 的 `static_assert(OP_K == 6144)` 会先红。

### 1.5 「需要改清单外文件吗」

**不需要。** 本 mission 的改动全部落在 scope 内的 5 个源码文件 + `evidence/attn_core/**`。
`git diff --name-status main...HEAD` 的实跑读数见 §7③。
（特别地：`m15_layer_kernel.h` / `m15_layer_loop.asc`（M100）、`m15_attn_kv*.h`（M98）、
`m10_attn_decode/**`（只读 donor）都**没有**被改。）

---

## 2. 第二步：机械抽取成段

### 2.1 生成物与生成器

| 文件 | 作用 |
|---|---|
| `m15_layer_loop/m15_attn_core.h` | **生成物**（勿手改）：include guard + 只含 inline body + 无 `main()` + 无 `#include` |
| `m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py` | 生成器 + `--check` |

**8 类机械替换（生成器逐类计数断言；重命名式，无算法改动）**：
① `namespace m10 {` → `namespace M15AC {`；② `}  // namespace m10` → `}  // namespace M15AC`；
③ `M10Aic` → `AttnCoreAic`；④ `M10Aiv` → `AttnCoreAiv`；⑤ `M10_DBG_STAGE0` → `M15AC_DBG_STAGE0`；
⑥ `M10_DEBUG_SKIP_AIC_FD` → `M15AC_DEBUG_SKIP_AIC_FD`；⑦ `M10_DEBUG_SKIP_COMBINE` → `M15AC_DEBUG_SKIP_COMBINE`；
⑧ printf 标签 `[M10K]` → `[M15AC]`（**纯文案**，不动数值/判据/同步）。
生成器还会断言：抽出的片段里**没有 `#include`**、`#if` 与 `#endif` 成对。

**＋ 第 9 类：整块替换（r2 新增，塔 2026-10-04 裁决方案 B）** —— **抽取物不再是"逐字 + 纯重命名"**，
见下面 §2.4。差异表因此是「8 类机械替换 + 1 类整块替换」。

### 2.2 暴露面（人类的原话要求：依赖哪些 pipeline / 输出哪些 pipeline / 要哪些 buffer id 与 cross-core id）

生成物头部有一段**由生成器从抽出的正文实测导出**的清单（所以它不可能与代码漂移），要点：

- **入口**：`M15AC::AttnCoreBody(q, k, v, out, seq, wsAcc, wsM, wsS, maskCol, dbgS, dbgP, dbgC, cfg, dbgC2, gP)`
  —— 参数表 = donor `__global__` 入口的参数表（AIC/AIV 两侧 `Init()` 的并集）。
- **依赖（消费）的 pipeline**：跨段 **无**（段起点自包含：调用方备好 q/k/v/seq/maskCol/workspace/dbg 即可）；
  段内 = cross-core（下面的 flag 表）+ 核内 BufferID（下面的 MutexID 表）。
- **输出（生产）的 pipeline**：GM `out`（`[2][12][256]` bf16，FD 归并收尾）、GM workspace
  `wsAcc/wsM/wsS`、GM 调试面 `dbgS/dbgP/dbgC/dbgC2`；跨段 set **无**（段尾 `PipeBarrier<PIPE_ALL>` 由调用方补）。
- **cross-core flagId 35 次调用**：`CC_MM0(0)/CC_MM1(1)` mode 4；`CC_P0(5)/CC_P1(6)/CC_P2(7)` mode 4；
  `CC_BAR(8)` mode 0；`CC_AIVDONE(9)/CC_RDY(10)/CC_ALLDONE(11)` mode 2；`AIV_CH = 16`。
  生成器同时给出**实测的 (Set|Wait, mode, pipe) 分布**（见头文件）。
- **MutexID**：AIC `B_Q..B_PL1` = 0..8；AIV `B_PC..B_DBG` = 0..6。生成器给出实测的 `Acq/Rls` 分布。
- **融合时必须重号的两条理由**（M102 逐条核；**② 已按复审 r1-F2 更正措辞**）：
  ① 本段 mode-2 的 9/10/11 与 hc 段 mode-2 的 8/9/10/11 在**同一 (核型, mode) 子空间**里撞号
  （`m15_layer_resources.h` §4 的登记表是唯一权威）；
  ② **mode-4 不是"新子空间"**：官方口径（本机 CANN 文档
  `asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md`
  「flagId取值范围说明」）明确 —— 模式 0/1/2 每核 16 个（0-15）；**模式 4 时 AIV 仍是同一批 0-15**
  （池没有变宽），**只有 AIC 侧放宽到 0-31**，且「AIC 发起 flagId 16-31 ↔ AIV1 的 wait 0-15」。
  同一核上同一 flagId 跨模式复用**合法**，前提是"模式切换前前一个 mode 的 set/wait 全部 drain 完"。
  ⇒ 本段 AIV 侧的 {0,1,5,6,7} 与顶层 AIV 上 mode0+mode2 已占满的 0-15 **是同一个物理池**，
  打平表必须按"drain 后复用"登记；本段 AIC 侧的 `ccMM + AIV_CH` = **16/17** 落在 AIC 放宽后的 0-31 里，
  而 `m15_layer_resources.h` 的 `FLAG_PER_CORE = 16` 会在 `id >= 16` 时直接把记录判成越界
  ⇒ **顶层登记表必须为 `AIC ∧ mode == 4` 单独放行**。截至本提交，主线上该文件已落
  `FlagIdLimit(core, mode)` + `FLAG_PER_CORE_AIC_MODE4 = 32`（以及 `ATTN_CORE_M4_AIV_MAX = 7` /
  `ATTN_CORE_M4_AIC_HI_LO = 16` 两个见证量与配套 `static_assert`）—— 以该文件为准。
- 另有 3 条 `static_assert` 把上述 id 的**数值**钉住（一旦重编号，编译先红并提示同步顶层打平表）。
- **足迹与 prefill 边界**（复审 r1-F5 要求写进本头，已加）：L1 288 KB / 512 KB、L0A 64 KB、
  L0B 64 KB、L0C 64 KB / 256 KB、AIV UB 峰值 64,512 B ≈ 63 KB / 248 KB，全部是段内自管理的编译期常量；
  以及"本段是 decode 形状（28 unit、`SPLITS=14`、`S2T=256`），**不能直接吃 prefill `m=4097`**"。

### 2.3 本段的「调试面」（M102 接线时要知道）

`dbgC`（`[28][16][256]` fp32 = 458,752 B）、`dbgC2`（同尺寸）、`dbgS`（`[56][8][256]` fp32 = 458,752 B）、
`dbgP`（`[56][8][256]` bf16 = 229,376 B）由段内**首 tile 无条件写**（donor 的定位遗留），
另有 17 条 `printf`。按"机械抽取"口径**原样保留**，逐条记在 §5。

### 2.4 第 9 类：整块替换（memory-based 计算 → RegBase VF）——**抽取不再是纯逐字**

**依据（人类逐字规则，塔 2026-10-04 裁决方案 B）**：
> 「……都应该都用 VEC_SCOPE aka simd vector function，**不应该使用 memory base 的 API**」

**改前**（donor `m10_attn_decode.asc:977-978` 与本抽取件 r1 逐字相同，位于 `AttnCoreAiv::Combine()`
的 FD 归并内循环 —— 数值路径：加权 + 跨 split 累加）：

```cpp
                Muls(tmpR, accR, w, S2T);
                Add(numR, numR, tmpR, S2T);
```

**改后**（`m15_attn_core.h:858` 新增的 `__simd_vf__` + `:1144` 的调用点）：

```cpp
__simd_vf__ inline void AccFmaRowVf(__ubuf__ float* numUb, __ubuf__ float* accUb, float w)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> acc, num;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t g = 0; g < (uint16_t)ROW_VL; ++g) {
            LoadAlign(acc, accUb + g * VL_F32);
            LoadAlign(num, numUb + g * VL_F32);
            Muls(acc, acc, w, fullM);
            Add(num, num, acc, fullM);
            StoreAlign(numUb + g * VL_F32, num, fullM);
        }
    }
}
```

- **登记方式**：生成器里是第 9 类 `BLOCK_REPLACEMENTS`（两处：插入 VF 函数 + 替换调用点），
  与第 1–8 类的"重命名式"**分表**，每处都断言"在 donor 里出现且只出现一次"。
- **累加次序一字未动**：`nsplit` 外层循环、逐 `i` 先乘后加，全部保持。
- **为什么可以期望逐位不变**：官方 Reg 规格表里 `Mul`/`Add`/`Muls` 都是 **0 ulp**（同表：Exp 1 ulp、Div 1 ulp）。
  **但这是论证不是见证** ⇒ 见 §3.2/§3.4 的三条见证。
- **`tmpR`（UB_TMPR 槽）在替换后不再被引用**，声明保留（不动段内那张编译期静态 UB 地址表），
  并以 `(void)tmpR;` 显式标注。
- **已知限度**：对这两行而言"与 donor 逐行相同"**不成立**；本文件里凡写"逐字一致"的地方，
  口径都是"**除第 9 类之外**逐字一致"。

---

## 3. 第三步：独立设备验证路

### 3.1 结构

| 文件 | 作用 |
|---|---|
| `m15_layer_loop/m15_attn_core.asc` | 入口壳：`__global__ __mix__(1,2) m15_attn_core_kernel`（薄壳）+ `main()` 模式分派 |
| `m15_layer_loop/m15_attn_core_host.h` | host 侧：数据生成 / 落盘 / 与归档对拍 / o_proj 的判据 |
| `m15_layer_loop/CMakeLists.txt` | `add_executable(m15_attn_core m15_attn_core.asc)`（该文件此前无人持有） |

**判据不复制**：host 侧的 `GenQKV` / `GenMaskCol` / 落盘文件名（`m10_case_s<seq>_*.bin`）逐字复刻 donor
host（`m10_attn_decode.asc:1120-1339`），于是 donor 的 `m10_attn_decode/check_ref.py` 可以**原样**在本目录跑
（它把前缀写死在 `load_case()` 的 `pre = f"m10_case_s{seq}_"`）。判据与被测实现的独立性因此是"两份不同的
产物"而不是"一份代码的两个副本"。
跑判据时必须 `PYTHONDONTWRITEBYTECODE=1`：`check_ref.py` 会 `import moe_block_ref`（来自 `<repo>/tools/golden`），
不加这个变量会在 scope 外产生 `__pycache__`。

### 3.2 读数（命令 → 输出 / rc，全部实跑）

**① 与 M10 归档 dump 逐字节对拍**（`M15AC_REFDIR=m10_attn_decode/data`；归档里缺的文件如实标"跳过"）：

| seq | 逐字节相同的文件（归档有的话） | 归档里没有的 |
|---|---|---|
| 256 | q, k, v, out, wsm, wss, dbgs, dbgc2, gp | wsacc, dbgp, dbgc |
| 300 | q, k, v, out, wsm, wss, dbgs, dbgc2, gp | wsacc, dbgp, dbgc |
| 4096 | q, out, wsacc, wsm, wss, dbgc2, gp | k, v, dbgs, dbgp, dbgc |

⇒ **归档里有的每一个文件都对上了**（日志：`logs/m101_core_run_20260927.log`，逐行 `[M101][bincmp]`）。

**② donor `check_ref.py` 的判定读数**（`logs/m101_core_check_ref_20260927.log`）：

```
[check] seq=256,300,4096 | 判定项 23/24 通过 | 报告项 27 单列 | guard 9/9 | 输入缺失 0 项
RESULT: FAIL (判定项 23/24 通过；判定 FAIL 1 项、guard FAIL 0 项；报告项 27 单列)   ← rc=1
```

| seq | 判据 A（out 边界感知 ulp） | 判据 B | C（P̃ 全数组） | D | E（独立 P·V） | F（设备 P·V） |
|---|---|---|---|---|---|---|
| 256 | **PASS** 常规 6130（越界 0）/ 边界 14（ulp 0、绝对界 0） | PASS | PASS 越界 0/8192 | PASS | PASS 0/8192 | PASS 0/8192 |
| 300 | **PASS** 常规 6086（越界 0）/ 边界 58（ulp 0、绝对界 0） | PASS | PASS 越界 0/9600 | PASS | PASS 0/16384 | PASS 0/16384 |
| 4096 | **FAIL** 常规 5924（越界 0）/ 边界 220（ulp 0、**绝对界越界 1**） | PASS | PASS 越界 0/131072（最大界占用 0.9965） | PASS | PASS 0/65536 | PASS 0/65536 |

报告项：位级一致率 **98.1771 / 98.6165 / 99.3652 %**；≤1ulp **100 / 100 / 99.9837 %**；
maxAbsErr 2.476e-04 / 2.442e-04 / 5.992e-05；4096 判据 E maxAbsErr 1.564e-03、F 2.980e-06。

**与 M10 README §0 的归档读数逐项相同**（判定项 23/24、报告项 27、guard 9/9、位级三档三个数、
≤1ulp 三档三个数、6130/14、6086/58、5924/220、4096 的 1 个绝对界越界、C 的界占用 0.9965）。
**4096 档按 M10 现状如实记账：不改判据、不自行放宽、不申请例外。**
本表与 §4 的读数取自**同一个二进制** sha256 `ab12e2e4…`（`logs/m101_bin_sha256.txt`；
r2 那一版是 `d5517787…`，两者只差 host 侧 FNV 常量，见 §4.3.1）。

**③ 第 9 类整块替换（§2.4）的等价性见证 = 上表 ① 的逐字节对拍。**
口径与读法写清楚：r1 的二进制是 `6e426cef…`（**纯逐字 + 8 类重命名**），r2/r3 的二进制
（`d5517787…` / `ab12e2e4…`，**多一条第 9 类整块替换**）上重跑三档后：

| 见证 | r1（纯逐字） | r2（第 9 类整块替换后） |
|---|---|---|
| 与归档 dump 逐字节相同的文件数 | 25 | **25**（不同的 0 个、归档里没有的 11 个） |
| `判定项 23/24 \| 报告项 27 \| guard 9/9 \| 输入缺失 0` | 同 | **同** |
| 位级一致率（三档） | 98.1771 / 98.6165 / 99.3652 % | **同**（逐位） |
| ≤1ulp（三档） | 100 / 100 / 99.9837 % | **同** |
| 判据 E/F 的 maxAbsErr（三档） | 1.818e-06 / 1.907e-06 / 1.564e-03（E） | **同** |

⇒ **对拍成立**（否则按塔的要求应立即停手报塔）。这也是"`Mul`/`Add` 官方 0 ulp"这条论证的**实测背书**：
换法不是"看起来等价"，而是**归档里每一个可比字节都没动**。

**④ 计算路径的判别计数**（复审 r1-F3 的替换物）：见 §7①，`audit_discipline.py --pair` 在 r2 上给出
**块外计算类调用 0 处**（r1 是 2 处 —— 正是第 9 类改掉的那两条）。

### 3.3 负向对照（`negkv`：调用点把 K/V 实参互换；抽出的段一字未改）

```
[check] seq=256,300 | 判定项 8/16 通过 | 报告项 18 单列 | guard 6/6 | 输入缺失 0 项   ← rc=1
```

| seq | 判据 A 打破条数 | C | E | F |
|---|---|---|---|---|
| 256 | 常规 **6120/6130**、边界 **14/14** | **6017/8192**（最大界占用 564.80） | **7756/8192** | **8191/8192** |
| 300 | 常规 **6069/6086**、边界 **58/58** | **7008/9600**（304.03） | **15869/16384** | **16384/16384** |

⇒ 8 项判定项由 PASS 翻成 FAIL（A/C/E/F × 两档）。判据 B（结构性）与 D（位置/掩码）仍 PASS ——
这两条本来就不咬数据内容，**如实记录**，不把它写成"全红"。

**为什么 D 一定不会红**（复审 r1-F5 的更正，已同步改进两处注释）：
判据 D 咬的是「未用槽位恒 0 + 尾 tile 无效列恒 0」这条**结构性**性质 —— 把 K/V 两个指针互换
改变不了任何槽位有没有被写过、也改变不了尾 tile 的列 mask 契约，所以 D **必然** PASS。
⇒ "必须变红"的清单是 **A/C/E/F**；`m15_attn_core.asc` 与 `m15_attn_core_host.h` 里原先写成
"(A/C/D/E/F)" 的两处注释在 r2 已改。

读数出处：`logs/m101_core_negkv_run_20260927.log` + `logs/m101_core_negkv_check_ref_20260927.log`；
跑法与 §3.2 同一二进制（sha256 `ab12e2e4…`）。

---

## 4. 第四步：o_proj + ×sigmoid(gate) 段

### 4.1 结构（`m15_layer_loop/m15_attn_oproj.h`）

按人类"模块化 + 暴露依赖/输出 pipeline"的要求，**按核型各暴露一个 body**（而不是一个 mix body）：

| 段 | 入口 | 依赖（消费） | 产出 | 跨核 | 核内 MutexID |
|---|---|---|---|---|---|
| §A ×sigmoid(gate) | `M15OP::GateMulAiv(attn, gate, t, m, mode)` | `attn` GM（[m][6144] bf16）、`gate` GM（同形，= prolog 的 `OUT_GATE` 平面） | `t` GM（[m][6144] bf16） | **无** | `OP_BUF_GIN(0)` MTE2→V、`OP_BUF_GOUT(1)` V→MTE3（本段独占） |
| §B o_proj | `M15OP::OProjGemmAic(t, w, y, m, mode)` | `t` GM（A 操作数）、`w` GM（B 操作数，**[N=2560, K=6144] 原始 layout**） | `y` GM（[m][2560] bf16 RNE）＝ 子层出口 `subOut` | **无** | `BUF_A0/A1/B0/B1/L0_0/L0_1/L0C` = 0..6（抄 donor m11） |

**为什么两段都不带 cross-core flagId**：独立验证路用**两次顺序 launch**（`__mix__(1,2)` 的门控 +
`__cube__` 的 GEMM），靠 stream 同步；跨核接线由 M102 按打平表决定 —— 段不知道自己被排在谁后面。
（这也消掉了配对写错导致死锁的风险。）

**抄 donor 的账**（`m11_bf16_gemm.asc`）：常量与类搬进 `namespace M15OP`；`Bf16Gemm` → `OProjGemm`；
`Process()` 增加 `mode` 入参（负向对照用）；其余 `CopyInA/CopyInB/LoadA/LoadB/CopyOut` 与
`Nd2NzParams` / `LoadData2DParamsV2` / `MmadParams` / `FixpipeParamsArch3510` 的字段值逐字保留。

### 4.2 判据与 ε 的逐项出处

官方规格表（硬件列）：`/workspace/asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md`
→ `Exp = 1ulp`、`Adds = 0ulp`、`Mul = 0ulp`、`Div = 1ulp`。
σ(g) = 1/(1+exp(−g))：`Muls(−1)` 精确、`Exp` ≤1 ulp（相对 ≤2u）、`Adds(+1)` 精确、`Div` ≤1 ulp（≤2u）、
分母误差传到 σ 上再乘 `e^{−g}/(1+e^{−g}) ≤ 1` 再加 2u ⇒ **|Δσ/σ| ≤ 4u**（u = 2^-24）。
`t = a·σ` 里 `Mul` 0 ulp ⇒ t 的相对误差完全来自 σ；bf16→fp32 的 `Cast` 精确。

- **判据 OP-A（GEMM，逐位）**：输入取**精确整数域**（t、W ∈ [-8,8] 整数；乘积 ≤ 64；K=6144 ⇒
  |Σ| ≤ 393,216 < 2^24 ⇒ fp32 任何累加次序都精确）⇒ 设备 bf16 位型必须与 fp64 参考 RNE 后**逐位相同**。
- **判据 OP-B（×sigmoid(gate)，T3）**：`|f32(t_dev) − t_ref| ≤ 0.5·ulp_bf16(t_ref) + 4u·|t_ref|`。
  **分辨率声明**：该判据对 t 的相对误差的分辨力 ≈ 0.5 bf16 半步 ≈ 0.39%（bf16 尾数 7 位）；
  比它更细的 σ 误差**不在此判据的可判范围内**（写法与 M10 判据 E 的分辨率声明同款）。
- **判据 OP-C（e2e，S1 定位声明）**：输入 = **设备产出的 t**（D2H 回读），参考 = fp64 `Σ W_j t_j`，
  tol = `K·u·Σ|W_j t_j|`（mmad k=6144 的 fp32 累加保守界，推导）。**被判量** = "GEMM 消费的 t 是否是
  门控段产出的那一份（同布局/同顺序）"；**t 本身不是本判据的被判量**（两侧同一份设备 t），t 由 OP-B 咬。
- **guard（OP-B 判别力对照）**：① 把参考按 RNE 量化到 bf16 当设备 ⇒ 越界须 0（验证容差 ≥ 纯量化半步，
  判据不空洞）；② 把设备值前 8 个元素再挪 +2 个 bf16 格点 ⇒ 越界须 > 0（验证有分辨力）。

### 4.3 读数（命令 → 输出 / rc，全部实跑）

二进制 sha256 = `ab12e2e4…`（`logs/m101_bin_sha256.txt`；§3 与 §4 的读数取自**同一个**二进制）。
完整日志：`logs/m101_oproj_run_20260927.log`（复跑：`logs/m101_oproj_rerun_20260927.log`）。

| 档 | 判定项 | 读数 | 负向对照（**必须红**） |
|---|---|---|---|
| m=1 | **OP-A** o_proj GEMM 逐位（精确整数域） | **PASS** 比较 2560 元素、位不等 **0** | `GEMM_MODE_KMINUS1` ⇒ 位不等 **2507/2560** |
| m=1 | **OP-B** ×sigmoid(gate)（T3） | **PASS** 比较 6144 元素、越界 **0**；maxAbsErr 1.9377e-03；**最大界占用 0.999834**；位差 0/1/≥2 格点 = **6144/0/0** | `NO_GATE` ⇒ 越界 **6144/6144**（界占用 351.5）；`SIGN σ(−g)` ⇒ 越界 **6104/6144**（281.7） |
| m=1 | **OP-C** e2e（S1：输入 = 设备 t） | **PASS** 比较 2560 元素、越界 **0**；maxAbsErr 0.89187；最大界占用 0.241964 | `GEMM_MODE_KMINUS1` ⇒ 越界 **1744/2560** |
| m=3 | **OP-A** | **PASS** 比较 7680 元素、位不等 **0** | ⇒ 位不等 **7531/7680** |
| m=3 | **OP-B** | **PASS** 比较 18432 元素、越界 **0**；位差 = **18432/0/0** | `NO_GATE` ⇒ **18432/18432**；`SIGN` ⇒ **18313/18432** |
| m=3 | **OP-C** | **PASS** 比较 7680 元素、越界 **0**；maxAbsErr 0.95138；最大界占用 0.254263 | `GEMM_MODE_KMINUS1` ⇒ 越界 **5097/7680** |

`[M101] o_proj：判定项 6/6 通过`，**rc=0**（负向/guard 任一项未变红则退出码非 0；本次全部变红）。
guard：`参考量化到 bf16 当设备 ⇒ 越界 0（须 0）`；`前 8 个元素 +2 格点 ⇒ 越界 8（须 >0）`。

#### 4.3.1 F1 的闭环：契约档的 y 落盘 + 指纹 + **可离线复跑的独立复核脚本**

**r1 的缺陷（复审 F1）**：`m101op_m{1,3}_y.bin` 是在循环**末尾**统一 dump 的，而那时 `hY` 已被
OP-C 的 `KMINUS1` 负向档覆盖 ⇒ 归档里唯一那份 `y` 是**故意弄坏**的输出；OP-A/OP-C 的设备结果
**没有任何可离线复核的工件**（复审拿它复算就吃到 2559/2560 假红）。

**r2 的改法**（`m15_attn_core_host.h`）：
1. **每份工件在产生的那一刻就落盘**，文件名自述身份：
   `m101op_m{m}_yA_contract.bin` / `_yA_kminus1.bin` / `_yC_contract.bin` / `_yC_kminus1.bin` /
   `_tdev.bin` / `_attn.bin` / `_gate.bin` / `_tint.bin`（各自的样例行见 `DumpAndPrint` 的调用点，
   旧的那个含义漂移的 `_y.bin` **已删除**）。
2. **输入指纹进日志**：`[M101][fp] <label> file=<name> len=<n> fnv1a64=<16 hex>` ——
   W（31.5 MB，不落盘）也给指纹。`.bin` 是运行产物、不入库，**指纹在入库的日志里** ⇒
   即使只有日志的机器也能把"我按公式重生成的输入"绑到"那次运行用的输入"。
3. **独立离线复核脚本**：`evidence/attn_core/check_oproj_dumps.py`（自己的 numpy fp64 参考，
   不 import 任何被测代码）：

```bash
cd m15_layer_loop/evidence/attn_core
PYTHONDONTWRITEBYTECODE=1 /usr/local/python3.12.13/bin/python3 check_oproj_dumps.py \
  out_oproj --log logs/m101_oproj_run_20260927.log   # 需要 numpy ⇒ 必须显式 python3.12（见 §6 的说明）
```

实跑读数（`logs/m101_oproj_offline_check_20260927.log`，**rc=0**）：

| 复核项 | 读数 |
|---|---|
| 输入绑定 | W `fnv1a64=29dd51e39fc752bb` == 日志；attn/gate/tint 三对指纹全部一致，且与磁盘 `.bin` **逐字节相同**（**标准 FNV-1a 64**，算法与自查向量见 §4.3.1） |
| OP-A 契约档 | 与 `RNE(fp64 Σ_k<6144)` **逐位相同 2560/2560（m=1）、7680/7680（m=3）** |
| OP-A 负向档 | 与 `Σ_k<6080` **逐位相同 2560/2560、7680/7680** ⇒ 该 dump 确实是"K-1 截断"那份 |
| OP-B | 越界 **0/6144**、**0/18432**；`max(dv/tol)=0.999834`；位差 `6144/0/0`、`18432/0/0` |
| OP-C 契约档 | 越界 **0/2560**、**0/7680**；最大界占用 0.241964 / 0.254263 |
| OP-C 负向档 | 对 K-1 参考越界 **0**（= 确实是截断结果）、对全 K 参考越界 **1744/2560、5097/7680**（与 C++ 日志逐项相同） |

⇒ **OP-A / OP-C 现在可以完全不碰设备地复核**，且复核脚本的数与 C++ 侧的判定数**逐项对上**。

**指纹算法（逐字文档化；复审 r2-N1 要求）**：`[M101][fp]` 里那个 `fnv1a64=` 是
**标准 FNV-1a 64** ——

```
offset basis = 14695981039346656037 = 0xCBF29CE484222325
prime        = 1099511628211        = 0x100000001B3
h = offset; for byte in buf: h = ((h XOR byte) * prime) mod 2^64
```

C++ 侧见 `m15_attn_core_host.h::Fnv1a64`，Python 侧见 `check_oproj_dumps.py::fnv1a64`，两处同实现。
**自查向量**（第三方可用它验证自己的实现，也可验证我这两个实现）：
`FNV1a64(b"") = 0xcbf29ce484222325`、`FNV1a64(b"a") = 0xaf63dc4c8601ec8c`
（Python 侧在 `main()` 入口用 `assert` 钉住，见日志首行 `[selfcheck] FNV-1a 64 标准向量通过…`）。

> **r1 版本的这个常量写错了**（复审 r2-N1 指出）：r1 用的是 `1469598103934665603`，
> **少写了末尾一位**，所以那份指纹**不是** FNV-1a 64 —— 用标准算法实现的第三方**复现不出来**，
> 而这恰恰是这个工件存在的意义。本轮按复审建议改**常量**（而不是改名），并**重跑了 oproj 档重发指纹**
> （独立复算已验：用一份与我的实现无关的标准 FNV-1a 64 写法算
> `out_oproj/m101op_m1_attn.bin` 得 `0c8e0636a7785231`，与日志该行**逐位相同**）。
> ⇒ 日志里 `fnv1a64=` 的数值**已是标准算法**；r1 那批旧数值只在旧日志里，不入库。

#### 4.3.2 设备确定性（r2 新增读数）

同一次持锁内把 o_proj 档**跑两遍**（`out_oproj` / `out_oproj_rep`），16 个 dump 对（m=1/3 × 8 个文件）
**全部"逐字节相同"**。⇒ 这两档的设备输出在两次运行间**可复现**（这条也顺带把 §4.3 下面那条
"第一版 vs 更正后"的数字差异限定在**判据侧**，而不是设备侧）。

#### 4.3.3 F4 的更正：倍数与数字出处

- **"1/4" 是错的，实测是 1/2。** 机制表述（符号-幅值 ⇒ 取到更靠近 0 的邻居）没问题，但
  该写法给出的是**半步**而不是整步 ⇒ 容差变成应有值的 **1/2**。复审在五个 FAIL 元素上量到
  `grid_ulp / lattice = 0.500`，与机制一致。r1 那个误差元素上：旧 tol 4.884e-4 vs 更正后 9.767e-4
  （|ref| ≈ 0.25）⇒ 比值 **1/2**。已改。
- **"按旧式则预测 18 处越界"的出处补上了，而且 11/18/19 的不一致已经定位（复审 r2-N2 替我定位的）**。
  现在 `check_oproj_dumps.py` 在**同一批 dump** 上把**三套口径**各算一遍：

| 档 | 更正后（本判据） | 第一版·**截断**口径（我 r2 的重建） | 第一版·**RNE** 口径（= r1 的 C++） |
|---|---|---|---|
| m=1 | **0/6144** | **18/6144** | **11/6144** |
| m=3 | **0/18432** | **54/18432** | **31/18432** |

  **根因 = 取 bf16 位型时的舍入方式**（不是我原先写的"未定位"）：
  - r1 的 C++ 用 `FloatToBf16()`，即 **RNE**（`(x + ((x>>16)&1) + 0x7FFF) >> 16`）；
  - 我 r2 的第一版重建在 `legacy_grid_ulp` 里写的是 `(x >> 16)`，即**截断**。
  同一批 dump 上，RNE 口径**恰好复现 r1 日志的 11 / 31**，截断口径给出 18 / 54，
  复审的 r1 重建（取两邻居较小者）给出 19 / 60 —— **三者是同一个已删除代码路径的三种重建，不是数据差异**。
  复现命令（`check_oproj_dumps.py` 的 `[OP-B]` 行同时打印三套）：

  ```bash
  cd m15_layer_loop/evidence/attn_core
  PYTHONDONTWRITEBYTECODE=1 /usr/local/python3.12.13/bin/python3 check_oproj_dumps.py \
    out_oproj --log logs/m101_oproj_run_20260927.log 2>&1 | grep '\[OP-B\]'
  #  [OP-B] n=6144  越界(更正后)=0 越界(第一版·截断口径)=18 越界(第一版·RNE 口径= r1 C++)=11 …
  #  [OP-B] n=18432 越界(更正后)=0 越界(第一版·截断口径)=54 越界(第一版·RNE 口径= r1 C++)=31 …
  ```

  **另一处更正（复审 r2-N2 同时指出）**：本表 r1 那一列原先写着"r1 日志只跑了 m=1 的判定行"——
  **这句是错的**，入库的 `logs/m101_oproj_r1_badcriterion.log` 第 48/62 行就是 m=3 那一档
  （`[op] m=3` → `判定项[OP-B…]: FAIL 比较 18432 元素、越界 31`）。已改成上面表里的 31。
  结论不受影响：更正后 0 越界、第一版 > 0 越界，三套重建在这一点上一致。
  （保留下来的做法：r1 的 C++ 那条路径已被替换，现在只有这三套**重建**可跑；我把它们都留在脚本里，
  不是为了"对上一个数"，而是为了让"11/18/19 从哪来"这件事**可复跑**、不再靠叙述。）
- 我在 r1 里报给塔的那条 `m10_attn_decode/check_ref.py::bf16_grid_ulp` finding **成立**
  （复审复验：负数分支返回半步），且它的三个调用点都作用在 `P̃ ≥ 0` 上 ⇒ **对 M10 读数无影响**。

---

## 5. 第五步：显式未完成项（逐条列，写明依赖）

| # | 未完成项 | 现状 | 卡在哪 / 依赖什么 |
|---|---|---|---|
| 1 | **接进层 kernel 的挂载点** | 未做 | 挂载点在 `m15_layer_kernel.h` 的 `KIND_ATTN` 相位 A（AIV 与 AIC 两个分支）与 `m15_layer_loop.asc`，**归并行的 M100**；本 mission 的边界明确禁止改它们。解封后交 **M102**：把 `M15AC::AttnCoreBody` 的调用插在 `M15AP::m15_attn_prolog_probe_body(...)` 之后，并按 §2.2 重号 flagId |
| 2 | **KV cache → 本段的绑定** | 未做（本段只吃裸指针） | 只用 `m15_attn_kv.h` 的 `M15KV_KV_*` 宏；**M98 正在更正主 KV 几何**（页 16,384 → 32,768 B、补 V 通道，分支 `feat/m98-attention-kv-cache-fill-math`）。本段与它无耦合（§1.3），但绑定必须等 M98 合入后按新宏写 |
| 3 | **prefill `m=4097` 的 per-row 化** | 未做（本段假定 decode 的 `m=1`，`seq` 只是 cache token 数） | M97 §5 第 7 项的同一件事：AIC 的 N-块分派与 AIV 的按头分派在 `m>1` 时要重排；o_proj 段本身对 m 无假设（GEMM 走 m-tile 循环、门控按扁平 chunk 分派），但 attention 核心段不是 |
| 4 | **QSA indexer 的选择语义（打分/topk/expand）** | 未做 | M35/M53/M88 已有前端（prolog 产出 `OUT_QIDX`/`OUT_KRAW` 与 raw ring）；本段目前的 `seq` 只用于切 tile，**没有"只 attend 选中的 ≤2048 token"这条语义**。要接 indexer 的 `packed indices`（`m15_attn_kv.h` 的 `PACK_*`） |
| 5 | **4096 档的残差与 T4 确定性** | 如实记账 | 4096 判据 A 的 1 个边界绝对界越界（已由 M10 完整归因到一个 P̃ 的 bf16 舍入翻转）；T4"确定性"在 M10 §8.1 记为**未隔离观测** ⇒ 本 mission **不宣称**该档通过 |
| 6 | **去掉段内的调试面** | 原样保留 | 4 个 dump（`dbgC/dbgC2/dbgS/dbgP`，首 tile 无条件写）+ 17 条 `printf`。M10 README §7 已记为"定位完成后应移除"。本 mission 按抽取口径**不改**；融合形态若要省这 4 次 MTE3/FIXP 与 printf，需在 `m15_attn_core.h` 里加编译期开关 —— 那会让"逐字一致"再多一条例外，留给 M102 决定 |
| 7 | **o_proj 段的跨核接线** | 未做 | §4.1 的两段目前靠 stream 同步；融合 kernel 里的跨核同步（AIV 写完 t → AIC 读）要按打平表定，并需要一条 AIV→配对 AIC 的 mode-2 通道（形态可抄 M10 的 `CC_AIVDONE`：2 set 配 1 wait） |
| 8 | **o_proj 的 m>64 / prefill** | 部分 | GEMM 段走 `mLoop = ceil(m/64)`，形状上支持 m>64，但本 mission 只实跑了 m=1 与 m=3；m=64 以上的尾块与 A 的越界可读行数（`m=1` 时多读 1 行的 donor 契约）未在 o_proj 上单独验证 |
| 9 | **donor 侧同一处 memory-based 计算**（r2 新记） | 未改，只报 | `m10_attn_decode.asc:977-978` 有**同样两处**经典 `Muls`/`Add`。本 mission 的第 9 类只改了**自己这份抽取件**（§2.4）；donor 是只读的，已按 finding 报塔（`.tower/comms/findings/…-m10-attn-decode-asc-fd-combine-2-memory-based-api…`）。**M10 的读数不受影响**，但要修的话属 M10 的 scope |
| 10 | **第 9 类的"纯逐字"性质**（r2 新记） | 已发生，如实声明 | 抽取物现在是「8 类机械替换 + 1 类整块替换」。凡本文件写"逐字一致"处，口径均为"**除第 9 类之外**"。若后续有人要求回到纯逐字，须把 §2.4 那条规则作废并接受 memory-based 计算留段 |

---

## 6. 复现命令（全部实跑过；`<wt>` = 本 mission 的 worktree）

**解释器说明（复审 r2-N3 的要求，先说在前面）**：本机 `python3` = `/usr/bin/python3`（**没有 numpy**）。
⇒ **凡是要 `import numpy` 的脚本（`check_oproj_dumps.py`）必须显式写
`/usr/local/python3.12.13/bin/python3`**；`lift_attn_core_segment.py` 与 `audit_discipline.py` 只用标准库，
`python3` 即可。r2 的文档里那三处写成裸 `python3` 的命令**当时跑不通**（复审实测
`ModuleNotFoundError: No module named 'numpy'`）—— 本轮已全部改成显式解释器，并在下面贴出逐字输出。

```bash
cd <wt> && source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release
cmake --build m15_layer_loop/build -j4 --target m15_attn_core

# 生成 / 校验抽取（--check 会重新抽取并与磁盘逐字节比；含第 9 类整块替换的计数断言）
python3 m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py
python3 m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py --check

# 计算路径的判别计数（复审 r1-F3；块外计算类调用应为 0）
python3 m15_layer_loop/evidence/attn_core/audit_discipline.py --pair
python3 m15_layer_loop/evidence/attn_core/audit_discipline.py m15_layer_loop/m15_attn_oproj.h
```

**上面四条与下面那条 numpy 命令的逐字输出**（本轮实跑，`LC_ALL=C`；输出里两处长得一样的绝对路径
前缀用 `…` 省略，其余逐字）：

```
$ python3 m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py --check
[ok] …/m15_layer_loop/m15_attn_core.h == 从 …/m10_attn_decode/m10_attn_decode.asc 重新抽取的结果（57593 字节，逐字节；锚点 asc 行 58..1005，donor sha256 e9ddfbd85e0f0540…）

$ python3 m15_layer_loop/evidence/attn_core/audit_discipline.py --pair
[结论]
  块内计算类调用 40 处；块外计算类调用 0 处；块外搬运用 22 处；块外 Reg 专用原语 0 处
  ⇒ '计算走 RegBase VF、memory-based API 只用于搬运/常量填充' 这条口径在本切片上**有判别力地**成立（不是靠模式匹配不到）。

$ python3 m15_layer_loop/evidence/attn_core/audit_discipline.py m15_layer_loop/m15_attn_oproj.h
[结论]
  块内计算类调用 9 处；块外计算类调用 0 处；块外搬运用 3 处；块外 Reg 专用原语 0 处
  ⇒ '计算走 RegBase VF、memory-based API 只用于搬运/常量填充' 这条口径在本切片上**有判别力地**成立（不是靠模式匹配不到）。

$ /usr/local/python3.12.13/bin/python3 -c "import numpy; print('numpy', numpy.__version__)"
numpy 2.5.1
$ python3 -c "import numpy"
ModuleNotFoundError: No module named 'numpy'        # ← 这就是 N3 说的那件事
```

**离线复核脚本的输出**（首行是新增的 FNV 自查；**没贴全的地方用 `…` 标出，其余每行逐字**，
完整输出见 `logs/m101_oproj_offline_check_20260927.log`）：

```
$ cd m15_layer_loop/evidence/attn_core && PYTHONDONTWRITEBYTECODE=1 \
    /usr/local/python3.12.13/bin/python3 check_oproj_dumps.py out_oproj --log logs/m101_oproj_run_20260927.log
[selfcheck] FNV-1a 64 标准向量通过：fnv1a64(b'')=0xcbf29ce484222325、fnv1a64(b'a')=0xaf63dc4c8601ec8c
[offline] dir=out_oproj  log=logs/m101_oproj_run_20260927.log  日志里 [M101][fp] **行数=18** / **不同标签个数=9**（两者不同：同一 label 每个 m 各一行）
[bind] W(full, 31457280 B) fnv1a64=29dd51e39fc752bb log=29dd51e39fc752bb -> 一致
…
[OP-B] n=6144 越界(更正后)=0 越界(第一版·截断口径)=18 越界(第一版·RNE 口径= r1 C++)=11 max|d|=0.00193773 max(dv/tol_new)=0.999834 位差 0/1/>=2 = 6144/0/0
[OP-A] 契约档：与 RNE(fp64 Σ_k<6144) 逐位相同 2560/2560（须 == 全部）；负向档：与 Σ_k<6080 逐位相同 2560/2560（须 == 全部，即负向 dump 确实是「被弄坏」的那份）
[OP-C] 契约档：越界 0/2560 max|d|=0.89187 最大界占用 0.241964；负向档：对 K-1 参考越界 0/2560（须 0）、对全 K 参考越界 1744（须 > 0）
…
[offline] 判定项 6/6 通过；输入缺失 0 项
RESULT: OK（离线复算的三条判据与负向档全部与入库读数一致）
```

> **那行两个数的区别（r3 复审抓到的一处措辞/数字错，这里说清）**：
> `行数=18` = 设备日志里 `[M101][fp]` 的**总行数**；`不同标签个数=9` = 去重后的**标签种类数**
> （本档跑 m=1 与 m=3，**同一个标签名每个 m 各一行** ⇒ 9 种标签 × 2 = 18 行）。
> 本脚本打印、以及上面前后文引用的，**以"不同标签个数 = 9"为准**。
> （r3 之前 README 这里贴的是 `标签数=18` —— 数字与措辞都错；脚本原来只打印标签个数、名字又写作
> "标签数"，容易被读成行数 ⇒ 现在脚本把**两个数各自命名**地打出来，从工件本身消掉这个歧义。）

**设备档一律走进程锁**（`-w` 必须在锁文件之前；进锁后脚本会先 `npu-smi info` 复查；一次进锁一条短命令）：

```bash
# 第三步：核心段三档 + 与归档逐字节对拍（第 9 类替换的等价性见证）
flock -w 300 /tmp/npu0.lock bash m15_layer_loop/evidence/attn_core/run_r2_device.sh core
cd m15_layer_loop/evidence/attn_core/out && PYTHONDONTWRITEBYTECODE=1 \
  /usr/local/python3.12.13/bin/python3 ../../../../m10_attn_decode/check_ref.py
# （判据用 donor 的脚本**原样**跑，不复制判据；PYTHONDONTWRITEBYTECODE 防止在 scope 外写 __pycache__）

# 第三步负向对照
flock -w 300 /tmp/npu0.lock bash m15_layer_loop/evidence/attn_core/run_r2_device.sh negkv
cd m15_layer_loop/evidence/attn_core/out_negkv && PYTHONDONTWRITEBYTECODE=1 \
  /usr/local/python3.12.13/bin/python3 ../../../../m10_attn_decode/check_ref.py 256 300

# 第四步：o_proj（契约档 y + 指纹；并复跑一次比确定性）——M15AC_CASES 在这里是 m 列表
flock -w 300 /tmp/npu0.lock bash m15_layer_loop/evidence/attn_core/run_r2_device.sh oproj

# 第四步的**离线**独立复核（自己的 numpy fp64 参考，不碰设备；rc=0 表示与入库读数一致）
cd m15_layer_loop/evidence/attn_core && PYTHONDONTWRITEBYTECODE=1 \
  /usr/local/python3.12.13/bin/python3 check_oproj_dumps.py out_oproj \
  --log logs/m101_oproj_run_20260927.log
```

**跑设备前的纪律**（本 mission 逐次执行）：`npu-smi` 复查（`run_r2_device.sh` 在锁内做，并打印输出）；
`df -h /` 确认根卷有空间（r2 期间读数 ≈ 308–319 G 可用 / 97% 已用）；
**一次进锁一条短命令**（core ≈ 8 s、oproj ≈ 22 s、negkv ≈ 8 s）、单进程、每条 `timeout 280`；
rc 写进日志尾部（`run_r2_device.sh` 落 `rc=<n>`）。
r1 期间 wt-100 的 `run_wire_evidence.sh` 在连续多轮占设备，本 mission 按"看到别人在跑就等"让位。

---

## 7. 纪律自查（实跑读数；命令与输出都在本 tip 上可复跑）

① **计算路径的纪律面**（复审 r1-F3：**原来那条 grep 是空洞的，已换掉**）

原写法（r1）：`grep -cE 'AscendC::(Cast|Duplicate|Exp|Add|Sub|Mul|Muls|Max|Min|Reduce|Select|Compare)\('`
⇒ 报"经典 memory-based 向量 API **0 处**"。**这条读数证明不了任何事**：抽取件里有 `using namespace AscendC;`，
向量/算术调用**都是非限定的**，那个模式**永远匹配不到**。复审指出的这一点成立。

替换成 `evidence/attn_core/audit_discipline.py`：**先把注释剥掉**，再在同一个"正文切片"上把向量/算术 API
按「在 `__VEC_SCOPE__` 块**内**」与「块**外**」分别计数。判别规则：
**块外的"计算类"调用即违规**；块外的**搬运/常量填充**（`DataCopy`/`Duplicate`/`Mmad`/`LoadData`）属豁免面。

```bash
python3 m15_layer_loop/evidence/attn_core/audit_discipline.py --pair
```

输出（本 tip，r2 实测；`A = m15_attn_core.h` 正文切片、`B = m10_attn_decode.asc` 正文切片）：

```
  块内计算类调用 40 处；块外计算类调用 0 处；块外搬运用 22 处；块外 Reg 专用原语 0 处
  资源管理函数 TPipe/TBuf/TQue/AllocTensor/TBufPool/Queue/Matmul/GetTPipePtr/TSCM 全部 A=0
  LocalTensor 出现数（本切片）  A=67  B=67  相同
  ⇒ '计算走 RegBase VF、memory-based API 只用于搬运/常量填充' 这条口径在本切片上**有判别力地**成立
```

**这条计数不是摆设 —— 它第一次跑就把一处真违规挖出来了**（详见 §2.4：FD 归并内循环里两处经典
`Muls`/`Add`，donor 同款；已按塔 2026-10-04 的裁决改成 RegBase VF，改后块外计算类调用归 0）。

**"2 → 0"要把两件事分开说（复审 r2 的 clarity point，这里照改）**：
(a) **真违规**：r1 抽取件与 donor 各**确有 2 处**块外计算类调用（就是上面那两条，工具会把它们逐处定位）；
(b) **工具的假命中**：本工具的第一版把**注释**也算进去了，于是第 9 类替换的注释里引用被删掉的原文时
    又报出 2 处。现在工具**先剥注释**再计数（`strip_comments()`），(b) 这一类不再出现。
⇒ "块外计算类调用 2 → 0" 里，(a) 是代码被修好的结果、(b) 是工具有了分辨力的结果，两者独立。
复审自己也复现了 (a)：把同一把尺子用在 r1 的头与 donor 上，都是"块内 38 / 块外计算类 2"且**逐处定位到那两行**；
并做了变异测试（在 `__VEC_SCOPE__` 外塞一条 `Muls` ⇒ rc=1 且定位到该行）⇒ 这个计数**有牙**。
同一把尺子也量了本 mission 新写的 `m15_attn_oproj.h`：
`python3 … audit_discipline.py m15_layer_loop/m15_attn_oproj.h` ⇒ **块内 9 处 / 块外 0 处，rc=0**。

（附：`CrossCore(Set|Wait)Flag` 35 处 = `SetFlag|WaitFlag` 35 处 ⇒ 本段没有核内 set/wait flag —— 这条
读数不受 F3 影响，因为它是**等式**而不是"某模式 0 命中"。`__VEC_SCOPE__` 6 处 = 5 个 `__simd_vf__`
函数体 + 1 处 donor 说明注释（构成见 §1.1）；number 的口径写在脚本里。）

② **本段不引用主 KV 几何**（M98 的同步要求）：

```bash
git diff --name-only main...HEAD | grep -v '^m15_layer_loop/evidence/' | xargs -r -I{} sh -c \
  'printf "%s: " {}; LC_ALL=C grep -c "M15KV_KV_" {} || true'
```

输出：5 个源码文件逐行 `<file>: 0`。

③ **边界**：`git diff --name-status main...HEAD`（三点式）—— 实跑输出（**20 个文件，全部在 scope 内**）：

```
M  m15_layer_loop/CMakeLists.txt
A  m15_layer_loop/evidence/attn_core/.gitignore
A  m15_layer_loop/evidence/attn_core/README.md
A  m15_layer_loop/evidence/attn_core/audit_discipline.py
A  m15_layer_loop/evidence/attn_core/check_oproj_dumps.py
A  m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py
A  m15_layer_loop/evidence/attn_core/logs/{m101_bin_sha256.txt, m101_core_check_ref_20260927.log,
   m101_core_negkv_check_ref_20260927.log, m101_core_negkv_run_20260927.log, m101_core_run_20260927.log,
   m101_oproj_offline_check_20260927.log, m101_oproj_r1_badcriterion.log, m101_oproj_rerun_20260927.log,
   m101_oproj_run_20260927.log}
A  m15_layer_loop/evidence/attn_core/run_r2_device.sh
A  m15_layer_loop/m15_attn_core.asc
A  m15_layer_loop/m15_attn_core.h
A  m15_layer_loop/m15_attn_core_host.h
A  m15_layer_loop/m15_attn_oproj.h
```

越界行数 = 0（用 scope 白名单正则过滤后计数）。特别地：`m10_attn_decode/**`、`m15_layer_kernel.h`、
`m15_layer_loop.asc`、`m15_attn_kv*.h`、`m15_attn_prolog*.h`、`m15_hc_*`、`m15_ple*`、`docs/**`、
`tools/**` 都**不在**这张表里。运行产物（`evidence/attn_core/out*/` 的 `.bin`）由
`evidence/attn_core/.gitignore` 排除，留在 worktree 里供就地复跑判据（§6）。

④ **绝对断言扫描**：对本轮新增/改的源码与脚本（含 3 个新脚本）扫
`已全部|无残留|0 命中|零命中|全部通过|一定|必然|肯定|绝对|全都`，**逐个命中归因**（不写成"扫净"）：

```bash
for f in m15_layer_loop/CMakeLists.txt m15_layer_loop/m15_attn_core.asc m15_layer_loop/m15_attn_core.h \
         m15_layer_loop/m15_attn_core_host.h m15_layer_loop/m15_attn_oproj.h \
         m15_layer_loop/evidence/attn_core/{lift_attn_core_segment,audit_discipline,check_oproj_dumps}.py \
         m15_layer_loop/evidence/attn_core/run_r2_device.sh; do
  LC_ALL=C grep -nE "已全部|无残留|0 命中|零命中|全部通过|一定|必然|肯定|绝对|全都" "$f"
done
```

输出：**1 处命中**，即 `m15_attn_core_host.h` 的 printf 文案
`"[M101] 与归档逐字节对拍（refDir=%s，用 \`head\` 而不是绝对断言）:"` —— 命中的是"绝对"两个字，
**本身就是反绝对断言**的措辞，属自匹配噪声。其余文件 0 处。
**r1 那两处 `必然 PASS` 的措辞已在 r2 改掉**（改成"本 mission 的 negkv 档里它仍 PASS，这是实测"）
—— 这是 r1 扫描的归因结论落地，不是新发现。

⑤ **`/tmp` 与运行产物**：本 mission 在 `/tmp` 只留过构建/运行日志与临时切片，r2 收工前已清；
运行 dump 落在 `evidence/attn_core/out{,_oproj,_oproj_rep,_negkv}/`（约 40 MB，**已被
`evidence/attn_core/.gitignore` 排除、不入库**），**刻意保留**在 worktree 里，让评审可以就地复跑
donor 的 `check_ref.py` 与 `check_oproj_dumps.py` 而不必重跑设备（各自约 8–22 s 设备时间）。

---

## 8. r1 复审 5 条 p2 的逐条对账（r2）

复审文件：`.tower/comms/reviews/review-feat-m101-attention-core-segment-lift-and-o-p-reviewer-m101-r1.md`
（`round: 1`，`reviewed_commit: 6e6e649…`）。**逐条列，不汇总。**

### F1 [p2] `m101op_m{1,3}_y.bin` 是 KMINUS1 负向档的输出 ⇒ OP-A/OP-C 无工件

- **改了什么**：(a) 每份工件在**产生的那一刻**落盘，文件名自述身份，含义漂移的旧 `_y.bin` 删除；
  (b) 每份 dump（含不落盘的 W）打印 `[M101][fp]` 指纹进**入库日志**；
  (c) 新增**离线**独立复核脚本 `evidence/attn_core/check_oproj_dumps.py`。
- **在哪一行**：`m15_layer_loop/m15_attn_core_host.h` —— `Fnv1a64`/`DumpAndPrint`（`DumpBin` 之后）、
  OP-A 契约档落盘（`DumpAndPrint("OP-A y (contract)"…)`）、OP-A 负向落盘、OP-B 契约 `tdev` 落盘、
  OP-C 契约/负向落盘、输入三件套 + W 指纹（循环开头）。旧 dump 块（`m101op_m%u_y.bin`）已删。
- **怎么自证**：`logs/m101_oproj_run_20260927.log` 里 9 条 `[M101][fp]` 行；
  `PYTHONDONTWRITEBYTECODE=1 /usr/local/python3.12.13/bin/python3 check_oproj_dumps.py out_oproj
  --log logs/…` ⇒ **rc=0**，
  且它用**自己的 numpy fp64 参考**算出：OP-A 契约 `2560/2560`、`7680/7680` 逐位相同；
  OP-A 负向与 `Σ_k<6080` 逐位相同（证明那份 dump 确实是"被弄坏"的）；OP-C 负向对全 K 参考
  越界 `1744/2560`、`5097/7680`（与 C++ 日志**逐项相同**）。读数表见 §4.3.1。

### F2 [p2] mode-4「新子空间」低估了别名，且 `FLAG_PER_CORE = 16` 装不下 AIC 侧 16/17

- **改了什么**：把"新子空间"的措辞换成官方口径（AIV 仍是同一批 0-15、只有 AIC 放宽到 0-31、
  16-31 ↔ AIV1、跨模式复用须先 drain），并写明本段 AIC 侧的 16/17 需要顶层为 `AIC ∧ mode==4`
  单独放行；同时点到主线已落的 `FlagIdLimit` / `FLAG_PER_CORE_AIC_MODE4`。
- **在哪一行**：`evidence/attn_core/lift_attn_core_segment.py` 的 HEAD 文本（生成物头部
  "② **mode-4 不是「新子空间」**…" 一段）+ 本 README §2.2 第 2 条。
- **怎么自证**：`grep -n "不是「新子空间」" m15_layer_loop/m15_attn_core.h`；
  官方依据原文：`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md`
  的「flagId取值范围说明」（本次实际 grep 到第 104–113 行与第 145–152 行两处）；
  顶层口径：`grep -n "FLAG_PER_CORE_AIC_MODE4\|FlagIdLimit" m15_layer_loop/m15_layer_resources.h`。
  **复核时的时点说明**：复审读的是我 worktree 里那份 `m15_layer_resources.h`（该文件第 443 行
  `FLAG_PER_CORE = 16`，与我 scope 无关、我未改）；主线在该文件上已落 32 的放行。
- **我没有按复审原话"改 `m15_layer_resources.h`"**：它不在本 mission scope（只读引用，需改先报塔）。
  本 mission 做的是"把本段的**需要**讲清"，接线归 M102。

### F3 [p2] §7① 的「经典 memory-based 向量 API 0 处」是空洞计数

- **改了什么**：删掉那条 grep 与它的读数表，换成**有判别力**的计数脚本：先剥注释，再按
  `__VEC_SCOPE__` 块内/块外分别计数"计算类"与"搬运/填充类"，块外出现计算类即违规。
- **在哪一行**：`evidence/attn_core/audit_discipline.py`（新，`strip_comments` / `vec_scope_spans` /
  `count_calls` / `main`）；README §7① 整段重写。
- **怎么自证**：`python3 …/audit_discipline.py --pair` ⇒
  **块内计算类 40 / 块外计算类 0 / 块外搬运 22 / 块外 Reg 专用原语 0**，rc=0；
  `LocalTensor` 出现数 A=67 B=67（与 donor 相同）。同一把尺子量 `m15_attn_oproj.h` ⇒ 块内 9 / 块外 0。
- **附带结论（这才是这条的真正价值）**：这个计数**第一次跑就报出 2 处真违规**（不是计数方法的错），
  已按塔的 B 方案改掉 ⇒ 见 §2.4 与下面「塔裁 B 方案」。

### F4 [p3] "1/4" 应为 1/2；"18 处" 与保留日志/复审重建都不符

- **改了什么**：① 倍数改成 **1/2** 并给出机制与两个具体数（旧 tol 4.884e-4 vs 更正后 9.767e-4）；
  ② 把"18 处"的来源变成**可复跑的脚本打印项**（同批 dump 三套口径各算一遍）；
  ③ 新增 §4.3.2 设备确定性读数（两次运行 16/16 dump 逐字节相同），把数字差异限定在**判据侧**；
  ④ **根因由复审 r2-N2 定位、本轮已落档**：取 bf16 位型时 **RNE vs 截断**——
  r1 的 C++ 用 `FloatToBf16`（RNE），我 r2 的重建用 `(x >> 16)`（截断）⇒ 同一批 dump 上
  RNE 恰好复现 r1 的 **11/31**、截断给出 **18/54**。详见 §4.3.3（含逐字复现命令）。
  ⑤ 并更正一处**我写错的引用**：§4.3.3 原写"r1 日志只跑了 m=1"，但入库的 r1 日志第 48/62 行
  就是 m=3 那一档（越界 **31**）—— 已按 artifact 改正。
- **在哪一行**：README §4.3.3（整节新增）；`check_oproj_dumps.py` 的 `legacy_grid_ulp` + `[OP-B]` 打印行。
- **怎么自证**：`logs/m101_oproj_offline_check_20260927.log` 的 `[OP-B] … 越界(更正后)=0 越界(第一版写法)=18`
  与 m=3 的 `54`；§4.3.2 的 `[det]` 16 行。

### F5 [p3] negkv 注释把 D 写进"必须变红"；头部缺 footprint 与 prefill 边界

- **改了什么**：① 两处注释改成 **A/C/E/F** 并说明 D 为什么不会红；② 生成物头部新增
  **资源足迹**（L1 288 KB/512 KB、L0A/L0B 64 KB、L0C 64 KB/256 KB、UB 峰值 64,512 B ≈ 63 KB/248 KB
  —— 全部是段内编译期常量）与 **prefill 边界**（decode 形状、不能直接吃 `m=4097`）。
- **在哪一行**：`m15_layer_loop/m15_attn_core.asc`（negkv 分支注释）；
  `m15_layer_loop/m15_attn_core_host.h`（文件头负向对照说明）；
  `evidence/attn_core/lift_attn_core_segment.py` 的 HEAD 文本（"资源足迹"与"prefill 边界"两段）。
- **怎么自证**：`grep -n "A/C/E/F" m15_layer_loop/m15_attn_core.asc m15_layer_loop/m15_attn_core_host.h`；
  `grep -n "资源足迹\|prefill 边界" m15_layer_loop/m15_attn_core.h`（生成物里）+
  `python3 …/lift_attn_core_segment.py --check`（rc=0 ⇒ 头部文本确实是生成器产出的、不会漂移）。
- **一处与复审数字的出入**：复审写「L1 ≈ 416 KB」，我按段内常量算出来是 **288 KB**
  （`L1_OFF_KV1 + L1_KV_BYTES` = 32,768 + 131,072 + 131,072 = 294,912 B），
  与 donor README §1 的「L1 = P 3×8KB + Q 8KB + KV 2×128KB = 288KB/512KB」一致。
  我按自己的算式入文（294,912 这个数即 §2.2 里 L1 的末端）。
  **复审 r2 已主动撤回它 r1 的「L1 ≈ 416 KB」**：它核出自己把 KV 区**重复计了一次**
  （它的式子 `L1_OFF_KV1 + 2×L1_KV_BYTES` = 163,840 + 262,144 = 425,984 B），
  并独立复算确认本段的 **294,912 B = 288 KB** 是对的（`L1_OFF_KV1 + L1_KV_BYTES`，
  其中 `L1_P_BYTES = M_PAD*S2T*2 = 8192`）。⇒ **本段 L1 足迹以 288 KB 为准，不存在另一个有效数字。**

### 塔裁 B 方案（不属于复审 5 条，单独列）

- **改了什么**：FD 归并内循环的两处经典 `Muls`/`Add` → RegBase VF（`AccFmaRowVf`），
  登记为生成器**第 9 类整块替换**，并**显式声明抽取不再是纯逐字**。
- **在哪一行**：`lift_attn_core_segment.py` 的 `VF_ACC_FMA` / `OLD9_A` / `NEW9_A` / `OLD9_B` /
  `NEW9_B` / `BLOCK_REPLACEMENTS` / `_block_replace`；生成物 `m15_attn_core.h:858`（VF 定义）与
  `:1144`（调用点）。
- **怎么自证（三条，全实跑）**：
  1. **与归档 dump 逐字节对拍**：25 个文件"逐字节相同"、**0 个不同**（§3.2 ①，r3 二进制 `ab12e2e4…`）；
  2. **donor `check_ref.py` 三档**：`判定项 23/24 | 报告项 27 | guard 9/9 | 输入缺失 0`，
     位级率与 E/F 的 maxAbsErr 与 r1 逐位相同（§3.2 ③ 的对照表）；
  3. **`audit_discipline.py --pair`**：块外计算类调用 **2 → 0**（§7①）。
  ⇒ 三条都成立，不存在"停手报塔"的情形。依据（人类逐字规则）逐字引在 §2.4 与生成器的差异表注释里。

---

## 9. r2 复审 3 条 p2 的逐条对账（r3）

复审文件：`.tower/comms/reviews/review-feat-m101-attention-core-segment-lift-and-o-p-reviewer-m101-r2.md`
（`round: 2`，`reviewed_commit: d9e3793…`）。**逐条列，不汇总。**

### N1 [p2] `Fnv1a64` / `fnv1a64` 名不符实（offset basis 少了一位）

- **选的处理**：复审建议的 **(a) 把常量改对**（不改名）。理由：这个工件存在的意义就是让第三方**按标准算法**
  复现指纹；改名只能免责，不能让它可用。
- **改了什么**：offset basis `1469598103934665603` → **`14695981039346656037`**（= `0xCBF29CE484222325`）。
  prime `1099511628211` 本来就对，未动。
- **在哪一行**：`m15_layer_loop/m15_attn_core_host.h::Fnv1a64`（常量 + 注释里写全算法与两个自查向量）；
  `evidence/attn_core/check_oproj_dumps.py::fnv1a64`（同实现 + docstring 写明标准算法）；
  `check_oproj_dumps.py::main` 入口加两条 `assert`（`fnv1a64(b"")==0xcbf29ce484222325`、
  `fnv1a64(b"a")==0xaf63dc4c8601ec8c`）。
- **怎么自证**：
  1. 脚本入口的自查 `[selfcheck] FNV-1a 64 标准向量通过…`（在入库日志首行，常量再写错这里立刻 AssertionError）；
  2. **用一份与我的实现无关的标准 FNV-1a 64 写法**独立复算 dump：
     `fnv1a64(out_oproj/m101op_m1_attn.bin) = 0c8e0636a7785231`，与日志该行**逐位相同**
     （八个 16 进制常量 `0x100000001b3` 形式写；自查向量 `0cbf29ce484222325` / `af63dc4c8601ec8c` 也对上）；
  3. **重跑了那条约 30 s 的设备档重发指纹**：`flock -w 300 /tmp/npu0.lock bash …/run_r2_device.sh oproj`
     ⇒ 新指纹（W `29dd51e39fc752bb` 等）全部进 `logs/m101_oproj_run_20260927.log`，
     且离线复核脚本绑定仍然 rc=0。
- **顺带**：因为 host 侧变了，r3 **把三条设备档都重跑了一遍**（core / negkv / oproj，各一条短命令），
  让 §3/§4 的全部读数落在**同一个二进制**（`ab12e2e4…`）上；读数与 r2 逐项相同（见 §3.2③/§3.3/§4.3）。

### N2 [p3] §4.3.3「r1 日志只跑了 m=1」是假的；11/18/19 可定位（复审替我定位了）

- **改了什么**：
  1. **更正引用错误**：入库的 `logs/m101_oproj_r1_badcriterion.log` **确有** m=3 那一档
     （第 48 行 `[op] m=3`、第 62 行 `判定项[OP-B…]: FAIL 比较 18432 元素、越界 31`）。§4.3.3 的表已改成 **31**。
  2. **把根因写进去**：取 bf16 位型时 **RNE vs 截断** —— r1 的 C++ 用 `FloatToBf16`（RNE），
     我 r2 的重建用 `(x >> 16)`（截断）。**同一批 dump 上**：RNE 口径 ⇒ **11（m=1）/ 31（m=3）**（= r1 日志，
     逐字对上）；截断口径 ⇒ **18 / 54**（= 我脚本原来的数）；复审的 r1 重建（取两邻居较小者）⇒ 19 / 60。
     ⇒ 三个数字是**同一个已删除代码路径的三种重建**，不是数据差异。
  3. **把口径差做成可复跑**：`legacy_grid_ulp(ref, rne=False|True)` 两个分支，`[OP-B]` 行同时打印三套。
- **在哪一行**：`evidence/attn_core/README.md` §4.3.3（表 + 复现命令）与 §8 的 F4 条目；
  `check_oproj_dumps.py::legacy_grid_ulp`（`rne` 形参 + docstring 写明两个口径各复现哪个数字）。
- **怎么自证**：
  ```bash
  cd m15_layer_loop/evidence/attn_core && PYTHONDONTWRITEBYTECODE=1 \
    /usr/local/python3.12.13/bin/python3 check_oproj_dumps.py out_oproj \
      --log logs/m101_oproj_run_20260927.log 2>&1 | grep '\[OP-B\]'
  #  [OP-B] n=6144  越界(更正后)=0 越界(第一版·截断口径)=18 越界(第一版·RNE 口径= r1 C++)=11 …
  #  [OP-B] n=18432 越界(更正后)=0 越界(第一版·截断口径)=54 越界(第一版·RNE 口径= r1 C++)=31 …
  ```
  以及 `grep -n "越界 31" logs/m101_oproj_r1_badcriterion.log`（第 62 行）。

### N3 [p3] 文档里的 `python3 check_oproj_dumps.py` 按字面跑会失败

- **改了什么**：§4.3.1 与 §6 里那三处调 `check_oproj_dumps.py` 的命令**全部改成显式
  `/usr/local/python3.12.13/bin/python3`**；§6 开头加一段**解释器说明**（本机 `python3` 没有 numpy；
  只有 numpy 依赖的两个脚本需要显式解释器，`lift_attn_core_segment.py` / `audit_discipline.py` 用 `python3` 即可）；
  并把 §6 的逐字实跑输出（含 `python3 -c "import numpy"` 的 `ModuleNotFoundError`）贴进文档。
- **在哪一行**：`evidence/attn_core/README.md` §6（解释器说明段 + 逐字输出块）、§4.3.1 的代码块。
- **怎么自证**：§6 里那段输出**就是本轮实跑的**（`LC_ALL=C`）；核心四行：
  `--check` rc=0、`audit --pair` 块外 0、`audit m15_attn_oproj.h` 块外 0、
  `python3.12 -c "import numpy"` ⇒ `2.5.1` 而 `python3 -c "import numpy"` ⇒ `ModuleNotFoundError`。

### 复审的另外两件（记录，不改代码）

- **复审撤回了它 r1 的「L1 ≈ 416 KB」**：它核出自己把 KV 区**重复计了一次**（`L1_OFF_KV1 + 2×L1_KV_BYTES`
  = 163,840 + 262,144 = 425,984 B），并独立复算确认本段的 **294,912 B = 288 KB** 正确。
  ⇒ 已在 §8 的 F5 条目里写明"复审已撤回、本段取 288 KB"，避免后来人看到两个互相矛盾的数字。
- **§5 的 10 条未完成项**：本轮**未增未减**，也未扩到别的 mission 的 scope。
