# docs/20 —— M107：M15 段体合规清盘（矩阵乘法 / mmad + 同族同步竞态）

**出处**：本文件由 mission **M107**（read-only survey）产出，落库供塔照它派活。
**范围**：`m15_layer_loop/**`（代码级通读）。本 mission 未修改 `m15_layer_loop/**` 下任何文件；
本文件是 M107 写出的唯一文件。

## §0 边界与口径（读之前先看这一节）

**读的是哪个版本**：不可变 rev **`20bd20d659b11cbb5e6e6d317921ecb3c16e2629`**（= M107 开工时的 `main`）。
**本文件里出现的所有行号都以这个 rev 为准**；为了抗漂移，每条都另给一个可 grep 的**内容锚点**
（符号名 / 逐字字符串）。行号只作查阅提示，**判据请用锚点**。

**零设备档**：本节全部结论是**代码级判定 + grep 证据**，没有新做实验、没有上设备。
凡是需要设备读数才能定的，一律写进 §5「待确认」，不写成结论。

**已知口径（人类原话，本 mission 的最高约束）**：

| 编号 | 逐字 | 在本文件里的落点 |
| --- | --- | --- |
| ① | 「凡是涉及矩阵乘法的操作都需要用 mmad 实现，不管 M=1 或者是多大」 | §2 清单 A |
| ② | 「发现问题之后，确实需要检查一下其他地方有没有同样的问题，不过快速读一遍代码 review 一下就行了」 | §2.3-B5 / §3.3 末 / §3.5 / §3.6（同族形态） |
| ③ | 「E=512/topk=10，打通阶段也需要使用真实 shape 呀」 | §1.4 / §2.5（形状前提） |
| ④ | 「本来就不应该用 scalar pipe 写 GM，为什么要这么做」 | §3.3 |
| ⑤ | 「不得写「已全部 / 无残留 / 0 命中」类绝对断言」 | 全篇：只写「本次命令的输出是什么」，不写「没有了」 |

**在飞占用（不要把别人的改动当既成事实）**：以下文件在 main 上的位置**照旧给出**，
但同一个文件正在被别的 mission 改，**命中即注明归属**：

| 文件 | 归属（在飞） | main 上的状态 |
| --- | --- | --- |
| `m15_layer_kernel.h` / `m15_layer_loop.asc` / `m15_hc_host.h` / `m15_ple_wire.h` | **M100** | **已随 `d348547` 并入 main**（见 §0.2）；`m15_ple_wire.h` 由 M100 机械生成 |
| `m15_attn_core.h` / `m15_attn_core.asc` / `m15_attn_core_host.h` / `m15_attn_oproj.h` | **M101** | main 上**仍不存在**（截至 `d348547`，r1 复审已核） |
| `m15_moe_layer.h` / `lift_moe_segment.py` | ~~**M105**~~ **无在飞归属** | 见 §3.2（第一例由 M105 修）；**M105 已合入**（`feat/m105-m15-moe-diagnostic-slot-race-fix`），但它做的是诊断槽竞态、**未做 WO-A2** ⇒ 该 P1 修单**仍待做**（见 §4 WO-A2 的归属更正） |
| `m15_ple.asc` / `m15_ple_wire.h` | **M111**（PLE 落盘通路修复） | 见 §4 的派单时序注 |

**仓内规则依据（引规则编号前已 grep 过原文）**：
`docs/05-megakernel-design.md` §6.1 的四条规则 —— 原文行号为 `:158`–`:176`，标题逐字为
**「§6.1 计算路径规范：四条 + 豁免清单」**。与本清盘直接相关的两条：

- **ⓐ** 计算路径必须走 vector function（`__VEC_SCOPE__` / `__simd_vf__`）；
- **ⓑ**「**搬运 / 矩阵类仍走 memory-based，不在禁令内**」——原文列举 `Mmad`/`MmadMx`/`Fixpipe`/`Nd2Nz`
  等，**理由：无 VF 等价物**。
  ⇒ 人类口径 ① 与仓内既有规则是**同向**的：矩阵类**本来**就该走 mmad 这条路；
  「用 VF 的逐元素 MAC 去做矩阵乘法」是**把矩阵乘法错放进了 ⓐ 的路**。

另有一条与「M=1 也要走 mmad」直接相关的仓内陈述（逐字）：
`docs/11-attn-analysis.md:103` ——「m=1 在 mmad 被 pad 成 16（`matmul.h:670-672` "m==1→16"——
**mega kernel 同样 m 走 16 行矩阵**）」。⇒「M=1 所以走向量」在本工程里**没有设计依据**。

### §0.1 塔裁三条口径（2026-09-27，M107；**统一判据**）

本节以下各节的判定一律以这条为准（它是塔对人类口径 ① 的正式解读，优于本文件先前按字面
「退化维都算」的读法）：

> **是否有「固定的矩阵 / 权重操作数」参与收缩 —— 有就必须 mmad；没有，就不是本规则管辖的矩阵乘法。**
> （塔给出的**理由**是**数据复用**：没有可复用的矩阵操作数时，强行 mmad 只能是几何填充、**零数据复用**，
> 违反的是规则的**目的**。）

三条具体裁决（逐字要点）：

| 裁决 | 内容 | 依据（逐字） |
| --- | --- | --- |
| **B1** | `M=N=1`、**无权重操作数**的单点积 → **不要求 mmad** | 「没有任何矩阵操作数（没有可复用的权重），强行 mmad 只能是 1×1 几何填充、**零数据复用** ⇒ 违反的是规则**目的**，不是遵守它」 |
| **B2** | `K=1` 的 rank-1 外积 / 逐元素 scale 后累加 → **不要求 mmad** | 「**K=1 意味着没有归约维**，数学上不是 contraction」 |
| **B4** | 系数随 `j` 变的门控归约 → **不是矩阵乘法** | 「不是固定线性算子」—— 与本文件用 `GateMixStage`/unpermute 的分界一致 |

塔同时给了四个**已裁决的样板**（本文件 §2 的违规表按它们逐条对齐）：
router 的 `x @ W[512,2560]^T`（W 是**真权重矩阵**）→ **必须 mmad，P1 不变**；
`PleGemv` 的 `[1,2560] @ [2560,12800]`（**权重 65.5 MB/token**）→ **必须 mmad，P1 不变**；
`GdnHeadRecurrence` 的 `S·k` / `S·q` / `k⊗d`（**S 是 128×128 状态矩阵**）→ **必须 mmad**，
但精度改造取决于 M106（塔**采纳**本文件 §4 WO-A3 的两段式建议）；
`m15_hc_layer.h:CombineStage`（`K=1`）与 `m15_ple.asc:PleGateItem` pass C（无权重）→ **移出违规表**。

⇒ 本文件据此把原 V5 / V6 挪进 §2.4（N10 / N11），并在 §2.4 逐条写明塔裁与理由。
**r1 复审期间塔又裁了两条**（B8 `SgateRows` ⇒ 在管辖内；B9 unpermute ⇒ 不在管辖内）⇒
原 V4 也挪进 §2.4（N12），**本文件已不再有任何「待定」条目** —— 详见 §2.3 与 §7.1。

**一处需要点明的表面张力（我按塔的分组归法记录，不自行改判）**：塔的样板里把
`GdnHeadRecurrence` 的 `S·k` / `S·q` / `k⊗d` **三个一起**判为「必须 mmad」，其中 `k⊗d` 单独看是
`K=1`（按 B2「`K=1` 不是 contraction」单看**不构成** contraction）。我的处置：
**按塔的分组记录为一条**（整个 delta-rule 状态更新步骤一起改），并注明其形式依据 = 这三者在同一
状态更新里不可分（mmad 形态下就是 `S ← diag(e^g)·S + k·d^T`，外积项是等式的右端项，不是独立算子）。
若塔认为 `k⊗d` 应单列剥离出「必须 mmad」的范围，**请直接改判** —— 在本工程里它不影响工作量
（V3 的改造粒度本来就是整个 `GdnHeadRecurrence`）。

### §0.2 合并状态（**时效事实，随复审轮次更新**）

- **本文件读的基线 rev 仍是 `20bd20d659b11cbb5e6e6d317921ecb3c16e2629`**（= M107 开工时的 main），
  所有行号以它为准；
- **但 main 在 r1 复审期间已前移到 `d348547a66c8148f8bad1bebce611694ef19adf8`**
  （`git log --oneline 20bd20d..main` → `d348547 Merge branch 'feat/m100-ple-segment-wiring-into-the-layer-k'`
  等）⇒ **M100 已并入**，`m15_ple_wire.h` 现是 main 的入库文件；
  **M101 截至 `d348547` 仍未并入**。
- ⇒ 本文件里凡写「归 M100 / main 上尚不存在」的段落，请按本节理解：
  **M100 的改动已是既成事实**（`m15_ple_wire.h` 存在，且 §2.2-V1 已按「两份副本」写）；
  凡引 `m15_ple_wire.h` 的行号，**都是 `d348547` 上的行号**（该文件在基线 rev 上不存在）。
- **C14 的闸门计数是 as-of 本轮 tip 的**；在 `d348547` 上重跑的读数见 §6 C14 的第二段
  （我在 `git archive main` 的干净导出树 + 本文件副本上实跑，rc=0）。

---

## §1 判定口径（我凭什么说「这是矩阵乘法」/「这是违规」）

### §1.1 什么算「数学上是矩阵乘法」（判据 M）

判据按**数学语义**，不按当前实现的形状。**最终以 §0.1 的塔裁判据为准**，下面这条是它的推导
（塔裁把「退化维」这一支收紧了两处：`K=1` 与「无矩阵操作数」——见本条末尾）：

> 存在收缩指标 `k`（求和长度 ≥ 2），把**两个不同的张量** `A`、`B` 配对求和，得到一个对 `A`、`B`
> 联合线性的量：`C[i,j] = Σ_k A[i,k]·B[k,j]`。
> **`M=1` 不改变判定**：GEMV（矩阵×向量）就是同一个双线性型在 `M` 维上取 1 的特例 ——
> 人类口径 ① 明说「不管 M=1 或者是多大」。
>
> **塔裁收紧的两处（§0.1）**：① `K=1` **不是 contraction**（没有归约维）⇒ 不算矩阵乘法；
> ② `M=N=1` 且**没有可复用的矩阵/权重操作数**的单点积 ⇒ 不算（零数据复用，mmad 只能是几何填充）。

按这个判据（**已并入 §0.1 塔裁的两处收紧**），下面的形态**不算**矩阵乘法（我逐条给了理由，避免把归约误伤成矩阵乘法）：

- **A 与 B 是同一张量**（`Σ_j x[j]²`、`‖x‖²`、softmax 分母、RMSNorm 的平方和、l2norm 的平方和）
  —— 那是**二次型 / 范数**，不是双线性型（没有两个独立操作数）；
- **`K=1`（没有归约维）** —— 塔裁 B2：`K=1` 的 rank-1 外积 / 逐元素 scale 后累加**数学上不是 contraction**；
- **`M=N=1` 且无可复用的矩阵/权重操作数** —— 塔裁 B1：没有可复用权重 ⇒ 零数据复用，mmad 只能是几何填充；
- **权重只沿对角出现**（深度可分离卷积：`out[c] = Σ_t w[t,c]·win[t,c]`，`w` 的第 `t` 行只与同一个
  通道 `c` 相乘）—— 数学上是 `(K×C) → C` 那个矩阵的**对角**，是逐元素加权和，不是通道混合。
  **按 §0.1 的塔裁判据复核后维持原判**：把 `C` 个通道的权重组进一个矩阵会得到**块对角**（非对角全是 0），
  没有可跨通道复用的矩阵操作数；且 `K=4` 远小于 mmad 的 `K` 粒度（16）⇒ 归入 B2 的精神（无真收缩、pad 浪费 4×）；
- **系数随输出列 `j` 逐元素变**（门控归约）—— 塔裁 B4：不是固定线性算子；
- **逐元素 / 广播 / 比较 / 选择 / 位宽转换 / 排序**（SwiGLU、sigmoid、RoPE、Sort32/MrgSort 等）。

### §1.2 什么算「违规」（判据 V）

> 判据 M 成立 **且** 该操作在设备侧是**用 AIV 向量/标量指令**（`Reg::Mul` + `Reg::Add` /
> `MulAddDst` + `Reduce<SUM>` / 逐元素标量循环）算出来的，**而不是**走 Cube 路（`Mmad`/`MmadMx`
> + `Nd2Nz` + `LoadData2D` + `Fixpipe`）。依据 = `docs/05` §6.1 ⓑ（矩阵类走 memory-based 的 mmad）。

### §1.3 严重度分级（我用的尺子，供塔核对）

| 级 | 判据 | 含义 |
| --- | --- | --- |
| **P1** | 有**固定的矩阵/权重操作数**参与收缩，且收缩维 `k ≥ 256`、输出维 `N ≥ 512`，在 decode 主路径上（每层/per-token 都跑） | 向量实现与 mmad 吞吐差一个量级的形态，优先改 |
| **P2** | 收缩维 `k = 128` 级（含状态矩阵），或 `k` 小但输出维 `N = 2560`；或改法要先定（见 §4 的依赖） | 该改，或需先定改法 |
| ~~待定~~ | ~~判据 M 成立但档位不清~~ | **本档已清空**：原两条（`SgateRows` / unpermute）已由塔裁 **B8（在管辖内 ⇒ 归 V2）** 与 **B9（不在管辖内 ⇒ §2.4-N12）** 定案 |
| ~~P3~~ | ~~`N=1` 单点积 / `K=1` 外积 / 系数随 `j` 变的门控归约~~ | **本档已由塔裁 B1/B2/B4 取消** —— 这三类都**不在本规则管辖范围**（见 §2.4 的 N4/N10/N11） |

⇒ **本文件不再有任何「待定」条目**：V1/V2/V3 三条 P1（V2 含 `SgateRows`），其余全部落在 §2.4。

### §1.4 形状前提（口径 ③ 的影响，写清楚免得读数被误读）

main 上 `m15_moe_resources.h` 的编译期档位**仍是缩形档**：
`NUM_EXPERTS = 4`、`TOPK_MAX = 4`（内容锚点：`constexpr uint32_t NUM_EXPERTS = 4;` /
`constexpr uint32_t TOPK_MAX = 4;`）。真实档是 **E=512 / topk=10**（人类口径 ③）。
moe 段的**规模无关化**已经做完（M84/M91/M95 已合入：`RT_ROWL = RT_SORTLN = 512` 固定 16 块，
`static_assert(RT_ROWL >= NUM_EXPERTS)`），所以「改常量到 512/10」这条路**在 main 上是通的**，
只是当前没有这么编。§2 里我按**真实档**标注每处的规模，缩形档下数字会小，**判定不变**。

---

## §2 清单 A：矩阵乘法合规

### §2.1 合规锚：已经是 mmad 的矩阵乘法（不要动它们）

全段体里 `Mmad(` 的站点（命令与输出见 §6 C2）：

| 文件 | 符号（内容锚点） | 数学形态 | 说明 |
| --- | --- | --- | --- |
| `m15_gdn_layer.h` | `Cube::Bf16Gemm` → `Mmad(cL0C, a2, b2, mmadParams)` | `S2` in_proj `[m,2560]@[2560,16480]`；`S6` out_proj `[m,6144]@[6144,2560]` | `GdnLayerChain` 的 AIC 段；`m=1` 时按 2 行抬升喂 mmad（`Nd2Nz` 行数=1 的 3510 契约，注释在 `RunTile` 里） |
| `m15_hc_layer.h` | `HcBf16Gemm` → `AscendC::Mmad(cL0C, a2, b2, mp)` | hc mixer 的 down / up 两个投影 | `HyperConnOp::ProcessAic` 的两段 GEMM |
| `m15_attn_prolog_probe.h` | `AttnGemm` → `Mmad(cL0C, a2, b2, mp)` | attention 前端 4 个投影：`q|gate`（`QG_W=12288`）、`k_proj`、`v_proj`（各 512）、`index_qk_proj`（640） | 由 `m15_layer_kernel.h` 的 `KIND_ATTN` AIC 分支调用（main 上已接线） |
| `m15_moe_layer.h` | `MXFP4GemmItem` → `AscendC::MmadMx(cL0C, a2, b2, mmadParams)` | gate_up `[t,HIDDEN]@[HIDDEN,GU_N]`；down `[t,INTER]@[INTER,HIDDEN]` | 槽位 = 4 routed + 1 shared（`NUM_SLOTS`），**这是 MXFP4 量化权重，必须 `MmadMx`** |

**在飞补充（不要记到 main 头上）**：
- **归 M101**：`feat/m101-…` 上新增的 `m15_attn_core.h` 用 `Mmad`（BMM1 = `Q·K^T`、BMM2 = `P·V`），
  `m15_attn_oproj.h` 的 `OProjGemm` 用 `Mmad`（o_proj）。**这四个文件在 main 上还不存在**
  （我核过：`git diff --stat main...feat/m101-…` 列出它们是新增文件；**截至 `d348547` 仍未并入**，
  r1 复审独立复核过同一结论）。⇒ `Q·K^T` / `P·V` / `o_proj`
  这三类**在 main 的交付形态里根本没有实现**（attention 相位 A 在 main 上是 `m15_attn_passthrough_body`
  直通占位），所以它们在 main 上**无从判违规**；M101 的在飞实现**已经是 mmad**。
- **归 M100（已并入）**：`m15_ple_wire.h` **已随 `d348547` 进 main**（机械生成物）——
  它把 `m15_ple.asc` 的 PLE device 段复制成了**第二份**，**层路径跑的就是它**
  ⇒ 见 §2.2-V1（P1-2 的订正）与 §0.2。
- **归 M100（已并入 `d348547`）**：`m15_layer_kernel.h` / `m15_layer_loop.asc` 在飞加 PLE 接线 ——
  **这一条已落进 main**（M100 已并入，见 §0.2），但下面这两条读数是在 **`20bd20d`（我的基线 rev）**
  上跑的，`m15_layer_loop.asc` 本次读数（命令见 §6 C8）：
  `grep -c "GetValue\|SetValue" m15_layer_loop/m15_layer_loop.asc`
  → 输出 `0`、rc=1；`grep -n "__global__\|__aicore__" m15_layer_loop/m15_layer_loop.asc`
  → 空输出、rc=1。⇒ 就**本次这两条命令在 `20bd20d` 上的范围**而言，该文件里没有设备侧算件
  （它是 host 驱动 + host 参考链）。`m15_layer_kernel.h` 里有设备算件，见 §2.1 表格与 §2.4-N9。
  **在 `d348547` 上这两个文件都已被 M100 改过** ⇒ 若要重核这两条读数，请在 `d348547` 上重跑
  （我没有在 `d348547` 上重跑这两条 grep —— 如实标注）。

### §2.2 违规清单（按严重度排序）

下表是 **V1–V3** 三个条目的索引（**V4 已按塔裁 B9 移出**，见下），每条下面给独立小节（含判定依据与建议修单）。
**注意 V1 在 main 上有两份逐字相同的副本**（r1 复审 P1-2），见 V1 小节。

| # | 级 | 文件:符号 | 数学形态（真实档规模） | 当前实现 |
| --- | --- | --- | --- | --- |
| V1 | **P1** | `m15_ple.asc:PleGemv` **与 `m15_ple_wire.h:PleGemv`（两份副本，逐字相同）** | `[1,2560] @ [2560,12800]`（GEMV，K=2560, N=12800） | AIV 向量：每列一次 40 chunk `Mul`+`Add` → `Reduce<SUM>` |
| V2 | **P1** | `m15_moe_layer.h:RouterStage::GemvGroupRow` + `SgateRows`（**两处都判违规** —— `SgateRows` 的档位已由塔裁 B8 定案，见 §2.3） | `logits[m,512] = x[m,2560]@W[512,2560]^T`（**W 是固定权重矩阵**，塔裁样板之一）；`sgate` = 同一权重族的 `N=1` 行 | AIV **单核**（`bid==0`）向量 MAC；**仅 decode 路径**（prefill B4 已是 cube，见 §4 WO-A2） |
| V3 | **P1** | `m15_gdn_layer.h:GdnHeadRecurrence` | per head：`S·k`（`[128,128]@[128]`）、`k⊗d`（rank-1）、`S·q`（`[128,128]@[128]`，**S 是 128×128 状态矩阵**，塔裁样板之一） | `__simd_vf__` 寄存器：`Mul`+`Add`+`ReduceSum`+`MulAddDst` |

**已按塔裁移出本表的条目**：
- **原 V4** = `m15_moe_layer.h:FmaChunk` / `UnpermuteStage`（`[1,topk]@[topk,2560]` 加权行和）
  → **塔裁 B9：不在本规则管辖内**（无固定权重矩阵参与收缩）⇒ 现 **§2.4-N12**，
  **降为性能事项**（原 WO-A4 不再进合规批次，见 §4）；
- **原 V5** = `m15_hc_layer.h:HyperConnOp::CombineStage`（`[2560]⊗[4]`，**K=1**）→ 现 **§2.4-N10**（塔裁 B2）；
- **原 V6** = `m15_ple.asc:PleGateItem` pass C（`M=N=1`，**无权重操作数**）→ 现 **§2.4-N11**（塔裁 B1）。
三条的详细论证都挪进 §2.4（保留原判定依据，加上塔裁与理由），**原 WO-A5 已撤销**（见 §4）。

（原表下曾有一句「V6 归在 P3、口径未定，见 §2.3-B1」—— 那句已随塔裁作废：`P3` 这一档已取消，
B1 已结案，V6 已是 §2.4-N11。）

---

#### V1 —— `m15_ple.asc:PleGemv`（PLE ③ kv 投影）：**P1**

- **锚点**：`__aicore__ inline void PleGemv(uint32_t bid, uint32_t nblk, BodyGm& G, uint32_t nTok, uint32_t negMask)`
  （main 行号 394）；段头注释逐字：「**③ kv 投影：kv[j] = Σ_k wcat[j,k]·emb[k]（AIV GEMV，fp32 逐 chunk 累加 → bf16）**」。
- **数学形态**：`kv = emb @ wcat^T`，`emb: [T,2560]`、`wcat: [12800,2560]`（`wcat = [key;value]`，
  `KVW = HYPER + HID = 12800`）。每个 token 一次 `M=1`、`K=2560`、`N=12800` 的 GEMV。
- **这是不是矩阵乘法**：是。`wcat` 是**权重矩阵**（`GlobalTensor<bfloat16_t> wcat; // [12800,2560]`），
  `kv[t,j] = Σ_k emb[t,k]·wcat[j,k]` 是标准的双线性型；`M=1` 是退化维，按口径 ① 不影响判定。
- **当前实现**：`for t`（token）→ `for j`（该核负责的列）→ 40 个 chunk 的 `Mul(pr, er, wr)` + `Add(acc, acc, pr)`
  → `Reduce<ReduceType::SUM>`。列**逐个**算，**每列一次向量归约**。
  证据（§6 C3/C9）：`grep -n "MulAddDst" m15_layer_loop/m15_ple.asc` → 空输出、rc=1；
  `grep -n "Reduce<" m15_layer_loop/m15_ple.asc` → 6 行输出（main 行号 `365/435/514/516/580/652`），
  其中 `PleGemv` 体内那一次是 `:435`。
- **为什么判违规**：判据 M 成立，且设备侧是 AIV 向量路 ⇒ 判据 V 成立。
  证据（§6 C9）：`grep -n "Mmad" m15_layer_loop/m15_ple.asc` → 空输出、rc=1。
  交叉核对过「不是因为我 grep 的 token 太窄」：把 token 放宽成
  `mmad|Mmad|MMAD|Matmul|MatMul|matmul|Gemm|GEMM|gemm` 时该文件**只有 1 处**命中，
  是第 8 行的 `Gemm` —— 而它逐字出自 `④ 逐 (token,stream) 分组 **Gemma**-RMSNorm 门控`，
  是 `Gemma` 的一部分，**是假阳性**，不是 GEMM 调用。
- **规模感受（真实档，per token）**：`12800 × 2560 = 32.8 M` MAC；权重字节 `12800×2560×2 B = 65.5 MB`
  **每 token 读一遍**。这是 m15 段体里**单点最大**的一处矩阵乘法（对比：GDN in_proj `16480×2560 = 42.2 M`，
  它已经是 mmad）。当前实现把它的**收缩维摊在 AIV 上**、并且**每列一次 `Reduce`**（12800 次归约/token）。
- **⚠ main 上有两份逐字相同的副本（r1 复审 P1-2）**：
  - **第 1 份**：`m15_ple.asc` 的 `PleGemv`（main 行号 394）—— M85 的独立 PLE kernel（自带 `main()`）；
  - **第 2 份**：`m15_ple_wire.h` 的 `PleGemv`（**该文件只存在于 `d348547` 起的 main，故这一处的行号
    是 `d348547` 上的：`:332`**）—— M100 的**机械生成物**
    （文件抬头逐字「**机械生成物，勿手改**」），由生成器 `lift_ple_device_segment.py` 从 `m15_ple.asc`
    逐字抽出 `namespace M85P { … }`；生成器带 `--check` 复跑（不一致即 rc=1）。
    **它已随 `d348547` 并入 main**（M100 合入），而 **层循环调用的正是这一份**：
    `m15_layer_kernel.h:444` 的 `PleGemv(bid, nblk, G, A.pleTok, A.pleNegMask);`
    （`:169` 逐字注明 device 实现 = `m15_ple_wire.h` 的 `PleGather/PleGemv/PleGateItem/PleConvItem`）。
  - ⇒ **本违规在 main 上是「两份、且活的层路径跑第 2 份」**。第 2 份同时含 ①`IdsOneToken` /
    ②`PleGather` / ④`PleGateItem` / ⑤`PleConvItem` 的副本 ⇒ **§3.3 的 G1（`outG.SetValue`）也有第二份**
    （`d348547` 上的 `:187`），**§2.4-N2（PleConvItem）与 N11（PleGateItem）同理**。
    G2/G3（两个 kernel 入口里的 `failG.SetValue` / `G.fail.SetValue`）**没有**副本
    （kernel 入口不在被抽出的 `namespace M85P` 内）。
  - **修它的正确做法 = 改 `m15_ple.asc` + 重跑生成器**（`lift_ple_device_segment.py`，跑 `--check` 确认同步），
    **不是**手改 `m15_ple_wire.h`（手改会被 `--check` 判红）。
- **建议修单（WO-A1）**：
  - scope：**两个文件** —— `m15_layer_loop/m15_ple.asc`（改源）+ **`m15_ple_wire.h`**（同一目录下的
    **机械生成物**，**重跑生成器**产出）；
    判据脚本若需改，`m15_layer_loop/m15_ple_check.py` 与 `ple/**` 的叙述同批；
  - **归属与时效**：这两个文件**当前归在飞的 M111**（PLE 落盘通路修复）⇒
    **本条修单需等 M111 合入后再派**，否则会与 M111 撞同一文件。
  - 改什么：把 `PleGemv` 从 AIV GEMV 改成 **AIC + Cube 的 bf16 mmad**（`emb` bf16 × `wcat` bf16 → fp32 累加
    → RNE 落 bf16，与现值 `Cast<bfloat16_t,float,castTraitB322B16>` 同舍入方向）；
    形状 `M=1 → 16`、`N` 按 `BASE_N=128/160` 分块、`K=2560/64`。
  - 判据：① 设备产物与**现值逐字节相同**（先用 A/B dump 锁住现值，再换实现，要求 `kv` 平面 sha256 相同）；
    ② 破坏对照：把 mmad 的 B 侧某一行换成垃圾 ⇒ `kv` 对应元素变红；
    ③ 非空洞：断言 `kv` 不是常量面（`H` 侧跨 token 不同）；
    ④ **两份一致性**：生成器 `--check` 通过（rc=0）。
  - 依赖/风险：PLE 是**独立 kernel**（`m15_ple_body_kernel`，自己的 main），要**新起 AIC**
    （`__mix__(1,2)` 已经在用，AIC 侧当前是空的）；`wcat` 是 bf16、无需 MX 路径。
    **层路径侧**（`m15_ple_wire.h`）已有 AIC 在跑 hc/gem 段 ⇒ 需要与既有 AIC 段排段序。

#### V2 —— `m15_moe_layer.h` 的 Router 打分：**P1**

- **锚点**：
  - `class RouterStage`（main 行号 757），段头注释逐字：「**S2：Router 段（m7_router_topk 移植：GEMV 向量 MAC → …），AIV0 单核**」；**本条只管 decode 路径**（`runs=all` 主交付链）：prefill B4 链的 router **已是 cube** —— `m15_moe_prefill.h::RouterMmadStage`（AIC bf16 `Mmad`，M132/M118 写、M26 设备验过 32,768/32,768 逐元素在 T3 界内，`max|Δ| = 2.4e-08`）⇒ **不要把本条读成整仓 router 都违规**。
  - `__aicore__ inline void GemvGroupRow(uint32_t r, uint32_t e0)`（main 行号 983）—— 专家打分；
  - `__aicore__ inline void SgateRows(uint32_t rows)`（main 行号 950）—— 共享专家门裸点积；
  - 调用点：`MoeLayerChain::ProcessAiv` 里 `if (isPrimary) { router.Run(p.subLimit); … }`，
    而 `isPrimary = (bid == 0)`（main 行号 ~2074）。
- **数学形态**：
  - `logits[r,e] = Σ_j x[r,j]·W[e,j]`，真实档 `x:[m,2560]`、`W:[512,2560]` ⇒ `M=m(=1 decode)`、`K=2560`、`N=512`；
  - `sgate[r] = Σ_j x[r,j]·w[j]` ⇒ 同一个矩阵乘法的 `N=1` 退化（`w` 是权重向量，`sgGm` 视图长度 `HIDDEN`）。
- **这是不是矩阵乘法**：`logits` 那条**是**（塔裁样板之一）—— `W` 是固定的权重矩阵
  （`rwGm.SetGlobalBuffer(… routerW …, NUM_EXPERTS * HIDDEN)`），标准双线性型；
  `GemvGroupRow` 里 8 个 `RegTensor<float> a0..a7` 与 `MulAddDst` 的 8 路累加，
  就是「8 个专家的点积共用一次 `x` 载入」的**手写 GEMV**。
  **`SgateRows` 那条：塔裁 B8 已定案 —— 「在规则管辖内」（违规）**，理由（塔的裁决原文要点）：
  「它有一个**固定权重向量、K=2560** ⇒ 满足我的判据『有固定的矩阵/权重操作数参与收缩』；
  我的 B1 裁决针对的是『**无权重操作数**』的单点积，**不含它**」。
  ⇒ 本条的划线范围**包含 `SgateRows`**（不再标「待定」）。它与 `logits` 是同一个权重族的两个输出，
  修法上**并进同一个 GEMM**（m13 donor 原本就是这样：注释逐字「第 `NUM_EXPERTS` 个累加器 `a4`」）。
- **当前实现**：`MulAddDst(acc, xf, w)` 逐 chunk + `Reduce<ReduceType::SUM>` + `Interleave` 打包树，
  **且只在 AIV0 上跑**（`isPrimary`）⇒ 单核 GEMV，其余 AIV 在本段空转。
  证据（§6 C3）：`grep -n "MulAddDst" m15_layer_loop/m15_moe_layer.h` → 10 行输出，
  逐行归属：`:949` 是注释文字，`:968` 在 `SgateRows`，`:1007`/`:1009`/`:1011`/`:1013`/`:1015`/`:1017`/`:1019`/`:1021`
  共 8 行在 `GemvGroupRow`。
- **为什么判违规**：判据 M 成立、判据 V 成立（`m15_layer_loop/m15_moe_layer.h` 里 `Mmad` 只出现在
  `MXFP4GemmItem` 的 `MmadMx` 调用与 `class MXFP4GemmItem` 段；router 段的矩阵乘法由 AIV 向量路算）。
- **规模感受（真实档，per token per MoE layer）**：`512×2560 = 1.31 M` MAC，权重 `2.62 MB` 读一遍；
  **而且是单核**。对比同段 `gate_up`（`MmadMx`）是多核 AIC 分块。
- **建议修单（WO-A2）**：
  - scope：`m15_layer_loop/m15_moe_layer.h` + `m15_layer_loop/lift_moe_segment.py`（**归属/状态更正**：原写 `归 M105`，但 M105 = m15 moe diagnostic slot race fix **已合入**、做的是诊断槽竞态、**没有改 router** ⇒ **本 P1 派单当时未派给它、至今未执行，仍待做**；§4 表与 WO-A2 标题里的同一标记同此更正）
    + `m15_layer_loop/m15_moe_resources.h`（UB/L1/L0 资源表若需新增）；
  - 改什么：把 `RouterStage` 的打分改成 AIC 上的 bf16 mmad（`x[m,2560] @ W[512,2560]^T`），
    `sgate` 作为**第 513 行**并进同一个 GEMM（m13 donor 原本就是这么做的：注释说改前它是
    「第 `NUM_EXPERTS` 个累加器 `a4` 与专家共用一份 x 载入」—— **回到那个形态 + 上 cube**）。
  - 判据：① `logits`/`ids`/`weights` 与现值**逐位**（`topk_ids`/`weights` 是 T1 逐位项，
    `router` 的 `Sort32` 之后是纯排序无算术，所以打分换实现后**仍应逐位**可要求，前提是累加序可控）；
    若 mmad 的 K 内归约次序与现值不同，**必须降级为 T3 并推导界**（`docs/17` §1.1 的 T3 触发条件
    第 ③ 条「含 mmad / cube 累加」—— r1 复审 P2-1 订正：T3 触发清单在 **§1.1**，
    `docs/17` 的 `## 4.` 是「什么算『非空洞』」）；
    ② 破坏对照：把某一列专家权重置零 ⇒ 对应 `logit` 变红；
    ③ 非空洞：`dev` 侧断言 `logits` 行内不是常量（跨专家有差）。
  - 依赖/风险：`Sort32`/`MrgSort` 的裁定例外**不受影响**（排序段本来就在 AIV，`docs/05` §6.1 ⓒ）；
    真实档 `E=512` 下 `W` 是 512 行 ⇒ mmad 的 `N` 要按 `BASE_N` 分块、跨 AIC 条带划分
    （`gate_up` 已有同款条带写法可抄）。**形状前提**见 §2.5。

#### V3 —— `m15_gdn_layer.h:GdnHeadRecurrence`：**P1**

- **锚点**：`__simd_vf__ inline void GdnHeadRecurrence(__ubuf__ float* stUb, __ubuf__ float* qUb, __ubuf__ float* kUb, …)`
  （main 行号 1004）；函数头注释逐字给出四步：
  「`decay` / `delta : w = Σ_j s[j]·k[j]` / `outer : S[i,:] += k·d` / `matvec : o[i] = Σ_j S_new[i,j]·q[j]`」。
- **数学形态**（per value head；`S` 是 `[128,128]` fp32 状态，`k/q/v` 是 128 维）：
  1. `W = S·k` —— `[128,128] @ [128,1]`；
  2. `d = β⊙(v − W)` —— 逐元素（不算）；
  3. `S ← S + k⊗d` —— **rank-1 外积**（`K=1`）；
  4. `O = S·q` —— `[128,128] @ [128,1]`；
  5. `S ← e^g·S` —— 逐元素（`e^g` 是 per-head 标量，`DIST_BRC_B32` 广播，不算）。
- **这是不是矩阵乘法**：**是，按塔裁的分组归法记录**（见 §0.1 末尾那一段）。
  - `W = S·k` 与 `O = S·q`：**S 是 `[128,128]` 的矩阵操作数、参与 `K=128` 的收缩** ⇒
    塔裁判据直接命中（塔把它列为四个样板之一：「S 是 128×128 状态矩阵」）；
  - `S ← S + k⊗d`：单独看是 `K=1`（按 §1.1 塔裁那一条、以及 **§2.4-N10 的 B2 裁决**，`K=1`
    **不是 contraction**）。**我不把它单列成一条矩阵乘法**，而是按塔的分组把它记成
    **同一状态更新步骤的右端项**：这三步合起来是 `S ← diag(e^g)·S + k·d^T`，
    改造成 mmad 时外积项与两个 matvec 是**同一次改造**（工程上不可分）。
  - ⇒ 本条的 form 依据落在 **`S·k` / `S·q` 这两步**上（`S` 是矩阵操作数）；
    `k⊗d` 只是随组记录。**若塔要把 `k⊗d` 剥离**，请直接改判 —— 不影响工作量。
- **当前实现**：寄存器 VF，`Mul` + `Add` + `ReduceSum`（算 `w`、`o`）+ `MulAddDst`（外积）。
  证据：`grep -n "MulAddDst" m15_layer_loop/m15_gdn_layer.h` → `:1036`/`:1037`（外积那两步）。
- **为什么判违规**：上一条命中塔裁判据（`S` 是矩阵、参与 `K=128` 的收缩），判据 V 亦成立
  （该文件 `Mmad` 只出现在 `Cube::Bf16Gemm`，即 in_proj/out_proj；递推段无 mmad）。
- **规模感受（per token per GDN layer，48 heads）**：`48 × 3 × 128×128 = 2.36 M` MAC。**状态是 fp32**。
- **设计上的可改性（这条对派单很关键）**：现实现是「**逐行 `i` 融合**」（`for i in ROWS`，每行内
  Load→四步→Store），但**行与行之间没有数据依赖** —— 给定 `k/q/v/β/e^g`，第 `i` 行的
  `decay→w→d→S[i]→o[i]` 与其它行完全独立。所以「逐行融合」是**性能手法，不是数学必需**；
  整体等价于「先 `S·k`（matvec）、再逐行 `S[i]+=k·d_i`（外积）、再 `S·q`（matvec）」
  ⇒ **存在自然的 mmad 形态**（这也正是 prefill 的 chunk 版在做的事）。
- **依赖/风险（必须一起写进修单）**：
  - 操作数是 **fp32**；`Mmad` 的 A/B 操作数是否支持 fp32、内部是否真做 fp32 乘累加 ——
    **M106（`cube fp32 operand capability probe`）正在探**。**这一条是 V3 的前置**。
  - 若 cube 只能吃 bf16 操作数，则把 `S`/`k`/`q` 降 bf16 会**改变数值**（状态是跨 token 常驻的
    RMW 量）⇒ 必须走 `docs/17` 的 T3 并推导界，**不能**按「换实现、判据不变」来处理。
  - 官方 donor 的 arch35 `recurrent_gated_delta_rule` 本身就是寄存器 VF（`docs/10` 把它列为
    「mega kernel 向量代码风格的最佳 donor」）⇒ 若不动它，需要一条**显式的裁定例外**（见 §4 的 WO-A3）。
- **建议修单（WO-A3）**：见 §4 —— 我把它写成「先出结论、再决定改不改」的两段式，不直接派改。

#### 原 V4 / V5 / V6 的去向（都已在 §2.4）

- **原 V4**（`m15_moe_layer.h:FmaChunk` / `UnpermuteStage`，top-k 加权行和）→ **§2.4-N12**：
  **塔裁 B9 = 不在本规则管辖内**（两操作数都不是固定权重/矩阵）⇒ **降为性能事项**；
- **原 V5**（`m15_hc_layer.h:HyperConnOp::CombineStage`，`K=1` 外积）→ **§2.4-N10**（塔裁 B2）；
- **原 V6**（`m15_ple.asc:PleGateItem` pass C，无权重单点积）→ **§2.4-N11**（塔裁 B1）。

完整论证都在 §2.4（保留原判定依据 + 塔裁与理由）。**本处只留指针**，避免同一段论证出现两份。

### §2.3 口径裁决（**塔裁已决的五条；本文件不再有待定项**）

#### 已由塔裁结案（§0.1 + r1 复审转达，2026-09-27）

| 原编号 | 原问题 | 塔裁 | 对本文件的影响 |
| --- | --- | --- | --- |
| **B1** | `M=N=1`、无权重操作数的单点积（`PleGateItem` pass C 的 `k_n^T q_n`）算不算矩阵乘法 | **不算**（无矩阵操作数 ⇒ 零数据复用） | 原 V6 → **§2.4-N11** |
| **B2** | `K=1` 的 rank-1 外积（`CombineStage` 的 `bo ⊗ injW`）算不算 | **不算**（`K=1` 没有归约维） | 原 V5 → **§2.4-N10**；**WO-A5 撤销** |
| **B4** | 系数随 `j` 变的门控归约（`GateMixStage`）算不算 | **不是矩阵乘法**（不是固定线性算子） | **§2.4-N4** 的判定被塔裁确认（原文由「我判」改为「塔裁」） |
| **B8** | `RouterStage::SgateRows` 的归属（`N=1`、**有固定权重向量 `w`**、`K=2560`） | **在规则管辖内（违规）** —— 理由（塔裁原文要点）：「它有一个**固定权重向量、K=2560** ⇒ 满足我的判据『有固定的矩阵/权重操作数参与收缩』；我的 B1 裁决针对的是『**无权重操作数**』的单点积，**不含它**」 | **§2.2-V2 的划线范围包含 `SgateRows`**（撤下「待定」） |
| **B9** | unpermute 加权行和（原 V4）的归属 | **不在规则管辖内（性能事项）** —— 按字面判据（无固定权重矩阵参与收缩），与 B1/B2 一致 | **原 V4 → §2.4-N12**；**原 WO-A4 不再进合规批次**，降为性能优化项 |

#### 仍待确认（**不是"待定档"** —— 它们的**判定**都已定，卡住的是**证据/时点**）

- **B3（GDN 递推里的 fp32 matvec 怎么改）**：V3 的**改法**取决于 M106 的结论
  （cube 能否吃 fp32 操作数）。**需要的证据**：M106 的 `probe_cube_fp32/README.md` 结论。
  V3 的**判定**不变（P1 违规）；塔已采纳本文件的两段式建议。
- **B5（同族形态的第二步）**：口径 ② 要求「检查其他地方有没有同样的问题」。本 mission 的 scope
  是 `m15_layer_loop/**`，**scope 外**我在读的过程中顺手 grep 到 `m22_router512` 有同形的
  标量写 GM（见 §3.3 的末条），已按 §3.3 记录并报了 TowerFinding —— 但**没有**做仓内全量扫。
  若塔要全量，需要单开一条 mission（本 mission 的边界是「快速读一遍代码 review」）。
- **B6 / B7 / B10**：见 §5（AIV 标量栈容量；`m15_ple.asc` 与副本的归属；**`PipeBarrier` 形态的
  同族待确认** —— B10 是 r1 复审 P2-2 带出来的新项）。

### §2.4 判「**不是**矩阵乘法」的（含塔裁移出的**三条**；逐条给理由，避免误伤）

| # | 文件:符号 | 形态 | 为什么不是矩阵乘法 |
| --- | --- | --- | --- |
| N1 | `m15_gdn_layer.h:PrologBlock`（S3 conv1d） | `acc = bias + Σ_{t=0..3} w[t,c]·win[t,c]` | **深度可分离**（depthwise）：`w` 的 tap 行只与同一个通道 `c` 相乘，权重矩阵在 `(t,c)` 上只取对角 ⇒ 无通道混合。仓内两处独立证据：`docs/10-gdn-analysis.md` 逐字「**conv1d depthwise K=4 over 10240ch** + bias + SiLU」；host 参考 `m15_layer_ref.h:H_RefProlog` 里权重下标是 `convW[j*CH + ch]`（同一 `ch`）。**按 §0.1 塔裁判据复核后维持**：组进一个矩阵会得到块对角（非对角全 0），没有可跨通道复用的矩阵操作数；`K=4` 也远小于 mmad 的 `K` 粒度 16。 |
| N2 | `m15_ple.asc:PleConvItem`（⑤ 膨胀深度卷积）**（在 `m15_ple_wire.h` 有第二份副本，见 §2.2-V1）** | `Σ_{k=0..3} w[k,c]·st[k,c]`（lag 9/6/3/0） | 同 N1：`wL` 每 tap 行按 `c` 取同一列（`G.wtap[k*HYPER + c0]`），depthwise。 |
| N3 | 各 norm / softmax 的平方和、分母 | `Σ_j x[j]²`、`Σ_j exp(x_j−max)` | `A` 与 `B` 是**同一张量** ⇒ 二次型/归约，不是双线性型（§1.1）。站点：`m15_gdn_layer.h` 的 `NormStage::ComputeRow`、`m15_hc_layer.h:NormStage` / `ComputeRstdNewtonRaphsonReg`、`m15_moe_layer.h:NormStage`、`m15_ple.asc:PleGateItem` 的 pass A/pass F、`m15_attn_prolog_probe.h` 的 AIV norm。 |
| N4 | `m15_hc_layer.h:GateMixStage`（S6） | `out[j] = (1/HC)·Σ_s sigmoid(gate[s,j])·x_s[j]` | 系数 `sigmoid(gate[s,j])` **随 `j` 逐元素变** ⇒ 不是固定的线性算子。**塔裁 B4 确认「不是矩阵乘法」（2026-09-27）**。与 V4 的分界仍成立：V4 的 `w[t,k]` 与 `j` 无关。 |
| N5 | `m15_moe_layer.h:CombineStage`（S9b） | `shared·sigmoid(sgate)`、`routed+shared` | 逐元素（`g` 是每 token 一个标量）⇒ 纯广播乘加。 |
| N6 | `m15_moe_layer.h:RouterStage::SoftmaxTopkRow` / `MergeTree` / `Merge2`、`m15_ple.asc:IdsOneToken` 的 id 值域判据 | 排序 / 归并 / 阈值比较 | `docs/05` §6.1 ⓒ 的**裁定例外**（`Sort32`/`MrgSort`：无 `Reg::` 等价物 + 官方 donor 同为 memory-based），本就不属计算路径禁令；id 值域判据是标量比较，不是计算路径。 |
| N7 | `m15_loop_layout.h` 全文 | 常量 / 槽号映射 | 纯 host 侧编译期常量，无设备算件。 |
| N8 | `m15_layer_loop.asc` 全文 | host 驱动 + host 参考链 | 该文件里没有 `__global__` / `__aicore__` 设备符号（§6 C8），`docs/05` §6.1 的豁免清单明文把「host 侧参考实现」排除在本规范之外。 |
| N9 | `m15_attn_layer.h:m15_attn_passthrough_body` | GM→UB→GM 逐字节直通 | 占位直通，无算术。真 attention 的 `Q·K^T`/`P·V`/`o_proj` **在 main 上不存在**（归 M101，且 M101 的形态已经是 `Mmad`，见 §2.1 末）。 |
| **N10** | `m15_hc_layer.h:HyperConnOp::CombineStage`（原 V5） | `bo ⊗ injW`（`[2560]⊗[4]`） | **塔裁 B2：`K=1` 没有归约维，数学上不是 contraction** ⇒ 不要求 mmad。详见下表后的 N10 段。 |
| **N11** | `m15_ple.asc:PleGateItem` 的 pass C（原 V6）**（在 `m15_ple_wire.h` 有第二份副本，见 §2.2-V1）** | `k_n^T q_n`（`M=N=1`） | **塔裁 B1：没有可复用的矩阵/权重操作数 ⇒ 零数据复用** ⇒ 不要求 mmad。详见下表后的 N11 段。 |
| **N12** | `m15_moe_layer.h:FmaChunk` / `UnpermuteStage`（原 V4） | `[1,topk] @ [topk,2560]`（加权行和，B 侧行来自 gather） | **塔裁 B9：不在本规则管辖内**（没有固定权重矩阵参与收缩）⇒ 不要求 mmad，**降为性能事项**。详见下表后的 N12 段。 |

#### N10（原 V5，塔裁移出）—— `m15_hc_layer.h:HyperConnOp::CombineStage`

- **锚点**：`__aicore__ inline void CombineStage(uint32_t bid, uint32_t nAiv)`（main 行号 653）；
  注释逐字：「**S1：combine（4 路残差流进 → H' 出），item = 整行第 c 个 64 元素 chunk**」。
- **我原先的判定（保留，供复审对照）**：数学形态是 `H'[mi, s*2560+j] = H[mi, s*2560+j] + bo[mi,j]·injW[mi,s]`，
  即 `bo ⊗ injW` = `[2560]⊗[4]` 的 **rank-1 外积**；我按「退化维不改变判定」把它算成矩阵乘法，
  同时**如实标注了保留**：「`K=1` 的外积没有真正的收缩维（求和长度 1），把它算成矩阵乘法与把任何
  标量乘向量都算成矩阵乘法只有一步之遥 —— 请塔按 B2 决定」。
- **塔裁（2026-09-27，逐字要点）**：「**B2（`K=1` 的 rank-1 外积 / 逐元素 scale 后累加）→ 不要求 mmad。
  `K=1` 意味着没有归约维，数学上不是 contraction**」。
- **结论与修法**：**从违规表移出**；**原 WO-A5 撤销**（不派改）。现形态
  （`LoadAlign<…DIST_BRC_B32>(w, iwTabUb + (mi*HC+s)*INJW_SLOT)` 取 per-(token,stream) 标量 →
  `Mul(t, bor, w)` → `Add(t, hr, t)`）**就是 `K=1` 外积的向量最优形态**；
  做成 mmad 要把 `K` pad 到 16 做 15/16 的无用功（塔裁的理由正是这一点）。
- **顺带的边界提醒（不是本条的合规问题）**：`InjwStage` 里 `injW` 的抽取用了
  `Reduce<ReduceType::SUM>` 从 one-hot 掩码取 lane（同一段代码，见 §6 C6 的 `m15_hc_layer.h`
  锚点区与 §6 C4 的归约计数），那是**归约到一个标量**，与 §2.4-N3 同族，同样不属矩阵乘法。

#### N11（原 V6，塔裁移出）—— `m15_ple.asc:PleGateItem` 的 pass C

- **锚点**：`__aicore__ inline void PleGateItem(uint32_t t, uint32_t s, BodyGm& G, uint32_t* bad, uint32_t negMask)`
  （main 行号 459）；pass C 的注释逐字：
  「**pass C：dot = fp32( bf16( Σ bf16(k_n·q_n) ) )（和先物化到 bf16 再进 pass D 的除法**…）」。
- **数学形态**：`dot = k_n^T q_n`，两个 2560 维向量 ⇒ `M=N=1`、`K=2560` 的**内积**；
  随后是逐元素的门控（`sigmoid(sign·bf16(sqrt(|d|/2560)))`）+ `gated = g·v`（逐元素）。
- **我原先的判定（保留）**：**口径未定** —— 按字面读口径 ①（`M=1` 也要 mmad），`[1,2560]@[2560,1]`
  也是矩阵乘法；但它**没有权重操作数**（两边都是本 token 的激活），且结果是一个标量。
  我当时**没有判它违规**，只是列进清单并标了 B1。
- **塔裁（2026-09-27，逐字要点）**：「**B1（`M=N=1`、无权重操作数的单点积）→ 不要求 mmad。
  没有任何矩阵操作数（没有可复用的权重），强行 mmad 只能是 1×1 几何填充、零数据复用 ⇒
  违反的是规则目的，不是遵守它**」。
- **结论**：**从违规表移出**；不派改。当前实现（`Mul(p, a, b)` + `Cast` 往返 + `Add` + 一次 `Reduce<SUM>`）
  就是该形态的合理写法。
- **未来若形态变了要重判**：塔裁的落点是「**无权重操作数**」。如果将来把 4 个残差流的
  `PleGateItem` 内积**合成 `M=4` 的一次调用**、或 `k_n`/`q_n` 改成来自某个**固定投影矩阵**，
  那就不再落 B1（M 侧出现可复用结构）⇒ 需要重判。此处只记判据，不预设结论。

#### N12（原 V4，塔裁移出）—— `m15_moe_layer.h:FmaChunk` / `UnpermuteStage`

- **锚点**：`__aicore__ inline void FmaChunk(…)`（main 行号 1772）、`class UnpermuteStage`（main 行号 1785）；
  段头注释逐字：「**S9a：unpermute 加权折叠（…）routed_out[t] = bf16( Σ_k fp32(bf16 w[t,k]) * fp32(y_sorted[inv[t,k]]) )**」。
- **数学形态**：per token `routed_out[1,2560] = w_row[1,TOPK] @ Y[TOPK,2560]`，
  其中 `Y` 的行由 `inv[t,k]` **gather** 出来，`w` 是 router 的 top-k 权重。真实档 `TOPK=10`。
- **我原先的判定（保留）**：**是**矩阵乘法 —— `w` 与列 `j` 无关、`Y` 在 `N=2560` 个输出列上被复用；
  但我**没有硬归**，而是写成「待定」并给出两种归法与各自工作量（塔裁要求的正是这个处置）。
- **塔裁（2026-09-27，逐字要点）**：「**B9 unpermute → 判为「不在规则管辖内」（性能事项）。
  按字面判据（无固定权重矩阵参与收缩），与 B1/B2 一致**」。
  （r1 复审的独立意见同向：「按塔裁字面判据（无固定矩阵/权重操作数）它落在规则外……
  建议塔按**字面判据判「规则外 ⇒ 降为性能项」**，与 B1/B2 同族、也让合规批次更干净」。）
- **结论与修法**：**从违规表移出**；**降为性能优化项**。**改法不变**（最优解不依赖归法）：
  把 `Σ_k w_k·Y_k` **融进 `down` GEMM 的 epilogue**（`Y` 行的两个消费者合并），
  **不是**新起一个 `M=1,K=16,N=2560` 的 mmad（A 侧只有 1 行，几何利用率极低）。
- **当前实现**：`FmaChunk` 逐 `TK` 做 `Mul(yF, yF, wF)` + `Add(acc, acc, yF)`（`wF` 由
  `ShiftLefts` 从打包的 int32 权重低 16 位还原成 bf16→fp32 广播）。
- **若将来当性能项做（WO-A4，归 M105）**：
  - 判据：`routed` 平面与现值**逐位**（m8#2 的既有判据就是逐位的）；破坏对照：把 `w` 某一路权重置 0
    ⇒ 对应 `routed` 分量变红；非空洞：`routed` 的行不是 `Y` 任一行本身。
  - 注意：`FmaChunk` 是 `TK` 模板实例化（`unperm1..unperm10`，`TK=1..10`），若走 mmad 就要处理
    `TK<16` 的 pad 语义（pad 行的权重必须为 0）。

### §2.5 形状前提（口径 ③ 的落点）

口径 ③「E=512/topk=10，打通阶段也需要使用真实 shape 呀」在 §2.2 的 V2 **与 §2.4-N12（原 V4）** 上直接生效：
main 上 `NUM_EXPERTS = 4` / `TOPK_MAX = 4`（缩形档），而 `V2` 的知识是「512 个专家的打分」、
`N12` 是「top-10 加权折叠」。**规模无关化已经做完**（M84/M91/M95 已合入，判据见 `m15_moe_resources.h`
的 `static_assert(RT_ROWL >= NUM_EXPERTS)` 与 `TOPK_MAX <= 32`）⇒ **切到 512/10 在 main 上是可行的**，
但它**不在本 mission 的改动范围**里。**建议**：把它作为 WO-A2 的**前置子任务**单列
（只改常量 → 跑一次 → 看哪些判据先红），这样 V2 的 mmad 改造是在**真实形状**上验证的，
而不是在 `E=4` 上过了、到 `E=512` 才知道不行（M95 的 README 已经把这条经验写下来了）。

---

## §3 清单 B：同族同步竞态（标量写 + 次序 token / 标量写落 GM）

### §3.1 「次序 token」这一族的机制（**M186 已在 all-false 下重述**；这一节是 §3.2–§3.4 的判据基础）

> **M186 重述（2026-10-05）**：全项目 release 侧已统一 `false`（M181/M182/M183 合入 main）。权威语义：
> `false` = CANN `ASC_LOCK_BLOCK` 默认「阻塞」、`true` = `ASC_LOCK_NON_BLOCK`；**两种模式都等本 pipe 已发射指令落地**，
> `true` 额外等此前同 id 的释放 ⇒ `true` 更保守。⇒ 本节的 `M103/M105 时点结论在 all-false 下已被重述`：
> 旧版（r1 订正轮）据「mode 0 = 立即生效、不承载可见性」推出的结论作废 —— mode 0 **就是** `false`，
> `Mutex::Unlock` 与 `BufRelease` 现在是**同一种模式**。`bUb[32]` 的「无序」**仍成立**，但理由减为一条（位置），见 §3.2。

> **订正历史（保留）**：r1 复审 P1-1 曾据当时 `docs/05:135` 把 `MutexUnlock<PIPE_S>` 判为「不承载可见性」；
> 该时点依据现已随 `docs/05` 改正作废，**结论方向以本轮 M186 重述为准**。

这一族的判定取决于两个**可核查的事实**：① `Mutex::Lock/Unlock<pipe>(id)` 与
`GetBufInternal/RlsBufInternal<pipe>(id)`（= 本仓的 `BufAcquire/BufRelease` 包装）是不是同一套 token；
② 两者的 **mode 模板实参**是不是同一个。

**事实①是；事实②现在也是**（M181/M182/M183 之后）。依据 = CANN 9.1.0 头文件
（`/usr/local/Ascend/cann-9.1.0/x86_64-linux/asc/include/basic_api/kernel_common.h` 的 `class Mutex`，
逐字；**该体在 `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)` 门控内** ——
非 3510 上 `Mutex::Lock/Unlock` 是空函数体，这条读数只在 3510 有效）：

```
template <pipe_t pipe> static __aicore__ inline void Lock(MutexID id)   { … GetBufInternal<pipe, 0>(id); }
template <pipe_t pipe> static __aicore__ inline void Unlock(MutexID id) { … RlsBufInternal<pipe, 0>(id); }
```

而 `GetBufInternal`/`RlsBufInternal` 在 3510 上分别展开到 `get_buf(pipe, bufId, mode)` /
`rls_buf(pipe, bufId, mode)`（`…/asc/impl/basic_api/kernel_event.h` 同名符号）。
本仓的包装（`m15_layer_loop/m15_moe_layer.h` 的 `BufAcquire`/`BufRelease`，**以符号为准**）现为：
`BufAcquire<PIPE>(id)` = `GetBufInternal<pipe,false>(id)`、**`BufRelease<PIPE>(id)` = `RlsBufInternal<pipe,false>(id)`**。

⇒ **acquire 侧与 release 侧现在同构：都是 `false`**（旧版这里写「release 侧不同 —— `Mutex::Unlock` 传 `0`/false、本仓 `BufRelease` 传 `true`」，那条在 all-false 下已作废）。

**事实②的后果由仓内规则给出（M186 已把 docs/05 的引文重对齐到现文本）**：

| 出处 | 逐字 |
| --- | --- |
| `docs/05` §2（`:19`，M1 reviewer **真机裁决 10/10 PASS**，当时档位为 `true`） | 「跨 pipe 数据交接（如 MTE2→V）用 **BufferID release**（`RlsBufInternal<pipe,false>` + 对侧 get）即可承载数据可见性（M1 reviewer 真机 10/10 PASS 实证，当时档位为 `RlsBufInternal<pipe,true>`）」 |
| `docs/05` §6 的硬件约束表（`:136`） | 「**全项目 release 一律 `false`**」 |
| `docs/05` §6.1 的表（`:143`） | 「公开 `Mutex::Lock/Unlock` 的 mode 硬编码为 0——**就用 mode 0**」 |
| `docs/06-m0-bringup.md:76`（**rev-pin 引文**：现文本已把其中的 drain BufferID 改为 BufferID 阻塞释放） | 「**跨 pipe 的 UB 数据交接必须走 `SetFlag/WaitFlag` 事件**（MTE2_V、V_MTE3、MTE2_MTE3、MTE3_S…）**或 drain BufferID**……**`PipeBarrier<PIPE_X>` 只能阻塞标量等 pipe 指令退休，不能保证 UB 数据路径对另一 pipe 可见**」 |

⇒ 结论（供 §3.2–§3.4 用；**all-false 重述处已标出**）：
1. `MutexLock<PIPE_S>(X)` 与 `BufAcquire<PIPE_MTE3>(X)` 是**同一个 id `X` 上的同一套 token、同一个 acquire mode（`false`）**；
2. `false` = CANN `ASC_LOCK_BLOCK`「等本 pipe 已发射指令落地后才释放」⇒ 它**覆盖**「它的 acquire 之后、release 之前」的写；`true` 额外等此前同 id 的释放（更保守）。**可见性的判据是位置**（写落在覆盖它的 release 之前），**不是 mode** —— 旧版「只有 mode `true` 承载可见性」在 all-false 下作废；
3. 「S 写 → `MutexUnlock<PIPE_S>(X)`（mode 0 = `false`）→ `BufAcquire<PIPE_MTE3>(X)` → MTE3 读」：S 侧这个 release 在 all-false 下**就是**阻塞释放、等 PIPE_S 落地 ⇒ **落在它之前的 S 写被覆盖、有序（重述）**；旧版按 `docs/05:19` **「unlock 之前的 S 写对 MTE3 可见」未被建立**，该判定在 all-false 下改判为**成立**（mode 0 = `false` 也等本 pipe 落地）。**限定**：写在 `MutexLock` **之前**的写不在本次 release 的覆盖范围内（由更早的 release 覆盖与否另判）；
4. 反过来，**写在 `BufAcquire<PIPE_MTE3>(X)` 之后**的 S 写（`bUb[32]` 形态）：它在该 token 的**任何 release 之后** ⇒ 不在覆盖范围内 ⇒ **无序**。这条**只由位置决定、与 mode 无关**（旧版「double 不成立」里 mode 那一半作废，位置那一半仍成立）。

### §3.2 已证实的第一例（**已在修，归 M105**）

- **位置**（main `20bd20d`）：`m15_layer_loop/m15_moe_layer.h` 的 `IndexGenStage::Run`
  （`class IndexGenStage`，main 行号 1276；函数体 main 行号 1299）；
  内容锚点 = **`bUb[32] = static_cast<int32_t>(oobCountUb);`**（main 行号 1387）。
- **形态（按 §3.1 的机制逐步核）**：
  - `:1372` `BufAcquire<PIPE_MTE3>(BUF_AIV_IDX)` —— MTE3 **已经拿到** `BUF_AIV_IDX` 这个 token；
  - `:1373`–`:1384` 一串 `DataCopyPad(...)`；`:1385` 起 `if (subLimit >= 9) { … }`；
  - `:1387` `__ubuf__ int32_t* bUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_CNT);`
    → `bUb[32] = static_cast<int32_t>(oobCountUb);`（**裸指针标量写 UB**）；
  - `:1389` `DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], ExtBlock1(4));` —— MTE3 **立刻**把它搬走。
  - 这一段里**没有同步原语**：把 `:1372`–`:1389` 这 18 行切出来 grep
    （命令与输出见 §6 C5b）`→ 空输出、rc=1`，在 `LC_ALL=C` 与 `LC_ALL=C.UTF-8` 下同样为空。
- **为什么次序不保（M186 在 all-false 下重述）**：按 §3.1 判定 4，这个 S 写发生在 MTE3 的
  acquire **之后** ⇒ 它在该 token 的**任何 release 之后**、不在覆盖范围内 ⇒ **无序**（这条**只由位置决定、与 release mode 无关**）。
  旧版把「S 侧连 drain release 都没有（`MutexUnlock` 是 mode 0）」当第二条理由 —— 在 all-false 下
  mode 0 **就是** `false` = `ASC_LOCK_BLOCK`，**该理由作废**；但仅凭位置这一条，`bUb[32]` 的无序仍成立。
  ⇒ 搬走的可能是 `UB_IG_CNT + 128B` 处的**旧内容**（塔已汇总的实测现象：搬走的是同层 router x 残值）。
- **⚠ 同函数里的另外 10 处（M186 在 all-false 下重述为「被 false release 覆盖 ⇒ 有序」）**
  同一函数内另有 10 处 UB 标量写（`cntUb[e]` / `offUb[...]` / `curUb[e]` / `srcUb[pos]` /
  `expUb[pos]` / `invUb[s]` / `wtkUb[...]`，main 行号
  `1312/1320/1323/1325/1335/1347/1348/1349/1351/1357`），位置分布：
  `cntUb`/`offUb` 在 `MutexLock<PIPE_S>(BUF_AIV_IDX)`（`:1333`）**之前**、
  其余在 `:1333`–`:1360`（`MutexUnlock<PIPE_S>(BUF_AIV_IDX)`）之间。
  **r1 订正曾据「那个 release 是 `MutexUnlock`（mode 0 = 立即生效）不承载可见性」判「未建立」——
  该依据在 all-false 下不成立**：mode 0 = `false` = `ASC_LOCK_BLOCK`，`MutexUnlock<PIPE_S>` 等 PIPE_S
  已发射指令落地后才释放 ⇒ 落在 `MutexLock`/`MutexUnlock` 之间的写**被它覆盖、有序**（`M103/M105 时点结论在 all-false 下已被重述`）。
  **限定**：写在 `MutexLock` **之前**的 `cntUb`/`offUb` 不在本次 release 的覆盖范围内（由更早的 release 覆盖与否另判）。
  **⚠ 需重读/待重验**：本条重述的依据是权威 mode 语义 + 官方 `asc_unlock.md:54`，**不是**新的设备读数；
  旧版从未对「写在 unlock 之前」单独做受控实验（m17 的注释与代码自相矛盾）。若要把它当**已决事实**派单或收单，
  建议先按本节末的 sha256 判据补一次设备对照。

  **两条与它相关、但不是受控实验的材料**（都在仓内、都不是我跑的，供 M105 与塔判断，**不能当结论**）：
  - **`m17_moe_real` 已经修过同一个 bug，并留下一次实测**：它的 `IndexGenStage::Run` 把诊断槽的
    标量写**挪到 `MutexLock` 之前**，注释逐字：「诊断槽的 UB 标量写必须落在 `MutexLock/Unlock<PIPE_S>`
    **之内或之前**……若把这次标量写放到 unlock 之后（**m13 原样**），就会与 `DataCopyPad` 抢跑 ——
    **实测 m=8 时诊断槽读到上一段遗留的 UB 残值（0x3ec5be6f），落盘 sha256 逐次不同**」。
    ⇒ 这是「写在 acquire 之后 ⇒ 红」的一次**正对照**，与第一例的实测现象同族；
    但它**没有**单独验证「写在 unlock 之前」是否真的有序（改法同时也换了落位）。
    另注：**m17 的这段注释里写的是「S 侧 unlock（drain）」，而 m17 自己的 `MutexUnlock` 也是
    mode 0**（`m17_moe_real/m17_moe_layer.asc` 的包装与 m15 同形）⇒ **该注释与 CANN 模板实参不一致**，
    我**不把它当作「mode 0 承载可见性」的证据**。
  - **`m13_moe_layer.asc` 有同一处原样代码**（`BufAcquire<PIPE_MTE3>(BUF_AIV_IDX)` 在 `:1157`、
    `bUb[32] = …` 在 `:1171`）⇒ 这一族在仓内**至少有三份**（m13 原版 / m15 副本 / m17 已修副本）。
- **⇒ 修单的范围（r1 订正后）**：**`bUb[32]` 这一处已证无序**；**另外 10 处未确认**。
  **不建议**把「另外 10 处不用动」当作既定前提去派单 —— 见下面选项的重排。
- **归属与状态**：**归 M105**（`feat/m105-m15-moe-diagnostic-slot-race-fix`，在改生成器
  `lift_moe_segment.py` + 重生成 `m15_moe_layer.h`）。我核过 M105 分支当前的 diff
  （`git diff --stat main...feat/m105-…`）**还是空的** ⇒ 截至本 mission 收尾，这一例**尚未落到分支上**。
  请以 M105 的 tip 为准，不要拿本文件当它的状态。
- **建议修单（WO-B1，实际由 M105 执行，这里只给判据）—— 选项已按塔裁重排**
  （**r2 订正**：初版把 `SetFlag/WaitFlag` 列为「首选」，依据是「与 `docs/06:76` 同向」；
  那是**选择性引用** —— §3.1 自己引的 `docs/05:19` **前半句就是禁用 set_flag/wait_flag 系列**的逐字禁令。
  已按塔的修正版裁决重排）：

  1. **【首选】把这次标量写挪到一个覆盖它的 release **之前**（`MutexLock` 与 `MutexUnlock` 之间即满足）。**
     判据 = §3.1 判定 3/4：all-false 下 `MutexUnlock<PIPE_S>`（mode 0 = `false` = `ASC_LOCK_BLOCK`）本身
     就等 PIPE_S 落地后释放 ⇒ 只要写落在它之前就被覆盖（**与 release 的 mode 无关**）。
  2. **【可选 / 加固】显式补一个 S 侧 `BufRelease<PIPE_S>(BUF_AIV_IDX)`**
     （= `RlsBufInternal<PIPE_S,false>`）：与 1 同模式，可作独立的显式 token 释放点。
     **注意覆盖面**：它只覆盖 release **之前**的写 ⇒ `bUb[32]` 仍必须挪位（回 1）。
  3. **【在 all-false 下等价于选项 1】** 把标量写挪进 `MutexLock`/`MutexUnlock` 段（= 初版选项②）。
     旧版按「mode 0 = 立即生效」判「不算修好」——**该判据在 all-false 下作废**：mode 0 就是 `false` = `ASC_LOCK_BLOCK`，
     落在 `MutexUnlock` 之前的写被覆盖 ⇒ **算修好**。m17 注释「S 侧 unlock（drain）」与它自己代码（`MutexUnlock` = mode 0）
     在 all-false 下**不再矛盾**（mode 0 也是阻塞释放）。
  4. **【禁用】`SetFlag/WaitFlag` 系列**（人类逐字禁令 + `docs/05:19` 逐字：
     「核内 pipeline 同步**只用 BufferID**（`get_buf`/`rls_buf`），**禁用 set_flag / wait_flag 系列**
     表达跨 pipe 数据可见性」）。**`docs/06:76` 的「事件……或 drain BufferID」里，
     两支规则都允许的只有 BufferID（阻塞释放，`false`）那一支**（该引文为 **rev-pin**：现文本已把「或 drain BufferID」改为「或 BufferID 阻塞释放」）。
     ⇒ **具体地说：初版列为首选的 `SetFlag<HardEvent::S_MTE3>` + `WaitFlag<HardEvent::S_MTE3>`
     属于被禁的这一系列，不得使用。**
  5. **【不足】** 只靠 `PipeBarrier` 也不够：`docs/06:76` 逐字「`PipeBarrier<PIPE_X>` 只能阻塞标量等
     pipe 指令退休，**不能保证 UB 数据路径对另一 pipe 可见**」。
  6. **【最省，可与选项 1 叠加】** 把 `oobCount` 并进已经在用的 UB 槽（少一次落点）—— 顺序形态按**选项 1**（把写挪到覆盖它的 release 之前；`MutexUnlock<PIPE_S>`（`false`）即满足，**不依赖选项 2**）。
  - **判据（必须能咬住）**：把 `oobCount` 人为置为非零（注入一个越界 id，或直接把 `oobCountUb`
    设成常量）⇒ `IG_DIAG_GM_SLOT` 处读回的值**必须**跟着变；修复前跑同一条判据应当**红**（可复现的
    回归判据）。**注意**：`oobCountUb` 现在的值是 `badIds`，正常输入下它就是 0 ⇒ 不注入的话
    「读回 0」是**空判据**（0 vs 0），这条本身是 M105 要小心的地方。
  - **m17 的 sha256 判据可复用**：它那次实测的判据形态是「同一二进制重复运行、落盘 sha256 逐次相同」
    —— 对**未确认**的 10 处，这是一个便宜且直接的受控判据（`bUb[32]` 那种残值污染会让 sha256 漂）。
  - **⚠ 自洽性提醒（M186 在 all-false 下重述）**：旧版这条提醒建立在「mode 0 不 drain」之上，**已作废**：
    all-false 下 mode 0 = `false` = `ASC_LOCK_BLOCK`，`MutexUnlock<PIPE_S>` 本身就是一个等 PIPE_S 落地的 release
    ⇒ **选项 1（把写挪到它之前）单独即可**；选项 2（补 `BufRelease<PIPE_S>`）是可选加固，不是前置。

### §3.3 用标量写落 GM 的站点（人类口径 ④）

**口径依据**（仓内已有，不是我发明的）：
`docs/06-m0-bringup.md:78` 逐字「…否则 `aclrtSynchronizeStream` 返回成功而 host 读回丢失/滞后
（**标量 `SetValue` 直写 GM 同样不可靠，角色探测因此改走 DMA**）」；
`docs/06-m0-bringup.md:91` 的表格行逐字「| 标量 `SetValue` 写 GM | kernel 退出时可见性无保证，
元数据一律走 MTE3 |」。

**站点清单**（main `20bd20d`；命令见 §6 C10）：

| # | 文件:符号 | 站点（内容锚点） | 归属 | 通路为什么不保 |
| --- | --- | --- | --- | --- |
| G1 | `m15_ple.asc:IdsOneToken` **（在 `m15_ple_wire.h` 有第二份副本，`d348547` 上的 `:187`；层路径跑的是副本）** | `outG.SetValue(static_cast<uint64_t>(t) * NG + g, id);`（main 行号 249） | **归在飞的 M111**（`m15_ple.asc` + 生成物 `m15_ple_wire.h`） | `outG` 是 `GlobalTensor<int64_t>` ⇒ **标量 pipe 直写 GM**；函数退出前**没有** `MTE3_S` 排空（唯一的同步是 kernel 末尾的 `PipeBarrier<PIPE_ALL>`，按 `docs/06:76` 那不是 GM 可见性契约） |
| G2 | `m15_ple.asc:m15_ple_ids_kernel` | `failG.SetValue(bid, bad);`（main 行号 891） | 同上 | 同 G1；且 `failG` 是 `GlobalTensor<uint32_t>` |
| G3 | `m15_ple.asc:m15_ple_body_kernel` | `G.fail.SetValue(bid, bad);`（main 行号 971） | 同上 | 同 G1；**且这一处有实测读数**（见下） |

**关于 G1 的「实测可用」正对照（必须一起写，否则会误判整条路都坏）**：
`m15_layer_loop/ple/REAL_TABLE.md` §6.4 逐字记录了 M92 的实测：
- body kernel 里同一写法 `G.hmMiss.SetValue(bid, miss)` ⇒ **读回 128 个槽位里只有 4 个核的值可见**
  （且那 4 个的 `miss` 都是 0）；改用「UB → GM 的 `DataCopy`」后 **56/56 核全部可见**；
- **正对照**：同一个 `GlobalTensor::SetValue` 写法在 **① 的 kernel**（`m15_ple_ids_kernel`）里是
  **可用**的 —— `M15_PLE_MUT=1024` 跑一次，`A_ids_meta.txt` 的 `dev_range_fail = 16` 能读回。
⇒ 现象**限定在 body kernel**（多 stage、带跨核 barrier、参数表更长），作者**未定位平台原因**
（如实记为现象、不作平台断言）。塔已就此立了 finding
（`.tower/comms/findings/20260927-tower-bug-kernel-gm-setvalue-4-56-m85.md`，`.tower/` 不在 git 里）。

**G3 还有一条独立缺陷（同一份 REAL_TABLE.md 的 U-K 条）**：`G.fail.SetValue(bid, bad)` 的
`bad` 在 `PleGateItem`/`PleConvItem` 里**只有 `(void)bad;`、从不自增** ⇒ `if (bad > 0u) { … }` 是
**死分支**，host 求和写进 body meta 的 `dev_fail` 恒 0 ⇒ **读 `dev_fail = 0` 不构成设备侧证据**
（`m15_ple_check.py` 不读它，所以现状不会造成假通过，但日志里这个 0 会被误读）。

**文件内部口径不一致（这是我要点出的一条）**：同一个 `m15_ple.asc` 在 §M92 段里逐字写了
「**不用 SetValue**：实测 SetValue 的落点只对部分核可见，故设备侧计数一律走与其它输出同一条
DataCopy 通路」（内容锚点：`// 累计器清零（设备侧判据的落点，最后整体走 DataCopy 落 GM —— **不用 SetValue**`），
而**同一个文件**的 `:891` / `:971` 仍是 `SetValue` 落 GM。⇒ 口径没有在文件内对齐。

**建议修单（WO-B2）**：
- scope：`m15_layer_loop/m15_ple.asc`（+ 若 `dev_fail` 的判定量要修，`m15_layer_loop/m15_ple_check.py`
  与 `m15_layer_loop/ple/**` 的叙述）；
- 改什么：① `G1` 的 `outG.SetValue` 改成「UB 暂存 → `DataCopyPad` 落 GM」（与同文件 `Hd`/`hmBad`
  那条已验证通路同一条）；② `G2`/`G3` 同改，并**顺手让 `bad` 真的自增**（否则搬到 DataCopy 上
  也只是把「恒 0」搬了个家）；③ 若短期内不改 G1（它有正对照、当前能读回），**必须在文件头/README
  里把它标成「已知偏离、有正对照、未修」**，而不是留着一句自相矛盾的注释。
- 判据：同 §3.2 —— **注入一个必然非零的场景**（例如 bit10 变异让 id 全部越界）⇒ `dev_range_fail`
  必须非零；`dev_fail` 必须非零。**不做这个注入就断言不了任何事**。
- 归属：**`m15_ple.asc` 与它的生成物 `m15_ple_wire.h` 当前归在飞的 M111**（PLE 落盘通路修复）
  ⇒ **本条修单需等 M111 合入后再派**（与 WO-A1 同因）。**G1 有两份副本**（`.asc:249` 与
  `m15_ple_wire.h` 的 `d348547` 上的 `:187`），层路径跑的是副本那份 ⇒ 两份都要修、且必须由生成器保证同步。

**scope 外的同形站点（口径 ② 的顺手发现，未做全仓分类）**：
命令 `grep -rnE '\.SetValue\(' --include=*.asc --include=*.h .`（§6 C11）在 `m15_layer_loop/**`
之外命中若干处，其中**最接近本族**的是 `m22_router512/m22_router512.asc:570`–`:571`
（`permSrcGm.SetValue(pos, static_cast<int32_t>(t))` / `permExpGm.SetValue(pos, e)`，两个
`GlobalTensor<int32_t>` ⇒ 与 G1–G3 同属「标量直写 GM」）。其余命中集中在
`m19_qsa_indexer/**`、`probe_aiv_sync/**`、`probe_host_dma/**`、`probe_sync_quirks/**`
（后者属 `docs/05` §6.1 的**豁免清单**「取证工程 `probe_*` —— 其 (a) 类是故意保留的反例/靶子」）。
**这些都不在本 mission 的 scope 内**：我只记 grep 命中的位置，**没有**逐个判「落点是不是 GM」
（那需要逐站核接收者类型，属另一条 mission 的工作量）；是否立修单请塔定。

### §3.4 形态上与「写在 acquire 之后」**不同**的站点（避免误伤；但「有序」这一判定**未建立**）

> **r1 复审 P2-2 订正 + r2 复审 P2-1 再订正**：初版的标题是「已正确排序」，判定依据只写了
> 「显式核内 drain 在 acquire 之前」。按 `docs/06:76`（引文含省略标记）「跨 pipe 的 UB 数据交接**必须走
> `SetFlag/WaitFlag` 事件……或 **drain BufferID**……**`PipeBarrier<PIPE_X>` 只能阻塞标量等 pipe
> 指令退休，不能保证 UB 数据路径对另一 pipe 可见**」（**rev-pin 引文**：现文本已把其中的 drain BufferID 改为 BufferID 阻塞释放）⇒ **`PipeBarrier` 不构成「有序」的证据**。
> 下表保留的是**形态分类**（这几处不是「写在 acquire 之后」那一族 ⇒ **不是本 mission 的违规**），
> 「有序」标签一律撤下。
> **r2 追加的两条边界**：① B10 的**依据够登记、不够定罪** —— 全表保持「未建立」，
> **不得读成「已坏」**；② **`m15_hc_layer.h` 是正对照**（见下表末行，塔已升格为舰队级正对照），
> **不是** B10 的例证。

| 文件:符号 | 站点 | 形态（与「写在 acquire 之后」不同在哪） | 「有序」是否已建立 |
| --- | --- | --- | --- |
| `m15_attn_cache.h:m15_attn_cache_body`（main 行号 196） | `flagL.SetValue(AC_FLAG_RING_WRITTEN, …)` 等 8 行（main 行号 452–459） | `flagL` 是 **`LocalTensor`（UB）不是 GM**；写完是 `AscendC::PipeBarrier<PIPE_ALL>();`（`:460`）**然后**才 `BufAcquire<PIPE_MTE3>(AC_BUF_IN)` → `DataCopyPad(flagG[0], flagL, …)`（`:462`）。**写与搬之间没有 `BufRelease<PIPE_S>`**（也没有 `PipeBarrier<PIPE>` 之外的顺序形态） | **未建立** —— 按 `docs/06:76`，`PipeBarrier` 不保证跨 pipe UB 可见性；本处也没有阻塞释放（`false`）的 release ⇒ 属 §5-B10 的同族待确认项 |
| `m15_attn_kv_probe.h:m15_attn_kv_probe_body`（main 行号 90） | `flagL.SetValue(0..3, …)`（main 行号 233–236） | 同上：`:237` `AscendC::PipeBarrier<PIPE_ALL>();` → `:238` `BufAcquire<PIPE_MTE3>` → `:239` `DataCopyPad(outFlagGm[0], flagL, …)` | **未建立**（同上一行） |
| `m15_moe_layer.h:IndexGenStage::Run` 的另外 10 处 UB 标量写 | `cntUb`/`offUb`/`curUb`/`srcUb`/`expUb`/`invUb`/`wtkUb` | 位置在 `MutexUnlock<PIPE_S>` 之前/之内（写的位置**不在** acquire 之后） | **已建立（M186 在 all-false 下重述）** —— `MutexUnlock<PIPE_S>`（mode 0 = `false` = `ASC_LOCK_BLOCK`）等 PIPE_S 落地后释放，覆盖 Lock/Unlock 之间的写；见 §3.2 末（**需重读/待重验**：依据是权威语义，非新设备读数） |
| **`m15_hc_layer.h:HyperConnOp::CombineStage`（main 行号 653）—— 正对照（r2 更正）** | `BufAcquire<PIPE_V>`（`:679`–`:681`，三个 token）→ VEC 计算 → **`BufRelease<PIPE_V>`（`:695`–`:697`）** → `BufAcquire<PIPE_MTE3>(BUF_AIV_OB)`（`:699`）→ `DataCopy` → `BufRelease<PIPE_MTE3>`（`:701`） | 写侧是 **V（vector）pipe**、交接**纯 BufferID**，**每一步 release 都是 `BufRelease` = `RlsBufInternal<pipe,false>`（阻塞释放，= CANN `ASC_LOCK_BLOCK` 默认）** —— 正是 `docs/05:19` 认可承载跨 pipe 数据可见性的形态 | **这个是本族「该长什么样」的样板**（塔已升格为**舰队级正对照**，并已发给 M111 照抄）。**r2 更正**：初版把它当成 B10 的例证，**核不实**（该文件 `PipeBarrier<PIPE_ALL>` 计数 = 0，只有 5 处 `PipeBarrier<PIPE_MTE2>`，属「同 pipe buffer 复用」那条规则） |
| `m15_ple.asc` 全文 | `PipeBarrier<PIPE_ALL>` 计数 = **40**（`20bd20d` 上实测；命令见 §6 C15） | 该文件**没有 acquire 调用**（`grep -c BufAcquire` = **2**，两处都在文件头注释 `:21`/`:22` 的「**没有**用 `M15H::BufAcquire/BufRelease`」里）⇒ 与同一栏「该文件不用 BufferID 软件流水」一致；它的标量落盘是 **GM 直写（不走 DMA）** ⇒ **不属本族** | 不适用（本族是 UB→DMA；GM 直写见 §3.3） |

**这一订正引出的同族待确认项**：**「标量写 UB → `PipeBarrier<PIPE_ALL>` → MTE2/MTE3 搬走」这个形态
在本仓规则下同样不构成可见性保证**（`docs/06:76`）。**已实测核到的站点 = 上表前两行**
（`m15_attn_cache.h:452-462` / `m15_attn_kv_probe.h:233-239`，两处都填「未建立」）。
**r2 订正**：初版在这里点名的另两处**都核不实**（`m15_ple.asc` 没有 acquire 调用；
`m15_hc_layer.h` 是**正对照**、方向相反）⇒ 已删除那两处点名，**只保留实测可复现的两行**。
**我没有逐个核其它站点**（本 mission 的边界是快速通读），把它登记为 **§5-B10**，
并建议塔按 `docs/06:76` 单列一条**同族排查**（判据：`grep -n "PipeBarrier<PIPE_ALL>"` 后逐个看
「这个 barrier 之后是否紧跟 MTE2/MTE3 的 UB 读，而**写侧**是标量/S pipe、且该 UB 槽**没有**
阻塞释放（`false`）的 release 覆盖」；**先按判据筛文件再逐个核**，不要拿本表当事先认定）。

### §3.5 同族形态尚存：**标量栈上的大数组**（口径 ② 的同族检查，**待确认**）

M84 的 `#5` 把 `IndexGenStage` 里的三个 AIV **标量栈**数组（`uint32_t cnt[E]` / `off[E+1]` /
`cursor[E]`）改成 **UB 静态槽位**，理由逐字（`m15_moe_layer.h` 的 `[M84-5]` 注释）：
「E=4 时 3×16 B 无感；**E=512 时 3×2 KB = 6 KB 压垮 AIV 标量栈** ⇒ 改 UB 静态槽位」。

**同族形态仍在的位置**：`m15_layer_loop/m15_moe_layer.h` 的 `VecQuantStage::Run`（main 行号 1524）
里仍是**标量栈数组**：
```cpp
uint32_t startOff[NUM_EXPERTS + 1];
startOff[0] = 0;
for (uint32_t e = 0; e < NUM_EXPERTS; ++e) { startOff[e + 1] = startOff[e] + countsGm.GetValue(e); }
```
真实档 `E=512` ⇒ `513 × 4 B = 2052 B` 的标量栈。**我不判它是缺陷**（当前缩形档 `E=4` 下是 20 B，
且 `VecQuantStage` 有 4 个实例、但调用是按序的），只记形态并列为 §5 的待确认：
**需要的证据** = ① AIV 标量栈的真实容量（CANN 文档/头文件里有没有上界），
或 ② 一次 `E=512` 编译 + 运行的实测（`moe_relift/check_stream.py` 若已覆盖可复用）。
另有一处同形但**不属此列**：`m15_attn_cache.h:m15_attn_cache_body` 的 `uint32_t candP[AC_MAX_GROUPS];`
（main 行号 247）与 `__ubuf__ bfloat16_t* mp[COMP_TOKENS_PER_STATE];`（main 行号 330）——
它们是**控制路径的定长小数组**（组数上限是编译期小常量），不随 `E` 增长。

### §3.6 其它「标量写落在 acquire 之后」的站点：本次扫描的命中与**限度**

`m15_layer_loop/**` 的 kernel 文件里，**设备段**的裸「标识符[下标] = …」赋值只出现在
`m15_moe_layer.h`（`IndexGenStage::Run` 的 11 处 + `VecQuantStage::Run` 的 2 处）
与 `m15_attn_cache.h` 的 5 处（`candP[nCand++]`×2 与 `mp[i] = <UB 指针>`×3，见 §3.5 末，
都不是「计算出的数值落数据面」）。命令与完整输出见 §6 C12（同一命令在
`m15_attn_cache_host.h` / `m15_attn_kv_host.h` / `m15_attn_prolog_host.h` / `m15_hc_host.h` /
`m15_layer_ref.h` / `m15_moe_host.h` / `m15_layer_loop.asc` 里也有命中，但那些都是 **host 侧
C 数组**，属 `docs/05` §6.1 豁免清单的「host 侧参考实现」）。
**限度（不要把这条读成「没有了」）**：
① 我的正则只覆盖「一行以内、`标识符[下标] = …`」这一种**书写形态**，它**不按变量名过滤**
  （`cntUb` 这类命名只是恰好如此）⇒ 跨行的赋值、以指针别名（`__ubuf__ T* p; p[i] = …`）形态出现的写、
  以及 `LocalTensor::SetValue`（已单列，见 §6 C11）都在**不同的**判别路径上；
② 我没有做仓内全量扫描（scope 是 `m15_layer_loop/**`）；
③ `BufAcquire<PIPE_MTE2>` 那一侧的「标量写落在 acquire 之后」我没有逐个核（本族的第一例是 MTE3 侧）。
⇒ 只能说到：「就这两条命令的范围而言，命中如上」。

---

## §4 可直接变成修单的清单

每条给 **scope（文件）→ 改什么 → 判据 → 依赖/风险**。归在飞 mission 的那几条我标了归属，
**派单前请核对方的工作树 tip**。**本表已按塔裁（B1/B2/B4 + r1 复审转达的 B8/B9）同步更新**：
`WO-A5` 因 `K=1` 被裁定为非 contraction 而**撤销**；`WO-A1` 的 scope **扩到两份副本**（见 §2.2-V1）；
`WO-A4` 因 B9 降为**性能事项**（不进合规批次）；`WO-A2` 的划线范围**含 `SgateRows`**（B8）。

| WO | 级 | 一句话 | scope | 归属预警 |
| --- | --- | --- | --- | --- |
| **WO-A1** | P1 | `PleGemv` 从 AIV GEMV 改成 bf16 mmad —— **两份副本**：改 `.asc` 源 + **重跑生成器**产出 `m15_ple_wire.h` | `m15_layer_loop/m15_ple.asc` + **`m15_ple_wire.h`**（同目录下的生成物） | **归在飞的 M111**（PLE 落盘通路修复）⇒ **需等 M111 合入后再派** |
| **WO-A2** | P1 | Router 打分 **+ `SgateRows`** 改成 bf16 mmad（**仅 decode 路径**；prefill B4 已 cube） | `m15_moe_layer.h`、`lift_moe_segment.py`、`m15_moe_resources.h` | **归属/状态更正：M105 已合入但未执行本派单 ⇒ 仍待做**（原写 `归 M105`）；同表 `WO-A2b`/`WO-A4`/`WO-B3` 三行也带同一失效标记、需一并重派，其中 `WO-A2b` 经核确实未执行（`m15_layer_loop/m15_moe_resources.h:70` 仍 `NUM_EXPERTS = 4`） |
| **WO-A2b** | 前置 | 把 MoE 档位切到真实 `E=512/topk=10` | `m15_layer_loop/m15_moe_resources.h` | **归 M105**（口径 ③） |
| **WO-A3** | P1 | `GdnHeadRecurrence` 的 matvec/外积改 mmad **或**申请裁定例外（两段式：注释先落、改造等 M106） | `m15_layer_loop/m15_gdn_layer.h` | 依赖 **M106** 结论 |
| **WO-A4** | ~~P1~~ **性能项** | unpermute 的 top-k 加权行和：**融合进 down epilogue**（最优解，不依赖口径） | `m15_moe_layer.h`、`lift_moe_segment.py` | **归 M105**；**不进合规批次**（塔裁 B9） |
| ~~WO-A5~~ | ~~P2~~ | ~~hc combine 的 rank-1 外积~~ → **已撤销**（塔裁 B2：`K=1` 不是 contraction） | — | 不派改 |
| **WO-B1** | P1 | MoE 诊断槽的 MTE3 抢跑（**第一例**）；**修法在 all-false 下重述**（首选=把 `bUb[32]` 的标量写挪到覆盖它的 release **之前**（`MutexUnlock<PIPE_S>` 即满足，mode 0 = `false`）；补 S 侧 `BufRelease<PIPE_S>` 为可选加固；`SetFlag/WaitFlag` 系列**禁用**） | `lift_moe_segment.py` → `m15_moe_layer.h` | **M105 正在做** |
| **WO-B2** | P2 | PLE 三处标量写 GM + `dev_fail` 死分支（**G1 有两份副本**，见 §3.3） | `m15_layer_loop/m15_ple.asc`（+ `m15_ple_check.py` / `ple/**` 叙述） | **归在飞的 M111**（同 WO-A1） |
| **WO-B3** | 待确认 | `VecQuantStage::Run` 的标量栈数组（E=512 ⇒ 2 KB） | `m15_moe_layer.h` | **归 M105**；先要证据 |
| **WO-B4** | 待确认 | **新增**：`PipeBarrier<PIPE_ALL>` 形态（标量写 UB → barrier → MTE2/MTE3 搬走）在 `docs/06:76` 下**不构成可见性保证** —— **够登记、不够定罪** | §3.4 表前两行（已实测的两处）；按 §3.4 末判据**先筛文件**（见 §5-B10） | 需先做同族排查（**未派**） |

### WO-A1（P1）PleGemv → mmad（**两份副本**；归在飞的 M111，需等其合入后再派）

- **scope**：**两个文件** —— `m15_layer_loop/m15_ple.asc`（**改源**）+ **`m15_ple_wire.h`**
  （同目录下的**机械生成物，勿手改**：改完源后**重跑生成器** `lift_ple_device_segment.py`，
  并跑它的 `--check` 确认两份一致、rc=0）。判据脚本若需改：`m15_layer_loop/m15_ple_check.py`、`ple/**`。
- **归属与时效**：这两个文件**当前归在飞的 M111**（PLE 落盘通路修复）⇒
  **本条需等 M111 合入后再派**，否则会撞同一文件。
- **改什么**：`PleGemv` 的 AIV 逐列 GEMV → **AIC + Cube**：`emb[1,2560]bf16 @ wcat[12800,2560]bf16^T`
  （`Nd2Nz` → `LoadData2D` → `Mmad` → `Fixpipe`，`M=1` 抬到 16 行，`N` 按 `BASE_N` 分块跨 AIC 条带，
  `K` 按 `BASE_K=64` 分块）。输出 fp32 累加 → RNE 落 bf16，**与现值的舍入点一致**。
- **判据**：① 先用 A/B dump 锁住现值的 `kv` 平面（sha256）；换实现后要求 **逐字节相同**
  （`K` 内累加次序会变 ⇒ 若逐字节达不到，就**降级为 T3 并推导界**，`docs/17` §1.1 第 ③ 条已把
  「含 mmad / cube 累加」列为 T3 的触发条件 —— r1 复审 P2-1 订正为 §1.1）；② 破坏对照：把 `wcat` 的某一行置垃圾 ⇒ 对应 `kv` 元素变红；
  ③ 非空洞：断言 `kv` 随 token 变（不是常量面）。
- **依赖/风险**：PLE 是独立 kernel（自己的 `main()`，`__mix__(1,2)` 已用、AIC 侧当前没有活）；
  `wcat` 是 bf16、**不需要** MX 路径；UB/L1/L0 资源要重新算（现有 `UB_END = 52512`，
  文件头有「峰值占用 < 248 KB」的说明）。

### WO-A2（P1）Router 打分 → mmad（**归属/状态更正：M105 已合入但未执行本派单 ⇒ 仍待做**；仅 decode 路径）

- **scope**：`m15_layer_loop/m15_moe_layer.h`（**生成物**）、`m15_layer_loop/lift_moe_segment.py`（**规则源**）、
  `m15_layer_loop/m15_moe_resources.h`（资源定尺）。**本条只管 decode 路径**：prefill B4 链已是 cube（`m15_moe_prefill.h::RouterMmadStage`，M26 设备验过 32,768/32,768 T3 内），**不在 WO-A2 的 scope**。**m7/m22 边界（同族、非阻塞）**：`m7_router_topk` / `m22_router512` 是同一 VF 形态的**概念来源**（m13 段头逐字写 `m7_router_topk 移植`），但**无交付链 include/符号调用、也不会被重新 lift**（生成器 `lift_moe_segment.py` 的抽取源是 `m13_moe_layer.asc`）⇒ **不是本合规项/交付的阻塞项**，WO-A2 的 scope 不含它们；但该形态**已被复制进 m13/m15/m17 三次**，登记一笔**低优先清理项**（防第四次复制）：按同族 grep 审 `m22_router512/m22_router512.asc` 的 router 打分段。
- **改什么**：`RouterStage::GemvGroupRow`（专家打分）与 `RouterStage::SgateRows`（共享门点积）
  从 AIV 向量 MAC → **AIC 上的 bf16 mmad**：`x[m,2560] @ W[E,2560]^T`，共享门作为**第 E+1 行**
  并进同一个 GEMM（m13 donor 原本就是这个形态：`m15_moe_layer.h` 的注释逐字说改前它是
  「第 `NUM_EXPERTS` 个累加器 `a4`，与专家共用一份 x 载入」）。
- **判据**：① `topk_ids` / `topk_weights` 的**逐位**判据（`docs/17` 的 T1）在缩形档与真实档都要过；
  若 mmad 的 `K` 内累加次序变了导致位不一致 ⇒ 走 T3 + 推界；
  ② 破坏对照：把某一行专家权重整行置零 ⇒ 该专家永不入选（`ids` 变红）；
  ③ 非空洞：`logits` 行内跨专家非常量；④ **`Sort32`/`MrgSort` 的裁定例外条款不变**
  （排序仍在 AIV，`docs/05` §6.1 ⓒ）。
- **依赖/风险**：需要 AIC 侧新增一段 GEMM + 与 AIV 的 mode-2 交接（`m15_gdn_layer.h` 的
  `gemmIn`/`BOUND_IN` 那一套可直接抄）；`E=512` 时 `N=512` ⇒ 需要 N 分块 + 跨 AIC 条带。**硬依赖（改 router 前必须先过）**：`m15_moe_layer.h` 是**机械生成物**，router 改动**必须回源** `m15_layer_loop/lift_moe_segment.py` 重生成（**不得手改 .h**），且**重生成前 `python3 m15_layer_loop/lift_moe_segment.py --check` 必须 rc=0**，否则重生成会把只落在 .h 的手改**静默回退**。历史先例：finding `20261004-agent-rlsmode-a-…` 记录 M181 未同步期间 `--check` 曾 rc=1（release-mode 改动只在 .h）；**该耦合已由 M181 r2/r4 收口**（`m13_moe_layer/m13_moe_layer.asc:81` 与生成器都改成 `RlsBufInternal<pipe,false>`，生成器并加断言挡 `true`），本 mission 在 `97620e7` 上实跑 `--check` 得 **rc=0** ⇒ 该前置条件**当前已满足**，但每次重生成前仍须复跑确认。

### WO-A2b（前置，归 M105）切到真实形状

- **scope**：`m15_layer_loop/m15_moe_resources.h`（+ 若权重清单/数据生成要换，`weights_manifest.txt`；
  **注意 `m15_layer_loop/slice_layer_manifest.py` 与 `weights_manifest.txt` 归在飞的 M100**
  —— 见 M100 的 diff 清单，改前请与 M100 对齐）.
- **改什么**：`NUM_EXPERTS = 4 → 512`、`TOPK_MAX = 4 → 10`，跑一遍，看哪些判据先红。
  `static_assert(RT_ROWL >= NUM_EXPERTS)`（`RT_ROWL = 512`）与 `TOPK_MAX <= 32` 在 `512/10` 下**都成立**
  ⇒ 编译这一关在 main 上是通的（这是 M84/M91/M95 买到的性质）。
- **判据**：**这一步本身不需要新判据** —— 它的产出是「真实形状下有哪些既有判据红了」的清单，
  用来给 WO-A2/WO-A4 定验收基线。**没有这一步，WO-A2 就会在 `E=4` 上绿、到 `E=512` 才暴露**。

### WO-A3（P1，两段式）GDN 递推的 matvec / 外积

- **第一段（现在就派，零设备）**：把「`GdnHeadRecurrence` 的逐行融合**不是数学必需**」这条
  写进 `m15_gdn_layer.h` 的注释与 README，并列出等价的三步形态
  （`S·k` → 逐行 `S[i] += k·d_i` → `S·q`）。判据：**不需要设备**，只需要把行间无依赖这一点
  在注释里论证清楚（`k/q/v/β/e^g` 是行无关的输入）。
- **第二段（等 M106）**：
  - 若 cube **能**吃 fp32 操作数：改 mmad，`M` 取 `ROWS=128`（或按 head 分块），
    `K=128`、`N=128`；数值形态保持 fp32 全链 ⇒ 判据**可以**保持现在的 T3 界。
  - 若 cube **不能**吃 fp32：走 `bf16` 操作数会**改变跨 token 常驻状态的数值** ⇒
    必须走 `docs/17` 的 T3 并**推导**界（不是报「实测最大 X ulp」），且要额外验证
    「状态演化跨 token 不发散」。
  - 若两者都不可接受：**申请裁定例外**（理由 = 官方 donor 的 arch35 `recurrent_gated_delta_rule`
    本身就是寄存器 VF、`docs/10` 把它列为最佳 donor），并按 `docs/05` §6.1 ⓒ 的样式
    **在使用点附一行注释指明依据**。
- **scope**：`m15_layer_loop/m15_gdn_layer.h`（+ 若走例外，`docs/20`/`docs/10` 的登记）。

### WO-A4（**性能项**，不进合规批次；归 M105）unpermute 的 top-k 加权行和

- **定案（塔裁 B9）**：**不在规则管辖内** ⇒ 本条**不是合规义务**，从合规批次移出。
- **改法（不依赖口径）**：首选**不做成独立 mmad** —— `routed_out = Σ_k w_k·Y_k` 与 `down` GEMM 的
  输出是同批 `Y` 行的两个消费者，融进 epilogue 更省一次整行读写；若要做成 mmad，则是
  `M=1, K=16(pad), N=2560`，**A 侧只有 1 行**，且 pad 行的权重必须置 0。
- 判据：`routed` 平面与现值**逐位**；破坏对照 = 把某一路权重置 0 ⇒ `routed` 对应分量变红。

### ~~WO-A5~~（**已撤销**，2026-09-27 塔裁 B2）

原内容（hc combine 的 rank-1 外积，建议申请裁定例外）**不再需要** —— 塔裁直接给了一般规则：
「**`K=1` 的 rank-1 外积 / 逐元素 scale 后累加 → 不要求 mmad。`K=1` 意味着没有归约维，
数学上不是 contraction**」。该条从违规表移出，论证保留在 **§2.4-N10**。
⇒ **不派改**：现形态（每 (token,stream) 一个标量 × 整行）本就是 `K=1` 外积的向量最优形态；
做成 mmad 要把 `K` pad 到 16 做 15/16 的无用功 —— 这与塔裁的理由完全一致。

### WO-B1（已由 M105 执行）第一例

见 §3.2 —— **修法在 all-false 下重述**（并与 `docs/05:19` + `docs/06:76` 对齐）：
**首选 = 把这次标量写挪到覆盖它的 release 之前**（`MutexLock`/`MutexUnlock<PIPE_S>` 之间即满足 ——
mode 0 **就是** `false` = `ASC_LOCK_BLOCK`，等 PIPE_S 落地后释放）；
**可选/加固 = 另补一个 S 侧 `BufRelease<PIPE_S>(BUF_AIV_IDX)`**（与首选同模式，非前置）；
**`SetFlag/WaitFlag` 系列禁用**（人类逐字禁令 + `docs/05:19` 逐字：核内同步只用 BufferID）；
只靠 `PipeBarrier` 也不足（`docs/06:76` 逐字）。
**⚠ 需重读/待重验**：本条重述依据是权威 mode 语义（+ 官方 `asc_unlock.md:54`），非新设备读数。
另补一条 M105 容易踩的坑：**现有 `oobCountUb` 在正常输入下恒 0**，
所以「读回诊断槽 = 0」是 0 vs 0 的**空判据** —— 修完必须配一次**注入**，证明它该非零时非零。

### WO-B2（P2，**归在飞的 M111**）PLE 的标量写 GM + `dev_fail` 死分支

见 §3.3。scope = `m15_layer_loop/m15_ple.asc` **+ `m15_ple_wire.h`（同目录下的生成物，重跑生成器）**
（+ `m15_layer_loop/m15_ple_check.py` 与 `ple/**` 的叙述）。
**G1 有两份副本**（`outG.SetValue` 在 `.asc:249` 与 `m15_ple_wire.h` 的 `d348547` 上的 `:187`），
而层路径跑的是副本那份 ⇒ 两份都要修，且必须由生成器保证同步。
**归属与时效**：`m15_ple.asc` / `m15_ple_wire.h` **当前归在飞的 M111** ⇒ **等 M111 合入后再派**。
**注意**：这条要**一次做完两件事** —— 把 `bad` 真的自增（否则搬到 `DataCopy` 上也还是恒 0），
以及把 `G1`（`outG.SetValue`）换到 DMA 通路（或在文件头显式标注「已知偏离 + 有正对照 + 未修」）。
判据 = 注入使 `bad` 必然非零的场景 ⇒ `dev_range_fail` 与 `dev_fail` 都必须非零。

### WO-B3（待确认，归 M105）标量栈数组

见 §3.5。**先要证据**（AIV 标量栈容量，或一次 `E=512` 的运行读数），再决定要不要改。

### WO-B4（待确认，**未派**）`PipeBarrier<PIPE_ALL>` 形态的同族排查

见 §3.4 末与 §5-B10：**「标量写 UB → `PipeBarrier<PIPE_ALL>` → MTE2/MTE3 搬走」这个形态**
按 `docs/06:76` **不构成可见性保证**。**依据够登记、不够定罪** —— 只登记「未建立」，
**下游不得读成「已坏」**。
**已知的两个站点 = §3.4 表的前两行**（`m15_attn_cache.h:452-462` / `m15_attn_kv_probe.h:233-239`）；
**r2 更正**：初版另点的 `m15_ple.asc`（无 acquire 调用）与 `m15_hc_layer.h`（是**正对照**、方向相反）
**都核不实**，已从见证清单删除 —— **本条的见证清单必须以可复现的 grep 为准**。
**先做排查（读代码即可，零设备）再决定派不派**；判据：对每个 `PipeBarrier<PIPE_ALL>` 站点问
「barrier 之后是否紧跟 MTE2/MTE3 的 UB 读，而**写侧**是标量/S pipe、且该 UB 槽**没有**
阻塞释放（`false`）的 release 覆盖」；**照本族「该长什么样」的样板 = `m15_hc_layer.h:HyperConnOp::CombineStage`
的纯 BufferID + `BufRelease`（阻塞释放，`false`）握手**（塔已升格为舰队级正对照）。

---

## §5 未决 / 待确认（需要什么证据，**不要猜**）

| # | 问题 | 需要什么证据 | 卡住了什么 |
| --- | --- | --- | --- |
| ~~B1~~ | ~~`M=N=1`、无权重操作数的单点积算不算~~ | **已结案（塔裁 2026-09-27：不算）** | 原 V6 → §2.4-N11；**不派改** |
| ~~B2~~ | ~~`K=1` 的 rank-1 外积算不算~~ | **已结案（塔裁 2026-09-27：不算，`K=1` 没有归约维）** | 原 V5 → §2.4-N10；**WO-A5 撤销** |
| ~~B4~~ | ~~系数随 `j` 变的门控流归约算不算~~ | **已结案（塔裁 2026-09-27：不是矩阵乘法）** | §2.4-N4 被确认 |
| ~~B8~~ | ~~`RouterStage::SgateRows` 归哪一档~~ | **已结案（塔裁 2026-09-27：在规则管辖内 ⇒ 违规；B1 只覆盖「无权重操作数」的单点积）** | §2.2-V2 的划线范围**含 `SgateRows`** |
| ~~B9~~ | ~~unpermute 加权行和（原 V4）归哪一档~~ | **已结案（塔裁 2026-09-27：不在规则管辖内 ⇒ 性能事项）** | 原 V4 → §2.4-N12；WO-A4 降为性能项 |
| **B10** | **（r1 复审 P2-2 带出、r2 复审 P2-1 定界）**「标量写 UB → `PipeBarrier<PIPE_ALL>` → MTE2/MTE3 搬走」这个形态算不算有序 | 按 `docs/06:76`**不构成**可见性保证（原文逐字：「`PipeBarrier<PIPE_X>` 只能阻塞标量等 pipe 指令退休，**不能保证 UB 数据路径对另一 pipe 可见**」）。**依据够登记、不够定罪** —— 只登记「未建立」，**不得读成「已坏」**。**已实测核到的站点只有两处**（`m15_attn_cache.h:452-462` / `m15_attn_kv_probe.h:233-239`，见 §3.4 表）；排查按 §3.4 末的判据做（零设备、**先筛文件再逐个核**） | §3.4 表前两行；WO-B4（**未派**） |
| **B3** | GDN 递推的 **fp32** matvec 怎么改 | **M106**（`probe_cube_fp32/README.md`）的结论：cube 能否吃 fp32 操作数、内部是否真 fp32 乘累加 | V3 的**改法**（判定本身已成立、P1 不变） |
| **B5** | 要不要做**仓内全量**的同族扫（口径 ② 的彻底版） | 塔定 —— 本 mission 的边界是「快速读一遍代码 review」，我只扫了 `m15_layer_loop/**` | 是否另立 mission |
| **B6** | AIV **标量栈**的容量上界 | CANN 文档/头文件里的上界，或一次 `E=512` 的实测 | §3.5 / WO-B3 |
| **B7** | `m15_ple.asc` / `m15_ple_wire.h` 的归属与时点 | **已解**：两份**当前归在飞的 M111**（PLE 落盘通路修复）⇒ WO-A1 / WO-B2 **等 M111 合入后再派**（见 §0.2） | WO-A1 / WO-B2 的派单时序 |

### §5.1 规范缺口（**已闭合**：`docs/05` 已补入 ⓔ 落盘通路；M186 已把该文件纳入 scope）

> **M186 注（2026-10-05）**：本节的「规范缺口」建议已落地为 **ⓔ 落盘通路**（M113 新增，见 `docs/05` 现文本）。下面保留的是当时（M107）的建议原文与依据，作为该规则的来路记录。

**问题**：`docs/05` §6.1 的「四条 + 豁免清单」（ⓐ 计算路径 / ⓑ 搬运·矩阵类 / ⓒ 无寄存器等价物的
原语例外＝`Sort32`/`MrgSort` / ⓓ 控制路径不在禁令范围）**没有一条覆盖「标量 pipe 写 GM」**；
`docs/05` §6.2 的 3510 硬约束表也没有。这条规则目前只散落在两处**教训式**记录里：

1. `docs/06-m0-bringup.md:78`（bring-up 正文）逐字：「…否则 `aclrtSynchronizeStream` 返回成功而
   host 读回丢失/滞后（**标量 `SetValue` 直写 GM 同样不可靠，角色探测因此改走 DMA**）」；
2. `docs/06-m0-bringup.md:91`（同章表格行）逐字：「| 标量 `SetValue` 写 GM | kernel 退出时
   可见性无保证，元数据一律走 MTE3 |」；
3. 现象级取证：`m15_layer_loop/ple/REAL_TABLE.md` §6.4（M92 归档，见 §3.3 的实测读数）。

**后果（本 mission 观察到的复发形态）**：`m15_ple.asc` 三处 + `m22_router512` 两处（已报 TowerFinding）；
且 `m15_ple.asc` **文件内自相矛盾**（M92 段写「不用 SetValue」，`:891`/`:971` 仍在用）。

**建议的补法（供塔派 mission 改 `docs/05` 时照抄）**：在 §6.1 加一条**并列**规则
（建议编号 **ⓔ 落盘通路**）：「**设备侧把数据/标志落 GM 一律走 DMA（AIV：MTE3 `DataCopy`/`DataCopyPad`；
AIC：`Fixpipe`）。标量 pipe 直写 GM（`GlobalTensor::SetValue`）不构成可见性契约** ——
依据 `docs/06` 上述两行 + `ple/REAL_TABLE.md` §6.4（同一写法在 body kernel 上 128 槽只 4 核可见、
① 的 kernel 上可用）。**配判据要求**：凡用标量落 GM 的地方必须给一次**正对照**（人为让它该非零时非零），
否则读回 0 不构成任何证据（`docs/17` §4 非空洞纪律）。**与 ⓓ 的边界要写清**：ⓓ 说的是
「计算出的**结果**写进 **UB** 属计算落盘、在禁令内」，ⓔ 说的是「落 **GM** 必须走 DMA」——
两者并列、不合并（与 §6.2「AIC 写 GM 必须走 Fixpipe」那条的行文风格一致）。

## §6 复现命令与读数（全部实跑；`LC_ALL=C`）

**纪律说明**：
- 所有命令在**我的 worktree 根**（`…/wt-107`）下跑；`HEAD = 20bd20d659b11cbb5e6e6d317921ecb3c16e2629`。
- 命令**不钉会漂的 ref**：本文件里出现的行号都配了内容锚点；§6 的命令只依赖**文件内容**。
- **正则字符类只用 ASCII**；做判据的两条命令在 `LC_ALL=C` 与 `LC_ALL=C.UTF-8` 下各跑过一次、
  读数一致（见 C13）。
- **rc 说明**：`grep` 无命中时返回 1，这是 grep 的正常语义、不是错误；下面凡写「空输出、rc=1」
  都是这个意思。
- **本文件自身是 `docs/*.md` 引用扫描的语料**：收尾时跑过一次 gate（见 **C14**，rc=0）。
  任何改动本文件的人在提交前都应重跑一次并确认 rc=0 —— 本文件将来会被逐字搬进 `main`，
  红闸门会被原样搬进去。

**C1 · rev**
```
$ git rev-parse HEAD
20bd20d659b11cbb5e6e6d317921ecb3c16e2629
```

**C2 · mmad 站点（`Mmad(`）**
```
$ grep -rn "Mmad(" m15_layer_loop/
m15_layer_loop/m15_attn_prolog_probe.h:140:            Mmad(cL0C, a2, b2, mp);
m15_layer_loop/m15_gdn_layer.h:1483:                    AscendC::Mmad(cL0C, a2, b2, mmadParams);
m15_layer_loop/m15_hc_layer.h:299:            AscendC::Mmad(cL0C, a2, b2, mp);
```
（`MmadMx` 另在 `m15_layer_loop/m15_moe_layer.h:204`；`MmadParams` 是参数结构体、不是调用。）

**C3 · 向量 MAC（`MulAddDst`）**
```
$ grep -rn "MulAddDst" m15_layer_loop/
… m15_moe_layer.h:968（SgateRows）、:1007/:1009/:1011/:1013/:1015/:1017/:1019/:1021（GemvGroupRow）、
   :949 是注释；m15_gdn_layer.h:1036/:1037（GdnHeadRecurrence 的外积）、:637（conv 的 tap 宏）、
   :1002 是注释；m15_layer_loop/lift_moe_segment.py 与 moe_relift/m91_README.md 里的是同一段代码/引述
```
（`m15_layer_loop/m15_ple.asc` 在这条命令下是**空输出、rc=1**。）

**C4 · 归约站点（`Reduce<`）**
```
$ grep -rn "Reduce<\|ReduceSum" m15_layer_loop/*.h m15_layer_loop/*.asc
→ 按文件计数：m15_gdn_layer.h 23、m15_moe_layer.h 19、m15_ple.asc 6、m15_hc_layer.h 6、
  m15_attn_cache_host.h 2、m15_moe_resources.h 1、m15_attn_prolog_probe.h 1
```

**C5 · 第一例的四个锚点（§3.2）**
```
$ grep -n "bUb\[32\]\|oobCountUb\|BufAcquire<PIPE_MTE3>(BUF_AIV_IDX)\|DataCopyPad(offsGm\[IG_DIAG_GM_SLOT\]" \
    m15_layer_loop/m15_moe_layer.h
1296:    __aicore__ inline uint32_t OobCount() const { return oobCountUb; }
1322:        oobCountUb = badIds;
1372:        BufAcquire<PIPE_MTE3>(BUF_AIV_IDX);
1387:            bUb[32] = static_cast<int32_t>(oobCountUb);
1389:            DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], ExtBlock1(4));
```
逐字上下文（`:1385`–`:1389`）：
```
        if (subLimit >= 9) {
            // [M84-6] 诊断槽：UB 源 32B 对齐（+128B）；GM 目的放 IG_DIAG_GM_SLOT
            //   （= SZ_OFFSETS 尾部 32B 槽起点，恒在真 offsets[0..E] 之后）
            __ubuf__ int32_t* bUb = reinterpret_cast<__ubuf__ int32_t*>(UB_IG_CNT);
            bUb[32] = static_cast<int32_t>(oobCountUb);
            LocalTensor<int32_t> bL(TPosition::VECCALC, UB_IG_CNT, 64);
            DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], ExtBlock1(4));
        }
```

**C5b · 第一例那一段里有没有同步原语（§3.2）**
```
$ sed -n '1372,1389p' m15_layer_loop/m15_moe_layer.h | grep -nE 'PipeBarrier|SetFlag|WaitFlag'
（空输出、rc=1）
$ for loc in C C.UTF-8; do LC_ALL=$loc sed -n '1372,1389p' m15_layer_loop/m15_moe_layer.h \
    | grep -cE 'PipeBarrier|SetFlag|WaitFlag'; done
0
0
```

**C6 · 符号锚点（§2.2 的 V1–V3 + §2.4 里被塔裁移出的 N10/N11/N12 + §2.4-N1/N2 的否证对象）**
```
$ grep -n "class RouterStage\|inline void SgateRows\|inline void GemvGroupRow\|inline void FmaChunk\|class UnpermuteStage" \
    m15_layer_loop/m15_moe_layer.h
m15_layer_loop/m15_moe_layer.h:757:class RouterStage {
m15_layer_loop/m15_moe_layer.h:950:    __aicore__ inline void SgateRows(uint32_t rows)
m15_layer_loop/m15_moe_layer.h:983:    __aicore__ inline void GemvGroupRow(uint32_t r, uint32_t e0)
m15_layer_loop/m15_moe_layer.h:1772:__aicore__ inline void FmaChunk(…
m15_layer_loop/m15_moe_layer.h:1785:class UnpermuteStage {

$ grep -n "__simd_vf__ inline void GdnHeadRecurrence\|__simd_vf__ inline void PrologBlock" m15_layer_loop/m15_gdn_layer.h
623:__simd_vf__ inline void PrologBlock(…
1004:__simd_vf__ inline void GdnHeadRecurrence(…

$ grep -n "inline void CombineStage\|inline void GateMixStage" m15_layer_loop/m15_hc_layer.h
653:    __aicore__ inline void CombineStage(uint32_t bid, uint32_t nAiv)
833:    __aicore__ inline void GateMixStage(uint32_t bid, uint32_t nAiv)

$ grep -n "inline void PleGemv\|inline void PleGateItem\|inline void PleConvItem" m15_layer_loop/m15_ple.asc
394:__aicore__ inline void PleGemv(…)
459:__aicore__ inline void PleGateItem(…)
690:__aicore__ inline void PleConvItem(…)
```

**C6b · 两份副本的核对（§2.2-V1 / P1-2；拷在 `d348547` 上的读数）**

（命令里把目录前缀放进变量 `P` —— **本文件刻意不写「目录 + 文件名」形式的 `m15_ple_wire.h` 路径**，
因为该文件不在本分支的基线 rev 上，写成路径式引用会让 `docs/scan_doc_refs.py` 判 out-of-range。）
```
$ P=m15_layer_loop/
$ diff <(sed -n '393,455p' ${P}m15_ple.asc) <(git show d348547:${P}m15_ple_wire.h | sed -n '331,393p') && echo IDENTICAL
IDENTICAL
$ git show d348547:${P}m15_layer_kernel.h | grep -n "PleGemv("
444:            PleGemv(bid, nblk, G, A.pleTok, A.pleNegMask);
$ git show d348547:${P}m15_ple_wire.h | grep -n "inline void PleGemv\|outG.SetValue\|inline void PleGateItem\|inline void PleConvItem"
187:        outG.SetValue(static_cast<uint64_t>(t) * NG + g, id);
332:__aicore__ inline void PleGemv(uint32_t bid, uint32_t nblk, BodyGm& G, uint32_t nTok, uint32_t negMask)
397:__aicore__ inline void PleGateItem(uint32_t t, uint32_t s, BodyGm& G, uint32_t* bad, uint32_t negMask)
628:__aicore__ inline void PleConvItem(uint32_t t, BodyGm& G, uint32_t* bad, uint32_t negMask)
```
⇒ ① 两份 `PleGemv` **逐字节相同**；② 层路径（`:444`）调用的正是副本那一份；
③ 副本里同时有 `outG.SetValue`（G1 的第二份，`:187`）与 `PleGateItem`/`PleConvItem`（`:397`/`:628`）。

**C7 · 「Router 是单核」的锚点（V2）**
```
$ grep -n "isPrimary" m15_layer_loop/m15_moe_layer.h
2074:        const bool isPrimary = (bid == 0);
2088:            if (isPrimary) {
```
即 `MoeLayerChain::ProcessAiv` 里 `if (isPrimary) { router.Run(p.subLimit); … }` ⇒ Router 段
**只在 AIV0 上跑**。

**C8 · `m15_layer_loop.asc` 里有没有设备算件**
```
$ grep -c "GetValue\|SetValue" m15_layer_loop/m15_layer_loop.asc
0                    （rc=1；`grep -c` 无命中时仍打印 0）
$ grep -n "__global__\|__aicore__" m15_layer_loop/m15_layer_loop.asc
（空输出、rc=1）
```

**C9 · PLE 的 mmad 与归约**
```
$ grep -n "Mmad" m15_layer_loop/m15_ple.asc
（空输出、rc=1）
$ grep -n "Reduce<" m15_layer_loop/m15_ple.asc
365 / 435 / 514 / 516 / 580 / 652        （6 行，全部是 Reduce<ReduceType::SUM>）
$ grep -noE 'mmad|Mmad|MMAD|Matmul|MatMul|matmul|Gemm|GEMM|gemm' m15_layer_loop/m15_ple.asc
8:Gemm          ← 逐字上下文是 `④ 逐 (token,stream) 分组 Gemma-RMSNorm 门控`，是 Gemma 的一部分
```

**C10 · PLE 的标量写 GM（§3.3）**
```
$ grep -nE 'SetValue\(' m15_layer_loop/m15_ple.asc
249:        outG.SetValue(static_cast<uint64_t>(t) * NG + g, id);
891:            failG.SetValue(bid, bad);
971:            G.fail.SetValue(bid, bad);
```

**C11 · 仓内 `.SetValue(` 站点（§3.3 的 scope 外部分）**
```
$ grep -rnE '\.SetValue\(' --include=*.asc --include=*.h . | grep -v '\.tower'
→ m15_layer_loop:  m15_attn_cache.h:452-459（8 行，LocalTensor）
                   m15_attn_kv_probe.h:233-236（4 行，LocalTensor）
                   m15_ple.asc:249 / :891 / :971（GlobalTensor，写 GM）
   m22_router512:  m22_router512.asc:570-571（GlobalTensor，写 GM）
   m19_qsa_indexer / probe_aiv_sync / probe_host_dma / probe_sync_quirks: 若干处（未逐个分类）
```
`LocalTensor::SetValue` 写的是 **UB**，不属「标量写 GM」；`m15_layer_loop` 里的 **GlobalTensor**
站点就是 G1/G2/G3 这三处，另加 `m22_router512` 的两处（两文件合计 5 处）。

**C12 · 裸「标识符[下标] = …」赋值（§3.6）**
```
$ grep -rnE '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])+[[:space:]]*=[^=]' m15_layer_loop/*.h m15_layer_loop/*.asc
→ 设备段命中：m15_moe_layer.h 的 :1312/:1320/:1323/:1325/:1335/:1347/:1348/:1349/:1351/:1357/:1387
  （IndexGenStage::Run）与 :1527/:1529（VecQuantStage::Run）；
  m15_attn_cache.h 的 :253/:260（candP）与 :330/:332/:334（mp = UB 指针）；
  其余命中集中在 *_host.h / m15_layer_ref.h / m15_layer_loop.asc（host 侧 C 数组，豁免清单内）
```

**C13 · locale 一致性（两条做判据的计数）**
```
$ for loc in C C.UTF-8; do echo "-- LC_ALL=$loc"; LC_ALL=$loc grep -c "MulAddDst" m15_layer_loop/m15_moe_layer.h; \
    LC_ALL=$loc grep -c "SetValue" m15_layer_loop/m15_ple.asc; done
-- LC_ALL=C
10
5
-- LC_ALL=C.UTF-8
10
5
```

**C14 · `docs/*.md` 引用扫描闸门（`docs/17` §10 的可跑附件；本文件是被扫语料之一）**

**第一段（as-of 本分支 tip；基线 `20bd20d`）**：
```
$ python3 docs/scan_doc_refs.py
docs scanned=16  doc refs=186 (unresolved=0)  section refs=169 (unresolved=0)
in-repo path refs=162  gating=0 (distinct=0)  allowlisted=2 (distinct=2)
    <2 行 allowlisted 明细：均为 docs/13 与 docs/18 引的两个 Bisheng 编译器自带头 —— 略>
RESULT: OK (16 files scanned, 186 doc refs, 169 section refs, 162 path refs, 2 allowlisted, 0 out-of-range)
rc=0
$ LC_ALL=C.UTF-8 python3 docs/scan_doc_refs.py >/dev/null 2>&1; echo rc=$?
rc=0
```

**第二段（同一份文件放进当前 `main` = `d348547` 的干净导出树；证明「搬进主线不会撞红闸门」）**：
```
$ git -C <wt-107> archive main | tar -x -C /tmp/m107_main_gate
$ cp <wt-107>/docs/20-kernel-compliance-sweep.md /tmp/m107_main_gate/docs/
$ cd /tmp/m107_main_gate && python3 docs/scan_doc_refs.py
docs scanned=17  doc refs=219 (unresolved=0)  section refs=212 (unresolved=0)
in-repo path refs=347  gating=0 (distinct=0)  allowlisted=2 (distinct=2)
RESULT: OK (17 files scanned, 219 doc refs, 212 section refs, 347 path refs, 2 allowlisted, 0 out-of-range)
rc=0
```
（`docs scanned` 从 16 变 17 = main 上多了一份 `docs/*.md`；计数变大是**别的文档**带来的，不是本文件。
两段的 `gating` 都是 0、`allowlisted` 都是 2。**两段的计数都是本轮（r2 订正后）重跑的**，
与当前 tip 逐字一致；历轮读数（r1 版 151/163/153 与 184/206/338、r2 首版 175/169/156 与 208/212/341）
**在各自的 tip 上**同样成立 —— 差别是文本增删，不是闸门回归。）

读法：`gating=0` / `0 out-of-range` ⇒ **本次运行闸门绿**（rc=0），两个 locale 下一致；
两段都在**各自的树**上实跑，故**计数只在各自 rev 上成立**（`docs/17` §9.3 的「断言性文字必须与同 commit
的事实一致」+ **§9.6** 规则⑥ 的「引用未合入内容必须标注分支与状态」：读数天然带**时间窗**，
`docs/17` §9.6 的 `:636` 就是同一形态的实例）。
两条 `allowlisted` 是**别的文档**（`docs/13`/`docs/18`）引的外部快照，与本文件无关；
本文件没有新增任何 `ALLOWLIST` 条目。
**（上面那两行 allowlisted 明细我刻意没有逐字抄进本文件** —— 它们是**本仓不持有**的外部路径，
写进来会平白增加两条只靠 `ALLOWLIST` 才不 gate 的引用；读者若要明细，跑一次该命令即可。）
（口径限制见该脚本 docstring：**裸 `§X` 号引用不在它的覆盖面内** —— 所以这个 rc=0 **不等于**
「本文件的章节点位引用全对」，只等于「三类被覆盖的引用里 0 条 out-of-range」。）

**C15 · §3.4 见证清单的读数（r2 复审 P2-1 要求：把悬空的「命令见 §6」落实）**
```
$ grep -c "PipeBarrier<PIPE_ALL>" m15_layer_loop/m15_ple.asc
40
$ grep -c "PipeBarrier<PIPE_ALL>" m15_layer_loop/m15_hc_layer.h
0                    （rc=1；grep -c 无命中时仍打印 0）
$ grep -n "PipeBarrier<" m15_layer_loop/m15_hc_layer.h
570:        PipeBarrier<PIPE_MTE2>();
671:            PipeBarrier<PIPE_MTE2>();
722:            PipeBarrier<PIPE_MTE2>();
803:            PipeBarrier<PIPE_MTE2>();
847:            PipeBarrier<PIPE_MTE2>();
$ grep -n "BufAcquire" m15_layer_loop/m15_ple.asc
21: *     **没有**用 `M15H::BufAcquire/BufRelease`（BufferID 软件流水）—— 那是"先打通再扣性能"的显式取舍，
22: *     列为未完成项（ple/README.md §7 U4）。旧版 PROLOGUE 声称用了 BufAcquire/BufRelease，是错的。
$ grep -n "BufAcquire<PIPE_V>(BUF_AIV_H)\|BufRelease<PIPE_V>(BUF_AIV_OB)\|BufAcquire<PIPE_MTE3>(BUF_AIV_OB)\|BufRelease<PIPE_MTE3>(BUF_AIV_OB)" m15_layer_loop/m15_hc_layer.h
679:            BufAcquire<PIPE_V>(BUF_AIV_H);
697:            BufRelease<PIPE_V>(BUF_AIV_OB);
699:            BufAcquire<PIPE_MTE3>(BUF_AIV_OB);
701:            BufRelease<PIPE_MTE3>(BUF_AIV_OB);
$ grep -c "BufRelease<PIPE_S>" m15_layer_loop/m15_moe_layer.h
0                    （rc=1）
```
读法：`m15_ple.asc` 的 `PipeBarrier<PIPE_ALL>` = **40**（`20bd20d` 与工作树同值；r2 复审也量到 40）
—— 初版写的「34–40 处（计数随版本变）」**已删掉那个无出处的下限**；
`m15_hc_layer.h` 的 `PipeBarrier<PIPE_ALL>` = **0**（其 5 处是 `PIPE_MTE2`，属同 pipe 复用那条规则），
它的 V→MTE3 交接是**纯 BufferID + 阻塞释放（`false`）**（`:679`/`:697`/`:699`/`:701`）⇒ **它是本族的正对照**，
初版把它当 B10 例证**核不实、已改**；`m15_ple.asc` 的 `BufAcquire` 两处**都在注释里**
⇒ 该文件没有 acquire 调用；`m15_moe_layer.h` 里**没有** `BufRelease<PIPE_S>`
⇒ 该函数现状没有 `BufRelease<PIPE_S>`（**M186**：all-false 下 `MutexUnlock<PIPE_S>`（mode 0 = `false`）已足以覆盖，见 §3.2/§4；此处仅记 grep 事实）。

---

## §7 这份文件自身的关系声明（供复审）

- 本 mission（M107）**没有修改** `m15_layer_loop/**` 下任何文件；本文件是它写出的唯一文件。
- 本文件**不声称**任何设备读数 —— 所有读数都是 grep / git 的**文本输出**；
  唯一的实测数值引用（PLE 的 4/56 核可见性、`dev_range_fail = 16`）**明确标注来自
  `m15_layer_loop/ple/REAL_TABLE.md` §6.4（M92 的归档）**，不是我跑的。
- **第一例（§3.2）**：`bUb[32]` 那一处**已证无序**（在修，归 M105）—— 理由在 all-false 下重述为**位置**（写在该 token 的任何 release 之后），与 mode 无关；
  **同函数另外 10 处的顺序性在 all-false 下重述为「被 false release 覆盖 ⇒ 有序」**（§3.2 末 —— M186 已把 M103/M105 时点的结论重述；
  旧版 r1 的「未确认」依据（`Mutex::Unlock` mode 0 不承载可见性）随 `docs/05` 改正作废）。**⚠ 需重读/待重验**：重述依据是权威 mode 语义，非新设备读数。
- 与在飞 mission 的关系见 §0 的在飞表 + **§0.2 合并状态**；**凡涉及 M100/M101/M105/M111 的文件，
  我引的位置都注明是在哪个 rev 上读的**。
- **README 的文档索引没有更新**：`README.md:22`–`:23` 有一张 docs 索引表（列到 `docs/18`），
  但 `README.md` **不在本 mission 的 scope**（scope = `docs/20-kernel-compliance-sweep.md`、`m15_layer_loop/**`）
  ⇒ 我没动它。要把它加进索引的话，请塔另派（历史上这类收尾是单独的 "land docs" mission，例如 M108）。
- **本文件改过两轮**（都在本 mission 内，都只动这一个文件）：
  1. **塔裁轮（§0.1，2026-09-27）**：原 V5/V6 移到 §2.4-N10/N11、WO-A5 撤销、V4 改「待定」、
     §1.1/§1.3 的判据与分级收紧。对照：**违规条目 6 → 4**（V1/V2/V3 + V4 待定）。
  2. **r1 复审轮**：见 §7.1。
  3. **r2 复审轮**：见 §7.2（修法序与见证清单的订正；**违规条目与 §4 行数都不变**）。
- **本文件是 `docs/*.md` 引用扫描的语料**（`docs/scan_doc_refs.py`，`docs/17` §10 的可跑附件）：
  本文件里的 `docs/NN` / `docs/NN §X` / 本仓路径式引用都会被算进 gate，**请任何改这份文件的人
  在提交前跑一次并确认 rc=0**（两段读数见 §6 C14）。

### §7.1 r1 复审（`p1-2items` + 2×p2）的处置

| 复审条目 | 处置 | 落在哪几节 |
| --- | --- | --- |
| **P1-1**（§3.1 的 mode 实参被混淆 ⇒「另外 10 处有序」不成立） | **订正**：§3.1 补引 `docs/05:19/135/143` + `docs/06:76` 并重述结论；「另外 10 处」**降为「顺序性未确认」**；WO-B1 的修法**重排**（见 §7.2 的 r2 再订正 —— 首选已不是 `SetFlag/WaitFlag`） | §3.1、§3.2 末、§3.4、§4（WO-B1）、§7（本条） |
| **P1-2**（漏了 `m15_ple_wire.h` 里的同一份 `PleGemv`） | **补**：§2.2-V1 把两处副本并列（含生成器与 `--check`、层路径 `m15_layer_kernel.h:444` 的调用点）；WO-A1 的 scope 扩到两文件并注明「生成物 ⇒ 改源 + 重跑生成器」；G1 的副本一并登记；**注明两份归在飞的 M111、需等其合入后再派** | §2.2-V1、§3.3（G1）、§4（WO-A1/WO-B2） |
| **P2-1**（`docs/17` 的 T3 触发条件在 §1.1，不在 §4） | **改**两处（`:282` 与 WO-A1 的判据）为 §1.1；**保留** §5.1 里那处本就正确的「`docs/17` §4 非空洞纪律」 | §2.2-V2、§4（WO-A1） |
| **P2-2**（§3.4 的「已正确排序」标签与 `docs/06:76` 未对齐） | **改**：§3.4 标题与列名改为「形态不同 / 有序未建立」，补引 `docs/06:76`；并**由此新增 §5-B10 与 WO-B4**（`PipeBarrier` 形态的同族待确认） | §3.4、§5（B10）、§4（WO-B4） |
| 复审的意见 3（§7 的计数措辞会误导） | **改**：改为明写「违规 6 → 4」「表内 9 行、WO-A5 已撤销 ⇒ 待派 8 条」这类可核对的形态 | §7（本条） |
| 塔裁 B8 / B9（复审转达） | **写入**：B8 ⇒ `SgateRows` **在管辖内（违规）**，§2.2-V2 撤下「待定」；B9 ⇒ unpermute **不在管辖内**，原 V4 → §2.4-N12，WO-A4 降为性能项 | §1.3、§2.2（表 + V2）、§2.3、§2.4（N12）、§4 |
| 复审的 notes（main 已前移、C14 会过期） | **写入**：新增 §0.2 合并状态（M100 已随 `d348547` 并入、M101 未并入）；C14 标注 **as-of `fad6c5b`** 并**补第二段**在 `d348547` 干净导出树上的实跑读数 | §0.2、§6 C14 |

**本轮的净变化（按 r2 复审 nit 1 的口径重写：以 §4 表的**行**为单位，不混口径）**：
§4 表共 **10 行**（`WO-A1/A2/A2b/A3/A4/A5/B1/B2/B3/B4`），其中
**已撤销 1 行**（WO-A5，划线）、**降级 1 行**（WO-A4 → 性能项，不进合规批次）、
**新增 1 行**（WO-B4，待排查）⇒ **待派 = 8 条**（A1/A2/A2b/A3/B1/B2/B3 + B4 的排查前置）。
违规条目侧：**6 → 4（塔裁 B1/B2/B4 轮）→ 3**（r2 的 B8/B9 轮：B8 并入 V2、B9 出表到 §2.4-N12）。
**没有任何一条被升级成违规**；两轮「加严」都是把断言**降保守**
（P1-1：一条「已有序」→「未确认」；P2-2/B10：一处「已正确排序」→「未建立」）。

### §7.2 r2 复审（`p1-1items` + 1×p2 + 2×nit）的处置

| 复审条目 | 处置 | 落在哪几节 |
| --- | --- | --- |
| **P1-1**（修法① `SetFlag/WaitFlag` 被列为首选，与人类纪律 + `docs/05:19` 冲突） | **订正**：承认初版对 `docs/06:76` 是**选择性引用**（那前半句禁令就在我引的 `docs/05:19` 里）；选项**按塔的修正版裁决重排** —— ①并进「覆盖 release 本来就是 drain」的槽、②S 侧 `BufRelease<PIPE_S>`（drain）、③「只挪进 mode-0 区域」**不算修好**、④`SetFlag/WaitFlag` **禁用**、⑤`PipeBarrier` 单独不足；并**按复审的自洽性要求把 ①②绑成一条**（现场约束：该函数 `grep -c "BufRelease<PIPE_S>"` = 0 ⇒ 现状不存在这样的槽）。**（M186 已在 all-false 下重述 ①/②/③：①=挪到覆盖 release 之前、`MutexUnlock`（`false`）即满足；②降为可选；③在 all-false 下算修好 —— 以 §3.2/§4 现文本为准）** | §3.2 末、§4（WO-B1 行 + 正文）、§7.1（P1-1 行的指针已改） |
| **P2-1**（B10 段见证两处核不实 + 一处悬空命令指针） | **改**：§3.4 表删掉核不实的两处点名（`m15_ple.asc` 无 acquire 调用；`m15_hc_layer.h` **是正对照**、方向相反），**只留实测可复现的两行**；**把 `m15_hc_layer.h` 写成正对照**（塔已升格为舰队级正对照）；`:662` 的「34–40 处（命令见 §6）」改成 **`20bd20d` 实测 40**；**新增 §6 C15** 把全部见证读数落成可复现命令（消除悬空指针） | §3.4、（§5-B10）、§4（WO-B4）、§6（C15） |
| **B10 的定性边界**（复审 + 塔） | **写入**：**依据够登记、不够定罪** —— 全篇保持「未建立」，**不得读成「已坏」**；WO-B4 标「未派」 | §3.4 表前、§5-B10、§4/WO-B4 正文 |
| **舰队级更正**（复审提出、塔采纳） | **写入**：`m15_attn_cache.h:452-462` / `m15_attn_kv_probe.h:233-239` **保持「未建立」，不作为先例**；本族的**正对照改为 `m15_hc_layer.h` 的阻塞释放握手（`false`）**（本文件已无任何「正确形态」的表述） | §3.4 表、§4/WO-B4 |
| nit 1（净变化计数口径不一致） | **改**：按「表内 10 行 / 已撤销 1 / 降级 1 / 新增 1 / 待派 8」显式分栏（见上），不再混口径 | §7（上一段） |
| nit 2（「同族里」措辞混两类状态） | **已按提示保持**：§3.2 标题与正文都写「`bUb[32]` 已证无序 / 另外 10 处**未确认**」两类状态并列（**M186 已把「另外 10 处」在 all-false 下重述为「被 false release 覆盖」—— 以 §3.2 现文本为准**） | §3.2 |

**r2 轮的净变化**：违规条目 **不变（3 条）**；§4 表的**行数不变（10 行）**，但 **WO-B1 的修法序全换**
（`SetFlag/WaitFlag` 从「首选」改为**禁用**）；**见证清单 2 处核不实 → 换成实测可复现的 2 处 + 1 处正对照**；
新增 1 条可复现命令（C15）。**仍然没有任何一条被升级成违规。**
