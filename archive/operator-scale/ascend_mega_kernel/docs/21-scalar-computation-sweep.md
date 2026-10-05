# docs/21 —— M123：M15 段体「标量做计算」通读扫描（规则 ⓕ 的一次横扫）

**出处**：本文件由 mission **M123**（read-only survey）产出，落库供塔照它派活。
**范围**：`m15_layer_loop/**` 的 device 代码（代码级通读）。本 mission **未修改** `m15_layer_loop/**`
下任何文件；**本文件是 M123 写出的唯一文件**（唯一例外见 §8 的落库说明）。

---

## §0 边界与口径（读之前先看这一节）

**读的是哪个版本**：不可变 rev **`9296e7946330162d51a57ba9fe58454a76c37c9d`**
（= 本 mission 开工时的 `main` tip：`9296e79 Merge branch 'feat/m120-docs-correction-ub-to-l1-operand-pa'`）。
**本文件里出现的所有行号都以这个 rev 为准**；为抗漂移，每条另给一个可 grep 的**内容锚点**
（符号名 / 逐字字符串 / 逐字代码行）。行号只作查阅提示，**判据请用锚点**。

**零设备档**：本节全部结论是**代码级判定 + grep 证据**（本 mission 未上设备、未做新实验）。
凡需要设备读数才能定的，一律写进 §8「未决」，不写成结论。

**已知口径（人类原话，本 mission 的最高约束）**：

| 编号 | 逐字 | 在本文件里的落点 |
| --- | --- | --- |
| ⓕ-1 | 「**矩阵乘法 ⇒ 必须 cube** ---这个是强制规则」 | §2（清单 ①） |
| ⓕ-2 | 「**其余的都用 VF 做**」 | §3（清单 ②） |
| ⓕ-3 | 「**不要用 scalar 做计算，scalar 只做控制流**」 | §1.2 / §1.3（界线）、§3（清单 ②） |
| ② | 「发现问题之后，确实需要检查一下其他地方有没有同样的问题，不过**快速读一遍代码 review 一下就行了**」 | 全篇体例：本次是**轻量通读**，不做新实验 |
| ⑤ | 「**不得写「已全部 / 无残留 / 0 命中」类绝对断言**」 | 全篇：只写「本次命令的输出是什么」+ 命令 + 范围 |

**规则 ⓕ 在仓内的落点（引编号前已 grep 过原文）**：
`docs/05-megakernel-design.md` §6.1 标题逐字为「**§6.1 计算路径规范：五条 + 豁免清单**」
（该节标题行 `### 6.1 规格 / API 约束` 在 `:137`；规范正文自 `:158` 起）。
**main 上现有的是 ⓐ–ⓔ 五条**（`grep -c "ⓔ" docs/05-megakernel-design.md` → `4`）。
**ⓕ 尚未落进 main**：命令 `grep -n "ⓕ" docs/05-megakernel-design.md` → **空输出、rc=1**（§7 C0）。
规则 ⓕ 的落库**归在飞的 M122**（`feat/m122-docs05-rule-f-matmul-cube-rest-vf-s`，塔广播
`20260927-tower-all-matmul-cube-vf-scalar-scalar.md` 逐字给出三条并列口径）。
⇒ 本文件**不引用「`docs/05` §6.1 ⓕ」这个编号**（它在 main 上还不存在），只引**塔广播**与 ⓐ/ⓑ/ⓔ/§6.2 的既有条文。

**两条塔裁（2026-09-27，本 mission 复审前）已就地写入**：对「n-gram id 算不算数据」与「`poolScale` 这种
常量选择算不算计算」两条边界项的裁决 + 由它们抽出的**统一判据**，集中在 **§3.5**；
对应的就地记录在 §3.2 / §3.3，判据行在 §1.3。**读本文件请先看 §1.3 的判据行，再看 §3.5 的定案。**

**这次扫描相对 `docs/20`（M107）的增量在哪**：M107 的合规清盘**只扫「矩阵乘法」这一类**
（`docs/20` §1.1 的判据 M + §2 的清单 A）。规则 ⓕ-2/ⓕ-3 把面**扩到「一切数值计算不得用 scalar」**
⇒ 本次要做的是：**站在「这个量是数据还是控制流」这条界线上，重新过一遍 m15 的 device 代码**。
因此 `docs/20` §2.4 里被判「**不是矩阵乘法**」的那几条（hc `GateMixStage`、PLE `PleGateItem`、
unpermute 加权行和、各 norm 的平方和、depthwise conv …）**在本规则下不是「已合规」的同义词** ——
它们**仍必须 VF**，只是**不上 cube**。本文件 §4 逐条去核了它们的**实现形态**（结论：实测已是 VF）。

**在飞占用（不要把别人的改动当既成事实）**：本次读的是 **main**，下列段体**不在 main 上**，
命中其名字的地方一律**注明归属**，不作为既成事实：

| 段体 | 分支（在飞） | 在 `main` 上的文件形态（本 mission 实测 `git diff --stat main...<branch>`） |
| --- | --- | --- |
| **B1** GDN prefill | `feat/m115-b1-gdn-prefill-chunk-scan` | 在该分支上新增三个文件 `m15_gdn_prefill.h`(+1140)、`m15_gdn_prefill_host.h`(+54)、`m15_gdn_resources.h`(+119)；**这三个名字在 `main` 上不存在**（本次文件名不写目录前缀，正是为了不让 `docs/scan_doc_refs.py` 把它们当**本仓路径**去解析） |
| **B2** attention prefill | `feat/m116-b2-attention-prefill-front-end-and` | `git diff --stat main...<branch> -- m15_layer_loop/` → **空**（改动不在 `m15_layer_loop/` 下，或其分支尚未落文件） |
| **B3** dense causal prefill core | `feat/m117-b3-dense-causal-prefill-core` | 同上 → **空** |
| **B4** MoE prefill | `feat/m118-b4-moe-prefill-segment` | 在该分支上新增 `m15_moe_prefill.h`(+1119)、`m15_moe_prefill_host.h`(+162)、`m15_moe_prefill_res.h`(+360)（`main` 上不存在） |
| **B5** hc prefill | `feat/m119-b5-hc-prefill-mtile` | 在该分支上新增 `m15_hc_prefill.h`(+444)、`m15_hc_prefill_host.h`(+164)（`main` 上不存在） |
| **M101** attn core | `feat/m101-attention-core-segment-lift-and-o-p` | 新增 `m15_attn_core.h` / `m15_attn_core.asc` / `m15_attn_core_host.h` / `m15_attn_oproj.h`（+5501 含 evidence） |
| **M105** MoE 诊断槽次序修复 | `feat/m105-m15-moe-diagnostic-slot-race-fix` | 改 `m15_layer_loop/m15_moe_layer.h`(**+22/-6**)、`lift_moe_segment.py`(+50/-6)；见 §6 WO-S1 的时效注 |

⇒ **B1–B5 的段体本身不在本扫描面上**（塔已在广播里逐个点名要求各段体自查、按 ⓕ 改）；
本文件对它们的责任只到「**mount 点**」（`m15_layer_kernel.h` 里 `M15L_PrefillPhaseA` 等，
§7 C9 给了 grep 读数：该文件里 `__VEC_SCOPE__` 计数 `0`，它只有行搬运与相位边界）。

**已知偏离（如实标注，不在本 mission 修）**：`m15_layer_kernel.h` 里 `M15L_PleIds` / `M15L_PleBody`
在 `bad > 0u` 时用 `failG.SetValue(bid, bad)`（`:428`）与 `G.fail.SetValue(bid, bad)`（`:500`）
—— 这一族（标量直写 GM）**已在 `docs/20` §3.3 单列**（G1/G2/G3，归在飞的 M111），
本文件**不重复计数**、也不把它当「标量做计算」（它是**诊断计数**，属控制/门控信息，不是数据面）。

---

## §1 判定口径（我凭什么说「这处在用 scalar 算数据」）

### §1.1 三类分栏（塔指定的排法，**不混着写**）

| 栏 | 定义 | 本文件位置 |
| --- | --- | --- |
| **①** | **matmul 却没用 cube**（双重违规：既违 ⓕ-1，又违 ⓕ-2/ⓕ-3 —— 把矩阵乘法放进了 AIV 的路） | §2 |
| **②** | **非 matmul 但用 scalar 算数据**（规则 ⓕ 新增出来的违规面） | §3 |
| **③** | **已合规**（作为对照；含 `docs/20` 的「非 matmul」条目在 ⓕ 下的实现形态核查） | §4 |

### §1.2 什么算「用 scalar 算」（判据 S）

在这套 device 代码里，一处「标量计算」的可判形态有三类（**本次三类都 grep 过**）：

- **(S-a) `__VEC_SCOPE__` / `__simd_vf__` 之外的裸 C++ 算术**（设备端 AIV 的标量单元）。
  例：`const int64_t t1v = p1 * m1;`（§3 S2-2 的锚点）。
  依据：`docs/05` §6.2 的硬规则行 **「VF（`__VEC_SCOPE__` / `__simd_vf__`）循环体内不得出现任何标量指令」**
  （原文位置以符号为准：`nest_o0i0b2_scal` 那一行）—— 该条**恰好把**「标量算术只允许出现在 VF **之外**」
  这一事实钉死了：**VF 之内**禁标量，于是**标量算术一律发生在 VF 之外**，也就一律是「没走 VF」的计算。
- **(S-b) `GlobalTensor::GetValue` / `LocalTensor::GetValue` 取出的值再参与算术**。
  例：`const float f = wGm.GetValue(s);` 后接整数 RNE（§3 S2-1 的锚点）。
- **(S-c) 标量算出的**值落进 UB/GM 的**数据面**（`identifier[idx] = <算术表达式>` 或 `SetValue`）。
  例：`wtkUb[t * 16 + k] = static_cast<int32_t>(rounded & 0xFFFFu);`。

**本次没有把** `Reg::*`（`RegTensor` 上的 `Mul`/`Add`/`Muls`/`Exp`/`Reduce`…）**算进「标量」** ——
它们是 VF（寄存器向量），正是 ⓕ-2 要求的那条路。

### §1.3 「数据」与「控制流」的界线（塔给的口径 + 三个边界子类）

**塔给的口径（逐字）**：「这个量是**数据**（要参与后续数值结果）⇒ VF；是**地址/下标/循环边界/分支条件**
⇒ scalar 合法」，并补一句「**『某个归一化系数/门控/掩码用标量算一算』这类小活算计算**，不是控制流」。

本文件据此把「标量算出的量」分三个边界子类，**判定强弱不同**
（**P 类的档位已由塔裁定案，见下表的「塔裁」栏**）：

| 子类 | 形态 | 本次判定 | 塔裁（2026-09-27） |
| --- | --- | --- | --- |
| **D（数据）** | 算出的量是**张量元素值**（权重/激活/系数），**作为操作数进入数值算式**（乘数/加数/系数） | **违规**（§3.1 S2-1） | 维持（塔已登记该条为本次唯一新增真违规） |
| **I（下标/地址）** | 算出的量是**行号/列号/槽号/偏移/前缀和**，只用于寻址 | **不判违规**（scalar 合法），登记形态（§3.4 S2-4） | **维持**：`IdsOneToken` 的 n-gram id 归此类 ⇒ **不在规则 ⓕ 的管辖内、不改**（§3.5 裁定 Q1） |
| **P（常量选择）** | 算出的量是**编译期常量、或从若干既有常量里按条件选一个**，随后作为 VF 指令的**标量操作数** | **边界项**（§3.3 S2-3） | **豁免**：「**选**」≠「**算**」⇒ 不在管辖内（§3.5 裁定 Q2） |

**塔裁给出的统一判据（本文件后续一律按它判，可直接引用）**：

> **看这个量有没有「作为操作数进入数值算式」。**
> 只进 `[ ]` / gather 的地址位 ⇒ scalar 合法；
> 一旦被当成**乘数 / 加数 / 系数**去算别人的值 ⇒ 回到 ② 侧。
> 另：「**在若干常量之间按条件选一个**」是**控制流**（分支/条件侧），不是「用标量把数据算出来」；
> 而**某个量作为 VF 指令的标量操作数**（例：`Muls(acc, acc, poolScale, all)`）**本身永远不是违规** ——
> VF 本来就吃标量系数。违规只可能发生在「这个系数是**用标量算术从数据里算出来**」的那种形态。

**一条必须一起写下的限度**：本文件的判据是**代码级**的（grep + 逐行读），
**没有**设备读数证明「某处标量算术真的在关键路径上耗时」——这与 `docs/20` 的零设备档同性质。

---

## §2 清单 ①：matmul 却没用 cube

### §2.1 本次的读数

**`m15_layer_loop/**` 里 `Mmad(` / `MmadMx(` 的站点**只有 4 处**（命令与逐行输出见 §7 C1），
位置与 `docs/20` §2.1 的「合规锚」表**逐条一致**：

| 文件:行（本 rev） | 锚点 | 形态 |
| --- | --- | --- |
| `m15_gdn_layer.h:1483` | `AscendC::Mmad(cL0C, a2, b2, mmadParams);` | GDN in_proj / out_proj |
| `m15_hc_layer.h:299` | `AscendC::Mmad(cL0C, a2, b2, mp);` | hc mixer down / up |
| `m15_attn_prolog_probe.h:140` | `Mmad(cL0C, a2, b2, mp);` | attention 前端 4 个投影 |
| `m15_moe_layer.h:204` | `AscendC::MmadMx(cL0C, a2, b2, mmadParams);` | MoE gate_up / down（**MXFP4 量化权重，必须 `MmadMx`**） |

### §2.2 与 `docs/20` 的关系（**不重复计数**）

`docs/20` §2.2 已把 m15 里的三处「矩阵乘法没用 cube」定案为 **V1（`PleGemv`）/ V2（router 打分 +
`SgateRows`）/ V3（`GdnHeadRecurrence`）**，各自带完整的数学形态、规模与修单（WO-A1/A2/A3）。
**本文件不重复列这三条，也不改它们的档位**；只在下面点两件与本次口径**直接相关**的事：

**(a) 它们的实现在本次读数里是 VF，不是标量 —— 广播里那句「标量 MAC」需要收窄。**
塔广播逐字把它们写成「**router 的单核标量 MAC**、`PleGemv` 的逐列 `Mul`+`Add`+`Reduce`」，判「双重违规」。
就本 rev 的代码而言，「**双重**」这个**归类**成立（它们同时违 ⓕ-1 与 ⓕ-2/ⓕ-3 的**精神**：
矩阵乘法跑到 AIV 上了），但**「标量」这个词不成立**：

- `RouterStage::GemvGroupRow`（`m15_moe_layer.h:983`）体内的累加是 **`MulAddDst(acc, xf, w, maskAll)`**
  （Reg 向量）—— 见 §7 C2 的逐行输出（`:1007`/`:1009`/…/`:1021` 共 8 行），**不是标量 MAC**；
  同段的 `SgateRows`（`:950`）在 `:968` 也是 `MulAddDst`。
  `docs/20` §2.2-V2 对它的措辞是**「AIV 单核向量 MAC」**（「单核」是 `isPrimary = (bid == 0)`），
  与本 rev 的代码一致 ⇒ **应以 `docs/20` 的措辞为准**。
- `PleGemv`（`m15_ple.asc:483`）体内是 `NormDonor::LoadRegForDtype` + `Mul` + `Add` + `Reduce<SUM>`
  （`:519`–`:524`），同属 Reg 向量。
- **塔已回应（2026-09-27）**：塔确认「你说的（措辞应收窄）正确」，
  **并已就那条广播发过更正** ⇒ 本节的措辞以 `docs/20` §2.2-V2 的**「AIV 单核向量 MAC」**为准。
  ⚠ 一条仍然成立的**限度**：这只改**措辞与归类**，**不改** ① 栏的判定与档位
  （它们仍是「矩阵乘法离开了 cube」，仍归 `docs/20` 的 V1/V2/V3）。

⇒ 本次对 ① 的处置：**① 栏在 m15 上等于「`docs/20` 的 V1/V2/V3 三条」**，
**没有**在本次扫描里另找出「用裸标量算矩阵乘法」的新站点（**限度**：本次 grep 面见 §7 C1/C2，
它覆盖 `Mmad`/`MulAddDst` 两个 token，不覆盖「以其它 API 名手写的 MAC」）。

**(b) 严重度排序上给塔一条建议**：按 ⓕ 的三条并列口径，① 栏的「双重违规」**不是因为它们用了标量**，
而是因为**矩阵乘法离开了 cube**；这一点与 `docs/20` §2.2 的归档理由（判据 M + 判据 V）**同向**，
⇒ §6 的修单排序**沿用 `docs/20` 的 WO-A1/A2/A3**，本文件不另立 ① 类修单。

---

## §3 清单 ②：非 matmul 但用 scalar 算数据（规则 ⓕ 的新增面）

本节是**本次扫描的主要增量**。按 §1.3 的 D/I/P 排序：**D 类排前**（**判违规 = 1 条**，§3.1 S2-1）；
**I 类与 P 类都为登记项**（§3.2 / §3.3 / §3.4），且两条边界项已由塔裁**移出管辖**（定案与统一判据见 §3.5）。

### §3.0 本次 grep 面的读数（先说清「查过什么」）

三条并跑的命令与逐行输出在 §7 C3：

- **S-a 面**：`__VEC_SCOPE__` 之外的浮点/整型标量声明与算术（`grep -nE "float [a-zA-Z_]+ *=|half …|bfloat16_t …|double …"`）。
  本次范围内命中的**少数**几处已逐条读到底；其中真正「算数据」的是下面 S2-1，其余见 S2-2/S2-3。
- **S-b 面**：`GetValue`（`grep -c "GetValue\|SetValue"` 的逐文件计数见 §7 C4：
  `m15_moe_layer.h` **11**、`m15_ple.asc` **24**、`m15_attn_cache.h` **8**、`m15_layer_kernel.h` **4**）。
  逐条读后：**绝大多数是取值后做下标/地址/门控**（合法），**只有 `m15_moe_layer.h:1353` 一处取出的权重
  值被拿去做数值换算**（S2-1）。
- **S-c 面**：`identifier[idx] = <算术>` 形态的裸标量写（命令与输出见 §7 C5）。
  本次范围内命中 18 行，**逐行归属**：`m15_moe_layer.h` 13 行（**S3 索引生成本体 9 行** + `wtk` 1 行 +
  诊断槽 1 行 + `VecQuantStage` 前缀和 2 行；**前两项也在 `IndexGenStage::Run` 里，故本体只算 9** —— `9+1+1+2 = 13`）、
  `m15_attn_cache.h` 5 行（`candP`/`mp` 两个**指针/组号**数组）⇒ `13 + 5 = 18`。

### §3.1 S2-1（**D 类，判违规**）—— `m15_moe_layer.h:IndexGenStage::Run` 里 top-k 权重的 bf16 舍入**用标量整数算术做**

- **锚点（逐字代码，本 rev 行号 `:1353`–`:1357`）**：

  ```cpp
  // bf16(RNE) 权重位打包进 int32 低 16 位（m8 w_tk_packed 协议）
  const float f = wGm.GetValue(s);
  uint32_t bits;
  __builtin_memcpy(&bits, &f, 4);
  const uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
  wtkUb[t * 16 + k] = static_cast<int32_t>(rounded & 0xFFFFu);
  ```

  外层函数锚点 = `class IndexGenStage`（`:1276`）、`__aicore__ inline void Run(uint32_t subLimit)`（`:1299`）。
- **它算的是什么量**：router 输出的 **top-k 权重** `w[t,k]`（`GlobalTensor<float> wGm`，`:1399`）的
  **bf16 舍入结果**（round-to-nearest-even，逐字 `0x7FFF + ((bits>>16)&1)` 就是 RNE 的舍入偏置）。
  产出的 `w_tk_packed` 随后经 `DataCopyPad(wtkGm[0], wtkL, …)`（`:1376`）落 GM。
- **为什么它算「数据」而不是「控制流」**：这个 pack 值是 `UnpermuteStage` 的**乘法操作数** ——
  见 §7 C6：`UnpermuteStage::CopyIn` 把它读进 `wL`（`:1820`），`ComputeRow` 用
  `LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(wRaw, wRowUb)` + `ShiftLefts(…, 16, …)` 还原成 bf16
  当 `wF`，然后 `Mul(yF, yF, wF, maskAll)`（`:1851`/`:1780`，`FmaChunk`）。
  ⇒ 它**直接参与 `routed_out` 的数值结果**，是**权重数据**，不是下标/地址/循环边界/分支条件。
- **当前实现**：`GetValue` 取标量 fp32 → `__builtin_memcpy` 取位型 → 标量整数加/移位 → 标量写 UB。
  **整条路都在 VF 之外**（S-a + S-b + S-c 三类同时命中）。
- **应当改成什么**：**VF 里的一次 Cast**。这条 bf16 RNE 在 VF 侧有一等公民写法：
  `Cast<bfloat16_t, float, castTraitB322B16>(dst, src, mask)`，其中
  `constexpr Reg::CastTrait castTraitB322B16 = {…, RoundMode::CAST_RINT}`（**同一文件** `:115`–`:116`
  就有这个常量，逐字 `RoundMode::CAST_RINT` = round-half-to-even）。
  ⇒ 与现值**同舍入方向**，且 `docs/05` §6.1 的「位宽转换 Cast」行**逐字**要求这条路。
  形态：把 router 的 `weights`（fp32，`[M,TOPK]`）**整行**读进 UB → `__VEC_SCOPE__` 里一次
  `Cast<float→bf16, CAST_RINT>` → 再按现有 `w_tk_packed` 的低 16 位协议落盘（或直接改协议、
  在 `UnpermuteStage` 侧改读 bf16 行 —— 后者动的是接口，需与 m8 的 `w_tk_packed` 协议对齐，
  故**建议只换舍入的实现、不换协议**）。
- **判据（可派）**：① `w_tk_packed` 与**现值逐位相同**（RNE 方向相同 ⇒ 应当可要求逐位；
  先用 A/B dump 锁住现值，再换实现）；② **破坏对照**：把 `CastRINT` 换成 `CAST_RINT` 之外的
  舍入模式（如 `CAST_ROUND`）⇒ `wtk` 平面必须变红；③ **非空洞**：断言 `wtk` 行内不是常量
  （不同 token 的权重不同）。
- **归属与时效（必须一起写进派单）**：`m15_moe_layer.h` **正在被在飞的 M105 改**
  （`git diff --stat main...feat/m105-…` → `m15_moe_layer.h | 22 +-`，改的是
  `IndexGenStage::Run` 里的**诊断槽标量写次序**，**与 `:1353`–`:1357` 不是同一处**）。
  ⇒ **本条修单需等 M105 合入后再派**，否则会撞同一函数、同一文件；若塔愿意，也可**并进 M105**
  （同一函数、同一次重生成 `m15_moe_layer.h`）。
  ⚠ `m15_moe_layer.h` 是**机械生成物**（抬头逐字见 §7 C7 的 `lift_moe_segment.py` 关系），
  **改源脚本 `lift_moe_segment.py` 再重生成**，不要手改 `.h`。

### §3.2 S2-2（**I 类；塔裁：不在管辖内，不改**）—— `m15_ple.asc:IdsOneToken` 的 n-gram id **用标量 int64 算术算**

- **锚点**：`__aicore__ inline uint32_t IdsOneToken(…, uint32_t* rangeFail)`（`m15_ple.asc:245`，
  **在 `m15_ple_wire.h:176` 有第二份逐字副本**）。核心逐字代码（`:276`–`:306`）：

  ```cpp
  const int64_t base = cur * m0;                          // int64 回绕（与 triton 一致）
  const int64_t t1v = p1 * m1;
  const int64_t t2v = p2 * m2;
  …
  int64_t mixed = base;
  if (order > 1) { mixed ^= t1v; }
  if (order > 2) { mixed ^= t2v; }
  const int64_t size = szG.GetValue(g);
  const uint64_t r0 = FloorMod64(mixed, size);
  const int64_t off = ofG.GetValue(g);
  … id = static_cast<int64_t>(rr) + off;  idsL.SetValue(g, id);
  ```

  其中的「取模」本身是**纯标量循环**：`UMod64`（`:188`）「**64 次迭代/元素**」的移位-减法（注释逐字
  「回避设备端 int64 除法；64 次迭代/元素」）；`ReqOf`（`:213`）是对 `qsl` 的**标量线性扫描**。
- **它算的是什么量**：① n-gram 的 **16 个 head id**（`outG[t*NG + g]`，即 `A_ids` 平面）。
- **两种读法（我提出、请塔裁的那两种）**：

  | 读法 | 依据 | 我提的结论 |
  | --- | --- | --- |
  | **是「下标」** | id 的**唯一消费者**是 ② 表行的行号：`PleGather` 的 `row = …;` → `G.table[row * HDIM]`（`m15_ple.asc:415`/`:419`）。它是 **gather 的下标** ⇒ 属 §1.3 的 **I 类**，scalar 合法 | 不违规 |
  | **是「数据」** | 它是 **① 这一段的输出张量**（`idsL.SetValue(g, id)` → `DataCopy(outG[…])`，`:307`/`:316`），且 T1 判据就是**逐位判这 16 个 int64** | 违规，须 VF |

- **塔裁（2026-09-27，M123 复审前）：采取「是下标」这一读 ⇒ 不在规则 ⓕ 的管辖内、不改。**
  塔给的依据（逐字要点）：ⓕ 的界线句已把「**下标**」明文放在 scalar 的**合法侧**；
  n-gram id 的角色**就是 gather 的下标** —— 它**本身从不出现在任何数值结果的算式里**
  （不参与乘加、不参与归约），它的消费者是**索引位**。
- **塔为此补的统一判据（写进 §1.3，本文件后续一律按它判）**：
  **看这个量有没有「作为操作数进入数值算式」** —— id 只进 `[ ]` / gather 的地址位 ⇒ 合法。
- **⇔ 本项由此的最终状态**：**不判违规、不派改**（原 WO-S2 关闭，见 §6）。
  判据（写给「下一个人」）：**检查全仓对这份 id 的使用**，只要出现「把 id 当乘数/加数/系数」的一处，
  这条裁定就**不再覆盖那一处**（那时它是新的 ② 类站点，要按 §1.2 的判据 S 重判）。
  本次的命令面见 §7 C8（`A_ids` 的 T1 判据在 `m15_layer_loop/m15_ple_check.py` 与 `m15_layer_loop/ple/README.md`）。
- **若将来形态变了要重判（与 `docs/20` §2.4-N11 的同款留口）**：若 id 被拿去做算术（例如对 id 做加权、
  做插值、参与任何数值结果），那就不再落本次裁定 ⇒ 需要按 §1.2 重判。
- **归属**：`m15_ple.asc` / `m15_ple_wire.h` 在 M111 合入后**没有其它在飞 mission 持有**
  （§0 的表格：B1–B5/M101/M105 的 diff 都不含这两个文件）。**本次不改**（塔裁：不在管辖内）。

### §3.3 S2-3（**P 类；塔裁：豁免，不在管辖内**）—— `m15_attn_cache.h:319` 的池化缩放系数**在 VF 外按模式选常量**

- **锚点（逐字，`:318`–`:320`）**：

  ```cpp
  // 池化的缩放：SUM_NOT_MEAN 档不除 4（`ops/qsa.py:431` 的方向反写）。**提到 VF 外**当标量。
  const float poolScale =
      (mode == AC_MODE_SUM_NOT_MEAN) ? 1.0f : (1.0f / static_cast<float>(COMP_TOKENS_PER_STATE));
  ```

  消费点：`:359` 的 `Muls(acc, acc, poolScale, all);`（VF 内，系数是**标量操作数**）。
- **它算的是什么量**：**池化的归一化系数**（`1/COMP_TOKENS_PER_STATE` = 1/4，或 1）。
- **为什么它落在边界上（我提出、请塔裁的理由）**：
  - 塔的口径有一句**逐字点到这一类**：「『某个归一化系数/门控/掩码用标量算一算』这类小活算计算，
    不是控制流」⇒ 按这句，它**像**是计算；
  - 但 `COMP_TOKENS_PER_STATE` 是**编译期常量**（`m15_layer_loop/m15_attn_kv.h:160`
    逐字 `constexpr uint32_t COMP_TOKENS_PER_STATE = 4;`，本次 grep 见 §7 C13），
    `1.0f / static_cast<float>(…)` 会被**折叠成常量**；真正运行期的只有**按 `mode` 在两个既有常量间选一个**
    （一个三元 / `Select`）⇒ 「算」的量是**选择**，不是逐元素算术（§1.3 的 **P 类**）。
- **塔裁（2026-09-27，M123 复审前）：豁免 —— 不在规则 ⓕ 的管辖内。**
  依据（逐字要点）：**「在若干常量之间按条件选一个」是控制流（界线句的分支/条件侧），不是「用标量把数据算出来」。**
  界线句里「用标量算一算归一化系数/门控/掩码」指的是**对该量做了算术**（乘/加/归约/函数），
  而**不是从既有常量里挑**。
- **塔同时要求一起写进文档的一条（判据的边界，很重要）**：
  **`poolScale` 随后作为 `Muls` 的标量操作数这件事本身永远不是违规** —— VF 指令本来就吃标量系数
  （人类也把这种用法认成 VF 的标量操作数）。**违规只可能发生在「这个系数是用标量算术从数据里算出来」的形态。**
  ⇒ 文档里凡看到「某 VF 指令带一个标量系数」，**不要**据此判 ② 类；要先问那个系数是**选来的**还是**算出来的**。
- **⇔ 本项由此的最终状态**：**不判违规、不派改**（原 WO-S3 关闭，见 §6）。
  **若将来形态变了要重判**：一旦 `poolScale` 从「两个常量选一」变成「由数据算出的系数」
  （例如对某个统计量做 `1/sqrt(...)`），那就不再落本次豁免 ⇒ 按 §1.2 的判据 S 重判（那种形态才是 ② 类）。
- **归属**：`m15_attn_cache.h` 与本 rev 的在飞段体**无交集**（§0 表格）。**本次不改**（塔裁：豁免）。

### §3.4 S2-4（**I 类登记，不派**）—— index / 前缀和 / 计数排序这一族

这一族的**共性**：算出的量是**行号、列号、槽号、偏移、前缀和**，只用于寻址或门控 ——
按 §1.3 的**塔裁判据**（「有没有作为操作数进入数值算式」），它们**只进地址位** ⇒ 归 **I 类**。逐条：

| 站点（本 rev） | 锚点（逐字） | 算出的量 | 归属子类 |
| --- | --- | --- | --- |
| `m15_moe_layer.h:1299` `IndexGenStage::Run` | `cntUb[e] = cntUb[e] + 1;`（`:1320`）、`offUb[e + 1] = offUb[e] + cntUb[e];`（`:1325`）、`curUb[e] = offUb[e];`（`:1335`）、`srcUb[pos] = …`（`:1348`） | 每专家计数 / 前缀和 / cursor / 计数排序的槽位散布 | **I**（`:1348`–`:1351` 是**槽号**） |
| 同上，注释逐字 | `//     标量实现（m*TOPK ≤ 256），UB 写 → MTE3 出（m3-AIV 同款 PIPE_S→MTE3 握手）`（`:1242`） | —— | 作者已**自认标量实现**；本次判据（有没有进数值算式）：**没有** ⇒ **I 类** |
| `m15_moe_layer.h:1437` `PermuteStage::Run` | `total += static_cast<uint32_t>(countsGm.GetValue(e));`（`:1441`） | 总行数（= 后续行分派的**下界**） | **I** |
| `m15_moe_layer.h:1524` `VecQuantStage::Run` | `startOff[e + 1] = startOff[e] + …countsGm.GetValue(e);`（`:1529`） | 专家槽起点（= `rowBase` 的前缀和） | **I** |
| `m15_ple.asc:213` `ReqOf` | `for (… ) { if (qsl.GetValue(i) <= t) { r = i; } }`（`:216`–`:220`） | request 号 | **I** |
| `m15_ple.asc:188`/`:201` `UMod64` / `FloorMod64` | `r = (r << 1) \| ((x >> i) & 1ULL); if (r >= d) { r -= d; }`（`:192`–`:195`） | 取模结果 | **I**（消费者是 §3.2 的 id ⇒ 随 §3.2 的塔裁一起**不改**） |

**处置**：**登记、不判违规、不派单**（按 §1.3 的 I 类 = 地址/下标，且已由 §1.3 的塔裁判据覆盖）。
**但一条必须一起写**：`VecQuantStage::Run` 的 `uint32_t startOff[NUM_EXPERTS + 1];`（`:1526`）是
**标量栈数组**，`docs/20` §3.5 已把它登记为 `WO-B3`（真实档 `E=512` ⇒ 2052 B 标量栈，**待证据**）——
**这与本文件无关，是另一条口径（栈容量），不重复计数**。

### §3.5 塔裁（2026-09-27，M123 复审前）：两条边界项的定案与**统一判据**

本节把塔对 §3.2 / §3.3 两条边界项的裁决**集中记一遍**（各节就地也写了），并给出**日后判 ② 类的一把尺子**。

**裁定 1 —— n-gram id（§3.2 S2-2）：不在 ③ 的管辖内，不改。**
依据（逐字要点）：ⓕ 的界线句已把「**下标**」明文放在 scalar 的**合法侧**；
n-gram id 的角色**就是 gather 的下标** —— 它本身**从不出现在任何数值结果的算式里**（不参与乘加、不参与归约），
它的消费者是**索引位**。

**裁定 2 —— `poolScale`（§3.3 S2-3）：豁免，不在管辖内。**
依据（逐字要点）：**「在若干常量之间按条件选一个」是控制流（分支/条件侧），不是「用标量把数据算出来」。**
界线句里「用标量算一算归一化系数/门控/掩码」指的是**对该量做了算术**（乘/加/归约/函数），
而不是**从既有常量里挑**。

**由两条裁定抽出的统一判据（写进 §1.3；本文件全篇按它判）**：

> **① 看这个量有没有「作为操作数进入数值算式」。**
> 只进 `[ ]` / gather 的地址位 ⇒ scalar 合法（**I 类**）；
> 一旦被当成**乘数 / 加数 / 系数**去算别人的值 ⇒ 回到 ② 侧（**D 类**）。
> **② 「在若干常量之间按条件选一个」是控制流（P 类）**，不是「用标量把数据算出来」。
> **③ 某个量作为 VF 指令的标量操作数（如 `Muls(acc, acc, poolScale, all)`）本身永远不是违规** ——
> VF 本来就吃标量系数；违规只可能发生在「这个系数是**用标量算术从数据里算出来**」的形态。

**⇒ 本次计数因此的变动（明写）**：
- **② 类的「判违规」条目：仍为 1 条**（§3.1 S2-1，`m15_moe_layer.h:1353`–`:1357` 的 top-k 权重 bf16 RNE）。
  两条边界项**没有**被计入，**也没有**因这次裁定新增/减少违规数。
- **登记项（不派）**：§3.2 **I 类 · 不在管辖内**、§3.3 **P 类 · 豁免**、§3.4 **I 类 · 不判违规**。
- **修单**：WO-S1 保留；**WO-S2 / WO-S3 关闭**（见 §6）。
- **复核判据若形态变化**：两条裁定都写了「若将来形态变了要重判」的触发条件
  （id 进数值算式 / 系数由数据算出）—— 这两句是本次裁定**唯一的适用范围声明**。

---

## §4 清单 ③：已合规对照（`docs/20` 的「非 matmul」条目在规则 ⓕ 下的实现形态核查）

**这一节是本次扫描的第二个增量**：`docs/20` §2.4 判「**不是矩阵乘法**」只说明**不上 cube**，
**不说明已合规**；规则 ⓕ-2/ⓕ-3 之下它们**仍必须 VF**。下表逐条核了**当前实现是不是 VF**。

| `docs/20` 条目 | 文件:符号（本 rev 锚点） | 本次核到的实现 | 结论（对规则 ⓕ） |
| --- | --- | --- | --- |
| **N4** 门控归约（随 `j` 变） | `m15_hc_layer.h:GateMixStage`（`:833`） | `:873` `NormDonor::SigmoidReg(sg, gr, one, maskAll);` → `:874` `Mul(t, sg, xr, maskAll);` → `:875` `Add(acc, acc, t, maskAll);` → `:877` `Muls(acc, acc, 1.0f / static_cast<float>(HC), maskAll);`，全在 `__VEC_SCOPE__`（`:862`）内 | **VF 合规**（系数是编译期常量） |
| **N10** `K=1` 外积 | `m15_hc_layer.h:HyperConnOp::CombineStage`（`:653`） | `:690` `Mul(t, bor, w, maskAll);` → `:691` `Add(t, hr, t, maskAll);`，`w` 由 `:687` `LoadAlign<float, LoadDist::DIST_BRC_B32>` 取 | **VF 合规** |
| **N11** 无权重单点积 | `m15_ple.asc:PleGateItem` pass C（`:548`，pass C 自 `:654`） | `:664` `Mul(p, a, b, maskAll);` → `:665`/`:666` 两条 `Cast`（对称的 bf16 物化）→ `:667` `Add(acc, acc, p, maskAll);` → `:669` `Reduce<ReduceType::SUM>(red, acc, maskAll);` | **VF 合规** |
| **N12** unpermute 加权行和 | `m15_moe_layer.h:FmaChunk`（`:1772`）/ `UnpermuteStage`（`:1785`） | `:1780` `Mul(yF, yF, wF, maskAll);` → `:1781` `Add(acc, acc, yF, maskAll);`（`FmaChunk`）；调用点 `:1852`–`:1861` | **VF 合规**（**但它的 `wF` 输入见 §3.1 S2-1**） |
| **N3** norm / softmax 的平方和、分母 | `m15_gdn_layer.h:NormStage`（**`class` 在 `:375`**；`:374` 是它的 `template <bool RES_F32>` 行 —— M123 r1 复审 p2-1 订正）、`m15_hc_layer.h:NormStage`（**方法**，`:706`）、`m15_moe_layer.h:NormStage`（`class` 在 `:595`）、`m15_ple.asc:PleGateItem` pass A/F、`m15_attn_prolog_probe.h:AivNormRope`（`:234`） | 逐条（**锚点取自本次 grep 输出，见 §7 C11**）：`m15_hc_layer.h:745` `Reduce<ReduceType::SUM>(red, acc, maskAll);`；`m15_gdn_layer.h:173`/`:223`/`:253` 的 `Reduce<ReduceType::SUM>`（在 `CalculateSquareReduceSum*` 里，由 `NormStage::ComputeRow` `:476` 调用）；`m15_moe_layer.h:713` `NormDonor::CalculateSquareReduceSum<float>(…)`；`m15_ple.asc:603` `Reduce<ReduceType::SUM>(r1, a1, maskAll);`；`m15_attn_prolog_probe.h:307` `ReduceSum(var, acc, all);`；`AivNormRope` 全程 `RegTensor`（`:278`/`:325`/`:358` 三个 `__VEC_SCOPE__`） | **VF 合规** |
| **N1** depthwise conv（GDN prolog） | `m15_gdn_layer.h:PrologBlock`（`:623`，**函数属性就是 `__simd_vf__`**） | `CONV_TAP` 宏展开到 `:637` `MulAddDst(acc, w, s, fullM)`；SiLU = `Muls`/`Exp`/`Adds`/`Div`（`:648`–`:656`） | **VF 合规** |
| **N2** depthwise conv（PLE ⑤） | `m15_ple.asc:PleConvItem`（`:779`） | `:866` `Mul(pr, wr, sr, maskAll);` → `:870` `Add(acc, acc, pr, maskAll);`；SiLU 段 `:875`–`:878` | **VF 合规** |
| **N5** MoE combine 逐元素 | `m15_moe_layer.h:CombineStage`（`:1890`） | `:1951`–`:1954` 的 sigmoid 四元组（`Muls`/`Exp`/`Adds`/`Div`）+ `:1959`/`:1964` `Mul`/`Add` | **VF 合规** |
| **N6** 排序 / 归并 | `m15_moe_layer.h:SoftmaxTopkRow`（`:1074`）/ `MergeTree`（`:1162`）/ `Merge2`（`:1188`） | `Sort32`（`:1111`）/ `MrgSort`（`:1204`）—— **`docs/05` §6.1 的裁定例外（ⓒ）**；其**算术部分**（max-shift+exp、renorm 除）在 `:1087`/`:1138` 两个 `__VEC_SCOPE__` 内 | **VF 合规 + 裁定例外** |
| **N7** `m15_loop_layout.h` 全文 | host 编译期常量 | —— | 不适用（无设备算件） |
| **N8** `m15_layer_loop.asc` 全文 | host 驱动 | `grep -c "__global__\|__aicore__" m15_layer_loop/m15_layer_loop.asc` → `0`、rc=1（§7 C9） | 不适用 |
| **N9** attention 直通占位 | `m15_attn_layer.h:m15_attn_passthrough_body`（`:45`） | 纯 `DataCopy`（无算术） | 不适用（且真 attention 归在飞 M101，其形态已是 `Mmad`） |

**另一条本次新核的正对照（不是 `docs/20` 的条目，但与 ⓕ-3 同向）**：
`m15_moe_layer.h:1887` 的段头注释逐字写着 —— **「sgate 的 sigmoid 在这里用向量做（S2 只出裸点积，
避免 AIV 内标量浮点）」**。⇒ 本仓**已经有一条自发的口径**：**把标量浮点从 AIV 上赶走、挪进向量**。
`m3-AIV 的标量量化器被替换` 那条同族注释也在（`:1491` 逐字「全 VEC 路径，**无 PIPE_S 挂载**」）。

---

## §5 与 `docs/20` 的边界（**不重复计数**，逐项对照）

| 本次栏 | 条目 | 与 `docs/20` 的关系 |
| --- | --- | --- |
| **①** | `PleGemv` / router 打分 + `SgateRows` / `GdnHeadRecurrence` | **= `docs/20` §2.2 的 V1/V2/V3**（见 `docs/20`）。**本文件不重复计数、不重排档位**；只补一句措辞订正（§2.2(a)） |
| **②** | S2-1 top-k 权重 bf16 RNE | **`docs/20` 未把它列为「计算」**。`docs/20` §3.4 只从**同步次序**角度登记过 `wtkUb[…]`（那是「标量写 UB → 搬走」的可见性家族，`docs/20` §3.2 末的 10 处之一）⇒ **本文件的角度是新的一层，不重复** |
| **②** | S2-2 n-gram id | **`docs/20` 未列**（`docs/20` §3.3 只登记过同一函数族的 **G1 落盘通路**：`outG.SetValue`，归在飞 M111）。⇒ **不重复**：那条是「怎么写盘」，本条是「怎么算」。**塔裁后此条归 I 类 · 不在管辖内**（§3.5） |
| **②** | S2-3 poolScale | `docs/20` 未列（`m15_attn_cache.h` 在 `docs/20` 只出现在 §3.4 的 `flagL.SetValue` 行）⇒ **不重复**。**塔裁后此条归 P 类 · 豁免**（§3.5） |
| **②** | S2-4 index / 前缀和族 | `docs/20` §3.5 登记过其中 **`VecQuantStage::Run` 的标量栈数组**（`WO-B3`，口径是**栈容量**）。⇒ 本文件只把同一站点按**另一条口径**（数据 vs 控制流）判为 **I 类不违规**，**并明确不复用它的计数** |
| **③** | §4 的一整张表 | 逐条引 `docs/20` 的 N 编号，**只核实现形态**，不新增计数 |

**一条必须一起写的限度**：`docs/20` 的扫描面是「matmul」；本文件的扫描面是「标量算术」
（`Mmad`/`MulAddDst` 两个 token + 三类标量形态）。**两个面不互相包含** —— 本文件**没有**重跑
`docs/20` 的判据 M，也不对它的 V1–V3 档位背书；两文各自的读数请分别按各自的命令核对。

---

## §6 可直接变成修单的清单

每条给 **scope（文件）→ 改什么 → 判据 → 依赖/风险**。**派单前请核对方的工作树 tip**。

| WO | 级 | 一句话 | scope | 归属预警 |
| --- | --- | --- | --- | --- |
| **WO-S1** | **P2** | S3 的 top-k 权重 bf16 舍入从**标量整数 RNE** 改成 **VF `Cast`（`CAST_RINT`）** | `m15_layer_loop/lift_moe_segment.py`（源）+ 重生成 `m15_layer_loop/m15_moe_layer.h` | **与在飞的 M105 同文件同函数** ⇒ **等 M105 合入后再派**，或并进 M105 |
| **WO-S2** | **已关闭**（塔裁：不在管辖内） | `IdsOneToken` 的 n-gram id 标量 int64 算术 —— **裁定为 I 类（gather 下标）⇒ 不改** | ——（原 scope `m15_layer_loop/m15_ple.asc` + 生成物 `m15_layer_loop/m15_ple_wire.h` **本次不动**） | 塔裁 2026-09-27；依据与判据见 §3.2 / §3.5 |
| **WO-S3** | **已关闭**（塔裁：豁免） | `attn_cache` 的池化系数**在既有常量间按条件选一个** —— **裁定为控制流 ⇒ 不改** | ——（原 scope `m15_layer_loop/m15_attn_cache.h` **本次不动**） | 塔裁 2026-09-27；依据与判据见 §3.3 / §3.5 |
| **WO-S4** | **不派（登记）** | index / 前缀和 / 计数排序族（I 类） | —— | 见 §3.4；`VecQuantStage` 标量栈属 `docs/20` 的 `WO-B3`，**本文件不复用** |

### WO-S1（P2）top-k 权重的 bf16 舍入 → VF

- **scope**：`m15_layer_loop/lift_moe_segment.py`（**规则源**）+ `m15_layer_loop/m15_moe_layer.h`
  （**生成物，勿手改**：改完源后重跑脚本）。若判据要跟着改：`m15_layer_loop/check_moe_ref.py`。
- **改什么**：`IndexGenStage::Run` 里 `:1353`–`:1357` 那 5 行（`GetValue` → `memcpy` → 标量 RNE →
  标量写 `wtkUb`）换成 **VF 的 `Cast`**：把 `wGm`（fp32 `[M,TOPK]`）整行读进 UB，
  在 `__VEC_SCOPE__` 里用 `Cast<bfloat16_t, float, castTraitB322B16>`（`RoundMode::CAST_RINT`）
  一次算完整行，再按**现有** `w_tk_packed` 协议（低 16 位）落 `wtkUb`。**协议不变** ⇒
  `UnpermuteStage` 侧**不需要动**。
- **判据**：① `wtk` 平面与现值**逐位**（RNE 方向相同；先用 A/B dump 锁住现值再换实现，
  要求 sha256 相同）；② **破坏对照**：把 `castTraitB322B16` 的 `CAST_RINT` 改成 `CAST_ROUND`
  ⇒ `wtk` 平面变红；③ **非空洞**：`wtk` 行内跨 token 不是常量；④ 既有 MoE 判据
  （`check_moe_ref.py`）全过。
- **依赖/风险**：`m15_moe_layer.h` 是 `lift_moe_segment.py` 的机械生成物（**M52 把第 7 条替换降级为
  `assert_upstream_vf()`**）⇒ 改源后必须重生成并核 `--check`（若脚本带该开关，见 §7 C7）。
  **与 M105 的时序见上表**。UB 侧需确认 `wGm` 的整行窗是否已在资源表里（`m15_moe_resources.h`），
  不够就加一格 `UB_RT_*`（那会动 `docs/20` 的 `WO-A2` 的同一张表 ⇒ 与 M105 的另一处潜在相交，
  **派单前请核 M105 的 tip**）。

### WO-S2（**已关闭**，塔裁 2026-09-27）n-gram id 的计算通路

- **裁定**：`IdsOneToken` 的 n-gram id 是 **gather 的下标**，**不在规则 ⓕ 的管辖内** ⇒ **本条关闭、不派改**
  （依据与统一判据见 §3.2 与 §3.5）。
- **原 scope（本次不动）**：`m15_layer_loop/m15_ple.asc` + 生成物 `m15_layer_loop/m15_ple_wire.h`；
  若日后形态变化要重判，叙述同步 `m15_layer_loop/ple/README.md`、`m15_layer_loop/m15_ple_check.py`。
- **触发重判的条件（§3.5 的唯一适用范围声明）**：一旦 id **作为操作数进入任何数值算式**
  （乘数/加数/系数），本条裁定**不再覆盖那一处** ⇒ 按 §1.2 的判据 S 重判。
- **若届时须改（保留两种改法，供未来使用）**：**(甲)** 只把 `cur*m0` / `p1*m1` / `p2*m2` / XOR 搬进 VF
  （取模 `FloorMod64` 是 64 迭代的移位-减法，需另找 `Reg` 等价物或「乘倒数 + 修正」）；
  **(乙)** 按 `docs/05` §6.1 的体例申请裁定例外并在使用点附一行注释。**两种本次都不做。**

### WO-S3（**已关闭**，塔裁 2026-09-27）池化缩放系数

- **裁定**：`poolScale` 是**在既有常量之间按条件选一个**（分支/条件侧）= **控制流** ⇒ **豁免、不派改**
  （依据与统一判据见 §3.3 与 §3.5）。
- **塔同时要求记住的一条**：该量随后作为 `Muls` 的**标量操作数**这件事本身**永远不是违规**；
  违规只可能发生在「这个系数是**用标量算术从数据里算出来**」的形态。
- **原 scope（本次不动）**：`m15_layer_loop/m15_attn_cache.h`。
- **触发重判的条件**：一旦这个系数**由数据算出**（例如对某个统计量做 `1/sqrt(…)`）⇒ 按 §1.2 重判。

### WO-S4（登记，不派）

- 见 §3.4。**这一族算出的量只用于寻址**（按 §1.3 的塔裁判据：不进数值算式）；
  且其中 `VecQuantStage` 的标量栈已有 `docs/20` 的 `WO-B3` 在跟（口径不同，是**栈容量**）
  ⇒ **不在本文件另立修单**。

---

## §7 复现命令与读数（全部实跑；`LC_ALL=C`）

以下每条都在本 worktree（分支 `feat/m123-scalar-computation-sweep`，rev `9296e79463…`）上
**实跑过并抄下输出**；命令一律**钉不可变 rev 的内容锚点**（符号名 / 逐字字符串），行号只作提示。

### C0 规则 ⓕ 是否已进 main

```console
$ grep -n "ⓕ" docs/05-megakernel-design.md
(空输出)
$ echo $?
1
$ grep -c "ⓔ" docs/05-megakernel-design.md
4
```

### C1 `Mmad` / `MmadMx` 站点（清单 ① 的分母）

```console
$ grep -n "Mmad(\|MmadMx(" m15_layer_loop/*.h m15_layer_loop/*.asc
m15_layer_loop/m15_attn_prolog_probe.h:140:            Mmad(cL0C, a2, b2, mp);
m15_layer_loop/m15_gdn_layer.h:1483:                    AscendC::Mmad(cL0C, a2, b2, mmadParams);
m15_layer_loop/m15_hc_layer.h:299:            AscendC::Mmad(cL0C, a2, b2, mp);
m15_layer_loop/m15_moe_layer.h:204:            AscendC::MmadMx(cL0C, a2, b2, mmadParams);
```

（4 行；与 `docs/20` §2.1 的合规锚表一一对应。**限度**：这条命令的 token 是 `Mmad(`/`MmadMx(`；
它**不**证明「没有以别的名字手写的矩阵乘法」—— 见 §8 Q4。）

### C2 router 打分的实现是 Reg 向量（不是标量 MAC）

```console
$ grep -n "MulAddDst" m15_layer_loop/m15_moe_layer.h
949:    //   但 **chunk 升序的 MulAddDst 累加序与 Reduce SUM 一字未改** ⇒ 逐位相同。
968:                    MulAddDst(acc, xf, w, maskAll);
1007:                MulAddDst(a0, xf, w, maskAll);
1009:                MulAddDst(a1, xf, w, maskAll);
1011:                MulAddDst(a2, xf, w, maskAll);
1013:                MulAddDst(a3, xf, w, maskAll);
1015:                MulAddDst(a4, xf, w, maskAll);
1017:                MulAddDst(a5, xf, w, maskAll);
1019:                MulAddDst(a6, xf, w, maskAll);
1021:                MulAddDst(a7, xf, w, maskAll);
```

（10 行：1 行注释 + `:968` 属 `SgateRows` + 8 行属 `GemvGroupRow` —— 与 `docs/20` §2.2-V2 的读数一致。）

### C3 标量声明面（S-a）

```console
$ grep -nE "float [a-zA-Z_]+ *=|half [a-zA-Z_]+ *=|bfloat16_t [a-zA-Z_]+ *=|double [a-zA-Z_]+ *=" \
    m15_layer_loop/m15_gdn_layer.h m15_layer_loop/m15_moe_layer.h m15_layer_loop/m15_hc_layer.h \
    m15_layer_loop/m15_attn_cache.h m15_layer_loop/m15_ple.asc m15_layer_loop/m15_layer_kernel.h \
    m15_layer_loop/m15_attn_layer.h m15_layer_loop/m15_attn_kv.h
m15_layer_loop/m15_gdn_layer.h:105:constexpr float RMS_EPS = 1e-6f;
m15_layer_loop/m15_gdn_layer.h:106:constexpr float RMS_AVG_FACTOR = 1.0f / static_cast<float>(HIDDEN);
m15_layer_loop/m15_gdn_layer.h:286:    static constexpr float POS_INF = 3.40282366920938E+38;
m15_layer_loop/m15_gdn_layer.h:547:constexpr float EPS = 1e-6f;                // l2norm eps
m15_layer_loop/m15_gdn_layer.h:548:constexpr float QSCALE = 0.08838834764831845f;  // 1/sqrt(128)
m15_layer_loop/m15_gdn_layer.h:549:constexpr float SPTH = 20.0f;               // softplus threshold（β=1）
m15_layer_loop/m15_gdn_layer.h:1215:constexpr float EPS_ = 1e-6f;                           // config.rms_norm_eps
m15_layer_loop/m15_gdn_layer.h:1216:constexpr float AVG_FACTOR_ = 1.0f / static_cast<float>(M15G::HEAD_D);
m15_layer_loop/m15_moe_layer.h:357:constexpr float RMS_EPS = 1e-6f;
m15_layer_loop/m15_moe_layer.h:358:constexpr float RMS_AVG_FACTOR = 1.0f / static_cast<float>(HIDDEN);
m15_layer_loop/m15_moe_layer.h:476:    static constexpr float POS_INF = 3.40282366920938E+38;
m15_layer_loop/m15_moe_layer.h:755:constexpr float RT_NEG_BIG = -3.0e38f;
m15_layer_loop/m15_moe_layer.h:1353:                const float f = wGm.GetValue(s);
m15_layer_loop/m15_hc_layer.h:151:    static constexpr float POS_INF = 3.40282366920938E+38;
m15_layer_loop/m15_attn_cache.h:319:        const float poolScale =
m15_layer_loop/m15_ple.asc:116:constexpr float EPS = 1e-6f;                    // rms_norm_eps
m15_layer_loop/m15_ple.asc:1487:    const double fillMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
m15_layer_loop/m15_ple.asc:1488:    const double regMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
m15_layer_loop/m15_ple.asc:1572:    double devRowFail = 0.0;
```

**逐行归属（本命令共 19 行输出）**：**14 行是 `constexpr` / `static constexpr` 的编译期常量**
（`RMS_EPS`/`RMS_AVG_FACTOR`/`POS_INF`/`EPS`/`QSCALE`/`SPTH`/`RT_NEG_BIG`… —— **编译期折叠，不构成
运行期标量计算**；按文件拆：`m15_gdn_layer.h` 8 + `m15_moe_layer.h` 4 + `m15_hc_layer.h` 1 +
`m15_ple.asc` 1 = **14**，命令 `grep -c constexpr` 的读数见 §7 C3b）；
**3 行是 `m15_ple.asc` 的 `main()` 里的 host 计时**（`:1487`/`:1488`/`:1572`，
在 `main()` 内、属 host 侧）；**只有 2 行落在设备函数体里**：
`m15_moe_layer.h:1353`（`const float f = wGm.GetValue(s);` —— **S2-1**）、
`m15_attn_cache.h:319`（`const float poolScale =` —— **S2-3**）。（`14 + 3 + 2 = 19`。）
**两处下面各抄一段原文**：

```console
$ sed -n '1350,1358p' m15_layer_loop/m15_moe_layer.h
                // Y 布局是 [slot][M_MAX][HIDDEN]（槽位 padding）→ inv 存 padding 行号
                invUb[s] = static_cast<int32_t>(pos);   // [M40-6a] 紧凑行号（= 计数排序产出位置）
                // bf16(RNE) 权重位打包进 int32 低 16 位（m8 w_tk_packed 协议）
                const float f = wGm.GetValue(s);
                uint32_t bits;
                __builtin_memcpy(&bits, &f, 4);
                const uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
                wtkUb[t * 16 + k] = static_cast<int32_t>(rounded & 0xFFFFu);
            }
$ sed -n '318,320p' m15_layer_loop/m15_attn_cache.h
        // 池化的缩放：SUM_NOT_MEAN 档不除 4（`ops/qsa.py:431` 的方向反写）。**提到 VF 外**当标量。
        const float poolScale =
            (mode == AC_MODE_SUM_NOT_MEAN) ? 1.0f : (1.0f / static_cast<float>(COMP_TOKENS_PER_STATE));
```

### C3b §3.0「19 行 / 14 行 constexpr」这两个数的**分子与分母各一条命令**

（本条是 **M123 r1 复审 p2-1** 的订正依据：初版写「20 行 / 15 行」，复审在同 rev、同 8 个文件上重跑
得到 **19 / 14**，并指出我自己贴的输出块本来就是 19 行 —— 订正见 §3.0 的归属句。）

```console
$ grep -nE "float [a-zA-Z_]+ *=|half [a-zA-Z_]+ *=|bfloat16_t [a-zA-Z_]+ *=|double [a-zA-Z_]+ *=" \
    m15_layer_loop/m15_gdn_layer.h m15_layer_loop/m15_moe_layer.h m15_layer_loop/m15_hc_layer.h \
    m15_layer_loop/m15_attn_cache.h m15_layer_loop/m15_ple.asc m15_layer_loop/m15_layer_kernel.h \
    m15_layer_loop/m15_attn_layer.h m15_layer_loop/m15_attn_kv.h > /tmp/c3.txt
$ wc -l < /tmp/c3.txt
19
$ grep -c "constexpr" /tmp/c3.txt
14
$ cut -d: -f1 /tmp/c3.txt | sort | uniq -c
      1 m15_layer_loop/m15_attn_cache.h
      8 m15_layer_loop/m15_gdn_layer.h
      1 m15_layer_loop/m15_hc_layer.h
      5 m15_layer_loop/m15_moe_layer.h
      4 m15_layer_loop/m15_ple.asc
```

（**分子与分母同源**：同一次 `grep` 的输出即分母 `19`；`grep -c constexpr` 在同一份输出上取分子 `14`。
按文件拆：`m15_gdn_layer.h` 8 行**全是** `constexpr`；`m15_moe_layer.h` 5 行 = 4 `constexpr` + **1 设备体**
（`:1353`）；`m15_hc_layer.h` 1 行 = `constexpr`；`m15_attn_cache.h` 1 行 = **设备体**（`:319`）；
`m15_ple.asc` 4 行 = 1 `constexpr` + 3 host 计时 ⇒ `constexpr` 合计 `8+4+1+1 = 14`，
设备体 `1+1 = 2`，host 计时 `3`，`14+2+3 = 19` ✓。
**限度**：`grep -c constexpr` 在 `m15_moe_layer.h` 上把 `:357`/`:358`/`:476`/`:755` 四行判为 `constexpr`
—— 那是**按行**计数、不解析语义；本条的归属句是**逐行看过**才写的，不单靠这条计数。）

### C4 `GetValue` / `SetValue` 的逐文件计数（S-b 的分母）

```console
$ grep -c "GetValue\|SetValue" m15_layer_loop/m15_moe_layer.h m15_layer_loop/m15_ple.asc \
    m15_layer_loop/m15_attn_cache.h m15_layer_loop/m15_layer_kernel.h
m15_layer_loop/m15_moe_layer.h:11
m15_layer_loop/m15_ple.asc:24
m15_layer_loop/m15_attn_cache.h:8
m15_layer_loop/m15_layer_kernel.h:4
```

（另有 `m15_attn_kv_probe.h` 6 行（probe 工程，`docs/05` §6.1 的豁免清单把 `probe_*` 排除在本规范之外）、
`m15_attn_kv.h` 3 行（本次逐行看过：三处**都在注释里**，是地址算式的说明）、
`m15_moe_resources.h` 1 行（同样在注释里）。**这条计数只说明「查过多少处」，不说明「哪些是计算」** ——
逐处归属在下面 C5 与 §3.0 里。）

### C5 裸标量写（S-c 的面）

```console
$ grep -rnE "^\s*(__ubuf__ )?[A-Za-z_][A-Za-z0-9_]*\[[^]]*\] *= " \
    m15_layer_loop/m15_moe_layer.h m15_layer_loop/m15_ple.asc m15_layer_loop/m15_attn_cache.h \
  | grep -v "://"
m15_layer_loop/m15_moe_layer.h:1312:            cntUb[e] = 0;
m15_layer_loop/m15_moe_layer.h:1320:            cntUb[e] = cntUb[e] + 1;
m15_layer_loop/m15_moe_layer.h:1323:        offUb[0] = 0;
m15_layer_loop/m15_moe_layer.h:1325:            offUb[e + 1] = offUb[e] + cntUb[e];
m15_layer_loop/m15_moe_layer.h:1335:            curUb[e] = offUb[e];
m15_layer_loop/m15_moe_layer.h:1347:                curUb[e] = static_cast<int32_t>(pos + 1);
m15_layer_loop/m15_moe_layer.h:1348:                srcUb[pos] = static_cast<int32_t>(t);
m15_layer_loop/m15_moe_layer.h:1349:                expUb[pos] = e;
m15_layer_loop/m15_moe_layer.h:1351:                invUb[s] = static_cast<int32_t>(pos);   // [M40-6a] 紧凑行号（= 计数排序产出位置）
m15_layer_loop/m15_moe_layer.h:1357:                wtkUb[t * 16 + k] = static_cast<int32_t>(rounded & 0xFFFFu);
m15_layer_loop/m15_moe_layer.h:1387:            bUb[32] = static_cast<int32_t>(oobCountUb);
m15_layer_loop/m15_moe_layer.h:1527:        startOff[0] = 0;
m15_layer_loop/m15_moe_layer.h:1529:            startOff[e + 1] = startOff[e] + static_cast<uint32_t>(countsGm.GetValue(e));
m15_layer_loop/m15_attn_cache.h:253:                    candP[nCand++] = q;
m15_layer_loop/m15_attn_cache.h:260:                candP[nCand++] = p;
m15_layer_loop/m15_attn_cache.h:330:                    mp[i] = rowUb + (q - chunkStart) * AC_UB_ROW_ELEMS;
m15_layer_loop/m15_attn_cache.h:332:                    mp[i] = zeroUb;
m15_layer_loop/m15_attn_cache.h:334:                    mp[i] = rngUb + (c * COMP_TOKENS_PER_STATE + i) * AC_UB_ROW_ELEMS;
```

（**18 行**。**`:1357` 是本文件 §3.1 的落点**（唯一一行右边是**数据面**算术）；
`:1312`–`:1351` 是计数排序的计数/前缀和/槽号（**I 类**）；`:1529` 是前缀和（**I 类**）；
`:1387` 是诊断计数（**控制/门控信息**，`docs/20` §3.2 的第一例）；`m15_attn_cache.h` 的 5 行分别是
**组号数组**（`candP`）与 **UB 指针数组**（`mp`），两级都不含数据面算术。
**限度**：这条正则只覆盖「一行以内、`标识符[下标] = …`」这一种**书写形态**，跨行赋值与
`SetValue` 形态在别的判别路径上（后者的清单见 `docs/20` §3.3 与本文件 §7 C4 的计数）。）

### C6 S2-1 的消费者（`wtk` → unpermute 的 `wF`）

```console
$ grep -n "ShiftLefts\|wRaw" m15_layer_loop/m15_moe_layer.h
453:                ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
1773:                                RegTensor<int32_t>& wRaw, RegTensor<float>& wF, __ubuf__ bfloat16_t* yUb,
1778:    LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(wRaw, wRowUb + TK);
1779:    ShiftLefts((RegTensor<uint32_t>&)wF, (RegTensor<uint32_t>&)wRaw, static_cast<int16_t>(16), maskAll);
1842:            RegTensor<int32_t> wRaw;
1850:                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(wRaw, wRowUb);
1851:                ShiftLefts((RegTensor<uint32_t>&)wF, (RegTensor<uint32_t>&)wRaw, static_cast<int16_t>(16), maskAll);
1853:                if constexpr (TK > 1) FmaChunk<1>(acc, yF, yB16, wRaw, wF, yUb, wRowUb, maskAll, off);
…（`:1854`–`:1861` 是 `TK = 2..9` 同款展开）
```

### C7 两份 PLE 副本的关系（S2-2 的 scope 依据）

```console
$ grep -n "^__aicore__ inline \(uint32_t\|uint64_t\|void\) [A-Z]" m15_layer_loop/m15_ple.asc
188:__aicore__ inline uint64_t UMod64(uint64_t x, uint64_t d)
201:__aicore__ inline uint64_t FloorMod64(int64_t x, int64_t d)
213:__aicore__ inline uint32_t ReqOf(int32_t t, GlobalTensor<int32_t>& qsl, uint32_t nReq)
245:__aicore__ inline uint32_t IdsOneToken(uint32_t t, GlobalTensor<int32_t>& idsG, …)
352:__aicore__ inline void PleGather(uint32_t bid, uint32_t nblk, BodyGm& G, uint32_t nTok, uint32_t negMask)
483:__aicore__ inline void PleGemv(uint32_t bid, uint32_t nblk, BodyGm& G, uint32_t nTok, uint32_t negMask)
548:__aicore__ inline void PleGateItem(uint32_t t, uint32_t s, BodyGm& G, uint32_t* bad, uint32_t negMask)
779:__aicore__ inline void PleConvItem(uint32_t t, BodyGm& G, uint32_t* bad, uint32_t negMask)
$ grep -n "^__aicore__ inline \(uint32_t\|uint64_t\|void\) [A-Z]" m15_layer_loop/m15_ple_wire.h
119:__aicore__ inline uint64_t UMod64(uint64_t x, uint64_t d)
132:__aicore__ inline uint64_t FloorMod64(int64_t x, int64_t d)
144:__aicore__ inline uint32_t ReqOf(int32_t t, GlobalTensor<int32_t>& qsl, uint32_t nReq)
176:__aicore__ inline uint32_t IdsOneToken(uint32_t t, …
283:__aicore__ inline void PleGather(…)
414:__aicore__ inline void PleGemv(…)
479:__aicore__ inline void PleGateItem(…)
710:__aicore__ inline void PleConvItem(…)
$ sed -n '245,320p' m15_layer_loop/m15_ple.asc > /tmp/a.txt
$ sed -n '176,251p' m15_layer_loop/m15_ple_wire.h > /tmp/b.txt
$ diff /tmp/a.txt /tmp/b.txt && echo IDENTICAL
IDENTICAL
```

（本条的**长行以 `…` 节选**（函数签名的后半段），**行号与函数名本身逐字**。
⇒ `IdsOneToken` 在**两份**里逐字相同；改源后必须**重跑生成器**，见 §6 WO-S2。）

### C8 既有判据的位置（S2-2 的判据从哪来）

- `A_ids` 的 T1 逐位判据在 `m15_layer_loop/m15_ple_check.py`（M85 的 13 条判据）与
  `m15_layer_loop/ple/README.md` 的叙述里；本文件**不重跑它们**（本 mission 零设备、不改判据）。

### C9 两个「不是设备算件」的文件

```console
$ grep -c "__global__\|__aicore__" m15_layer_loop/m15_layer_loop.asc
0
$ echo $?
1
$ for f in m15_layer_loop/m15_layer_kernel.h m15_layer_loop/m15_attn_kv.h m15_layer_loop/m15_attn_layer.h; do
    echo -n "$f VEC_SCOPE="; grep -c "__VEC_SCOPE__" $f; done
m15_layer_loop/m15_layer_kernel.h VEC_SCOPE=0
m15_layer_loop/m15_attn_kv.h VEC_SCOPE=0
m15_layer_loop/m15_attn_layer.h VEC_SCOPE=0
```

（⇒ `m15_layer_kernel.h` 在本次读数里只有**搬运 + 相位边界**（`VEC_SCOPE` 计数 0）；
`m15_attn_kv.h` 是 host/device 共用的**地址算式表**；`m15_attn_layer.h` 是纯抄写占位。**限度**：
「`VEC_SCOPE` 计数 0」只说明**没有词法 VF 块**；`__simd_vf__` 形态在这三个文件里也是 0，
但这条推理**只对这两个 token 成立**（本次没有做「任意 API 名手写的算术」的全量扫描）。）

### C10 `__VEC_SCOPE__` / `__simd_vf__` / `Reg` 算子的逐文件计数（② / ③ 的分母）

```console
$ for f in m15_gdn_layer.h m15_moe_layer.h m15_hc_layer.h m15_ple.asc m15_ple_wire.h \
           m15_attn_cache.h m15_attn_prolog_probe.h m15_layer_kernel.h m15_attn_kv.h m15_attn_layer.h; do
    printf "%-32s VEC_SCOPE=%-3s simd_vf=%-3s Reg_Mul=%-3s Reduce=%-3s\n" "$f" \
      "$(grep -c '__VEC_SCOPE__' m15_layer_loop/$f)" "$(grep -c '__simd_vf__' m15_layer_loop/$f)" \
      "$(grep -c '\bMul(\|\bMuls(\|\bAdd(\|\bAdds(' m15_layer_loop/$f)" "$(grep -c 'Reduce' m15_layer_loop/$f)"; done
m15_gdn_layer.h                  VEC_SCOPE=10  simd_vf=8   Reg_Mul=60  Reduce=25
m15_moe_layer.h                  VEC_SCOPE=25  simd_vf=0   Reg_Mul=40  Reduce=24
m15_hc_layer.h                   VEC_SCOPE=8   simd_vf=0   Reg_Mul=34  Reduce=7
m15_ple.asc                      VEC_SCOPE=16  simd_vf=0   Reg_Mul=39  Reduce=6
m15_ple_wire.h                   VEC_SCOPE=16  simd_vf=0   Reg_Mul=39  Reduce=6
m15_attn_cache.h                 VEC_SCOPE=2   simd_vf=0   Reg_Mul=5   Reduce=0
m15_attn_prolog_probe.h          VEC_SCOPE=9   simd_vf=0   Reg_Mul=21  Reduce=1
m15_layer_kernel.h               VEC_SCOPE=0   simd_vf=0   Reg_Mul=0   Reduce=0
m15_attn_kv.h                    VEC_SCOPE=0   simd_vf=0   Reg_Mul=0   Reduce=0
m15_attn_layer.h                 VEC_SCOPE=0   simd_vf=0   Reg_Mul=0   Reduce=0
```

（⇒ §4 的「已合规」对照表就建立在这张表上：**算式的主体在这三个 VF 计数非零的文件里**。
两个 PLE 文件的 `VEC_SCOPE=16` 逐字相同，与 C7 的 `diff → IDENTICAL` 互相印证。）

### C11 §4 的逐条 VF 证据（抄自 `grep`，非手打）

```console
$ grep -rn -F "Mul(t, sg, xr" m15_layer_loop/m15_hc_layer.h
m15_layer_loop/m15_hc_layer.h:874:                    Mul(t, sg, xr, maskAll);
$ grep -rn -F "Mul(t, bor, w" m15_layer_loop/m15_hc_layer.h
m15_layer_loop/m15_hc_layer.h:690:                Mul(t, bor, w, maskAll);              // bo · injW[s]
$ grep -rn -F "Mul(p, a, b" m15_layer_loop/m15_ple.asc
m15_layer_loop/m15_ple.asc:664:            Mul(p, a, b, maskAll);
$ grep -rn -F "Mul(yF, yF, wF" m15_layer_loop/m15_moe_layer.h
m15_layer_loop/m15_moe_layer.h:1780:    Mul(yF, yF, wF, maskAll);
$ grep -rn -F "Mul(pr, wr, sr" m15_layer_loop/m15_ple.asc
m15_layer_loop/m15_ple.asc:866:                Mul(pr, wr, sr, maskAll);
$ grep -n "MulAddDst(acc, w, s" m15_layer_loop/m15_gdn_layer.h
637:    MulAddDst(acc, w, s, fullM)
```

### C12 本文件自身的一致性（`docs/scan_doc_refs.py`；**双 locale**）

```console
$ LC_ALL=C python3 docs/scan_doc_refs.py > /tmp/scan_c.log 2>&1; echo "LC_ALL=C rc=$?"
LC_ALL=C rc=0
$ head -2 /tmp/scan_c.log
docs scanned=18  doc refs=275 (unresolved=0)  section refs=250 (unresolved=0)
in-repo path refs=486  gating=0 (distinct=0)  allowlisted=2 (distinct=2)
$ tail -1 /tmp/scan_c.log
RESULT: OK (18 files scanned, 275 doc refs, 250 section refs, 486 path refs, 2 allowlisted, 0 out-of-range)
$ LC_ALL=C.UTF-8 python3 docs/scan_doc_refs.py > /tmp/scan_u.log 2>&1; echo "LC_ALL=C.UTF-8 rc=$?"
LC_ALL=C.UTF-8 rc=0
$ head -2 /tmp/scan_u.log
docs scanned=18  doc refs=275 (unresolved=0)  section refs=250 (unresolved=0)
in-repo path refs=486  gating=0 (distinct=0)  allowlisted=2 (distinct=2)
$ tail -1 /tmp/scan_u.log
RESULT: OK (18 files scanned, 275 doc refs, 250 section refs, 486 path refs, 2 allowlisted, 0 out-of-range)
```

（**两个 locale 的读数逐字相同、rc 都是 `0`**（`docs scanned=18` = 本文件已在 `docs/` 里）。
本条在**落库前**的基线是 `17 files / 224 doc refs / 217 section refs / 350 path refs / 0 out-of-range`；
**塔裁小节（§3.5）+ 各处就地改标 + C13 写入后**的读数是 `259 / 245 / 457`；
**r1 复审 p2-1 的订正（§7 C3 归属句 + **§7 C3b** + §8.2）写入后**的读数是 `259 / 246 / 470`；
**r2 复审 p2-1 的订正（C12 归属句 + **§7 C14** + §8.3）写入后**的读数是 `268 / 250 / 478`；
**r3 复审的订正（§8.3 两个数 + **§8.4 自检小节**）写入后**的读数是 `274 / 250 / 485`；
**r4 复审的订正（§3.0 的 9/1/1/2 + **把自检里的硬编码期望改成真算**）写入后**的读数就是上面这一组。
**两个 delta（`245→246` 与 `457→470`）与 r2 这一轮的 delta 都按“谁的 token 变了”逐条查过**
（用扫描器**自己的** `SECREF` / `PATHREF` 正则对相邻两个 rev 的本文件做 **token diff**，
命令、读数和逐条 `added/removed` 清单见 **§7 C14**）：

- **`section refs` 的增量**：**唯一**新增的 section-ref token 出现在 **§8.2 的正文里**，
  **不是** C3b（初版 C12 曾把 r1 那一处归给 C3b —— **r2 复审 p2-1 订正**，见 §8.3）。
  ⚠ **判据要点（扫描器语义）**：`SECREF` 要求 **`docs/NN` 前缀**
  （``docs/NN`` + 可选反引号 + 间隔 + `§X`）⇒ **裸 `§7` 一律不被匹配**；
  §7 C14 的 token diff 实测：本文件里裸 `§7` 出现 **27 次**，
  而 **`SECREF` 命中的 `§7` token = 0** ⇒ **C3b 对 `section refs` 的贡献是 0**
  （**逐 token 的清单以 §7 C14 为准**，此处不复述具体 token）。
- **`path refs` 的增量**：C3b 按 r1 复审要求把那条命令里的 **8 个文件路径**（各 1 次）连同
  `cut | uniq -c` 的**按文件计数** 5 行（`attn_cache`/`gdn`/`hc`/`moe`/`ple` 各 1）逐字落库
  ⇒ `8 + 5 = 13`（**零删除**），与 §7 C14 的 `PATHREF` token diff 逐条一致。
  **r2 这一轮的 `path refs` 增量**同样来自落库的新命令块（§7 C14 的 `PATHREF` diff 输出块），
  逐 token 清单见 §7 C14。

**`gating=0` 与 `out-of-range=0` 未变**。⇒ 本文件里 `docs/NN`、`docs/NN §X`、
本仓路径三类引用**都可解析**（`unresolved=0`、`gating=0`）。
⚠ **这些归属句的限度（三条）**：① 它们只对**本文件自身的** token 增量成立（在 `git show <rev>:docs/21-…` 上做 diff），
**不能**直接读成全局 `in-repo path refs` 变动的**唯一**来源（后者是**全部 18 个 docs 文件**的合计）；
② 归属只在**相邻两个 rev** 之间成立 —— 本文件每被订正一次，**它自己的引用集合就会变**，
⇒ **跨多轮的增量必须逐段 diff，不能从头一次基线一路相减**；
③ 逐 token 清单只在 §7 C14 里出现一次，C12 **不复述**（复述会再造出新的引用、把读数再推一次）。
⚠ **一条先例（本文件踩过并改掉）**：初稿曾在 §0 的在飞表里**把分支期文件名写成「本仓目录前缀 + 文件名」**
（三个只在 B1/B4/B5 分支上存在的头文件），扫描器把「带 `mNN_*/` 前缀的路径」当**本仓路径**解析 ⇒
`gating=3`、**rc=1**（落库前的 rc 是 `1`，不是 `0`）。改法 = **分支期文件只写文件名、不写目录前缀**（见 §0 的表）。)

### C13 §3.3 的塔裁依据（`poolScale` 里的那个常量确实是编译期常量）

```console
$ grep -rn "COMP_TOKENS_PER_STATE *=" m15_layer_loop/*.h
m15_layer_loop/m15_attn_kv.h:160:constexpr uint32_t COMP_TOKENS_PER_STATE = 4;             // indexer_compress_ratio
```

（⇒ `1.0f / static_cast<float>(COMP_TOKENS_PER_STATE)` 是**编译期可折叠**的常量，
运行期只剩「在两个既有常量间按 `mode` 选一个」—— 这正是 §3.3 / §3.5 塔裁判「选择 ≠ 算术」的**事实前提**。
**限度**：这条只证明它是 `constexpr`；**没有**反汇编证明编译器一定折叠，故 §3.3 的措辞是
「会被折叠成常量」而不是「必然已折叠」。）

### C14 §7 C12 两条 delta 归属的**取证**（用扫描器自己的正则做 token diff）

（本条是 **M123 r2 复审 p2-1** 的订正依据。判定用的是**扫描器自己那一对正则**（下面逐字抄出，
**未修改 `docs/scan_doc_refs.py`**）；对两个 rev 的本文件取 token 后做 `collections.Counter` 差分。
**可复现性说明（如实标注限度）**：驱动脚本本次只放在 `/tmp`（**未入仓**），但它只做
「读 `git show <rev>:docs/21-…` → 用下面两条正则 `finditer` → Counter 差分 → 打印 added/removed」，
把这两条正则抄进去即可重建；**入仓的是“命令 + 逐字输出”**，不是脚本本身。）

```console
# SECREF = r'docs/([0-9]{2})(?:-[A-Za-z0-9_.-]+)?(?:\.md)?' + r'`?' + r'\s*(?:的\s*)?§\s*([0-9]+(?:\.[0-9]+)*)'
$ git show 995eec0:docs/21-scalar-computation-sweep.md > /tmp/a.md
$ git show 28f3313:docs/21-scalar-computation-sweep.md > /tmp/b.md
$ python3 /tmp/secref_diff.py     # 用上面那条 SECREF 对 /tmp/a.md 与 /tmp/b.md 取 token + Counter 差分
SECREF tokens rev995eec0: 28  rev28f3313: 29
added  : [('docs/20` §2.2', 1)]
removed: []
bare '§7' (no docs/NN) raw count: 27
matched SECREF containing §7: []
```

（⇒ **唯一新增的 section-ref token 是 `` docs/20` §2.2 ``**；**裸 `§7` 出现 27 次、而 `SECREF` 命中的
`§7` token 为 0** ⇒ 「C3b 新增了一处 `§7` 段内引用」**不成立**，初版 C12 的归属句已按 r2 订正。
**本条与 C12 里 `section refs` 245→246 的差**（每条 §-引用出现次数）**不是同一个量**：
token 计数按**出现次数**，扫描器的 `section refs` 也是按**出现次数**（`SECREF.finditer` 的命中数），
两者在这里恰好都是 `+1`；上述 `28 → 29` 是本文件自身的 token 总数。）

```console
# PATHREF = re.compile(
#     r'(?<![A-Za-z0-9_./-])('
#     r'(?:m[0-9]{1,2}_[A-Za-z0-9_]+|tools|baseline_env|probe_sync_quirks|build|data|evidence|logs|golden|weights)'
#     r'/[A-Za-z0-9_./-]+\.(asc|h|hpp|cpp|cu|cuh|py|sh|md|json|toml|txt|log|cmake|yaml|yml|cfg|ini)'
#     r')(?![A-Za-z0-9])')
$ python3 /tmp/pathref_diff.py     # 同一对 rev 的 PATHREF token + Counter 差分
pathref total: 107 -> 120
added (+13):
  + 2 m15_layer_loop/m15_attn_cache.h
  + 1 m15_layer_loop/m15_attn_kv.h
  + 1 m15_layer_loop/m15_attn_layer.h
  + 2 m15_layer_loop/m15_gdn_layer.h
  + 2 m15_layer_loop/m15_hc_layer.h
  + 1 m15_layer_loop/m15_layer_kernel.h
  + 2 m15_layer_loop/m15_moe_layer.h
  + 2 m15_layer_loop/m15_ple.asc
removed (-0):
```

（⇒ `path refs` 的 `+13 / 零删除` 与 C3b 落库的内容逐条吻合：命令里 8 个路径各 1 + `uniq -c` 的
5 行各 1 = 13。**限度**：这是**本文件自身**的 token 增量；扫描器报的 `in-repo path refs=470`
是**全部 18 个 docs 文件**的合计。）

**同一个方法对「`28f3313`（r1 tip）→ 当前 tip」的读数**（用上面同一对正则，同一段脚本；
**逐 token 明细不在此复述** —— 复述本身会给本文件再添引用、把被测对象自身的读数再推一次。
**注意**：这一格自 r2 起就写「→ 本轮 tip」，而 tip 每轮都在动 ⇒ 本表按**当前 tip**重测过，
**跨轮不得相减**）：

| 分栏 | 本文件 token 数 | 净增量 | 与 C12 记录的读数对照 |
| --- | --- | --- | --- |
| `SECREF`（section refs） | `29 → 33` | **+4**（`removed = 0`） | `246 + 4 = 250` ✓ |
| `DOCREF` 且**非** `§` 命中（doc refs） | `35 → 51` | **+16**（`removed = 0`） | `259 + 16 = 275` ✓ |
| `PATHREF`（in-repo path refs） | `120 → 136` | **+16**（`removed = 0`） | `470 + 16 = 486` ✓ |

（⇒ **三个分栏的净增量与该轮扫描器读数逐项自洽，且三轮都是零删除**。**限度**：
① 「本文件 token 数」与扫描器的分栏计数是**同一套正则**下的量，但扫描器的分栏是**全部 18 个 docs 文件**的合计，
本表只是**本文件**那一份；② `DOCREF(非 §)` 的列是**扣掉**与 `SECREF` 起点重叠的命中之后的值
（即分子/分母同源的那一列），故它与 `DOCREF` 的原始命中数不同。）

---

## §8 未决 / 待确认（需要什么证据，**不要猜**）+ 本文的关系声明

**先一句**：**原来的 Q1 / Q2 两条边界项已由塔裁结案**（2026-09-27，M123 复审前）—— 定案与**统一判据**见
**§3.5**，就地记录见 §3.2 / §3.3。下表把这两行保留为**已结案**（`~~删除线~~`）供复审对照，
**`未决` 档现在只剩下面这三条（Q3 / Q4 / Q5）**。

| # | 问题 | 需要什么证据 | 卡住了什么 |
| --- | --- | --- | --- |
| **~~Q1~~** | ~~n-gram id 的**计算**（S2-2）算不算「数据」~~ | **已结案（塔裁 2026-09-27：是 gather 下标 ⇒ 不在管辖内、不改）** | 原 WO-S2 **已关闭**；依据与统一判据见 §3.2 / §3.5 |
| **~~Q2~~** | ~~「常量选择」（S2-3 的 `poolScale`）算不算计算~~ | **已结案（塔裁 2026-09-27：选择 ≠ 算术 ⇒ 豁免、不改）** | 原 WO-S3 **已关闭**；依据与统一判据见 §3.3 / §3.5 |
| **Q3** | S2-1 的 `Cast` 改造**能否要求逐位** | 需要**一次设备档**：先把现值 `wtk` 落 dump 取 sha256，换实现后比 sha256 | WO-S1 的判据强度（不能逐位就降 T3 + 推界，`docs/17` §1.1） |
| **Q4** | 有没有「**以其它 API 名手写的 MAC**」（C2 的 token 面没覆盖的形态） | 需要一次**更宽 token** 的 grep（`Matmul|Gemm|Dot|Mac|I8|fp4` 一族）并逐条看是不是走 cube | ① 栏「本次未另找新站点」这句的**限度** |
| **Q5** | B1–B5 段体内部的标量算术（§0 的在飞表） | **不在本 mission 的 scope**；塔已在广播里点名各段体自查 | 本文件的覆盖面**只到 main 上的 mount 点**；B 段体落地后**建议另派一次同形扫描**（命令可照 §7 C3/C5 复用） |

**Q5 一条补充（时限事实）**：`m15_layer_kernel.h` 的 `M15L_PrefillPhaseA`（§0 提到的 mount 点）
在 `pfWired != 0` 时**什么都不写**（逐字注释「段体还没落地 ⇒ **什么都不写** ⇒ host 的逐行标记判据
必然变红（响亮失败，不静默通过）」）⇒ **B 段体合入前，那一路不存在「标量算数据」的现场**。

### §8.1 本文自身的关系声明（供复审）

- **本文是 M123 的唯一产物**。`m15_layer_loop/**` 下**没有**被本 mission 改动
  （`git status --short` 在本文件之外应为空；**这一句请复审在 tip 上用 `git show --stat` 核**）。
- **塔裁（2026-09-27）已就地写入**：§1.3 的判据行 + **§3.5 的集中定案小节**，并在 §3.2 / §3.3 / §5 / §6 / §8
  的相应条目上就地改标（Q1/Q2 由「待塔裁」→「已结案」；WO-S2/WO-S3 由「待塔裁」→「已关闭」）。
  **本次判据的净变动**：② 类**判违规仍为 1 条**（§3.1 S2-1），**没有新增/减少违规计数**；
  两条边界项由「登记待裁」变为「登记 + 裁定依据」。§7 为此新增 **C13**（`COMP_TOKENS_PER_STATE` 是 `constexpr`）。
- **本文与 `docs/20` 的分工**（重复一遍，供复审对照）：
  `docs/20` = **判据 M（是不是矩阵乘法）+ 判据 V（有没有走 cube）**；本文 = **判据 S（有没有用 scalar 算数据）**。
  两文对同一处代码的结论**可以并存**（例：`PleGemv` 在 `docs/20` 是 V1「该走 cube」，
  在本文的 S 判据下是「**已经是 VF**，不属 S 面的违规」）。**`docs/20` 的 V1/V2/V3 与本文的 ① 栏是同一批站点**，
  引用时**注明「见 `docs/20`」、不重复计数**。
- **本文的判据是代码级的**：`S-a`/`S-b`/`S-c` 三个面都有可复现的 grep（§7 C3/C4/C5），
  但**没有**设备读数证明任何一处在关键路径上的耗时 —— 与 `docs/20` 的零设备档同性质。
- **本文引用的 rev** 是 `9296e7946330162d51a57ba9fe58454a76c37c9d`（不可变）。
  凡引**在飞分支**的地方（§0 的表）都写成 **branch 名 + `git diff --stat main...<branch>` 的读数**，
  **并已注明它是「时效事实」** —— 分支前移后请以 `git diff` 重跑为准（塔的第六变体纪律）。

### §8.2 r1 复审的处置（**p2-1items**，fix-then-merge）

复审（M123 r1）**除下面一条外全部 PASS**；它逐字核过 §3.1 的源文件链
（`m15_moe_layer.h:1353-1357` → `:1376` → `UnpermuteStage` 的 `:1820` → `:1778`/`:1779` → `:1780` 的
`Mul(yF,yF,wF,…)`）、`docs/20` §2.2 与本文 ① 栏的关系、以及双 locale 的 `scan_doc_refs.py` 读数。

| # | 复审内容 | 处置 |
| --- | --- | --- |
| **p2-1** | §7 C3 的归属句写「**20 行**输出 / **15 行** `constexpr`」，而**我自己贴的输出块就是 19 行**；复审在同一 rev、同样这 8 个文件上重跑得 **19 行 / 14 行 `constexpr`**（拆解：`constexpr` 14 + host 计时 3 + 设备体 2 = 19） | **已改**：§7 C3 的归属句改为「**19 行**」+「**14 行** `constexpr`」并补 `14+3+2=19` 的算式与**按文件拆解**；另新增 **§7 C3b** 把这两个数的**分子与分母各一条命令**（同一次 `grep` 的输出即分母，`grep -c constexpr` 在同一份输出上取分子）+ `cut \| uniq -c` 的按文件计数落库。**结论未变**（两行设备体的定位本来就对） |
| **顺带** | `§4 N3` 引 `m15_gdn_layer.h:NormStage`（`:374`）—— `class NormStage {` 实际在 **`:375`**（`:374` 是它的 `template <bool RES_F32>` 行） | **已改**：`§4 N3` 的行号改为 `:375` 并注明 `:374` 是 `template` 行；顺手把同格里 `m15_hc_layer.h:NormStage` 标注为**方法**（该文件里它是方法、不是 `class`） |

**这一轮的性质**：**纯订正**（数字与行号），**不改任何判定、不新增/减少任何计数**
（② 类判违规仍为 1 条；WO-S2/WO-S3 仍为已关闭）。**没有**触碰 `m15_layer_loop/**`。

### §8.3 r2 复审的处置（**p2-1items**，再次 fix-then-merge）

复审（M123 r2）确认 r1 的订正**全部成立**（§7 C3 的 19 行 / 14 行 `constexpr`、C3b 块、§4 N3 的 `:375`、
`m15_hc_layer.h` 那处是**方法**），并**独立复现**了 `path refs 457→470`（= 8 + 5 = 13、零删除）这一归属；
它打回的是**另一条归属句**。

| # | 复审内容 | 处置 |
| --- | --- | --- |
| **p2-1** | §7 C12 写「`section refs` 245→246 **是因为 C3b 新增了一处 `§7` 段内引用**」——**这不是扫描器数的东西**：`SECREF` 要求 `docs/NN` 前缀，**裸 `§7` 不被匹配**；复审用 `SECREF` 做 token diff 指出**唯一新增的 section-ref token 是 `` docs/20` §2.2 ``，出现在 §8.2** | **已改**：C12 的归属句改为「唯一新增的 section-ref token = `` docs/20` §2.2 ``，出现在 §8.2」+ 明写「**裸 `§7` 不被 `SECREF` 匹配 ⇒ C3b 的贡献是 0**」；并**新增 §7 C14** 把这次 token diff（`SECREF` 与 `PATHREF` 两条）连同读数落库；**数字 `246` 未动** |

#### 同类扫描（r2 要求的「一次性扫干净」）

**做法**：在**本文件全文**上 grep 所有「某个读数变化是因为 XXX」这一类**归属句**，
逐条按扫描器的 `SECREF` / `PATHREF` **语义**自查（token diff 的读数见 §7 C14），
涉及 section refs 的**一律**按「**必须有 `docs/NN` 前缀才被计数**」重写。

```console
$ P='所致\|是因为\|导致\|随之增加\|增量来自\|的来源\|新增了一处\|贡献是\|零删除'
$ grep -c "$P" docs/21-scalar-computation-sweep.md
22
$ H=$(grep -n "^### §8.3" docs/21-scalar-computation-sweep.md | cut -d: -f1)
$ sed -n "1,$((H-1))p" docs/21-scalar-computation-sweep.md | grep -c "$P"
7
$ sed -n "${H},\$p" docs/21-scalar-computation-sweep.md | grep -c "$P"
15
```

⚠ **先说一条自指（否则这个数会被误读）**：这条正则命中的 **22 行里有 15 行在本节之内**
（含 §8.4 的自检脚本、本节的限度说明与 §8.4.3 的取证命令）——
因为本节既要**引述**被判的那句原话、又要**写明**判据词（`$P` 那一行本身就是一次命中），
于是**自己也被自己匹配**。⇒ 下面只对**§8.3 之外的 7 行**逐条归类（行号取自这次实跑，仅作查阅提示）。

| 位置（本次实跑行号） | 归属句 | 自查结论 |
| --- | --- | --- |
| `:169`+`:170`（**同一句，跨两行**） | 「① 栏的双重违规**不是**因为它们用了标量」（`§2.2(b)`） | **不是扫描器读数的归属句**（是判据/措辞论断，与 `scan_doc_refs.py` 无关） |
| `:793`（`§7 C12`） | 「C3b 对 `section refs` 的**贡献是 0**」 | **成立**（负向归属；依据 = §7 C14 的 `SECREF` token diff：命中 `§7` 的 token = 0） |
| `:797`（`§7 C12`） | 「`path refs` 457→470 … **零删除**」（C3b 落库 13 处） | **成立**（`PATHREF` token diff 复核：+13 / 零删除，逐条吻合） |
| `:845`（`§7 C14`） | 「『C3b **新增了一处** `§7` 段内引用』**不成立**」 | **成立**——它是**对 r1 那句错归属的订正记录**（原句本身已不存在于 §7 C12） |
| `:870`（`§7 C14`） | 「`path refs` 的 `+13 / **零删除**` …」 | **成立**（同上，是 C14 里对该归属的复述与限度说明） |
| `:883`（`§7 C14`） | 「三个分栏的净增量 … **三轮都是零删除**」 | **成立**（r2 轮的对照表；依据 = 同一脚本对 `28f3313 → 本轮 tip` 的读数） |

**同类扫描的结论**（**就这条命令的范围而言**）：**全部命中 22 行；其中 §8.3 自身占 15 行（自指）**，
**正文归属句 7 行 / 6 句**（`:169`+`:170` 是一句）——**1 句已按 r2 改写**（原 `§7 C12` 那句），
**4 句经自查成立**，**1 句判定不属于计数归属句**。
**§7 里唯一涉及 `section refs` 的归属句就是 C12 那一条**；另两处 `docs/20 §…` 是**正文引用**
（证明两文分工），不是「谁导致了计数变化」的归属。

**限度（两条）**：① 这条 grep 用的是「**所致 / 是因为 / 导致 / 随之增加 / 增量来自 / 的来源 /
新增了一处 / 贡献是 / 零删除**」这一组词，它**不覆盖**用别的措辞写的归属句（例如「由此」「源于」「归因于」）；
本文件里另已**逐行读过 §7 全文**作为补充，但这仍不是「任意措辞都能抓到」的保证。
② 上面的行号是**本次实跑**的读数，本文件每订正一次行号都会漂 ⇒ **判据请用引号里的原句**，不要用行号。

**这一轮的性质**：**纯订正**（两个数 + 新增 §8.4 自检），**不改任何判定、不新增/减少任何计数**
（② 类判违规仍为 1 条；WO-S2/WO-S3 仍为已关闭）。**没有**触碰 `m15_layer_loop/**`。

### §8.4 全篇**数字自检**（r3 要求：做成可复跑的脚本）

**为什么要有这一节**：本文件到 r3 已是**第四轮**，前三轮里有三轮打回的**都是同一族**——
「**散文里的数字与它自己的命令输出不一致**」（r1：C3 的 20/15；r2：C12 的 section-ref 归属；
r3：§8.3 结论段的 16/9）。⇒ 只靠人来核不够，**把检查做成脚本**。

**做法**：脚本把「散文里出现的计数」（用固定正则从本文件里抠出来）与**同一条命令的实跑读数**
逐条比对，输出 `检查项 | 散文声称 | 实跑 | 判定`；有 FAIL 就 `rc=1`。
**`--rev <rev>` 模式**是**负向对照**：把「散文」换成该 rev 的旧版，看判据会不会变红
（判据**能咬住**才算判据 —— 本仓的既有纪律）。

**脚本（落到仓外任意路径后，在仓库根跑）**：

```python
#!/usr/bin/env python3
# M123 数字自检：把 docs/21 散文里的计数与「同一条命令的实跑读数」逐条对账。
# 用法：python3 selfcheck.py [--rev <rev>]   —— --rev 模式 = 负向对照（散文取该 rev 的旧版）
import re, subprocess, glob, sys, collections
DOC = 'docs/21-scalar-computation-sweep.md'
REV = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--rev' else None
S = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
doc = S('git show %s:%s' % (REV, DOC)) if REV else open(DOC, encoding='utf-8').read()
def c(rx, g=1):
    m = re.search(rx, doc, re.S)
    return m.group(g) if m else None
def G(pat, files):
    return S("grep -nE '%s' %s" % (pat, files)).splitlines()
MF = ['m15_gdn_layer.h', 'm15_moe_layer.h', 'm15_hc_layer.h', 'm15_attn_cache.h', 'm15_ple.asc',
      'm15_layer_kernel.h', 'm15_attn_layer.h', 'm15_attn_kv.h']
ML = ' '.join('m15_layer_loop/' + f for f in MF)
RX3 = r'float [a-zA-Z_]+ *=|half [a-zA-Z_]+ *=|bfloat16_t [a-zA-Z_]+ *=|double [a-zA-Z_]+ *='
C3 = G(RX3, ML)
k3 = lambda f: len([l for l in C3 if l.startswith('m15_layer_loop/' + f) and 'constexpr' in l])
C5 = [l for l in S(r"""grep -rnE "^\s*(__ubuf__ )?[A-Za-z_][A-Za-z0-9_]*\[[^]]*\] *= " m15_layer_loop/m15_moe_layer.h m15_layer_loop/m15_ple.asc m15_layer_loop/m15_attn_cache.h | grep -v "://" """).splitlines()]
k5 = lambda f: len([l for l in C5 if l.startswith('m15_layer_loop/' + f)])
GV = {f: int(S('grep -c "GetValue\\|SetValue" m15_layer_loop/' + f).strip() or 0) for f in
      ['m15_moe_layer.h', 'm15_ple.asc', 'm15_attn_cache.h', 'm15_layer_kernel.h',
       'm15_attn_kv_probe.h', 'm15_attn_kv.h', 'm15_moe_resources.h']}
FMA = G('MulAddDst', 'm15_layer_loop/m15_moe_layer.h')
sc = S('LC_ALL=C python3 docs/scan_doc_refs.py')
SR = re.search(r'doc refs=(\d+) \(unresolved=(\d+)\)\s+section refs=(\d+)', sc)
PR = re.search(r'in-repo path refs=(\d+)\s+gating=(\d+)', sc)
RXS = r'docs/([0-9]{2})(?:-[A-Za-z0-9_.-]+)?(?:\.md)?' + r'`?' + r'\s*(?:的\s*)?§\s*([0-9]+(?:\.[0-9]+)*)'
RXD = r'docs/([0-9]{2})(?:-[A-Za-z0-9_.-]+)?(?:\.md)?'
RXP = r'(?<![A-Za-z0-9_./-])((?:m[0-9]{1,2}_[A-Za-z0-9_]+|tools|baseline_env|probe_sync_quirks|build|data|evidence|logs|golden|weights)/[A-Za-z0-9_./-]+\.(asc|h|hpp|cpp|cu|cuh|py|sh|md|json|toml|txt|log|cmake|yaml|yml|cfg|ini))(?![A-Za-z0-9])'
old = S('git show 28f3313:' + DOC)
ss = lambda t: {m.start() for m in re.finditer(RXS, t)}
donly = lambda t: [m.group(0) for m in re.finditer(RXD, t) if m.start() not in ss(t)]
P = '所致|是因为|导致|随之增加|增量来自|的来源|新增了一处|贡献是|零删除'
tot = len(S('grep -nE "%s" %s' % (P, DOC)).splitlines())
H = int(S('grep -n "^### §8.3" %s' % DOC).split(':')[0])
before = int(S('sed -n "1,%dp" %s | grep -cE "%s"' % (H - 1, DOC, P)).strip() or 0)
after = int(S('sed -n "%d,\\$p" %s | grep -cE "%s"' % (H, DOC, P)).strip() or 0)
R = []
def k(name, claimed, actual):
    R.append((name, str(claimed), str(actual), 'PASS' if str(claimed) == str(actual) else 'FAIL'))
k('§2.1 Mmad 站点"只有 N 处"', c(r'`Mmad\(` / `MmadMx\(` 的站点\*\*只有 (\d+) 处\*\*'),
  len(G(r'Mmad\(|MmadMx\(', ' '.join(sorted(glob.glob('m15_layer_loop/*.h')) + sorted(glob.glob('m15_layer_loop/*.asc'))))))
k('§2.2(a) GemvGroupRow :1007…:1021 共 N 行', c(r'共 (\d+) 行）'), len([l for l in FMA if 1007 <= int(l.split(':')[0]) <= 1021]))
k('§7 C2 首行归述"（N 行：1 行注释"', c(r'（(\d+) 行：1 行注释'), len(FMA))
k('§7 C3 "本命令共 N 行输出"', c(r'本命令共 (\d+) 行输出'), len(C3))
k('§7 C3 "N 行是 constexpr"', c(r'\*\*(\d+) 行是 `constexpr`'), k3('m15_gdn_layer.h') + k3('m15_moe_layer.h') + k3('m15_hc_layer.h') + k3('m15_ple.asc'))
m4 = re.search(r'`m15_gdn_layer\.h` (\d+) \+\s*`m15_moe_layer\.h` (\d+) \+\s*`m15_hc_layer\.h` (\d+) \+\s*`m15_ple\.asc` (\d+) = \*\*14\*\*', doc)
k('§7 C3 按文件 8 + 4 + 1 + 1', '/'.join(m4.groups()) if m4 else '?',
  '/'.join(str(k3(f)) for f in ['m15_gdn_layer.h', 'm15_moe_layer.h', 'm15_hc_layer.h', 'm15_ple.asc']))
k('§7 C3b uniq 行 moe/ple/hc/attn_cache = 5/4/1/1',
  '/'.join([c(r'(?m)^\s+(\d+) m15_layer_loop/m15_moe_layer\.h$') or '?', c(r'(?m)^\s+(\d+) m15_layer_loop/m15_ple\.asc$') or '?',
            c(r'(?m)^\s+(\d+) m15_layer_loop/m15_hc_layer\.h$') or '?', c(r'(?m)^\s+(\d+) m15_layer_loop/m15_attn_cache\.h$') or '?']),
  '/'.join([str(len([l for l in C3 if l.startswith('m15_layer_loop/' + f)])) for f in ['m15_moe_layer.h', 'm15_ple.asc', 'm15_hc_layer.h', 'm15_attn_cache.h']]))
k('§7 C4 四个文件 = 11/24/8/4',
  '/'.join([c(r'`m15_moe_layer\.h` \*\*(\d+)\*\*、`m15_ple\.asc`') or '?', c(r'`m15_ple\.asc` \*\*(\d+)\*\*、`m15_attn_cache\.h`') or '?',
            c(r'`m15_attn_cache\.h` \*\*(\d+)\*\*、`m15_layer_kernel\.h`') or '?', c(r'`m15_layer_kernel\.h` \*\*(\d+)\*\*') or '?']),
  '/'.join(str(GV[f]) for f in ['m15_moe_layer.h', 'm15_ple.asc', 'm15_attn_cache.h', 'm15_layer_kernel.h']))
k('§7 C4 另有三个文件 = 6/3/1',
  '/'.join([c(r'`m15_attn_kv_probe\.h` (\d+) 行') or '?', c(r'`m15_attn_kv\.h` (\d+) 行') or '?', c(r'`m15_moe_resources\.h` (\d+) 行') or '?']),
  '/'.join(str(GV[f]) for f in ['m15_attn_kv_probe.h', 'm15_attn_kv.h', 'm15_moe_resources.h']))
k('§3.0/§7 C5 "命中 N 行"', c(r'本次范围内命中 (\d+) 行'), len(C5))
k('§3.0 moe/attn_cache = 13/5', '%s/%s' % (c(r'`m15_moe_layer\.h` (\d+) 行（\*{0,2}S3 索引生成') or '?', c(r'`m15_attn_cache\.h` (\d+) 行（`candP`/`mp`') or '?'),
  '%d/%d' % (k5('m15_moe_layer.h'), k5('m15_attn_cache.h')))
moe_rows = [l for l in C5 if l.startswith('m15_layer_loop/m15_moe_layer.h')]
n_wtk = len([l for l in moe_rows if 'wtkUb[' in l])
n_diag = len([l for l in moe_rows if 'bUb[' in l])
n_vq = len([l for l in moe_rows if 'startOff[' in l])
n_s3 = len(moe_rows) - n_wtk - n_diag - n_vq
m5 = re.search(r'S3 索引生成(?:本体)?\s*(\d+) 行\*{0,2} \+ `wtk` (\d+) 行 \+\s*诊断槽 (\d+) 行 \+ `VecQuantStage` 前缀和 (\d+) 行', doc)
k('§3.0 拆解 S3(本体)/wtk/诊断槽/VecQuant（真算：按内容把 %d 行分类）' % len(moe_rows),
  '/'.join(m5.groups()) if m5 else '?', '%d/%d/%d/%d' % (n_s3, n_wtk, n_diag, n_vq))
sum4 = sum(int(x) for x in m5.groups()) if m5 else -1
m_moe = re.search(r'`m15_moe_layer\.h` (\d+) 行（', doc)
k('§3.0 moe 小计 == 四项之和（文档内部自洽）',
  '%s==%d' % (m_moe.group(1) if m_moe else '?', sum4), '%d==%d' % (len(moe_rows), n_s3 + n_wtk + n_diag + n_vq))
k('§3.0 moe 小计 == 实测行数（散文 vs 源）', m_moe.group(1) if m_moe else '?', str(len(moe_rows)))
m_attn = re.search(r'`m15_attn_cache\.h` (\d+) 行（`candP`/`mp`', doc)
m_tot = re.search(r'本次范围内命中 (\d+) 行', doc)
k('§3.0 总 18 == moe 小计 + attn_cache（文档内部自洽）',
  '%s==%d+%d' % (m_tot.group(1) if m_tot else '?', int(m_moe.group(1)) if m_moe else -1, int(m_attn.group(1)) if m_attn else -1),
  '%d==%d+%d' % (len(C5), len(moe_rows), k5('m15_attn_cache.h')))
k('§0 grep -c ⓔ = 4', c(r'`grep -c "ⓔ" docs/05-megakernel-design\.md` → `(\d+)`'), int(S('grep -c "ⓔ" docs/05-megakernel-design.md').strip()))
n_ff2 = len(G('ⓕ', 'docs/05-megakernel-design.md'))
k('§0 ⓕ 在 main 上（散文写「空输出」）', ('空输出' if 'ⓕ" docs/05-megakernel-design.md` → **空输出' in doc else '?'),
  ('空输出' if n_ff2 == 0 else '%d 行' % n_ff2))
S("sed -n '245,320p' m15_layer_loop/m15_ple.asc > /tmp/v_a.txt; sed -n '176,251p' m15_layer_loop/m15_ple_wire.h > /tmp/v_b.txt")
same_ple = S('diff /tmp/v_a.txt /tmp/v_b.txt').strip() == ''
k('§7 C7 两份 PLE 副本（散文写 IDENTICAL）', ('IDENTICAL' if re.search(r'(?m)^IDENTICAL$', doc) else '?'),
  ('IDENTICAL' if same_ple else 'DIFFERENT'))
k('§7 C9 layer_loop.asc 设备符号 = 0', c(r'`grep -c "__global__[^`]*` → `(\d+)`'), int(S('grep -c "__global__\\|__aicore__" m15_layer_loop/m15_layer_loop.asc').strip() or 0))
RX10 = re.compile(r'(?m)^(m15_[A-Za-z0-9_]+\.(?:h|asc))\s+VEC_SCOPE=(\d+)\s+simd_vf=(\d+)\s+Reg_Mul=(\d+)\s+Reduce=(\d+)\s*$')
doc10 = {m.group(1): '%s/%s/%s/%s' % m.groups()[1:] for m in RX10.finditer(doc)}
act10 = {}
for f in doc10:
    p = 'm15_layer_loop/' + f
    act10[f] = '%s/%s/%s/%s' % (S('grep -c "__VEC_SCOPE__" ' + p).strip(), S('grep -c "__simd_vf__" ' + p).strip(),
                                S(r"""grep -c '\bMul(\|\bMuls(\|\bAdd(\|\bAdds(' """ + p).strip(), S('grep -c "Reduce" ' + p).strip())
for f in sorted(set(doc10) | set(act10)):
    if doc10.get(f) != act10.get(f):
        print('  C10 mismatch', f, 'doc=', doc10.get(f), 'actual=', act10.get(f))
k('§7 C10 表 %d 行的读数（真算：从文档表格解析期望值）' % len(doc10), 'True', doc10 == act10)
k('§7 C12 doc/section refs', '%s/%s' % (c(r'doc refs=(\d+) \(unresolved=0\)'), c(r'section refs=(\d+) \(unresolved=0\)')), '%s/%s' % (SR.group(1), SR.group(3)))
k('§7 C12 path refs/gating', '%s/%s' % (c(r'in-repo path refs=(\d+)'), c(r'gating=(\d+) \(distinct')), '%s/%s' % (PR.group(1), PR.group(2)))
for nm, rx, fn in [('SECREF token 数', r'\| `SECREF`（section refs） \| `(\d+) → (\d+)`', lambda t: len(re.findall(RXS, t))),
                   ('DOCREF(非 §) token 数', r'\| `DOCREF` 且\*\*非\*\* `§` 命中（doc refs） \| `(\d+) → (\d+)`', lambda t: len(donly(t))),
                   ('PATHREF token 数', r'\| `PATHREF`（in-repo path refs） \| `(\d+) → (\d+)`', lambda t: len(re.findall(RXP, t)))]:
    m = re.search(rx, doc)
    k('§7 C14 %s' % nm, '%s→%s' % m.groups() if m else None, '%s→%s' % (fn(old), fn(doc)))
k('§8.3 "全部命中 N 行"', c(r'\*\*全部命中 (\d+) 行；其中 §8\.3 自身占 \d+ 行（自指）\*\*'), tot)
k('§8.3 "§8.3 自身占 N 行"', c(r'全部命中 \d+ 行；其中 §8\.3 自身占 (\d+) 行（自指）'), after)
k('§8.3 "§8.3 之外的 N 行"', c(r'下面只对\*\*§8\.3 之外的 (\d+) 行\*\*'), before)
k('§8.3 tot = before + after', str(tot), str(before + after))
if REV:
    print('== 负向对照：散文取自 rev %s（源文件与扫描器按工作区跑）==' % REV)
w = max(len(r[0]) for r in R)
print('%-*s | %-11s | %-11s | %s' % (w, '检查项', '散文声称', '实跑', '判定'))
print('-' * (w + 42))
for r in R:
    print('%-*s | %-11s | %-11s | %s' % (w, r[0], r[1], r[2], r[3]))
bad = [r for r in R if r[3] != 'PASS']
print('-' * (w + 42))
print('TOTAL %d checks, PASS %d, FAIL %d' % (len(R), len(R) - len(bad), len(bad)))
sys.exit(1 if bad else 0)
```

**A. 自检（工作区 tip）**：输出 A，`rc=0`。
**B. 负向对照①**：`--rev 6d4a86e`（r3 tip —— 它**正是** r4 打回 p2-1 的那一版），输出 B，`rc=1`。
**C. 负向对照②**：`--rev fd02939`（r2 tip —— r3 打回的那一版），输出 C，`rc=1`。

**输出 A**（`python3 selfcheck.py`）：

```console
检查项                                             | 散文声称        | 实跑          | 判定
-----------------------------------------------------------------------------------------
§2.1 Mmad 站点"只有 N 处"                            | 4           | 4           | PASS
§2.2(a) GemvGroupRow :1007…:1021 共 N 行          | 8           | 8           | PASS
§7 C2 首行归述"（N 行：1 行注释"                          | 10          | 10          | PASS
§7 C3 "本命令共 N 行输出"                              | 19          | 19          | PASS
§7 C3 "N 行是 constexpr"                          | 14          | 14          | PASS
§7 C3 按文件 8 + 4 + 1 + 1                         | 8/4/1/1     | 8/4/1/1     | PASS
§7 C3b uniq 行 moe/ple/hc/attn_cache = 5/4/1/1   | 5/4/1/1     | 5/4/1/1     | PASS
§7 C4 四个文件 = 11/24/8/4                          | 11/24/8/4   | 11/24/8/4   | PASS
§7 C4 另有三个文件 = 6/3/1                            | 6/3/1       | 6/3/1       | PASS
§3.0/§7 C5 "命中 N 行"                             | 18          | 18          | PASS
§3.0 moe/attn_cache = 13/5                      | 13/5        | 13/5        | PASS
§3.0 拆解 S3(本体)/wtk/诊断槽/VecQuant（真算：按内容把 13 行分类） | 9/1/1/2     | 9/1/1/2     | PASS
§3.0 moe 小计 == 四项之和（文档内部自洽）                     | 13==13      | 13==13      | PASS
§3.0 moe 小计 == 实测行数（散文 vs 源）                    | 13          | 13          | PASS
§3.0 总 18 == moe 小计 + attn_cache（文档内部自洽）        | 18==13+5    | 18==13+5    | PASS
§0 grep -c ⓔ = 4                                | 4           | 4           | PASS
§0 ⓕ 在 main 上（散文写「空输出」）                         | 空输出         | 空输出         | PASS
§7 C7 两份 PLE 副本（散文写 IDENTICAL）                  | IDENTICAL   | IDENTICAL   | PASS
§7 C9 layer_loop.asc 设备符号 = 0                   | 0           | 0           | PASS
§7 C10 表 10 行的读数（真算：从文档表格解析期望值）                 | True        | True        | PASS
§7 C12 doc/section refs                         | 275/250     | 275/250     | PASS
§7 C12 path refs/gating                         | 486/0       | 486/0       | PASS
§7 C14 SECREF token 数                           | 29→33       | 29→33       | PASS
§7 C14 DOCREF(非 §) token 数                      | 35→51       | 35→51       | PASS
§7 C14 PATHREF token 数                          | 120→136     | 120→136     | PASS
§8.3 "全部命中 N 行"                                 | 22          | 22          | PASS
§8.3 "§8.3 自身占 N 行"                             | 15          | 15          | PASS
§8.3 "§8.3 之外的 N 行"                             | 7           | 7           | PASS
§8.3 tot = before + after                       | 22          | 22          | PASS
-----------------------------------------------------------------------------------------
TOTAL 29 checks, PASS 29, FAIL 0
```

**输出 B**（`python3 selfcheck.py --rev 6d4a86e`，**负向对照①**，r3 tip）：

```console
== 负向对照：散文取自 rev 6d4a86e（源文件与扫描器按工作区跑）==
检查项                                             | 散文声称        | 实跑          | 判定
-----------------------------------------------------------------------------------------
（节选：下面只抄**与输出 A 逐字不同**的行 —— 连同 TOTAL 共 7 行；其余 26 行与 A 逐字相同）
§3.0 拆解 S3(本体)/wtk/诊断槽/VecQuant（真算：按内容把 13 行分类） | 10/1/1/2    | 9/1/1/2     | FAIL
§3.0 moe 小计 == 四项之和（文档内部自洽）                     | 13==14      | 13==13      | FAIL
§7 C12 doc/section refs                         | 274/250     | 275/250     | FAIL
§7 C12 path refs/gating                         | 485/0       | 486/0       | FAIL
§7 C14 DOCREF(非 §) token 数                      | 35→50       | 35→50       | PASS
§7 C14 PATHREF token 数                          | 120→135     | 120→135     | PASS
-----------------------------------------------------------------------------------------
TOTAL 29 checks, PASS 25, FAIL 4
```

**怎么读输出 B（这一条是本轮 p2-2 要的「自检有没有牙」的示范）**：
**`§3.0` 那 2 处 FAIL 正是 p2-1** —— `10 vs 9`（拆解里的 1-off）与 `13==14 vs 13==13`
（小计与四项之和不自洽）。**这两条只有把「期望值」改成真算之后才可能出现**
（改前它们是写死 `'10/1/1/2'` 的字面量，两个数一致 ⇒ **永远 PASS、抓不到任何东西**）。
另 2 处 `§7 C12` 的 FAIL 是 `--rev` 模式的**已知偏差**（散文取旧 rev、扫描器按工作区跑），
按 §8.4.2 限度 3 不当结论。**口径**：输出 B 全文 **34 行** = 1 行模式表头 + 33 行；
与 A **逐字不同 7 行**（6 个检查行 + TOTAL）、**其余 26 行逐字相同**。

**输出 C**（`python3 selfcheck.py --rev fd02939`，**负向对照②**，r2 tip）：

```console
== 负向对照：散文取自 rev fd02939（源文件与扫描器按工作区跑）==
（节选：与输出 A 逐字不同的 9 行 —— 6 个检查行 + TOTAL 及 2 行列名相同者，如下）
§3.0 拆解 S3(本体)/wtk/诊断槽/VecQuant（真算：按内容把 13 行分类） | 10/1/1/2    | 9/1/1/2     | FAIL
§3.0 moe 小计 == 四项之和（文档内部自洽）                     | 13==14      | 13==13      | FAIL
§7 C12 doc/section refs                         | 268/250     | 275/250     | FAIL
§7 C12 path refs/gating                         | 478/0       | 486/0       | FAIL
§7 C14 DOCREF(非 §) token 数                      | 35→44       | 35→44       | PASS
§7 C14 PATHREF token 数                          | 120→128     | 120→128     | PASS
§8.3 "全部命中 N 行"                                 | 16          | 22          | FAIL
§8.3 "§8.3 自身占 N 行"                             | 9           | 15          | FAIL
-----------------------------------------------------------------------------------------
TOTAL 29 checks, PASS 23, FAIL 6
```

**怎么读输出 C**：它同时抓到 r2 的两类问题 —— **§3.0 的 1-off（同 B）** 与 **§8.3 的 `16/9`**（r3 打回的那两个数）。
**口径**：输出 C 全文 **34 行**；与 A **逐字不同 9 行**、**其余 24 行逐字相同**；
两处 `§7 C12` FAIL 同 B，是 `--rev` 模式的已知偏差。

#### §8.4.1 检查结果表（「检查了哪些数字 / 哪些对不上 / 怎么改的」）

| # | 检查项（数字的出处） | 散文声称 | 实跑 | 判定 | 对不上时的处置 |
| --- | --- | --- | --- | --- | --- |
| 1 | §2.1 `Mmad(`/`MmadMx(` 站点数 | 4 | 4 | PASS | — |
| 2 | §2.2(a) `GemvGroupRow` 的 `:1007…:1021` 行数 | 8 | 8 | PASS | — |
| 3 | §7 C2 `MulAddDst`（`m15_moe_layer.h` 全体）行数 | 10 | 10 | PASS | — |
| 4 | §7 C3 标量声明面总行数 | 19 | 19 | PASS | — |
| 5 | §7 C3 其中 `constexpr` 行数 | 14 | 14 | PASS | — |
| 6 | §7 C3 按文件 8 + 4 + 1 + 1 | 8/4/1/1 | 8/4/1/1 | PASS | — |
| 7 | §7 C3b `uniq -c` 五行的 moe/ple/hc/attn_cache | 5/4/1/1 | 5/4/1/1 | PASS | — |
| 8 | §7 C4 四文件 `GetValue\|SetValue` | 11/24/8/4 | 11/24/8/4 | PASS | — |
| 9 | §7 C4 另三文件 | 6/3/1 | 6/3/1 | PASS | — |
| 10 | §3.0 / §7 C5 裸标量写行数 | 18 | 18 | PASS | — |
| 11 | §3.0 其中 moe/attn_cache | 13/5 | 13/5 | PASS | — |
| 12 | §3.0 拆解 **S3(本体) / wtk / 诊断槽 / VecQuant**（**真算**：把 moe 的 13 行按内容分类） | 9/1/1/2 | 9/1/1/2 | PASS | **本轮曾为 `10/1/1/2`**（r4 p2-1）⇒ 已改；见下方「r4 实际改掉」 |
| 13 | §3.0 moe 小计 == 四项之和（**文档内部自洽**；**本轮新增**） | 13==13 | 13==13 | PASS | **本轮曾为 `13==14`** ⇒ 同上 |
| 14 | §3.0 moe 小计 == 实测行数（散文 vs 源；**本轮新增**） | 13 | 13 | PASS | — |
| 15 | §3.0 总 18 == moe 小计 + attn_cache（**本轮新增**） | 18==13+5 | 18==13+5 | PASS | — |
| 16 | §0 `grep -c "ⓔ"` | 4 | 4 | PASS | — |
| 17 | §0 `ⓕ` 在 main 上（期望值**从散文措辞取**「空输出」） | 空输出 | 空输出 | PASS | 本轮由写死 `'0'` 改成取措辞 |
| 18 | §7 C7 两份 PLE 副本（期望值**从散文措辞取** `IDENTICAL`） | IDENTICAL | IDENTICAL | PASS | 本轮由写死 `'True'` 改成取措辞 |
| 19 | §7 C9 `m15_layer_loop.asc` 设备符号 | 0 | 0 | PASS | — |
| 20 | §7 C10 表 **10 行**的四元组读数（期望值**从文档表格解析**） | 表值 | 一致 | PASS | 本轮由写死 10 组字面量改成解析表格 |
| 21 | §7 C12 扫描器 `doc refs` / `section refs` | 见 C12 | 一致 | PASS | — |
| 22 | §7 C12 扫描器 `path refs` / `gating` | 见 C12 | 一致 | PASS | — |
| 23 | §7 C14 `SECREF` token 数（`28f3313` → tip） | 29→33 | 29→33 | PASS | — |
| 24 | §7 C14 `DOCREF`（非 `§`）token 数 | 35→51 | 35→51 | PASS | — |
| 25 | §7 C14 `PATHREF` token 数 | 120→136 | 120→136 | PASS | — |
| 26 | §8.3 全部命中行数 | 见 §8.3 | 一致 | PASS | — |
| 27 | §8.3 自身命中行数 | 见 §8.3 | 一致 | PASS | — |
| 28 | §8.3 之外的命中行数 | 见 §8.3 | 一致 | PASS | — |
| 29 | §8.3 `tot = before + after`（恒等式） | 恒等 | 恒等 | PASS | — |

**r4 实际改掉的两处**（p2-1 / p2-2；**两者是同一件事的两面**：散文的加总不自洽，而自检**抓不到它**）：

| 位置 | 改前 | 改后 | 依据 |
| --- | --- | --- | --- |
| **§3.0 的归属拆解**（p2-1） | 「S3 索引生成 **10** 行 + `wtk` 1 + 诊断槽 1 + `VecQuantStage` 2」= **14**，而同句小计写 **13**（自相矛盾） | 「**S3 索引生成本体 9 行** + `wtk` 1 + 诊断槽 1 + `VecQuantStage` 2」= **13** ✓ | 按**内容**把 moe 的 13 行分类：`wtkUb[` 1 行、`bUb[` 1 行、`startOff[` 2 行、其余 **9** 行（`9+1+1+2 = 13`，与 `13+5 = 18` 一同自洽） |
| **§8.4 自检的 check #12**（p2-2） | 「**实跑**」栏是**写死的字面量** `'10/1/1/2'` ⇒ 它**永远 PASS**，抓不到任何 1-off | 改成**按内容真算**，并**新增**两条：拆解四项之和 == 文档小计、且 == 实测行数 | 见输出 B（`--rev 6d4a86e`）：改后的 check 在**含 p2-1 的那一版**上判 **2 处 FAIL**（`10 vs 9`、`13==14 vs 13==13`） |

**本轮（r3）实际改掉的两处**（它们是**这一轮之前**就存在的，被本节的检查判据抓出/由复审抓出）：

| 位置 | 改前（r2 tip 上的散文） | 改后（当前文件） | 依据 |
| --- | --- | --- | --- |
| §8.3 **结论段** | 「全部命中 **16 行**；§8.3 自身占 **9 行**」 | 「全部命中 **22 行**；§8.3 自身占 **15 行**」（`7 + 15 = 22`） | 当前文件的 `grep -c` 实跑：**22 / 7 / 15**（见 §8.3 的 console 块与输出 A 的 4 行 §8.3 检查） |
| 同上 —— **改前那一版自己的**实跑值 | 同一命令在 `fd02939` 上量 = **19 / 7 / 12** ⇒ 散文的 `16/9` 与**它自己**的输出矛盾 | —— | §8.4.3 的命令（`git show fd02939:` 后同法量） |
| 读数为何又从 `19/12` 变成 `22/15` | —— | —— | **§8.4 自身有 3 行**会被 `P` 匹配到：自检脚本里的 `P = …`（1 行）、§8.4.2 限度 4 的引述（1 行）、§8.4.3 的 `P='…'`（1 行）⇒ `tot`/`after` 各 +3。**已披露**（见 §8.4.2 限度 4） |

### §8.4.3 三段"改前"取证（可复跑）

```console
$ git show fd02939:docs/21-scalar-computation-sweep.md > /tmp/prev.md
$ P='所致\|是因为\|导致\|随之增加\|增量来自\|的来源\|新增了一处\|贡献是\|零删除'
$ grep -c "$P" /tmp/prev.md
19
$ H=$(grep -n "^### §8.3" /tmp/prev.md | cut -d: -f1); echo $H
941
$ sed -n "${H},\$p" /tmp/prev.md | grep -c "$P"
12
$ grep -o "全部命中 [0-9]* 行；其中 §8.3 自身占 [0-9]* 行" /tmp/prev.md
全部命中 16 行；其中 §8.3 自身占 9 行
```

（⇒ r2 tip **自己的**三项读数 = `19 / 7 / 12`，而它**自己的散文**写 `16 / 9` —— 这是 r3 打回的
**直接证据**，与复审的读数一致。）

#### §8.4.2 这一节自身的限度（**四条**）

1. **覆盖面 = 脚本里列出的那 29 项**。散文里**还有**不受脚本检查的数字 —— 例如 §0 在飞表的
   `+1140` / `+22/-6` 这类 `git diff --stat` 读数（它们是**分支的时效事实**，脚本不去重跑别人的分支）、
   §3.2 的「16 个 head id」「64 次迭代/元素」（**源码事实**，不是本文件自产的命令输出）、
   §3.1 的「`:1353`–`:1357` 那 5 行」（**行号区间**，不是计数）。
   ⇒ **不得**把「29 项 PASS」读成「全篇每一个数字都核过」。
2. **行号类断言不在脚本的可判范围内**（脚本核的是**计数**）。行号会随每次订正漂 ⇒ 全篇的行号一律
   只作**查阅提示**，判据请用**内容锚点/原句**。
3. **`--rev` 模式的语义**：它只把「散文」换成旧版，**源文件与扫描器仍按工作区跑** ⇒
   该模式**只对「与文档内部一致的检查项」有判别力**（本轮的两组负向对照命中的正是 §3.0 与 §8.3 那几行）；
   对源文件类检查项它**不是**负向对照。
4. **`§8.3` 那一条是自指的**：`P` 会匹配到**本节自己**的 **3 行** —— 自检脚本里的 `P = '所致|是因为|…'`、
   本限度条的这句引述、以及 §8.4.3 的 `P='…'` 命令 ⇒ `tot`/`after` 里各含 **3 行**来自本节。
   这一点已写进 §8.3 与 §8.4.1 的对照表，**不是**未披露的偏差。

#### §8.4.4 「硬编码期望值」自查（r4 p2-2 要求：同一类的都要改）

**为什么单列一条**：「期望值写死在脚本里」= 判据**永远 PASS**，与 §8.2 已经栽过的那类（空洞判据）
同族。第 12 项就是这么栽的：它的「实跑」栏是字面量 `'10/1/1/2'`，所以 §3.0 的 1-off 从未被它发现。

**就本节这 29 项逐条过了一遍**，按「期望值从哪来」分三类：

| 类 | 含义 | 本轮处理 | 涉及项（**共 29 项，三类相加 = 29**） |
| --- | --- | --- | --- |
| **真算** | 期望值**从数据/命令实跑推导**（或从**文档自己的输出块/表格**里解析），两边独立 | —— | **22 项**：#1–#11、#14、#16、#19、#21–#28 |
| **改掉** | 期望值**写死字面量** ⇒ 本轮改成真算 | **已改** | **4 项**：**#12**（`'10/1/1/2'` → 按内容分类真算，并顺带新增 #13/#14/#15 三条加总判据）、**#20**（写死 10 组四元组 → **从文档 §7 C10 的表格里解析**）、**#17**（写死 `'0'` → 改为**取散文措辞「空输出」**）、**#18**（写死 `'True'` → 改为**取散文措辞 `IDENTICAL`**） |
| **恒等式 / 文档内部自洽** | 期望值**不是一个源事实**，而是「文档自己的数应满足的关系」 | 保留（这类**本就该**写死成关系式），并在 §8.4.1 表里如实标出 | **3 项**：#13（文档小计 == 四项之和）、#15（总 == 两部分之和）、#29（`tot == before + after`） |

（**计数自检**：`22 + 4 + 3 = 29` ✓，与 §8.4.1 表的 29 行、输出 A 的 29 行一致。）

**改后的牙口证明（负向对照，见输出 B / C）**：
- `--rev 6d4a86e`（r3 tip，含 p2-1）⇒ **#12 与 #13 判 FAIL**（`10 vs 9`、`13==14 vs 13==13`），
  **这两条在改前是抓不到任何东西的**；
- `--rev fd02939`（r2 tip）⇒ 另加 **#26/#27 判 FAIL**（§8.3 的 `16/9`）。
⇒ **这 4 项都已改成真算**（#12 / #17 / #18 / #20），并**逐条列在 §8.4.1 表里**
（**不以笼统概述代替逐条清单**）。

