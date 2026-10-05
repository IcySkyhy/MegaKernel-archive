# probe_mask_lanes —— M94：`CreateMask<T, ALL>()` 的 lane 数与落盘字节数实测

> 本目录是 M94 的交付物：一个**只报事实**的设备探针 + 一个**独立**判据脚本（模型不在探针里）。
> 读数与结论的绑定见 §4–§7；未完成项见 §10。

## 0 结论（先给答案）

**悬案裁决：`40dd6de:m17_moe_real/m17_moe_layer.asc:1138` 的「int32 的 ALL mask 是 128 lane ⇒ 每行 store 512B ⇒ 跨行越界」
这一前提在本探针覆盖的 dtype × 形状下不成立。**

1. `CreateMask<int32_t, MaskPattern::ALL>()` 在设备上激活 **64 条 lane**，一次 `StoreAlign` 落盘
   **256 B = 1 VL**（§4 表 1）。「128」是 **b16 位宽模式**的元素数（`CreateMask<bfloat16_t, ALL>()`），
   人类的「VL 恒 256B、不同 dtype 元素数不同」的表述与读数一致；m17 注释把 **b16 的 128** 当成了 int32 的 lane 数。
2. `StoreAlign` 的单次落盘量由**目标寄存器的位宽**封顶（= VL = 256B），**不由 mask 声明的元素数**决定：
   即使把 b16 的 ALL mask（128 元素）、甚至 b8 的 ALL mask（256 元素）喂给 int32 store，
   落盘仍是 **256 B**（§5 交叉位宽读数；`row_m16_b16mask` / `row_m16_b8mask` 与 `row_m16_cur` 的 arena **FNV 逐位相同**）。
3. ⇒ `m15_layer_loop/m15_moe_layer.h` 的 `SoftmaxTopkRow`（`maskAllI = CreateMask<int32_t, ALL>()` + ids 行距
   256 B）在 tip 上**没有观察到跨行越界**：16 行的 ids/ws 落盘恰好覆盖 `[0,4096)` 与 `[4096,8192)` 两个区，
   C1(越行槽)/C2(越 VL 窗)/C3(写入者归属)/C4(每行写满) 四项判据的越界计数全为 0（§6）。
   判据的**判别力**由同一套检测器在负向对照上咬住越界证明（`row_nc_p192` 咬住 64 B、`row_nc_p128_b16` 咬住 128 B）。
4. **`UpdateMask<int32_t>(64)` 与 `CreateMask<int32_t, ALL>()` 在该落盘上不可区分**：m17 的改动是**语义无变化**
   的显式化（同一形状下 arena FNV 逐位相同）—— 这**不是**说 m17 的改动有害，而是它**不是**在修一个真越界。
5. 真正能一次写 2×VL = 512 B 的形态只有**双寄存器（intlv）落盘** `StoreAlign<SD, DIST_INTLV_*>(dst, r0, r1, mask)`
   （编译器对 `vsts` dual 的 4th 参数是**静态断言**只收 `DIST_INTLV_*`）—— 而 m15/m17 的 ids 落盘都是**单寄存器**形态（§5 的 `dual_*`）。

## 1 悬案与两种说法（逐字对照）

| # | 说法 | 来源 | 本探针读数 |
|---|---|---|---|
| ① | 「int32 的 ALL mask 是 **128 lane** ⇒ 每行 store **512B**、跨行越界（RT_IDS 行距只有 256B）」 | `40dd6de:m17_moe_real/m17_moe_layer.asc:1138-1140` 的注释（该引用由 `tools/row_trigger_table.py` 的 `m17_comment_128` 锚点逐 rev 复核；在 `e37909e`/`40dd6de` 上内容相同） | **不成立**：64 lane / 256 B（§4、§6） |
| ② | 「SIMD API 固定 VL 256B，不同数据类型的 VL 一致，element 的数量不同」 | 人类原则（逐字） | **成立**：9 种 dtype 的 ALL 落盘都是 256 B（§4） |

被审落点（**旧形态**，钉在不可变 rev `e37909e` = M95 并入 main 之前的最后一版）：
`git show e37909e:m15_layer_loop/m15_moe_layer.h` → `:1099` `MaskReg maskAllI = CreateMask<int32_t, MaskPattern::ALL>();`
、`:1108` `StoreAlign(idsUb, idxs, maskAllI);`、`:1047` `idsUb = … UB_RT_IDS + r * 64 * 4`（行距 256 B）。
同一形态在 `m13_moe_layer/m13_moe_layer.asc:1006/1015`（**内容锚点**，在现 main `40dd6de` 上仍成立）。

> **锚点口径（第 2 轮复审 P2 后的定版；三点都在 §8 表里可复跑）**
> 1. **一律钉不可变 rev，不钉 moving ref**：历史形态钉 `e37909e`（旧形态；`RT_RB=8` 几何）、M95 分支钉
>    `cf34829`、M95 并入 main 后的状态钉 `40dd6de`。**`main` / 特性分支 ref 会前移** —— M95 并入 main
>    （`40dd6de`）后 `main` 上旧形态已不存在（`grep -c 'CreateMask<int32_t, MaskPattern::ALL>'
>    m15_layer_loop/m15_moe_layer.h` = **0**），钉 `main` 的命令就不再回它当初写的输出。
> 2. **能用内容锚点就不用行号**：例如 `m13_moe_layer.asc:1006/1015`（内容未变，行号也稳）；
>    `tools/row_trigger_table.py` 现在**报每条锚点的命中数**，命中数为 0 的锚点会显式打印（不让命中列表悄悄变短）。
> 3. 行号一律写作「`<rev>:<path>:<line>`」，并在 §8 给出产出它的命令；下表同。
> M94 立项时（旧 base `0bd80c6`，`RT_RB=16`）同一处是 `0bd80c6:m15_layer_loop/m15_moe_layer.h:983/992`、
> `RT_RB` 在 `0bd80c6:m15_layer_loop/m15_moe_resources.h:187`（= 16）。

## 2 方法（为什么这些读数可信）

* **探针只报事实**：哨兵 `0xA5` 由 host 侧 `aclrtMemset(dArena, ARENA, 0xA5, ARENA)` 在**设备侧**灌入 GM
  arena；kernel 只把该 arena 原样 MTE2 搬进 UB（不重新填充），V 段只写它该写的窗口，再 MTE3 搬回 GM，
  host D2H 回读后逐字节扫 ⇒ **非哨兵字节必为设备侧写入**。
  `WRANGES` = 「与哨兵不同的字节区间」，逐字节、无容差（docs/17 **T1**）。
* **判据在 host 侧独立脚本里**（`tools/check_mask_lanes.py`）：**期望值全部在脚本里**，模型来源是官方文档的
  位编码表（`asc_create_mask.md` 的 b8/b16/b32 位宽模式说明 + `asc_storealign.md` 的「单次搬出量 = VL 256B」），
  **不是**探针自己的代码 ⇒ 判据与被判物不共享代码路径（docs/17 §1.2 反自指）。
* **lane 身份与时序**：`LoadAlign` 从图案 buffer 载入 `pat[i]`（i = 窗口内字节偏移）再带 mask 落盘
  ⇒ 落盘字节的值直接给出「哪条 lane 被写」。每次落盘跑两遍（图案 A @+0、图案 B @+1024），
  同 launch 内重复 + 跨进程 3 rep（row 组）⇒ 稳定性判据有两条独立来源。
* **图案盲点（如实登记）**：检测「被写」靠「字节 ≠ 哨兵」，所以图案值恰等于 `0xA5` 的字节是盲点。
  图案 A 的盲点 = 偏移 254，图案 B 的盲点 = 偏移 100，**两者互补** ⇒ 任一 lane 至少在一种图案下可见。
  判据里盲点是**模型的一部分**（不是事后解释）：观测集合 = 模型应写集合 − 盲点（§4 的 `blind_n` 列）。
* **行距组的图案按 (区,行) 取常数值**（`pat` slot idx = 区×16+行，字节值 `0x11+idx`）⇒ 被写字节的**值**能指认
  「是哪一个 (区,行) 的 store 写的」，于是「别的行写进来了」这类**自愈后仍可指认**的越界也能被抓到（C3）。

## 3 变体矩阵与判据的四条通道

`./build/probe_mask_lanes list` 打印全部变体（`evidence/logs/oplist.txt` 是归档副本）。分组：

| 组 | 数量 | 目的 |
|---|---|---|
| `lane-own` | 9 | 9 种 dtype 各自用**本位宽** ALL mask 落盘 ⇒ 任务 1 的主表 |
| `lane-cross` | 6 | 同一 ALL 图案用在**别的位宽** store 上 ⇒ 检验「落盘量由寄存器封顶」 |
| `lane-enc` | 40 | 同宽/异宽**部分掩码**（VL1..VL128 / H / Q / M3 / M4）⇒ 定 mask 的位编码 |
| `masktype` | 8 | **同宽不同 C++ 类型**的 mask（half vs bf16、int16 vs uint16、float vs int32、uint8 vs int8） |
| `dual` | 3 | 双寄存器（intlv）落盘 ⇒ 「>VL 的单次落盘只有这一形态」 |
| `row` | 10 | 复刻 `SoftmaxTopkRow` 的 ids/ws 落盘（行距 256B）+ 越界负向对照 + 稀疏正对照 |

判据（`tools/check_mask_lanes.py`）—— **门**与**报告项**分列：
* **L 组门（两条）**：**C-L1** 单次落盘的所有被写字节必须落在两个落盘窗口内（⇒ 落盘 ≤ 1 VL = 256 B）；
  **C-L2** 本位宽 ALL 的落盘必须把窗口**写满**（256 B）。
* **L 组报告项（不是门）**：观测集合与 doc 先验位扩展模型的逐字节比对（`all1`/`any1` 两种假设各算一遍）。
  交叉位宽下有 10/62 与 doc 先验不同（§5.1）—— **该分歧不会让判据变红**，它是关于文档的证伪读数。
* 行距组（row）四条门：**C1** 被写字节 ⊆ ∪行槽（宽度 = 行距）；**C2** 被写字节 ⊆ ∪[槽起点, 槽起点+256)；
  **C3** 槽内字节值 = 该 (区,行) 的常数图案值；**C4** 每行的 256B 窗口必须被写满。
* 负向对照：`row_nc_p192`（**行距** 192 B < VL）、`row_nc_p128_b16`（b16、行距 128 B；mask 不变）⇒ **C1 必须咬住越界**。
* 稀疏正对照：`row_sparse4_p512`（行距 512B > VL，窗口之间留 256B 空洞）⇒ 越界必须 = 0，且空洞必须保持哨兵。
* **rc 契约**：全部门（含常量绑定）都反映到退出码（0 = 全通过；1 = 有门不成立；2 = 没得比）。

<!-- RESULTS-START -->
## 4 任务 1 主表：dtype → ALL mask 的 lane 数 → 一次落盘字节数（**实测**）

读法：`lane-own` 组的每个变体 = 用**本位宽**的 `CreateMask<MD, ALL>()` 做一次
`LoadAlign` + `StoreAlign`；探针每次落盘跑两个窗口（图案 A @+0、图案 B @+1024），
所以「可见字节」= 两个窗口合计，「落盘字节」= 可见字节 + 图案盲点（见 §2；本表 9 行的盲点各 2 B）。

| # | dtype | `CreateMask<T, ALL>()` 的有效元素数（实测） | 单次 `StoreAlign` 落盘字节 | 变体（`evidence/logs/run_<op>_r1.log`） | 可见字节（2 窗口） |
|---|---|---|---|---|---|
| 1 | `int8_t` | **256** | **256 B = 1 VL** | `lane_s8_m8_all` | 510 |
| 2 | `uint8_t` | **256** | **256 B** | `lane_u8_m8_all` | 510 |
| 3 | `int16_t` | **128** | **256 B** | `lane_s16_m16_all` | 510 |
| 4 | `uint16_t` | **128** | **256 B** | `lane_u16_m16_all` | 510 |
| 5 | `half` | **128** | **256 B** | `lane_f16_m16_all` | 510 |
| 6 | `bfloat16_t` | **128** | **256 B** | `lane_bf16_m16_all` | 510 |
| 7 | `float` | **64** | **256 B** | `lane_f32_m32_all` | 510 |
| 8 | `int32_t` | **64** | **256 B** | `lane_s32_m32_all` | 510 |
| 9 | `uint32_t` | **64** | **256 B** | `lane_u32_m32_all` | 510 |

**逐条读数（判据 C-L2：本位宽 ALL 必须把窗口写满 256 B ⇒ 9 行全 PASS）**：
`grep "lane-own" evidence/check_mask_lanes.log` → 9 行，`exp_n=510 obs_n=510 OK y`（`510 = 2×(256−1)`）。
⇒ 人类的表述「VL 固定 256 B、不同 dtype 的 VL 一致、element 数量不同」在 9 种 dtype 上逐条成立；
**int32 的 ALL = 64 lane**（不是 m17 注释说的 128），`64 × 4 B = 256 B = 一个 VL`。

**同一「位宽」下换 C++ 类型不影响**（`masktype` 组 8 个变体全 `OK`，均 510）：`half` vs `bfloat16_t`、
`int16_t` vs `uint16_t`、`float` vs `int32_t`、`uint8_t` vs `int8_t` 两两 arena FNV 相同。
⇒ 决定语义的是**位宽**（b8/b16/b32），不是具体类型。

## 5 任务 2：把「128 lane」钉死 —— 它是 **b16 的元素数**，不是 int32 的

### 5.1 交叉位宽读数（store 位宽 ≠ mask 位宽）

`lane-cross` / `lane-enc` 组的 46 个变体给出三条结论：

1. **落盘量由 store 的寄存器封顶（恒 ≤ 1 VL = 256 B）**，与 mask 怎么造无关：
   - `lane_s32_m32_all`（int32 mask）→ 64 个 int32 元素 = 256 B；
   - `lane_s32_m16_all`（bfloat16 mask，b16 的 ALL = 128 元素）、`lane_s32_m8_all`（b8 的 ALL = 256 元素）
     → **同样 256 B**（三者 `WRANGES` 与 FNV 逐字节相同）；
   - `row_m16_b16mask` / `row_m16_b8mask`（把 b16/b8 的 ALL mask 喂给 `StoreAlign(idsUb, idxs, ...)`）
     → 与 `row_m16_cur`（int32 mask）**arena FNV 逐位相同**（`0xc04ca12f9151c383`）。
   ⇒ 即使把「128 元素」的 b16 ALL mask 用在 int32 store 上，落盘仍是 256 B、**不跨行**。
2. **mask 的位扩展方式**：`CreateMask<W, pat>` 在 256-bit 谓词寄存器里按下标标位；store 的元素
   `e'` 覆盖字节段 `[e'*sizeof(T'), (e'+1)*sizeof(T'))`，**段内任一位被标**该元素即参与。
   - 官方文档（`asc_create_mask.md` 的位宽模式表：b16 每 2 bit/元素、b32 每 4 bit/元素）的
     「整组置位」读法，在**交叉位宽**上与实测**不一致**：62 个 lane/masktype 变体里 **10 个**不命中
     （`lane_s16_m32_*`、`lane_s8_m16_*`、`lane_u8_m16_all`、`lane_s8_m32_*` 等；全在 store 位宽 ≠ mask 位宽处）。
   - 与实测**一致**的后验读法是「**一个被选元素只置它那一组的第 0 位**」（等价于**按字节的位图**：
     b8 元素 e 标 byte e、b16 标 byte 2e、b32 标 byte 4e）⇒ 该模型 62/62 命中。
   - **口径（重要）**：后验模型是对本批读数**拟合**出来的 ⇒「后验 100% 命中」**不是**独立证据；
     有信息量的是「官方文档的整组置位读法在交叉位宽上不命中」这条**证伪**读数。
     两个模型的复现率由 `tools/fitted_mask_model.py` 打在 `evidence/fitted_mask_model.log`（可重跑）。
   - 本 mission 的结论**不依赖**上一条的取舍：无论 mask 怎么扩展，**单次落盘 ≤ VL** 这一点由
     §4（9 种 dtype 恒 256 B）+ §5.1 第 1 条 + §5.2 的形态证据三方独立支撑。
3. **交叉位宽的元素数例子**（store=b32，mask 位宽不同时的有效 int32 元素数）：
   `m32` VL1/2/4/8 → 1/2/4/8；`m16` VL1/2/4/8 → 1/1/2/4；`m8` VL1/4/8 → 1/1/2。
   这些读数与「b32 mask 的 4-bit 组」的 doc 读法在 VL1/VL2 上就有差别（见 §5.1 第 2 条），
   但**都不超过 64 个 int32 元素**。

### 5.2 「一次能写 2×VL」的形态只有一种，而 m15/m17 都没用它

`dual` 组（`StoreAlign<SD, DIST_INTLV_*>(dst, r0, r1, mask)`，即 `asc_storealign_intlv` 的双寄存器形态）
落盘跨度 **512 B**（3 个 dtype 全 `OK`，可见 510 = 512 − 2 盲点）。而编译期证据更硬：
`vsts` 的 dual 形态对第 4 个（`dist`）参数有**静态断言**，只接受 `DIST_INTLV_B8/B16/B32`
（本探针最初写 `DIST_NORM` 时构建直接报 `error: static assertion failed ... The 4th argument of vst is not valid`）。
⇒ **带掩码的单寄存器落盘在 m15/m17 的写法下就是 1×VL**；512 B 需要双寄存器形态，而
`SoftmaxTopkRow` 与 m17 的 ids 落盘都是单寄存器。

## 6 任务 2/3：`SoftmaxTopkRow` 行距组的判定（含负向对照）

复刻几何：arena `[0,4096)` = i32[16][64] ids 区（行距 256 B）、`[4096,8192)` = fp32[16][64] ws 区（行距 256 B），
`[8192,12288)` 是尾哨兵。**每 (区,行) 一个常数值图案**（0x11 + 区×16 + 行）⇒ 被写字节的**值**能指认是谁写的。

四条判据：**C1** 被写字节 ⊆ ∪行槽（宽度 = 行距）；**C2** 被写字节 ⊆ ∪[槽起点, 槽起点+256)；
**C3** 槽内字节值 = 该 (区,行) 的图案值；**C4** 每行的 256 B 窗口必须被写满。

| 变体 | 形状 | rep | C1 越行槽 | C2 越 VL 窗 | C3 归属错 | C4 未写满行 | 判定 |
|---|---|---|---|---|---|---|---|
| `row_m1_cur` | m=1，ids mask = `<int32_t, ALL>`，行距 256 B | 3 | 0 | 0 | 0 | 0 | OK |
| `row_m8_cur` | m=8（M91 把 `RT_RB` 改 8 后**最小触发档**） | 3 | 0 | 0 | 0 | 0 | OK |
| `row_m15_cur` | m=15（`rows < RT_RB`，非整块） | 3 | 0 | 0 | 0 | 0 | OK |
| `row_m16_cur` | m=16 = `RT_RB`（复审说的整块触发档） | 3 | 0 | 0 | 0 | 0 | OK |
| `row_m16_upd64` | 同上，但 ids mask = `UpdateMask<int32_t>(64)`（**m17 的形态**） | 3 | 0 | 0 | 0 | 0 | OK |
| `row_m16_b16mask` | 同上，但 ids mask = `<bfloat16_t, ALL>`（128 元素假设） | 3 | 0 | 0 | 0 | 0 | OK |
| `row_m16_b8mask` | 同上，但 ids mask = `<int8_t, ALL>`（256 元素假设） | 3 | 0 | 0 | 0 | 0 | OK |
| `row_nc_p192` | **负向对照**：同 dtype 同 mask，行距缩到 192 B | 3 | **64**（首坏 3072） | — | — | — | OK（检测器咬住） |
| `row_nc_p128_b16` | **负向对照**：b16 落盘、行距 128 B | 3 | **128**（首坏 512） | — | — | — | OK（检测器咬住） |
| `row_sparse4_p512` | **稀疏正对照**：行距 512 B（窗口间 256 B 空洞） | 3 | 0 | 0 | 0 | 0 | OK |

读数要点：
* **`row_m16_cur` 的被写区间 = `[0,8192)`**：恰好是 16 行 ids + 16 行 ws 的窗口并集，一个字节都没写到窗口外，
  也没有任何窗口未被写满 ⇒ 在 tip 的写法下**没观察到跨行越界**（§0 第 3 条的读数版本）。
* **`row_m1_cur` 只写 `[0,256)` 与 `[4096,4352)`** ⇒ 链上 MoE 恒 m=1 的形状下，单行落盘恰 256 B。
* **`row_m16_upd64` / `row_m16_b16mask` / `row_m16_b8mask` 与 `row_m16_cur` 的 arena FNV 逐位相同**
  （`0xc04ca12f9151c383`，3 rep × 4 个变体共 12 次读数一致）⇒ 在这条落盘上
  「int32 ALL」「`UpdateMask<int32_t>(64)`」「b16 ALL」「b8 ALL」**四种 mask 构造产出的字节完全一致**。
* **负向对照咬住**：行距 < VL 时 C1 报出「越出本行槽」的字节数 = `VL − 行距`（192 → 64 B；
  128 → 128 B），首坏偏移 = 末行槽尾 + 1 —— 这条证明**同一套检测器**在真有越界时会变红
  （即 §6 的「0」不是检测盲区，而是真的没写出去）。
* **稀疏正对照**：行距 512 B 时窗口之间留 256 B 空洞（`slot_hole_bytes=1024`），空洞保持哨兵（C1/C2 = 0）
  ⇒ 并集判据不是「怎么跑都过」。
* **3 rep 跨进程 FNV 逐位一致**（`stable=y` 列）⇒ 稳定性（docs/17 T4）。

### 6.1 与被复刻代码的一处**如实登记**的差异
m15 的 `SoftmaxTopkRow` 里两笔落盘的顺序是 **ws 先、ids 后**（`e37909e:m15_moe_layer.h:1107-1108`），
本探针的 `row` 组是 **ids 先、ws 后**。两笔在正常几何下触及不相交的字节段；若发生「ids 落盘跨行」，
两种顺序下**最后**写进 ws 第 0 行的都是末行的 ids 落盘（顺序不同不改变结论），故本探针的 C1–C4 判定不受该差异影响。

### 6.2 新旧对照（M95 前后两版 `m15_layer_loop/m15_moe_layer.h`）——历史缺陷取证

| | 版本 | ids 落盘写法 | 行距 | 实测判定 |
|---|---|---|---|---|
| 旧 | 钉 `e37909e:m15_layer_loop/m15_moe_layer.h:1099/1108`（行距 `:1047`）；`m13_moe_layer/m13_moe_layer.asc:1006/1015` 同形（**现 main 上仍成立**）。**M95 并入 main 后 `main` 上此形态已不存在** | `MaskReg maskAllI = CreateMask<int32_t, MaskPattern::ALL>();` … `StoreAlign(idsUb, idxs, maskAllI);` | `r * 64 * 4` = 256 B | 单次落盘 **256 B**、恰一行 ⇒ **在本探针覆盖的形状下不构成跨行越界**（§6 的 m=1/8/15/16 四档全 0；§4 表 第 8 行） |
| 新 | M95 分支钉 `cf34829`（`:1145-1146/1155`；`RT_WROW = 64` 在 `:m15_moe_resources.h:221`）；**并入 main 后**状态钉 `40dd6de`（`:1145-1146/1155`、`RT_WROW` 在 `m15_moe_resources.h:223`） | `uint32_t nw = RT_WROW; MaskReg mwi = UpdateMask<int32_t>(nw);` … `StoreAlign(idsUb, idxs, mwi);` | `r * RT_WROW * 4` = 64×4 = 256 B | 同上（`row_m16_upd64` 与本行同形：C1–C4 全 0） |

* 自主核对（不依赖转述，**钉不可变 rev**）：`git show cf34829:m15_layer_loop/m15_moe_layer.h | grep -c 'CreateMask<int32_t'` → **0**；
  同式钉现 main `40dd6de` 也是 **0**（形态已在 main 上被重写，见 §1 锚点口径）
  ⇒ 旧形态确实不在 M95 的生成物里；`... :m15_moe_resources.h | grep -n 'RT_WROW ='` → `221:constexpr uint32_t RT_WROW = 64;`。
* 因此 m17 注释里那条「int32 的 ALL mask 是 128 lane ⇒ 每行 store 512 B ⇒ 跨行越界」的前提**不成立**：
  128 是 **b16 位宽模式的元素数**（§4 第 6 行 `bfloat16_t` = 128 lane），int32 的 ALL 是 **64 lane = 256 B**。
  该注释是本 mission 能确证的**文档性错误**（代码侧的 `UpdateMask<int32_t>(64)` 本身无害，见 §5.1 第 1 条）。

### 6.3 新写法（`UpdateMask<int32_t>(RT_WROW)`）为什么对 —— 实测支撑

* `UpdateMask<int32_t>(64)` 置位的是**最低 64 个 b32 元素** ⇒ `64 × 4 B = 256 B` = `RT_WROW*4` = **恰一行行距**。
* 设备证据：`row_m16_upd64`（与本行逐字同形的复刻：`uint32_t n64 = 64; mi = UpdateMask<int32_t>(n64);`
  + `StoreAlign(base + r*256, ri, mi)`）**3 个独立进程 rep**：`exp_n=8192 obs_n=8192`、C1/C2/C3/C4 全 0、
  arena FNV `0xc04ca12f9151c383` 与 `row_m16_cur`（`CreateMask<int32_t, ALL>`）**逐位相同**。
* 未覆盖的边界（如实登记）：`UpdateMask` 是 **post-update** 形态（参数是 `uint32_t&`，消费后归 0）。
  M95 的 `nw` 在每个 `SoftmaxTopkRow` 调用（每行）内重建，故不存在「同一 mask 被多次落盘复用」；
  本探针**没有**实测「同一 mask 变量被复用两次」的行为（§10 未完成项）。

## 7 任务 3：触发条件的算术 + 可达性（条件式结论）

`tools/row_trigger_table.py` **按指定 rev 读**：参数用 `argparse` 解析（`repo`（位置，可省）|
`--rev <sha>` / `--rev=<sha>` | `--tip` | `--m-max <N>`；默认钉不可变 rev `e37909e`，`--tip` 读工作树）。
**rc 契约**：脚本**开头就报锚点齐备性**（旧形态核心锚点 3/3 或新形态 3/3 且 `RT_RB` 读得到），
末尾打 `RESULT:`；**锚点不齐备 ⇒ rc=1**（不静默出降级表）。
并逐条打印引用行（每条锚点的命中数也会报，命中数为 0 的也显式打印）。**本节的常量与行号都取自 `e37909e`**
（= 旧形态所在的最后一版 main；旧形态现在 main 上已不存在，见 §1 锚点口径）：
`e37909e:m15_moe_layer.h:787` `nblk = (M + RT_RB - 1) / RT_RB`、`:790` `rows = (M - b0 < RT_RB) ? … : RT_RB`、
`:1047` `idsUb = … UB_RT_IDS + r * 64 * 4`（**行距 256 B**）、`:1099` `maskAllI = CreateMask<int32_t, ALL>()`、
`:1108` `StoreAlign(idsUb, idxs, maskAllI);`；以及 `e37909e:m15_moe_resources.h:197` `RT_RB = 8`、
`:206` `UB_RT_IDS = UB_RT_LOG + RT_RB*64*4`、`:207` `UB_RT_WS = UB_RT_IDS + RT_RB*64*4`。
归档：`evidence/row_trigger_table.log`（默认 rev `e37909e`）。

> **`e37909e` 上 `RT_RB = 8`**（M91 的 16→8 已合入该 rev）⇒ 条件式触发集合（前提是 `store_bytes = 512`）
> 就是 `m ∈ {8,16,…,64}`（最小触发 m = 8）；实测 `store_bytes = 256` ⇒ 该集合为空（§4/§6）。
>
> **M95 重写后的几何（`--tip` / `evidence/row_trigger_table_tip.log`）**：`40dd6de` 上 ids 行距 =
> `RT_WROW * 4` = **64×4 = 256 B**（`m15_moe_resources.h:223` `RT_WROW = 64`）⇒ **行距没变**，
> 「一次落盘（≤1 VL = 256 B）≤ 行距」这条关系在新形态下同样成立；变的只是存 ids 的写法（§6.2）。

**条件式推导**（设一次 ids 落盘写 `store_bytes`，行距 = 256 B）：
* `store_bytes ≤ 256` ⇒ 落盘不超出本行槽 ⇒ **任何 m 都不越界**；
* `store_bytes = 512`（m17 注释主张的前提）⇒ 末行（`r = rows−1`）落盘覆盖本行 + 下一行：
  `rows < RT_RB` 时溢进 **未被使用的 ids 槽**（输出不可见）；`rows == RT_RB` 时溢到
  `UB_RT_IDS + RT_RB*256 = UB_RT_WS` ⇒ 覆盖 **ws 第 0 行的 top-k 权重**（它在 `r=0` 时写下、之后不再写）
  ⇒ 首坏字节 = `UB_RT_WS + 0`。
* 触发集合（**条件式**，前提是 `store_bytes = 512`）：`RT_RB=16` → `m ∈ {16,32,48,64}`；
  `RT_RB=8`（M91 分支）→ `m ∈ {8,16,…,64}` ⇒ 最小触发 m 由 16 降到 8（与复审 D 段的量化一致）。

**可达性（本探针的读数）**：§6 的 4 个 `rows`/mask 档位（m=1/8/15/16，含 m17 形态）全部 **C1–C4 = 0**，
且这 4 档的 `row_*` 读数在两版代码里对应同一几何 ⇒ **在本探针覆盖的形状下，该越界不可达**；
「最小触发 m 由 16 降到 8」这条量化只在 `store_bytes = 512` 的假设成立时才有意义，而该假设被 §4/§5 推翻。
* **能在设备上真复现「越界」的档位**（把行距弄小，而不是把 mask 弄大）：`row_nc_p192`
  （int32、行距 192 B）在设备上真复现了 64 B 越界、首坏偏移 3072；`row_nc_p128_b16`（b16、行距 128 B）
  复现了 128 B 越界、首坏偏移 512。这两档说明「越界」这一类现象在设备上**可复现且可检出**，
  只是 m15 的（行距 256 B = 1 VL）几何不产生它。

## 8 「命令 → 输出/rc」表（本节每条都在本 tip 上实跑）

| # | 命令（在 `probe_mask_lanes/` 下） | 输出/rc |
|---|---|---|
| 1 | `npu-smi info` | 开工前核：NPU 0，`No running processes found in NPU 0`。**矩阵跑当时的快照（`evidence/logs/commands.txt:23/39`）里跑前与跑后都有 2 个其它 mission 的 `m15_layer_loop` 进程占核** ⇒ 本批读数是在**并发背景**下取得的（如实记录；本探针每变体 ~1–7 s、判据只比字节集合，不依赖时序，但读数口径必须带这条背景） |
| 2 | `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release` | rc=0（`evidence/logs/cmake_configure.log`） |
| 3 | `cmake --build build -j8` | rc=0（`evidence/logs/build.log`） |
| 4 | `./build/probe_mask_lanes list \| grep -c '^lane\|^mt_\|^dual_\|^row_'` | `75`（`evidence/logs/oplist.txt`） |
| 5 | `bash run_probes.sh` | rc=0；`total=95 ok=95 fault=0 no_output=0`（`evidence/logs/matrix.txt` 末段；每个变体一个独立进程） |
| 6 | `python3 tools/check_mask_lanes.py .` | rc=0；`variants=75 ok=75 wrong=0 skip=0` + `RESULT: PASS`（`evidence/check_mask_lanes.log`） |
| 7 | `python3 tools/fitted_mask_model.py .` | rc=0；`doc 先验模型不命中 = 10`、`后验拟合模型不命中 = 0`（`evidence/fitted_mask_model.log`） |
| 8 | `python3 tools/row_trigger_table.py`（默认 rev `e37909e`）／`--rev e37909e`／`--rev=e37909e`／`--rev 40dd6de`／`--tip` | 前四条 rc=**0**（`--rev e37909e` 的输出与默认**逐字节相同**；`40dd6de` 走新形态锚点 3/3）；归档 `evidence/row_trigger_table.log`（旧形态：`RT_RB=8@197`、行距 `:1047` 64×4=256B、`:1099/:1108`）与 `evidence/row_trigger_table_tip.log`（新形态：`RT_WROW=64@223`、行距 `:1082`、`:1146/:1155`）；两模式都逐条报锚点命中数，**命中数为 0 的会显式打印** |
| 9 | `bash check_locale.sh` | PASS：`LC_ALL=C / C.UTF-8 / POSIX` 三份读数行逐字节相同（`evidence/locale_check.txt`） |
| 10 | `grep -E '^[0-9a-f]{64}  (probe_mask_lanes.asc\|CMakeLists.txt\|run_probes.sh\|check_locale.sh\|tools/check_mask_lanes.py)$' evidence/logs/commands.txt \| sha256sum -c -` | **5 条全 `OK`**（含产出设备读数的 `probe_mask_lanes.asc`）⇒ 归档读数与当前树同源 |
| 11a | `git show e37909e:m15_layer_loop/m15_moe_layer.h \| grep -n 'CreateMask<int32_t, MaskPattern::ALL>\|StoreAlign(idsUb, idxs'`（**旧形态，钉不可变 rev**） | `1099: MaskReg maskAllI = CreateMask<int32_t, MaskPattern::ALL>();`、`1108: StoreAlign(idsUb, idxs, maskAllI);` |
| 11b | 同式钉现 main `40dd6de`（`git show 40dd6de:…`） | 只回 `1155: StoreAlign(idsUb, idxs, mwi);` —— **旧形态在该 rev 已不存在**（M95 重写），这正是要钉 rev 的原因 |
| 12 | `git show cf34829:m15_layer_loop/m15_moe_layer.h \| grep -c 'CreateMask<int32_t'`（钉不可变的 M95 分支 rev；钉分支 ref 会因它前移而变） | `0` |
| 13 | `git show cf34829:m15_layer_loop/m15_moe_resources.h \| grep -n 'RT_WROW ='`（钉 rev）；同式钉 `40dd6de` 得 `223:` | `cf34829` → `221:constexpr uint32_t RT_WROW = 64;`；`40dd6de` → `223:constexpr uint32_t RT_WROW = 64;`（**两者都是 64**，行号因 M95 重写而移） |
| 14 | `grep -n '期望可见字节总数\|取值集合' evidence/check_mask_lanes.log`（**输出栏为逐字引用**） | `98:  L 组模型期望可见字节总数 = 14998（期望 > 0；=0 说明在比 0 vs 0）`、`102:  可变性守卫：L 组模型期望可见字节数取值集合 = [0, 2, 4, 8, 16, 32, 64, 128, 175, 255, 510]（须 > 1 个取值，否则判据不区分形态）`；其中「**11 个取值**」是我们读出来的（不是原行文字） |
| 15 | 常量绑定对照（**/tmp 副本**注入）：把副本 `probe_mask_lanes.asc` 的 `WS_OFF` 4096→4352 后 `python3 tools/check_mask_lanes.py <副本>` | `WS_OFF probe=4352 checker=4096 **MISMATCH**` + `RESULT: FAIL` + **rc=1**（修前是 rc=0 —— 见 §9 的 rc 契约） |
| 16 | locale 断言对照（**/tmp 副本**注入）：把副本 `run_row_nc_p192_r1.log` 的 `WRANGES` 改成单段 `0-3136` 后 `bash check_locale.sh` | `ASSERT FAIL: 越界档 nrange 必须 > 1：p192=1` + `RESULT: FAIL` + **rc=1**；归档原样则 rc=0 |
| 17 | `python3 tools/row_trigger_table.py --rev deadbeef`（对照：读不到锚点的 rev） | `锚点齐备性` 段打印 `旧形态核心锚点命中 0/3；新形态核心锚点命中 0/3；… RT_RB = None` + `RESULT: FAIL（锚点不齐备 ⇒ 上表是降级表，勿采信）` + **rc=1**；（修前：`--rev <sha>` 的 sha 被当成位置参数 `repo`、锚点全未命中却 **rc=0** —— 第 3 轮复审 P2-1） |

**口径与遗留（必须连同读数一起看）**：
* 表里 #7 的「后验模型 100% 命中」是**拟合**结果，不作为独立证据（§5.1 第 2 条已声明）。
* `evidence/logs/commands.txt` 里的 sha256 是每次 `run_probes.sh` **跑之前**记的。本批归档是在**修完
  第 1 轮复审的 4 条 P2 之后重跑**的整轮，因此 **5 个文件的指纹全部 OK**（见上表 #10）。
* **读数可重复性（本轮实测，可直接复核）**：把本批归档与上一批（同一份探针源码，仅注释不同、相隔约 40 分钟、
  并发背景不同）逐文件比对 ⇒ **`evidence/logs/run_*.log` 95 个文件 0 差异、`evidence/dumps/*.bin` 95 个文件
  0 差异**（`git diff --name-only -- probe_mask_lanes/evidence/dumps | wc -l` = 0；`git diff --name-only --
  'probe_mask_lanes/evidence/logs/run_*.log' | wc -l` = 0），变的只有 `commands.txt`（日期/指纹）、
  `locale_check.txt`（新增断言文本）与构建日志 ⇒ 设备读数逐字节可重复（docs/17 T4）。
* `evidence/locale_check.txt` 的 `date:` 行是**最后一次复核重跑**的时刻（内容与 `run_probes.sh` 那次 逐字节相同：三 locale 六行读数行 + 6 条 ASSERT 行；见上表 #9 与 #16）。其余读数行不受该时刻影响。
* **本表所有引用都钉不可变 rev**（`e37909e` / `cf34829` / `40dd6de`）或改用**内容锚点**（如
  `m13_moe_layer.asc:1006/1015`）；`main` 只在「写作时的状态」这类句子里出现，且都注明 rev。
  判断依据（第 2 轮复审 P2 实测）：M95 并入 main（`40dd6de`）后，钉 `main` 的旧形态命令**不再回它当初写的输出**。
* **常量绑定守卫**：`check_mask_lanes.py` 会把探针源码里 `ARENA/IDS_OFF/WS_OFF/PAT_LANE_A/PAT_LANE_B/
  PAT_SLOT/SENTINEL` 的取值与判据侧的同名常量逐条比对（避免两个文件里的同名裸常量静默各说各话），
  不一致则 `RESULT: FAIL` **且 rc=1**（修前的 rc 通道漏了这一条，对照见上表 #15）。`VL=256` 不由常量绑定，
  而由 **C-L2 实测**兜住（若 VL 不是 256，C-L2 必然变红）。
* `tools/check_mask_lanes.py` 与 `check_locale.sh` 的**退出码就是门**：任何一条门不成立（含常量绑定不一致、
  locale 正对照断言不成立）都返回 rc=1；因此按 `rc` 接入脚本/CI 不会静默通过。**复用时按 rc 判**，不要只看
  输出文本里的 `RESULT:` 行（文本行只是给人读的）。`run_probes.sh` 会把它俩的 rc 打进输出。
<!-- RESULTS-END -->

## 9 机械清单：若要改 `SoftmaxTopkRow`，需要动什么（本 mission **不改**产品代码）

按本探针的读数，**没有**必须改的理由；下面这份清单是「若后续 mission 仍要改」的机械步骤，
以及**为什么不该在本 mission 改**：

1. **不需要改 `maskAllI`**：`CreateMask<int32_t, ALL>` 在本形状下 = `UpdateMask<int32_t>(64)`（读数逐位相同）。
   **该形态只存在于旧 rev**：`e37909e:m15_layer_loop/m15_moe_layer.h:1099/1108`（`maskAllI = CreateMask<int32_t, ALL>()`
   与它的 `StoreAlign(idsUb, idxs, maskAllI)`）—— 并入 M95 后 **`main` 上已无此形态**
   （`40dd6de` 同处是 `:1146/:1155` 的 `UpdateMask<int32_t>(nw)`，见 §6.2），所以下面这条改法**在现 main 上已经做过**，
   只在「回退或另开分支重做旧形态」时才有意义。
   若真要照旧形态改：把 `e37909e:…:1099` 的 `MaskReg maskAllI = CreateMask<int32_t, MaskPattern::ALL>();` →
   `uint32_t n64 = 64; MaskReg maskAllI = UpdateMask<int32_t>(n64);`（注意 `UpdateMask` 取 `uint32_t&` 非 const 引用，
   且**有 post-update 语义**：同一 mask 只能消费一次，重复使用需重建）。**该改动会改变 arena 的字节？
   实测不变**（FNV 相同）⇒ 但会改变 SASS/指令，位级对拍基线（M91）需要重跑。
2. **真正值得动的不是 mask，而是行距/掩码的显式契约**：把「ids 行槽 = 64 × 4B = 256B = 1 VL」写成
   `static_assert`（如 `static_assert(RT_IDS_ROW_BYTES == 256, ...)`）并加一条注释说明
   「单寄存器落盘上限 = VL」。这属于行距重排 mission（本 mission 的 #4 边界），
   **动了会撞 M91 的位级对拍基线**（`40dd6de:m15_layer_loop/evidence/m52_relift_diff.txt:79-80` 记着这一段的来源）。
3. **m17 的注释应当修正**（`40dd6de:m17_moe_real/m17_moe_layer.asc:1138-1140`）：把「128 lane ⇒ 512B」改成
   「b32 位宽模式下 ALL = 64 lane = 256B = 行距；`UpdateMask<int32_t>(64)` 与之等价（本探针读数 §5）」。
   该文件属**只读引用**（M94 不改），登记为 TowerFinding。
4. **若真要做「跨行越界」的护栏**：正确形态是把落盘宽度与行距一起断言，而不是只断言 mask 的元素数；
   可复用本目录的 `row` 组判据（C1–C4）与 `tools/row_trigger_table.py` 的算术。
   **复用口径（rc 契约）**：`tools/check_mask_lanes.py` 与 `check_locale.sh` 的**退出码就是门**（0=全通过、
   1=有门不成立、2=没得比）——接进脚本/CI 时**按 rc 判**，不要只 grep 输出里的 `RESULT:` 文本；
   `check_mask_lanes.py` 还会把「判据侧几何常量 vs 探针源码同名常量」逐条比对（不一致即 rc=1）。

<!-- DISCIPLINE-START -->
<!-- DISCIPLINE-END -->

## 10 边界与未完成项（如实登记）

* 本探针**只覆盖** `dav-3510`（Ascend950PR）+ 单 AIV + 1 launch 的形态；未覆盖多核并发、硬件 loop、
  地址寄存器偏移落盘、`StoreDist` 的其它打包形态（PACK_B16/PACK4_B32/…）下的落盘宽度。
* 「一次落盘 ≤ 1 VL」的**结构性**证据是「2×VL 需要双寄存器形态 + 编译器静态断言只收 intlv」，
  本探针**没有**穷举所有 `StoreDist`（例如 `DIST_PACK_*` 的落盘宽度未逐一实测）。
* `UpdateMask` 的 post-update 语义（mask 只能消费一次）在本探针里每次迭代都重建 mask，
  **没有**实测「同一 mask 复用两次」的行为。
* 行距组的 arena 是 12288 B、ids/ws 各 16 行的**镜像布局**，不是 m15 的真实 UB 地址（真实布局里有
  MTE2/MTE3 流水与 ping-pong）；本探针判的是**落盘窗口与行距的几何关系**，不是 m15 的端到端输出。
* 本探针的 arena 是 12288 B 单一缓冲；未实测「落盘地址跨越 UB 边界（含分配尾）」时的行为。
* **环境事件（如实登记）**：本轮出现过一次**容器根文件系统写满**（`df` 报 `Use% 100`、可用 248 MB），
  当时 `tools/check_mask_lanes.py` 的输出被中途截断（python rc=120）。清理本目录 `build/` 后重跑成功，
  归档里的 `check_mask_lanes.log` 是**完整重跑**的结果（106 行、`RESULT: PASS`）。
  该事件与判据无关，但它会**静默截断**任何写盘动作，故记在此处；同一条 Finding 已报塔。
* 未覆盖：`CreateMask<T, ALL>()` 在 **b64 / complex32**（`RegTraitNumTwo`）形态下的语义。

## 11 交付要点对照（塔给的四条 → 本目录的证据落点）

| 塔的要点 | 结论 | 证据落点 |
|---|---|---|
| ① dtype → lane → 落盘字节表；判「m17 的 128-lane 注释是否错」 | 表见 §4（9 种 dtype 全 256 B）；int32 ALL = **64 lane**；m17 注释的「128 lane ⇒ 512 B」**不成立**（128 是 b16 的元素数） | `evidence/logs/run_lane_*_r1.log`、`evidence/check_mask_lanes.log`、`evidence/dumps/ml_lane_*_r1.bin` |
| ② 旧/新对照（M95 前后两版），把跨行越界定性为历史缺陷 | 旧写法（tip `:1099/1108`）实测 256 B/行**不跨行** ⇒ 该「缺陷」在本探针覆盖的形状下不成立；M95 新写法与旧写法在这条落盘上**字节相同** | §6.2、§8 第 11–13 行命令（`git show` 定版）、`row_m16_cur` vs `row_m16_upd64`（FNV 相同） |
| ③ 核新写法 `UpdateMask<int32_t>(RT_WROW=64)` 是否恰覆盖 64 个 int32 元素 | **成立**：64 元素 × 4 B = 256 B = 一行行距；`row_m16_upd64` 3 rep C1–C4 全 0 | §6.3、`evidence/logs/run_row_m16_upd64_r{1,2,3}.log` |
| ④ 判据必须能变红（第五变体）、常量显式绑定、不得绝对断言 | 负向对照 `row_nc_p192`（64 B）/`row_nc_p128_b16`（128 B）咬住越界；**两条 rc 契约对照**（注入常量不符 ⇒ checker rc=1；注入单段 WRANGES ⇒ locale rc=1）；常量绑定 PASS；全文用「本探针覆盖的形状下」限定 | §6、§8（含 #15/#16 对照与口径段）、`evidence/check_mask_lanes.log`、`evidence/locale_check.txt` |
