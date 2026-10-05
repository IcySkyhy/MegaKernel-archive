# M22：真实规模 MoE router（512 专家 / top-10）

M42 交付：把 `m7_router_topk` 的**真实规模**路由器（512 专家、top-10、真实 checkpoint
`mlp.gate.weight` 逐字节切片）移植成独立工程，补上**多级归并树的选择与每级代价**、
**权重按行 staging 的布局与预算**，并按 M40 定的 MoE 段输入契约补出**路由计数输出段**
（每专家 token 计数 / 紧凑槽起点 / count 模式 group_list / active_num / perm 映射）。

一句话结论：**512/10 在真实规模下可工作，且是布局问题不是算力问题**——把 m13 的
「一次性预转全部专家权重进 UB」换成「按行 streaming + 固定 8 行 fp32 解包窗」后，
UB 占用 178.6KB / 可用 248KB，m=4097（prefill）在真机上与 golden **逐元素一致**。

> ⚠️ **本交付是单核（1 个 AIV block），不是 `mix(1,2)` 全核启动**——不满足本项目
> 「mix(1,2) 全核」的约束。本 kernel 内部**未使用 CrossCore 同步**，以
> `<<<1, 0, stream>>>` 单 block 启动；m=4097（prefill）单核实测 **106.3~161.0ms**
> （多次重跑，本机 10+ worker 共用）。多核设计（按 token 行切核 + 1~2 次 CrossCore
> mode0 AIV barrier 归约 counts + `perm_*` 跨核游标）与代价模型见 **§9.1**，**未实现**。
> 本交付满足的是**功能与数值**验收，不是全核性能验收。

真机结果（Ascend950PR，`m22_router512` 与 `m22_router512_m4` 两个目标，**单 AIV block**，
均 ALL PASS）：

| 档 | 权重 | m | logits max abs | ids 差异 | weights max ulp | 活跃专家 | 空专家 | 单槽专家 | 最大槽 | kernel(ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| real_m1 | 真实 checkpoint | 1 | 6.87e-05 | 0 | 0 | 10 | 502 | 10 | 1 | 52.4 |
| real_m2 | 真实 | 2 | 6.87e-05 | 0 | 0 | 19 | 493 | 18 | 2 | 0.2 |
| real_m16 | 真实 | 16 | 1.22e-04 | 0 | 1 | 117 | 395 | 84 | 4 | 0.5 |
| real_m33 | 真实 | 33 | 2.06e-04 | 0 | 1 | 179 | 333 | 100 | 7 | 1.0 |
| real_m64 | 真实 | 64 | 2.06e-04 | 0 | 1 | 229 | 283 | 89 | 14 | 1.8 |
| **real_m4097** | 真实（prefill） | 4097 | 3.13e-04 | **0** | 1 | 479 | 33 | 5 | 1003 | 161.0 |
| nu_m1 | 合成 one-hot | 1 | 0（逐位） | 0 | 0 | 10 | 502 | 10 | 1 | 0.2 |
| nu_m7 | 合成 one-hot | 7 | 0（逐位） | 0 | 0 | 67 | 445 | 65 | 3 | 0.3 |
| nu_m33 | 合成 one-hot | 33 | 0（逐位） | 0 | 0 | 301 | 211 | 298 | 20 | 1.0 |
| nu_m64 | 合成 one-hot | 64 | 0（逐位） | 0 | 0 | 404 | 108 | 262 | 60 | 1.8 |

（`real_*` = m7 同款输入生成器 + 真实 checkpoint 权重；`nu_*` = 合成 one-hot gate +
**按 profile 指定每专家槽数**的受控输入，用于「非均匀分布」契约验证。

**两条列的来源必须说清**（`docs/17` §2.3）：`logits max abs` 是 **T3 判定之外**的报告项，
量的是**设备 vs 文件内 C++ 顺序 fp32 host 参考**（判定形式是 §6.1 的 T3 元素界；对 numpy
golden 的同一量在 `evidence/check_ref_*.log` 的 R1 行，m=4097 档为 1.373e-04）。
`kernel(ms)` 列取自归档的 **`evidence/run_mode2.log`（本轮 clean rebuild 的那次运行）**——
本机与 10+ 个 worker 共用，同一档跨次运行差异极大（两棵树的归档日志里
real_m1 = 52.4ms vs 0.8ms、real_m4097 = 161.0ms vs 106.3ms，见 `evidence/run_mode{2,4}.log`；
首档 `real_m1` 还含设备初始化开销），**故该列只作报告项，不作为设计输入被引用**；
§4.3/§9 引用的是 106.3~161.0ms 这个区间。）

---

## 1. 接口契约（M40 MoE 段输入契约）

本 kernel 的输出即 M40（`agent-moefuse`，MoE FFN per-layer 融合）段的输入。契约取自
tower 给 M40 的裁决 + `docs/15`（M33 对官方 vllm-ascend 路径的逐句转写）：

```
router: moe_gating_top_k(x[m,2560], k=10, renorm=0, norm_type=0)
init_routing_v2(x, topk_ids[m,10], expert_num=512, active_num=m*10,
                expert_tokens_num_type=1 /*count*/)
   → sorted_hidden[Σt_e,2560] + expanded_row_idx[Σt_e] + expert_tokens[512]
GMM#1: group_list = counts（**count 模式，不是 offsets**）
```

| 输出 | 形状 | dtype | 说明 | 依据 |
|---|---|---|---|---|
| `router_logits` | [m,512] | fp32 | **max-shifted**（`l - rowmax`），与 `moe_block_ref` 的返回一致 | M4 golden |
| `topk_ids` | [m,10] | int32 | 降序；并列取小 id | `moe_block_ref.router_topk` |
| `topk_weights` | [m,10] | bf16 | `norm_topk_prob=True` 归一后（RNE） | 同上 |
| `expert_counts` | [512] | int32 | `t_e` = 每专家 token 计数（**count 模式 group_list 载荷**） | `docs/15` §3.3 ★3 |
| `expert_slot_base` | [512] | int32 | `t_e` 的**独占前缀和** = **紧凑专家槽**起点（Σt_e 布局，非"每专家固定槽"） | `docs/15` §3.3 ★3 |
| `group_list_i64` | [512] | int64 | `expert_counts` 的官方 count 模式编码投影（zero-extend，低位=t_e、高位=0） | `docs/15` §2.6（`expert_tokens_num_type=1`） |
| `route_scalars[0]` | [1] | int32 | `active_num` = **m·topk**（官方语义：扩展槽总数，调用方已知） | `docs/15` §2.6 |
| `route_scalars[1]` | [1] | int32 | **活跃专家数** `|{e : t_e>0}|`（count 模式循环上界） | 本 mission 补充，见下 |
| `route_scalars[2]` | [1] | int32 | 脏 id 诊断计数（**恒 0**，guard） | 本 mission |
| `route_scalars[3]` | [1] | int32 | Σt_e（**恒 == m·topk**，guard） | 本 mission |
| `perm_src_token` | [Σt_e] | int32 | 槽位 → 源 token；(expert, token) 稳定序 | `moe_block_ref.moe_permute` |
| `perm_expert` | [Σt_e] | int32 | 槽位 → 专家 id（同序） | 同上 |

### 契约闭环：M40 已确认的实际消费面（**下游 M40 已确认**）

**M40（`agent-moefuse`）于 2026-09-26T12:09:30Z 书面确认本表为其段内实际消费面**
（其 README §5.7 于 commit `80769d0`@12:08:58Z 落定）；tower 亦于 12:09:50Z 转达并存档。
⇒ **契约闭环，M42 不需要改任何输出。**

| M40 消费 | 形状/dtype | 用途（M40 侧） |
|---|---|---|
| `topk_ids` | [m,10] int32 | S3 计数排序的输入 |
| `topk_weights` | [m,10] **bf16 可用**（M40 打包进 `w_tk_packed` 低 16 位） | unpermute 加权折叠 |
| `expert_counts` | [512] **int32（要求 int32）** | 空专家跳过（count 驱动）+ 行数上界 |
| `expert_slot_base` | [512] int32 = 前缀和 | **紧凑槽起点**（M40 已把 A/scale/GU/H/Y 全改成用它寻址） |
| `perm_src_token` / `perm_expert` | [Σt_e] int32，(expert,token) 稳定序 | permute 的 gather |

**M40 不消费**（段内自行推导）：`route_scalars[0]`（`active_num`）、`route_scalars[1]`（活跃专家数）、
`group_list_i64` —— 行数由 `off[NUM_EXPERTS]`（= Σt_e）推出，活跃专家由 `counts[e] != 0` 判定。
**故本 mission 把两种 `active_num` 语义、两种 dtype 都显式输出的防御性做法保留**（不冲突、
下游各取所需），无需改契约。

**接线注意（1 条，非阻塞）**：M40 侧用的是 **`expert_offsets[E+1]`（尾部哨兵 = Σt_e）**，
本 mission 导出的是 **`expert_slot_base[512]`**（`E` 项）——直接对接需补尾部哨兵
（或按 counts 求和），这是一个 1 行的适配，不是接口错误。

**已知缺口（M40 自报）**：tower 要求 M40 做的「接口契约测试」（活跃专家数可变 + 每专家
token 数不均 + 空专家/单 token 专家）**M40 未交付**（预算耗尽）。因此本 mission 的
**4 个受控 `nu_*` 档**（`x.bin` + `expert_counts.bin` + `expert_slot_base.bin` + `perm_*`，
已归档 `evidence/mode2/nu_*/`）就是补这个测试最合适的入口 —— 见 §6.4。

### 一处曾需 M40 拍板的歧义（已确认，不影响对接）

tower 裁决里的 `active_num` 与官方 `moe_init_routing_v2` 的 `active_num` **不是同一个量**：

* 官方那个 = **m·topk**（扩展槽总数 S，调用方已知）→ 我放在 `route_scalars[0]`；
* 「count 模式 group_list 的循环上界」需要的是**活跃专家数** `|{e: t_e>0}|` → 我放在 `route_scalars[1]`。

两者我都显式输出了，避免下游各猜一个；M40 已确认它两个都不用（见上表）。同理 dtype：官方
count 模式是 **int64[512]**，kernel 原生的计数是 int32，所以我**两个都给**
（`expert_counts` int32 + `group_list_i64` int64），M40 确认按 **int32** 读。
**若将来仍需改 dtype/加字段**，改动点是：`m22_resources.h`（UB 布局 + 预算断言）、
`RouteCounts()`（kernel 侧生成）、`check_ref.py` 的 J8/J9 与机内自检的 J8/J9/J10
（判据侧要同步改，否则同一 commit 的判据与实现又不一致）——算 4 处量级，仍属低成本。

`perm_src_token`/`perm_expert` 严格说属于 MoE 段的 permute（m13 S3 / M40 的职责）——本 mission
输出它们，是为了让「非均匀分布接口契约测试」的两侧能拿**同一批输入**互相印证（见 §6）。
M40 已确认它的段内**不消费** `active_num` / 活跃专家数 / `group_list_i64`（见上表），
所以这三个是防御性输出。

---

## 2. 移植边界：与 `m7_router_topk` 的逐文件差异表

`m7_router_topk/` 的源文件**未被本 mission 修改**（上游变更不会自动流入本工程）。

| 上游（m7） | 本工程（m22） | 差异 |
|---|---|---|
| `m7_router_topk.asc`（764 行，kernel+host 同文件） | `m22_router512.asc`（kernel+host 同文件） | 移植基底；改动见下表 |
| `m22_resources.h` | **新增** | 全局静态资源表（形状常量 / BufferID / UB 偏移 / 预算表 / `static_assert`）；m7 把常量散在 .asc 里 |
| `CMakeLists.txt` | `CMakeLists.txt` | 同结构（`find_package(ASC)` + `--npu-arch=dav-3510`）；**新增第二个目标** `m22_router512_m4`（`-DM22_MERGE_MODE=4`） |
| `check_ref.py` | `check_ref.py` | 由「对 golden `router_topk`」扩为「对 golden + `moe_permute`」，10 项判定 + 报告项 + guard 分栏；**新增设备 FTZ 口径建模**（见 §5） |
| `extract_router_weight.py` | `extract_router_weight.py` | 同工具链抽同一张量；**新增与 m7 切片的 sha256 逐字节互证断言** |
| `data/router_weight.bin`（2.5MB） | `data/router_weight.bin`（2.5MB） | 同内容：sha256 `966e4d1c…6ac6`（两处完全一致，脚本会断言） |
| `README.md` | `README.md` | 本文 |

`m22_router512.asc` 相对 m7 的实质改动（其余为改名与注释）：

| # | 改动 | 为什么 |
|---|---|---|
| 1 | `RB` 16 → **8** | 为契约输出段腾 UB 余量（m7 余量只有 17.5KB，见下） |
| 2 | 归并树编译期可选 `M22_MERGE_MODE` = 2（m7 现行）/ 4（新增） | 任务 2：核验 m7 经验能否复用 + 给出可行的改进树与每级代价 |
| 3 | **新增 `RouteCounts()` 契约输出段** | 任务 4：按 M40 契约输出 counts/base/active_num/group_list/perm |
| 4 | host 参考 `RefRouterTopk` **新增设备 FTZ 口径建模** | §5 的实证结论；不建模则 m=4097 真实权重档有 20/4097 行 top-10 集合不可复现 |
| 5 | host 用例表由 5 档扩为 **10 档**（含 m=4097 与 4 个受控非均匀档） | 任务 5：prefill 规模 + 非均匀分布契约验证 |
| 6 | host 新增 dump（counts/base/perm/group64/scalars）+ 逐档 json | 任务 5：可离线复跑的证据 |
| 7 | UB 预算从注释里的一个数字改为 `m22_resources.h` 的**逐块表 + static_assert** | m7 的注释少算 80KB（见下），不想继承这个坑 |

**顺带发现（已 TowerFinding）**：`m7_router_topk.asc:67` 的注释与 README 写「总 150.5KB」，
按同文件 offset 常量逐项相加实际是 **230.5KB**（236032B）——少算了正好一个 `UB_XB`（x 块 80KB）。
不影响行为（`static_assert` 用的是同一批常量，230.5KB < 248KB 成立），但把可用余量从真实的
17.5KB 高估成 97.5KB。本工程按 `m22_resources.h` 的逐块表核对，总量 **178.6KB / 余量 69.4KB**。

---

## 3. 归并树（任务 2）

### 3.1 m7 的 Sort32 + 2 路 MrgSort 经验**可直接复用**

结论：**可以直接复用，不需要重新设计**。依据三条独立证据：

1. **机制层**：`m7_router_topk.asc` 的树是 `Sort32`（16 个 32 对块内降序）→ 4 级 2 路
   `MrgSort(validBit=0b0011, elementLengths=32)`，每级「取输入前 32 对、输出 64 对」；
   归纳不变量 `top-10(并集) ⊆ top-32(各输入) ⊆ 本级输出` 对 top-10 严格精确。
2. **指令层**：`probe_sync_quirks` 探针 A 的 A00/A01 把「2 路 `vb=0b0011`」与「2 路归并树
   （m7 现行实现）」作为基线变体实测，逐元素 dump 全序 PASS（`probe_sync_quirks/README.md` §3）。
   同一节 A02/A15/A20 还证明**4 路**（`vb=0b1111`、4×32 对）在 dav-3510 上同样正确，
   当年 #19 的「丢 src3/src4 索引」是误用 `MrgSort4` API（该 API 在 3510 上是静默 no-op）。
3. **端到端层**：本 mission 实测 m=1/2/16/33/64/4097 六档，ids 与 golden 逐元素相等（§6）。

`elementLengths` 单位是 **8B 对**、`validBit` 必须等于非空路数、`repeatTimes=0` 或
`vb` 与 `lens` 不自洽会**静默不写**——这三条是本工程仅有的三个必须遵守的约束
（全部来自探针 A07/A21/A23，代码里已在 `Merge2/Merge4` 处注明）。

### 3.2 可选改进：4 路树（`m22_router512_m4`）

因为探针 A 已证 4 路可用，本工程实现了第二棵树并做了 A/B：

```
Level1  16 块 × 32 对  --4 次 4 路 MrgSort(len=32×4)-->  4 组 × 128 对
Level2   4 组 × 前 32 对 --1 次 4 路 MrgSort(len=32×4)-->  1 组 × 128 对
       --Extract 前 64 对--> top-10
```

正确性不变量同样是 `top-10 ⊆ top-32(各输入)`（每组的 128 对是它自己 4 个块的精确全序，
global top-10 的成员必在该组的 top-10 ⊆ top-32 内）。

### 3.3 每级代价（`tools/cost_model.py` 输出，已归档 `evidence/cost_model.txt`）

| 树 | 级 | MrgSort 次数 | 每级读(对) | 每级写(对) | 累计读 | 累计写 |
|---|---|---|---|---|---|---|
| 2 路（m7） | 1 | 8 | 512 | 512 | 512 | 512 |
| | 2 | 4 | 256 | 256 | 768 | 768 |
| | 3 | 2 | 128 | 128 | 896 | 896 |
| | 4 | 1 | 64 | 64 | 960 | 960 |
| | **合计** | **15** | 960 | 960 | 960 | 960 |
| 4 路 | 1 | 4 | 512 | 512 | 512 | 512 |
| | 2 | 1 | 128 | 128 | 640 | 640 |
| | **合计** | **5** | 640 | 640 | 640 | 640 |

外加每行各 1 次 `Sort32`（16 repeat，输出 16×32 对 = 4KB）与 1 次 `Extract`（取前 64 对）。

⇒ **4 路树把 MrgSort 调用从 15 次降到 5 次（-67%）、UB 归并对流量 1920 → 1280 对（-33%）**。

**但它是零收益的**，因为归并根本不是瓶颈：实测两棵树在**全部 10 档上逐字节等价**
（`evidence/merge_mode_equiv.txt`，90 个张量对象 `cmp` 全同），端到端时间差被共用设备的
噪声淹没（m=4097 两棵树在 106.3~161.0ms 区间内互有高低，见 `evidence/run_mode{2,4}.log`；
m=64 两棵树均为 **1.8ms**——`real_m64` 在两份日志里都是 1.8ms，1.7ms 是另一档 `nu_m64`，
出处同上）。按 `tools/cost_model.py` 的
指令模型，router 是**向量指令发射受限**（m=4097 估算 1.89e8 条向量指令 ≈ 135ms @1.4GHz，
与实测 106.3~161.0ms 同量级），归并树的几千条指令可以忽略。故默认目标仍是 m7 的 2 路树
（成熟、已验），4 路树作为已验的备选保留。

---

## 4. 权重 staging 布局与预算（任务 3）

### 4.1 问题

m13 的缩形档路由器（`m13_resources.h:139`）把**全部**专家权重一次性预转进 UB：
`UB_RT_WF = (NUM_EXPERTS+1)*HIDDEN*4`。真实 512 专家下 = `513×2560×4 = 5.25MB ≫ 248KB`
——这是**布局问题**，不是调参问题。m7 已经用按行 streaming 解掉了它，本工程沿用并把它
写成显式预算表（`m22_resources.h`）。

### 4.2 数据流图（全部基础 API + 指令级 intrinsic，无 TPipe/TBuf/TQue/AllocTensor）

```
                       ┌─ 每行 5120B bf16，ping-pong 双缓冲（UB_WB，2×5120B）
GM: w[512][2560] bf16 ─┤   LoadW(e)/LoadW(e+1) 预取 → DataCopy(MTE2)
                       └→ UB_WB ──UNPACK_B16 载入 + Cast──→ fp32 权重行
                                                        │（每行一次，40 步）
                                                        ▼
                              UB_WF = 固定 8 行 fp32 解包窗（80KB，**与专家总数无关**）
                                                        │
GM: x[m][2560] bf16 ──DataCopy(MTE2)──→ UB_XB（RB=8 行 bf16，40KB）
                                             │  UNPACK_B16 + Cast（每步 64 连续元素）
                                             ▼
                                      MulAddDst ×8（8 专家并行，fp32 向量 MAC）
                                             │  8 路 Reduce + Interleave 树 → 8 连续 lane
                                             ▼
                                   UB_LOG = logits[RB][512] fp32（16KB）
                                             │  Reg Max/Sub/Exp（行内，max-shift）
                                             ▼
                              UB_VAL e_i[512] → Sort32(16) → MrgSort 树 → Extract
                                             │  Reg Div(renorm) + 整数位打包 → bf16
                                             ▼
             GM: logits / topk_ids / topk_weights ──MTE3 DataCopy/DataCopyPad──┘
                                             │
     标量段（PIPE_S 值依赖，阻塞释放 BufferID 交接，无 set/wait flag）
     ids 回读 → 直方图 counts[512] → 独占前缀和 base[512] → 游标落位 perm_src/perm_exp
                                             │  UB_CNT/UB_BASE/UB_SCL/UB_G64 → MTE3
                                             ▼
     GM: expert_counts / expert_slot_base / group_list_i64 / route_scalars / perm_*
```

### 4.3 UB 预算（RB=8，逐块；`m22_resources.h` 末尾有同表）

| 块 | 字节 | 内容 | 与专家数的关系 |
|---|---|---|---|
| `UB_XB` | 40960 | x row-block（RB=8 行 bf16） | 固定（RB×5120） |
| `UB_WB` | 10240 | w 行 ping-pong（2×5120B bf16） | **固定 2 行** |
| `UB_WF` | 81920 | 8 行 fp32 解包窗 | **固定 8 行**（m13 是 513 行 ⇒ 5.25MB） |
| `UB_LOG` | 16384 | logits[RB][512] fp32 | 固定（RB×E×4） |
| `UB_VAL` | 2048 | 行内 e_i[512] | 固定（E×4） |
| `UB_IDX` | 2048 | arange 索引模板[512] | 固定（E×4） |
| `UB_TMP` | 4096 | Sort32 输出（16×32 对） | 固定（2E×4） |
| `UB_MA` / `UB_MB` | 4096 + 4096 | 归并缓冲 A/B | 固定（2E×4） |
| `UB_OUTV` / `UB_OUTI` | 256 + 256 | Extract 值/索引（64 对） | 固定 |
| `UB_IDS` | 4096 | ids 行 staging[RB][128] | 固定（RB×512） |
| `UB_WS` | 2048 | weights 行 staging（bf16 位打包） | 固定（RB×256） |
| `UB_CNT` | 2048 | 每专家计数（count 模式载荷） | **512 项，与专家数线性** |
| `UB_BASE` / `UB_CUR` | 2048 + 2048 | 紧凑槽起点 / 落位游标 | 512 项 |
| `UB_SCL` | 64 | active_num / n_active / 诊断 / Σt_e | 固定 |
| `UB_G64` | 4096 | i64[512] 官方 count 模式编码投影 | 512 项 |
| **`UB_END`** | **182848 = 178.6KB** | **可用 248KB ⇒ 余量 69.4KB** | |

**为什么在 248KB UB 下可行**：整条数据通路里唯一**随 E×HIDDEN 增长**的 buffer 只有
`UB_WF`——而它是**固定的 8 行窗口**（80KB），权重行以 bf16 从 GM 流入、转 fp32、用完即弃，
所以 512 专家和 5120 专家占用相同 UB。这正是 m13 的 513 行预转布局做不到的原因。

**L1 占用 = 0 字节**：本核是纯 AIV 向量 GEMV（`__vector__ __global__`），既不用 cube 也不
搬 L1；m7 的选择（文档 §设计：m≤64 时 router 是带宽受限 GEMV，免 AIC 参与与跨核同步）
在真实规模下同样成立。AIC/L1（512KB）留给同融合域里的 grouped GEMM。

**权重字节流量（设计输入，非调优）**：每个 row-block 要重读全部 512 行权重（2.6MB），
m=4097 + RB=8 ⇒ 513 个 row-block × 2.5MiB = **1.3448e9 B（1.34 GB）**；实测 106.3~161.0ms
（两棵树的归档日志 `evidence/run_mode{2,4}.log`）⇒ **8.3~12.6 GB/s**（按同一批归档的两端算：
1.3448e9 B ÷ 0.1610 s = 8.35e9、÷ 0.1063 s = 12.65e9），
远低于 HBM 峰值，**说明不是带宽受限**（与 §3.3 的指令发射结论一致）。
**口径声明**：上面两端是 **host 墙钟**读数，取值来自归档那次运行；按 tower 2026-09-26 的测量规则，
**host 墙钟在共享卡上只作定性**（同一二进制连跑的抖动可达数倍），本段只用来判"是不是带宽受限"这个
**结构性结论**，不构成任何性能数字。

---

## 5. 参考口径的必要修正：设备 FP32 FTZ（本轮实证）

这一条是本 mission 最重要的**方法论**结论，已按流程报 tower（TowerFinding: bug/high，
指向 `tools/golden/moe_block_ref.py::router_topk`）并抄送 agent-moefuse。

**现象**：m=4097 + 真实 checkpoint 权重档，设备与 golden 有 **20/4097 行**的 top-10
**集合**不同（125 个槽位、weight 最大 ulp 96），但**设备与 numpy golden 的 logits 只差 1.37e-4**
（注意：本 README 里 m=4097 的 logits 误差有两个数，量的是**两条不同的参考链**——
**1.37e-4 = 设备 vs numpy golden**（本节，`evidence/ftz_modeling_scan.txt` 的复现路径）；
**3.13e-04 = 设备 vs 文件内 C++ 顺序 fp32 host 参考**（`run_mode2.log` 的 `logMaxAbs`，
§6.3）。两者都真，差异来自 host 参考的 2560 次顺序相加与 golden 的 numpy 归约次序不同）。

**机制**（逐行核查）：设备 softmax 分数落在 fp32 **次正规区**时被硬件 **FTZ flush 成 0**
（`docs/05` §6.1 明确「fp32 次正规 FTZ 是硬件模式，**不在规避范围**」），而 golden 的
`np.exp` 保留 `1e-38..1e-45` 的次正规值并据此排序 ⇒ 深尾 logits（`|l| ≳ 88`，本档实测
max-shifted logits 下探到 **-221**）时，top-10 的后几名整体落在"设备全 0、参考各不相同"的
区间里，**并列规则分歧**（设备：同值取小 id；参考：按次正规实数值排）。

**口径扫描**（同一份设备 dump，`tools/ftz_scan.py`，输出归档 `evidence/ftz_modeling_scan.txt`）：

| 参考口径 | ids 差异 |
|---|---|
| golden 原样（保留次正规） | 125 槽 / 20 行 |
| **scores < 2^-126（fp32 最小正规数）→ 0** | **0 槽 / 0 行** |
| scores < 1e-35 / 1e-30 / 1e-25 / 1e-20 / 1e-15 → 0 | 103 / 251 / 689 / 2210 / 5262 槽 |
| score=0 当 logit < -64 / -70 / -80 / -87 / -100 | 356 / 236 / 113 / 8 / 117 槽 |

⇒ 只有「次正规 flush」这一条与设备逐元素吻合。把参考按此口径修正后，**全部 10 档 0 差异**。

**本工程的处置**：kernel 内 host 参考与 `check_ref.py` 都按设备 FTZ 口径复算 top-k
（并列规则仍取小 id，与 golden 同规则），并把「未建模 FTZ 的 golden 与本参考的差异」列为
**报告项 R0**（M22 当时的读数：m=4097 档 = 125 槽 / 20 行）。kernel 侧**未做任何规避**——FTZ 是硬件模式，
按 `docs/05` §6.1 不在规避范围。

**⚠️ R0 在 M46（`e330796`）之后语义改变，M56 已重跑刷新快照（2026-09-26）**：

| 时期 | `moe_block_ref.router_topk` | R0 量的是什么 | 全档读数 |
|---|---|---|---|
| M22 归档（pre-M46） | **不建模 FTZ** | 设备 FTZ 的**真实影响**：未建模 FTZ 的 golden vs 本参考 | m=4097 = **125 槽 / 20 行**，其余 9 档 0/0 |
| **现在（post-M46，本次重跑）** | **已建模 FTZ**（`ftz_f32` + 3 个点位） | 两套 FTZ 实现的**一致性**（golden 内部 vs `check_ref.py`），**不再是** FTZ 影响量 | **10 档全部 0 槽 / 0 行**（含 m=4097） |

⇒ **R0 退化为 0/0 是预期结果，不是"信号丢失"或"档位退化"**：M46 把 golden 侧缺口补上后，
R0 从「参考两侧口径不同的差异量」变成了「同一口径的两个实现是否一致」的一致性检查；
它现在**只在两侧 FTZ 建模不一致时才会非零**（仍是一条有效的回归护栏，值 = 0 即两实现一致）。
设备 FTZ 影响的**原始证据**已冻结在 `evidence/ftz_modeling_scan.txt`（扫描表首行 125 槽 / 20 行）
与本表首行，不受本次重跑影响。**注意 R0 是报告项、不是判定项**：它不参与 PASS/FAIL，
本档的判定项仍为 J1..J10（m=4097 档 9 项，J3 因 logits 未入库转报告项）。

`evidence/check_ref_mode{2,4}.log` 已按当前代码重跑刷新（两棵树各 10 档，全部 `RESULT: OK`），
重跑脚本：`tools/rerun_check_ref_logs.sh`。刷新前后**唯一实质变化就是 R0 那一行**（125/20 → 0/0）
与新增的每档 `RESULT: OK` 计数行；命令回显里 m=4097 那档补上了此前遗漏的
`--w data/router_weight.bin`（照抄旧回显会因缺权重参数而 rc=2）。

**Ver §5 证明什么 / 不证明什么**：
- 证明：设备 e_i 在次正规区被 flush（扫描表只有 2^-126 这一条吻合）；把它建进参考后
  m=4097 真实权重档 ids 逐元素相等。
- 不证明：设备 `Exp` 在**正规区**的精度（本档设备 vs golden 的 logits 差 1.37e-4 已覆盖，但未做逐指令级
  表征）；也不证明 golden 该不该改（`tools/golden` 不在本 mission 作用域，已报 tower 裁决并已由 M46 修复）。

---

## 6. 判据与结果（`docs/17` L0 口径：判定项 / 报告项 / guard 分栏）

### 6.1 容差分档声明（`docs/17` §1.1）

先声明档位再给数字（本工程**没有**使用 T2/T3 的例外来放宽要求；T1 项一律逐位，T3 项实测
远在界内）：

| 判据 | 数据域 | 档位与理由 | 判据形式 |
|---|---|---|---|
| J1 `router_logits` | fp32，长累加链 | **T3**（2560 项含 40 深 lane 链 + 64 lane 树归约，属"长 fp32 链"） | 逐元素 `\|dev−ref\| ≤ ε·Σ\|terms\| + 0.5·ulp(out)`，ε = (40+6+2560)·2⁻²⁴ = **1.553e-04**（设备 40 深 lane 链 40·2⁻²⁴、树归约 6·2⁻²⁴、**本参考** 2560 次顺序相加 2560·2⁻²⁴——参考链更长故必须计入）；Σ\|terms\| 用上界 `max_k\|x_k\|·Σ_k\|w_ek\|` |
| J2 `topk_ids` | 整数/索引域 | **T1** | 逐位（int32 全等），无例外 |
| J3 `topk_weights` | bf16，含 VF `Exp` + fp32 除法 | **T3**（含超越函数近似） | 推导界：renorm 后各项为正且归一到 1 ⇒ `Σ\|terms\|/den = 1` ⇒ 界 = `ε + 0.5·ulp(out)`，ε = (≈2 ulp 的 `Exp` + 10·2⁻²⁴ 归约) ≈ 8.4e-7 相对 = **4.3e-4 bf16 ulp** ⇒ 判据即 **≤1 ulp**（实测 max 1 ulp，未逼近界） |
| J4..J7 counts/base/perm | 整数/索引域 | **T1** | 逐位（int32 全等） |
| J8 `group_list_i64` | 整数域 | **T1** | 逐位（int64 全等） |
| J9/J10 active_num / Σt_e | 整数域 | **T1** | 逐位（int32 全等） |
| guard g1..g4 | 结构性 | **T4** | 见下 |

报告项（不参与 PASS/FAIL）：`R0` golden 与本参考的 ids 差（**M46 后语义为「两套 FTZ 实现的一致性」，全档 0/0**；M46 前是「未建模 FTZ 的差异」，m=4097 = 125 槽/20 行 —— 见 §5 的语义表）；
`R1` logits max abs / max rel
（m7 兼容的绝对口径，**仅报告**）；`R2` 实到的非均匀分布画像；`R3` 每行 topk 权重行和；
`T3 界占用`（最大 `|dev−ref|/bound`；**对机内 C++ host 参考** m=4097 实测 **0.23%**；
**对 numpy golden** 实测 **0.11%**——两个数都是报告项，量的是两条不同的参考链）。

### 6.2 三层校验链

1. **kernel 内自检**（`m22_router512` / `m22_router512_m4` 自身，进程退出码 0）：
   逐档与**文件内 host 参考**比。**判定 100/100**（10 档 × 10 项，J1 用的是上表的 T3 界：
   越界元素 0），**档内 guard 40/40**（10 档 × 4 项），外加 3 条非空洞性 guard
   （确定性 ×2、输入敏感性 ×1）= 全 PASS。
2. **离线交叉校验**（`check_ref.py`，对 `moe_block_ref` 的 `router_topk` + `moe_permute`）：
   **判定 99/99**（5 档 ×10 + m=4097 档 9 + 4 档 ×10；m=4097 的 J3 因 8.4MB logits dump
   未入库降为报告项），**guard 29/29**（9 档 ×3 + m=4097 档 2）。两种归并树各跑一遍，均全过。
   **x 重建是硬闸门**（tower 条件 ①）：走 `--x-seed` 路径时 `--x-sha256` 缺失或不符一律
   `FAIL` + 退出码 1、不进入判定（`check_ref.py` 的 X1/X2/X3；见 `evidence/README.md` §3.1）。
3. **golden 自洽**：`check_ref` 的 J4..J7 同时校验了「紧凑槽 = 独占前缀和」与
   「perm 序 = (expert, token) 稳定序」，与 `moe_permute` 的 CSR/stable-argsort 语义逐项一致。
4. **跨编译/跨进程确定性**：clean rebuild + 新进程重跑后，10 档 × 9 张量 × 2 棵树的 dump
   与首次归档**逐字节一致**（`evidence/determinism_rerun.txt`）。

### 6.3 判定项清单（逐项可复算）

机内自检（每档 10 项，顺序 = 打印里 `J:xxxxxxxxxx`；档位见 §6.1）：

| 项 | 档 | 内容 |
|---|---|---|
| J1 | **T3** | `router_logits` 逐元素 ≤ `ε·Σ|terms| + 0.5·ulp(out)`（ε=1.553e-04，推导见 §6.1；**触发条件 ②：K=2560 被切成 40 段跨 split 累加 + 64 lane 树归约**）；越界 0 元素，最大界占用 **0.23%（对机内 C++ host 参考）/ 0.11%（对 numpy golden）**；实测 max abs 3.13e-04（对机内参考）作为报告项 |
| J2 | **T1** | `topk_ids` 与参考逐元素相等（int32） |
| J3 | **T3** | `topk_weights` 与参考 fp32→bf16(RNE) 的 \|ulp\| ≤ 1（推导界见 §6.1；实测 max 1 ulp） |
| J4 | **T1** | `expert_counts` == 参考每专家计数 |
| J5 | **T1** | `expert_slot_base` == 参考独占前缀和 |
| J6 | **T1** | `perm_src_token` == 参考（(expert,token) 稳定序） |
| J7 | **T1** | `perm_expert` == 参考 |
| J8 | **T1** | `group_list_i64[e]` == (int64)`expert_counts[e]`（官方编码投影） |
| J9 | **T1** | `route_scalars[0]` == `route_scalars[3]` == m·topk |
| J10 | **T1** | Σ`expert_counts` == m·topk |

离线 `check_ref.py` 的 J1..J10 是**上表的重排**（J1=ids、J2=weights、J3=logits、J4..J10 同名，
档位相同），两者都跑、都归档。

guard（报告项，不参与 PASS/FAIL 计数）：
`g1` 脏 id 诊断 == 0；`g2` 活跃专家数 ∈ (0,512]；`g3` perm 值域合法且与 counts 自洽；
`g4` logits 每行 max == 0（max-shift 语义）；`G4x`（离线）重生成的 x 与 dump 的 x 逐字节一致；
`确定性` 同档连跑两遍逐字节相同；`输入敏感性` seed 7 vs 8 → ids 必须变。

报告项（不参与判定）：`R0` golden 与本参考的 ids 差（M46 后 = 两套 FTZ 实现的一致性，全档 0/0，见 §5）；
`R1` logits max abs / max rel；
`R2` 实到的非均匀分布画像（活跃/空/单槽/最大槽）；`R3` 每行 topk 权重行和。

### 6.4 任务 5-③：非均匀分布 / 接口契约用例

用**合成 one-hot gate**（`W[e][k] = 1 ⟺ k == e`，e < 512 ≤ 2560）把 `logit[t][e]` 精确
控制成 `x[t][e]`，再按 profile 指定每专家槽数、以"发牌"（token t 取位置 t, t+m, …, t+9m）
把槽位分给 token —— **构造性保证**每 token 恰好 10 槽且各类均满足（`m22_resources.h` 之外，
见 `m22_router512.asc` 的 `BuildProfile`/`GenSyntheticOneHot`）。logit 取 12.0 − 0.5·rank
（bf16 精确、同一 token 内互不相同且间隔 0.5，无并列歧义）。

| 档 | profile | 活跃 | 空专家 | 单槽专家 | 最大槽 | 覆盖的边界 |
|---|---|---|---|---|---|---|
| nu_m1 | 全单槽 | 10 | 502 | 10 | 1 | 极端稀疏：每活跃专家恰好 1 槽 |
| nu_m7 | 重槽 3/2 + 单槽 | 67 | 445 | 65 | 3 | 空专家 + 单 token 专家 + 轻微不均 |
| nu_m33 | 重槽 20/10/2/1 + 单槽 | 301 | 211 | 298 | 20 | 活跃专家数可变 + 明显不均 |
| nu_m64 | 重槽 60/30/8/4 + 二槽 138 + 单槽 | 404 | 108 | 262 | 60 | 重槽 + 二槽 + 单槽 + 空专家 |

四档的 `logits` 与参考**逐位相等**（logMaxAbs = 0），ids / weights / counts / base /
perm 全部精确 ⇒ 证明输出接口在**非均匀、可变活跃数、含空与单槽专家**下正确。
`real_*` 档（真实权重）顺带给出自然分布画像：m=1 是 10 活跃 / 502 空 / 10 单槽，
m=4097 是 479 活跃 / 33 空 / 5 单槽 / 最大槽 1003（真实路由下专家负载高度不均）。

这与 M40 的接口契约测试**可以互相印证**：两侧用同一批输入（4 个 nu 档的
`x.bin` / `expert_counts.bin` / `expert_slot_base.bin` / `perm_src_token.bin` /
`perm_expert.bin` 已归档在 `evidence/mode2/nu_*/`），构造脚本也在本工程内。
**但需如实说明**：M40 自己那份「接口契约测试」**未交付**（M40 自报预算耗尽，见 §1 末尾），
所以目前**只有本 mission 这一侧**跑过这 4 个档；M40 的融合段本 mission 未做过集成验证，
本节的「互相印证」是**同一批输入已备好、可供后续接测**，不是"两侧都已跑过"。

### 6.5 非空洞性（`docs/17` §4，T4）

| 要求 | 本工程的证据 |
|---|---|
| 输入敏感性 | seed 7 vs 8（m=33 真实权重）→ ids 必须不同，PASS |
| dump 非零且位置正确 | `g3` perm 与 counts 自洽 + counts 只在活跃专家处非零（`R2` 画像逐档报出） |
| 槽位/层号可区分 | `expert_counts` 由 golden `moe_permute` 独立算出（不是同一指针算两遍）；权重 sha256 断言与 m7 切片相同 |
| 确定性前提实测 | 每档连跑两遍、位级比较全部张量（`determinism real_m33` / `nu_m33` PASS） |

---

## 7. 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m22_router512/build -S m22_router512 -DCMAKE_BUILD_TYPE=Release
cmake --build m22_router512/build -j4          # 两个目标：m22_router512 / m22_router512_m4

M22_OUT=./m22_out     ./m22_router512/build/m22_router512       # 2 路归并树（默认）
M22_OUT=./m22_out_m4  ./m22_router512/build/m22_router512_m4    # 4 路归并树（A/B）

# 离线交叉校验（对 moe_block_ref）：
/usr/local/python3.12.13/bin/python3.12 m22_router512/check_ref.py ./m22_out/real_m33 \
      --w m22_router512/data/router_weight.bin
/usr/local/python3.12.13/bin/python3.12 m22_router512/check_ref.py ./m22_out/nu_m64 --w onehot

# 报告项复现器：
/usr/local/python3.12.13/bin/python3.12 m22_router512/tools/cost_model.py
/usr/local/python3.12.13/bin/python3.12 m22_router512/tools/ftz_scan.py <dumpdir> --w <w|onehot> [--x-seed N]

# 权重切片（与 m7 切片 sha256 互证）：
/usr/local/python3.12.13/bin/python3.12 m22_router512/extract_router_weight.py
```

环境变量：`M22_ROUTER_W`（权重路径，默认 `m22_router512/data/router_weight.bin`）、
`M22_OUT`（设置则逐档落盘 dump 供 `check_ref.py`）。numpy 只在
`/usr/local/python3.12.13/bin/python3.12` 下。

## 8. 同步与 3510 约束

**同步表（只用 BufferID，无 set_flag/wait_flag）**：

| BufferID | 交接 | release 模式 | 说明 |
|---|---|---|---|
| 0 `BUF_X` | MTE2 → V | 阻塞释放 | x row-block |
| 1/2 `BUF_W0/W1` | MTE2 → V | 阻塞释放 | w 行 ping-pong |
| 3 `BUF_LOG` | V → MTE3 | 阻塞释放 | logits |
| 4 `BUF_TOP` | V → MTE3，再由 **PIPE_S** 取 | 阻塞释放（MTE3 侧）/ mode 0（PIPE_S 侧，= `false` 取得） | ids/weights 写出的**值依赖**：标量段要消费 MTE3 写出的 ids，故用阻塞释放 + PIPE_S get 表达（`docs/05` §2 允许的 PIPE_S 例外），**不引入 set/wait flag** |
| 5 `BUF_RT` | PIPE_S 标量写 UB → MTE3 读 | mode 0 / 阻塞释放 | 计数段 staging 出 GM |

**核间同步：本交付未使用 CrossCore**——单 block（`<<<1, 0, stream>>>`）内完成，路由的输出
按 token 行天然可切分（见 §9 的后续设计）。若后续按 token 维切多核，注意 `docs/05` §6.2 的
pipe 类匹配规则（AIV 侧只能 `PIPE_S/V/MTE2/MTE3`）。

**3510 quirk 规避（沿用 m7，逐条）**：GM 一律 bf16 位视图寻址；每 `GlobalTensor` 只
`SetGlobalBuffer` 一次；`LocalMemBar` 只在 `__VEC_SCOPE__` 内；非 32B 对齐 `StoreAlign` 会
507035（短向量写用 mask=8 对齐槽）；`StoreUnAlign` 是整寄存器写不能当短向量用；
**4 路 `MrgSort4` API 是静默 no-op，必须用 `MrgSort(dst, srcList, MrgSort4Info)`**；
`Reg::Arange` 只填 64 lane（索引模板按 64 分块建）；普通 128×bf16 寄存器的 `Cast<float>`
只取偶数元素（用 `DIST_UNPACK_B16` 载入）；fp32→bf16 cast 输出落 32bit lane 低半（用整型
位运算打包）；VF 循环归纳变量必须 `uint16_t`；VF 内禁止标量读 GM（标量段放在 VF 外）。

## 9. 已知限制与后续

1. **单核（未做全核并行）**。本交付是单 AIV block 串行：m=4097 时 kernel 实测 106.3~161.0ms
   （多次重跑；共用机器，见首表注），按
   `tools/cost_model.py` 是**向量指令发射受限**（1.89e8 条向量指令 ≈ 135ms @1.4GHz），
   不是带宽受限（权重流量 1.3448e9 B ≈ **8.3~12.6 GB/s**，口径与出处见 §4.3 与
   `evidence/run_mode{2,4}.log`，远低于峰值；**host 墙钟只作定性**，同 §4.3 的口径声明）。设计（未实现）：
   - 最省事的切法是**按 token 行切多核**——logits / topk_ids / topk_weights 都是逐行独立的，
     这三项**零跨核同步**；但 `expert_counts` / `expert_slot_base` / `perm_*` **都需要跨核**，
     见下面两条。
   - **`expert_counts` 的归约**：各核写自己的 partial counts 到 GM，用 1 次
     **CrossCore mode 0 AIV barrier**（AIV 侧 set 挂 `PIPE_MTE3`，wait 挂 `PIPE_S`
     或窄 pipe）后由 0 号核求和 → 写 counts/base/active_num；再 1 次 barrier 供他核消费。
     注意 `docs/05` §2：mode 0 只能同类型（全 AIV）all-to-all。
   - **`perm_*` 的跨核依赖（易漏）**：槽位序必须是全局 (expert, token) 稳定序、总长 Σt_e。
     因此 0 号核算出的 `base[e]` **还不够**——k 号核若也从 `base[e]` 起写，各核会互相覆盖。
     k 号核必须用 `base[e] + Σ_{j<k} partialCnt_j[e]` 作为自己专家的起始游标
     （partial counts 按上一条已落在 GM，所以只需在 barrier #2 之后各核自己前缀求和），
     或者由 0 号核直接发布「每核每专家的起始偏移」表。按token行切分时**每个核的 token 子集
     内部仍要保持 (expert, token) 序**，否则与 golden 的 `moe_permute` 语义不一致。
   - 另一种切法是**按专家维切**（每核吃 512/N 个专家，各自出 local top-32，再跨核归并），
     能同时减向量指令量与权重流量，但跨核归并需要 S 规模的中间缓冲（走 GM）。
   本 mission 的任务清单未含多核，故如实标为**未完成项**——**这一条已在 README 首屏
   （标题下第一段后的警示块）显著声明**：本交付是单核，不是 `mix(1,2)` 全核启动。
2. **prefill 的 token 维分块留在调用方**：本 kernel 的 `m` 是运行期参数、内部按 RB=8 的
   row-block 循环，m=4097 一次调用即可（实测通过）；若要按 tile 切分以配合融合，
   host 侧循环调用即可，无需改 kernel。
3. **并列语义**：完全相等的 softmax 分数下，硬件 Sort32/MrgSort 不做 index tie-break
   （与 CANN donor op 一致），参考不可复现——本档通过**设备 FTZ 口径建模**把这一类
   （次正规并列）消掉了（§5）；理论上仍存在「参考侧完全相等且在正规区」的并列输入，
   但用 12.0−0.5·rank 这类构造输入不会产生（见 §6.4），真实路由数据亦不会。
4. **未做性能调优**（按用户裁决 ②「现在不关心性能」）：RB/EGRP/KCHUNK 按 UB 与
   指令形态选取，未做 tile 形状搜索；§4.3 的流量数字是**设计输入**，不是调优结论。
5. **未接 vllm-ascend 槽位**：本算子的边界是"一次 router + 路由计数"，最终应由
   vllm-ascend 的 `qwen4_exp` 层模块调用（`docs/15` §8 的接入点由 M37 出方案）。

## 10. 证据索引（`evidence/`）

| 路径 | 内容 |
|---|---|
| `evidence/mode2/<case>/`、`evidence/mode4/<case>/` | 10 档 dump（`topk_ids`/`topk_weights`/`expert_counts`/`expert_slot_base`/`perm_src_token`/`perm_expert`/`group_list_i64`/`route_scalars`/`router_logits`/`x.bin`；m=4097 的 `x.bin`(21MB) 与 `router_logits`(8.4MB) 未入库，**也无对应 `.json`**，shape/dtype 见 `evidence/README.md`） |
| `evidence/dump_sha256.txt` | 全部 dump 的 sha256（含未入库的两个大张量） |
| `evidence/run_mode2.log`、`run_mode4.log` | 两棵归并树的机内自检完整输出（判定/guard 计数） |
| `evidence/check_ref_mode2.log`、`check_ref_mode4.log` | 离线 `check_ref.py` 完整输出（判定/报告/guard 分栏）。**已由 M56 按当前代码重跑刷新**（此前为 pre-M46 快照）；重跑脚本 `tools/rerun_check_ref_logs.sh` |
| `tools/rerun_check_ref_logs.sh` | 一条命令重生成上面两份日志（10 档 × 2 棵树；m=4097 自动从 `dump_sha256.txt` 取 x 的 sha256）；退出码 0 = 20 档全部 rc=0 |
| `evidence/ftz_modeling_scan.txt` | §5 的口径扫描原始输出（`tools/ftz_scan.py`）。**设备 FTZ 影响的原始证据冻结在此** |
| `evidence/r0_semantics_recheck.txt` | **R0 非空洞性对照**（`tools/r0_semantics_check.py`）：同档上同时算「当前 golden（已建模 FTZ）= 0/0」与「golden 退回不建模 FTZ = 125 槽 / 20 行」⇒ 证明 0/0 是语义变化的预期值，不是匹配器失效 |
| `tools/r0_semantics_check.py` | 生成上面这份对照读数（`--mode mode2|mode4 --m 4097`） |
| `evidence/cost_model.txt` | §3.3/§4.3 的每级代价与指令/流量模型（`tools/cost_model.py`） |
| `evidence/merge_mode_equiv.txt` | 2 路 vs 4 路归并树的逐字节等价性与两侧结论 |

判据 → 命令映射：每个判定项都能用
`python3.12 check_ref.py evidence/mode2/<case> --w <data/router_weight.bin|onehot>`
（m=4097 追加 `--x-seed 5 --x-sha256 <见 dump_sha256.txt> --no-logits`）在归档数据上重跑。

## 11. 修订记录（M51，2026-09-26）

本 mission 评审已指出、因当时 tip 冻结而留下的两处数字/口径缺口，本轮一并订正（**不改任何判定项与结论**）：

| # | 修订 | 出处 |
|---|---|---|
| 1 | §3.2 的 m=64 行：原写「两棵树 1.7~1.8ms」→ **两棵树均为 1.8ms**。`real_m64` 在两份归档日志里都是 `t=1.8ms`；1.7ms 是另一档 `nu_m64`（mode4 那次），两者不同档不可混写 | `evidence/run_mode2.log`（`real_m64 … t=1.8ms`）、`evidence/run_mode4.log`（同） |
| 2 | §4.3 与 §9.1 的带宽口径统一为 **`8.3~12.6 GB/s`**：§4.3 原写 `≈12.6~8.3 GB/s`（方向反了）、§9.1 原写 `≈8.3~12.5GB/s`（12.5 来自已删除的旧值 107.5ms）。两处现均写作 `8.3~12.6 GB/s` 并指向归档：权重流量 1.3448e9 B ÷ 两端实测时间 0.1610s / 0.1063s | `evidence/run_mode{2,4}.log`（`real_m4097 t=161.0ms` / `106.3ms`） |
| 3 | 根因一并清掉：`tools/cost_model.py` 结论段原**硬编码**「实测 107.5~161.0ms ≈ 8.3~12.5 GB/s」，日志更新后即成过期值 —— 现改为**运行时从 `evidence/run_mode{2,4}.log` 现场解析**（共用机器活值，缺失时明确报缺、不回落到记下来的数），`evidence/cost_model.txt` 已按新脚本重跑刷新 | `tools/cost_model.py`、`evidence/cost_model.txt` |
| 4 | 对拍脚本 `check_ref.py` 按 tower 新规则改**三态退出码**（`0` 通过 / `1` 有差异 / `2` 没得比-输入缺失）：dump 缺失不再抛 traceback 而是 `RESULT: SKIPPED` + rc 2；判定项为 0 时不再可能 PASS（`all([])` 恒真那条路已堵）；OK 文案补计数 | `m22_router512/check_ref.py` 头注释；自测读数归档在 `baseline_env/evidence/13-number-audit-script-audit.md` §7 |
（**注**：M51 那次**未重跑** `evidence/check_ref_mode{2,4}.log` —— 重跑会同时改变报告项 R0：M46（`e330796`）修好 golden 的 FTZ 建模后，m=4097 档的 R0 由归档里的 `125 槽 / 20 行` 变成 `0 槽 / 0 行`。该漂移不是 M51 引入的，当时单独报了 finding、归档保留为 pre-M46 快照。
**M56（2026-09-26）已用当前代码重跑刷新该快照**，并补齐 m=4097 那档命令回显里此前遗漏的 `--w data/router_weight.bin`（照抄旧回显复跑会 rc=2）；重跑脚本 `tools/rerun_check_ref_logs.sh`，语义解释见 §5 的「R0 在 M46 之后语义改变」表。）

**原则**（与 `docs/17` §8.1 同）：活值（计时、带宽、随共用机器漂移的读数）**不以字面量出现**在任何断言或打印里；
要么运行时从归档现场解析，要么只保留「活值 + 指向采集时刻」。§3.2 那张 `kernel(ms)` 表本来就是报告项、且注明了活值与出处，保持不变。

> ⚠ **本段在 `c0558b3` 那版曾被误删**（评审 `reviews/…m56-r1.md` 判 P2-2）：我加 §12 时用定点替换改动了 §11 末尾，结果把紧跟在后的
> 这一整段（含「§3.2 表为何保持原样」的免责声明）静默删掉了，只核对了**章节号与文件尾**，没发现
> 段落级缺失。现已按 base 原文放回。自查方法已随之加严：见 §13。

## 12. 修订记录（M56，2026-09-26）

| # | 修订 | 出处 |
|---|---|---|
| 1 | `evidence/check_ref_mode{2,4}.log` **重跑刷新**（此前是 pre-M46 快照）：两棵树各 10 档全部 `RESULT: OK`；**唯一实质变化是报告项 R0 一行**（m=4097 档 `125 槽 / 20 行` → `0 槽 / 0 行`），另新增每档 `RESULT: OK (...)` 计数行、m=4097 命令回显补 `--w` | `tools/rerun_check_ref_logs.sh`；§5 的语义表 |
| 2 | §5 增「**R0 在 M46 之后语义改变**」表：修前 R0 量设备 FTZ 的真实影响（125/20），修后量两套 FTZ 实现的一致性（0/0）⇒ **0/0 是预期，不是档位退化**；补齐非空洞性对照（同档上「golden 退回不建模 FTZ」仍得 125/20） | `evidence/ftz_modeling_scan.txt`、`evidence/r0_semantics_recheck.txt` |
| 3 | 过期断言清理（tower 规则：运行时打印的断言性文字也是证据）：R0 的**运行时标签**原写「未建模 FTZ 的 golden 与本参考的 ids 差」，而 M46 后 golden 已建模 FTZ ⇒ 改为「R0 golden(已建模 FTZ) 与本参考的 ids 差（0 槽/行 = 两套 FTZ 实现一致）」；`check_ref.py` 头注释与 R0 报告项说明同步改写 | `m22_router512/check_ref.py` |
| 4 | `check_ref.py` 的 golden 导入路径由**硬编码绝对路径** `/workspace/ascend_mega_kernel/tools/golden` 改为**优先本仓相对** `../tools/golden`（不存在才回落绝对路径）—— 此前在别的 checkout / worktree 上会 `ImportError` | `m22_router512/check_ref.py` |
| 5 | **`c0558b3` 那版**（评审 `reviews/…m56-r1.md` P2-2）：§11 末尾那段「原则」（含「§3.2 表为何保持原样」的免责声明）被**静默删除**，已在 `96e7e96` 按 base 原文放回（见 §11 末尾的警示块）；自查方法加严见 §13 | `git show ea4c818:m22_router512/README.md` |

## 13. 编辑纪律：改完必须做**段落级**差异核对（M56 起，首次随 `96e7e96`）

**起因**：`c0558b3` 那版用「定点替换改 §11 末尾」时静默删掉了紧随其后的「原则」整段。当时核对的是
**章节号连续 + 文件尾还在** —— 两件都通过，因为被删的是**节内的最后一段**，不是章节也不是文件尾。
（这与 13:07 tower 广播里 M36 的 `s[:index(marker)]` 事故同族：**diff 看起来像"改了一段"**。）

**可执行的核对动作**（每次改完 README/长文档后跑；先按空行切块，再比块集合）：

```bash
BASE=ea4c818          # 该文件本次改动前的 tip（或 merge-base）
F=m22_router512/README.md
diff <(git show $BASE:$F | awk 'BEGIN{RS="";ORS="\n\n"} {print}' | sort) \
     <(awk 'BEGIN{RS="";ORS="\n\n"} {print}' $F | sort) | head -40
# 期望：只出现"我这次真要改的块"（新增的 §12/§13 + 恢复的那段）；出现别的整块 ⇒ 误删/误动
```

要点：**按空行分块后比块集合**，比 grep 章节号或 `tail` 更强 —— 它能把"节内某一段被端掉"
直接显示成缺失块。`96e7e96` 已对四份 README（`m20_hyperconn`、`m18_gdn_prefill`、`m22_router512`、
`m7_router_topk`）各跑了一遍，见 review-request 的说明。

### 13.1 静态标注核对（M56 起，首次随 `c4b7b0e`；**与上一条并列的硬要求**）

上一条管「**段落还在不在**」；这一条管「**写下的那句话对不对**」。起因是 M56 里连续出现的同一形态：
**新写的静态文字没有被自身核对**。三处实例**一律引 commit**（轮次标签不可核，故不写）：
`tools/scan_stale_runtime_prints.py` 的 `[refused]` 文案（说"仍会扫描"、代码没做到；该版 `96e7e96`）、
`tools/golden/selfcheck.py` 运行时注记里的「3 处命中都在 `MxQuantComputeScale` 内」（只对 2/3 个文件成立；
该版 `96e7e96`，原始那句见 `0c163b5`）、`m20_hyperconn/README.md` 把 `BUF_AIV_GJ(8)` 标成「injW 预转」
（该版 `cecf0ae`；**它声明处的注释**写的是 `预转: hc_norm bf16 行 MTE2 -> V（整核一次）`，
`injW` 是 `BUF_AIV_IW` 的用途）。

**规则**：**凡在文档/注释里对符号（常量、BufferID、宏、函数名）的用途或归属下判断，
必须引用"声明处原文"，或附"当场计数"** —— 不许自造用途描述。三类合法写法：

1. **引述**：逐字抄声明处注释，并给出 `文件:行`（如 m20 README §3 的死声明表）；
2. **计数**：`grep -c <符号> <实现文件>` 的当场读数（如 `asc_refs`）；
3. **工具并排**：跑 `tools/check_symbol_claims.py`，把「文档该行」与「声明处注释原文」两栏贴进 review-request。

（工具只能咬「借用他符号词汇」这一类机械信号，**不是判定器**；它的定位、能自动咬什么、
假阳性实测多少，都写在 `tools/check_symbol_claims.py` 的 docstring 与
`tools/evidence/symbol_claims_check.log` 里 —— **引用它时必须照抄这个定位，不许说成"工具判过了"**。）

### 13.2 数字对账（M56 起，首次随 `6c48163`）

**规则**：**重跑任何一个读数块之后，本文件里引用同一读数的其它数字必须同步改** ——
具体动作：`grep` 本文件**全部数字**，逐个确认它引用的那份读数已经更新（尤其是同一节里的
"关键读数 / 小结表 / 关于…" 这类散文句）。**别只改代码块**：读数块与引用它的散文是一对。

可执行做法（工具化，**候选不是判定**）：
```bash
python3 tools/check_symbol_claims.py --numbers --docs <file…>
# 它只在「同一文件 + 同一节」内比较；命中即列出「同一条线出现不同数值」的两处（值 + 行号 + 行文）
```
已知盲区与两种假阳性形态（同一名词不同谓语 / 同一张表不同行）写在工具 docstring 的
`--numbers` 一节里，读候选时直接跳过。

**读数必须标注「在哪个 commit 上取」（M56 收尾轮起）**：**跨文件 / 全仓口径的读数**（例如
`check_symbol_claims.py` 默认扫全仓 `*.md` 得到的那组计数）**会随任何一次 `.md` 改动作废**，
所以引用它时必须写清**它是在哪个 commit 上取的**，否则读者拿同一命令在 tip 上跑会得到另一个数。
本条是评审建议舰队保留的那条自查。

### 13.3 历史叙述一律引 commit，不引轮次标签（M56 起，首次随 `6c48163`）

**规则**：写「当时那句是什么样」「某版曾经如何」时，**引用 commit（+ `文件`）而不是"第 N 轮"**。
理由：**轮次标签不可核**，而 commit 可核 —— 这与已有的「引用要能被独立定位」「不写裸行号」同族。
若必须指评审出处，写 `reviews/…-m56-rN.md` 这个**文件名**（它也是可核的产物），而不是"第 N 轮说…"。

**自查命令（可复算；凡引用本节的结论，请连同命令一起核）**：

```bash
cd <repo>
grep -rnE '\br[0-9]+\b|第[0-9一二三四五六七八九十]+轮|round[ _-]?[0-9]' \
  tools/ m20_hyperconn/ m18_gdn_prefill/ m22_router512/ m7_router_topk/ \
  --include='*.md' --include='*.py' --include='*.log' --include='*.txt' \
  --include='*.h' --include='*.asc' --include='*.sh'
```

**本节刻意不写聚合计数**：命中数是**活值**（任何一次 `.md`/`.log`/`.py` 改动都会让它漂移；要改的
恰恰就是这些文件，所以它随每次收尾改动变）。所以这里只给：**命令 + 逐类归因 + 复核动作**，
读者跑一遍就能自己得到当下的数。

**归类必须按「匹配片段」而不是按行** —— 一行里可能同时躺着「允许的文件名引用」与「该改的裸标签」：
`tools/evidence/stale_runtime_print_scan.log` 的 §5 表就有一行同时含
`reviews/…m56-rN.md` 与一个裸轮次标签（形如「字母 r + 数字」、前后无 mission / commit 限定），
**按行排除会把整行剔掉、连带把该改的标签一起藏起来**（M56 收尾轮就是这么漏掉那两处的）。
判类动作：对**每个匹配**，看它所在的 token 上下文（前后各 ~30 字），而不是看整行里有没有别的允许项。

按片段判类后，命中只会落在下面这几类里。**下表是「本节所在 commit」上实测的文件级清单**，
跑上面的命令可逐条对照；同一文件里逐行仍按上面的片段规则判类（第 2、3 类可落在同一个文件上）。

| 类 | 含义 / 处置 | 实测命中的文件 |
|---|---|---|
| 1 | **原样归档的读数引文块**：记录的是**当时捕获到的原文**，按归档规则**不得为了"好看"而改动** | `tools/evidence/symbol_claims_check.log`（其读数 A/E 的引文块） |
| 2 | **其它 mission 的评审轮次标签**：不属本 mission 的历史叙述，按 scope 隔离**留给各自 owner** | `m18_gdn_prefill/README.md`、`m18_gdn_prefill/audit_readme_numbers.py`、`m18_gdn_prefill/check_ref.py`、`m20_hyperconn/README.md`、`m20_hyperconn/check_ref.py`、`m20_hyperconn/evidence/check_ref_negative_control.log`、`m22_router512/evidence/README.md`、`m22_router512/evidence/determinism_rerun.txt`、`m22_router512/evidence/dump_sha256.txt`、`m22_router512/evidence/merge_mode_equiv.txt`、`tools/golden/moe_block_ref.py`、`tools/golden/selfcheck.py` |
| 3 | **规则允许的两类自命中**：(a) `reviews/…m56-rN.md` **文件名引用**（可核产物，本身不是轮次标签）；(b) 「字母 r + 数字」形式的**标识符**（局部变量 / 函数名，与轮次无关） | (a) 本文件（§13 表）、`tools/evidence/stale_runtime_print_scan.log`（§5 表）、`m18_gdn_prefill/evidence/readme_number_audit_readings.log`；(b) `tools/golden/evidence/ftz_reachability_audit.py`、`m22_router512/tools/r0_semantics_check.py` |
| 4 | **本 mission 其它文本里的裸标签** ⇒ **这一类就是漏网**，改成引 commit | 复核动作见下（不给清单：这一类的成员正是要动掉的东西） |

**复核动作**：把命令输出逐文件与上表第 1–3 类对照 —— 上表**没列到的文件**、以及列到的文件里**落在
第 1–3 类之外的匹配行**，都属第 4 类。改完再跑一遍，重复到没有第 4 类为止。

⇒ **不要写「已无残留 / 已全部改净」这类绝对断言**：它们不可核，且本节文字自身也可能漂移。
要么给命令让人自己看，要么写「在 <commit> 上跑 <命令> 的输出是 …」，并说明它**只在那个 commit 上成立**。
（M56 收尾轮就是因为写了绝对断言、而实际还剩两处裸标签被按行排除藏住，才被复审打回。）
