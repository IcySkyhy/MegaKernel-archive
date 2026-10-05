# Layer Kernel 集成设计（MoE block 链 + GDN 层链）

> 设计：M18 survey（2026-09-26，agent-layerspec）。输入：main 上 m0-m9 全部算件 README + 资源表源码核对 + docs/05 §5 + docs/09/10/11 + tools/golden。
> **修订：M38（2026-09-26，agent-docsland）**——按 `docs/14-hyperconnection-ple-indexer-spec.md` §9.1 的逐行清单改层界语义 / 资源表 / 风险节。**本文所有层边界结论以 docs/14 为准**；修订前版本见 `git show main:docs/12-layer-integration.md`（M18 原文，仅历史参考，层界语义已被本文取代）。
> 原则（docs/05 §5.2）：单一全局编译期资源表，模块只声明 footprint，地址/BufferID/flagId 以常量注入；段间 mode-0 barrier 分隔 → 段内活跃 buffer 地址可叠放（编译期 linear allocator 按生命周期）。
> **接口约束（用户裁决 2026-09-26）见 §10，先读 §10 再读本文其余部分**：每个 kernel 只跑一层、48 层循环留在框架侧、**层内核含 hc mix**、PLE/ngram 表走 host + PCIe-through MTE2、现阶段只求正确性；验收口径见 `docs/17-verification-standard.md`（QSA 固有偏差见其 §7 附则）；**prefill m=4097 的逐段形状、decode/prefill 共用方案 B 与 14 条不返工清单见 `docs/15-prefill-design.md` §5/§6.3**。

## 0. 结论溯源（本文件 ↔ docs/14 / docs/15 / 用户裁决）

先读 `docs/14-hyperconnection-ple-indexer-spec.md`、`docs/15-prefill-design.md` 与本文 §10（用户裁决），再读本文。下表给出本文每节结论的权威出处，便于后续读者判断"该信哪一条"。

| 本文节 | 权威出处 | 关系 |
|---|---|---|
| §1 集成顺序 / 资产就绪度 | docs/14 §8.2、§8.3、§9.5 | 顺序不变，落位与降本项按 docs/14 |
| §1 item 5 共用裁决 / §5 同步图 | docs/15 §5（**方案 B**） | **新增**：单 TU、两个入口符号（decode/prefill）、一套编译期静态资源表；kernel 内无按 m 的运行时分叉 |
| §2 算件接口盘点 | docs/14 §3.4（归一化位置总表）、§4.3（每层 HC 调用次数）、§9.1 L16 | **改写**：m6 行降级，新增 `hc_combine_norm`/`hc_gate_mix` 两行 |
| §3 GM 平面图 | docs/14 §4.1（48 层骨架）、§4.2（残差流宽度与跨层张量）、§9.1 L28/L30 | **改写**：层界从 1 张量变 3 张量 |
| §4 全局资源表 | docs/14 §9.1 L34/L36/L38/L40/L42、§9.2（新增权重槽/UB/L1 预算）、§5.2/§5.3（带宽账） | **逐行修订 + 新增预算表** |
| §5 同步图 | docs/14 §9.3、docs/15 §5.3（不返工清单第 8/14 条：flagId 按 mode 分节、pipe 类） | **增补** HC 段与新 flagId 压力 |
| §6 小改清单 | docs/14 §9.1 L54、§8.2 #9；docs/15 §5.3（★1/★3/★4/★12 四条高危项） | D 取消，改 D′/D″；并入 prefill 不返工清单 |
| §7 icache / 标量发射 | docs/14 §9.2（HC 权重槽 387、形状只 4 种）、docs/15 §5.2 | 增补 |
| §8 失效假设 / 仍然有效的资产 | docs/14 §8.2、§8.3；docs/15 §2.3（HC 权威数学） | **新增节** |
| §9 风险与遗留 · HC/PLE | docs/14 §10（未取证/存疑）、§11（两处地雷）、§6.3（PLE 常驻）、§7（QSA）；PLE 落地方案改按用户裁决 ④ | **新增地雷；PLE 阻塞项已由裁决解除** |
| §9 风险与遗留 · prefill | docs/15 §0 #1/#5、§6.3、§6.4（瓶颈翻转与 AI 表） | **改写**：prefill 除 MoE 外全部翻转为算力支配 |
| §9 风险与遗留 · QSA | `docs/17` §7 附则（tower 裁决 2026-09-26）+ docs/15 §0 #4、§9 存疑 3 | **新增**：首期基准 = 自建稠密 causal 参考，必须标注与官方固有不一致 |
| §10 接口约束 / 验收口径 | 用户裁决 7 条（tower 广播 2026-09-26）+ `docs/17-verification-standard.md` + docs/15 §8（vllm-ascend 接入点） | **新增节**；不是 docs/14 的结论，来自用户裁决，优先级更高 |

## 1. 总判断与集成顺序

1. **MoE block 链优先**（资产就绪度最高）：m6#1 → [m7 router，**main 已合并**] → m8#1 permute → GMM#1(m3 AIC) → swiglu_quant(m5，替换 m3 的 scalar quantizer) → GMM#2(m3) → m8#2 unpermute → shared expert（1 槽位 m3 形态）→ m6#2。golden 数据集（m1/m33，topK=2/4 缩小形态、K/N 真实）直接做整链验收。
2. **GDN 层链次之**：m6 → in_proj（**m11 bf16 GEMM，main 已合并**）→ m9 prolog → m4 recurrence → RMSNormGated（**m12，main 已合并**；m6 变体：sigmoid(z)·o 前置 + RMSNorm + bf16 出）→ out_proj（m11）→ m6。m9+m4 契约已对齐成链。
3. **attention 层**（M17）不在本方案范围；但注意 **48/48 层都有 MoE 段**（docs/14 §8.2 #10），12 个 `full_attention` 层实为 QSA 稀疏检索（docs/14 §7、§11 #5）。
4. **现状（main 事实）**：m7/m11/m12 已合并；**m13 MoE 整层 kernel、m14 GDN 整层 kernel 均已合并**（单次 `mix(1,2)` 启动跑全链，682+118 / 114+103 条判据 PASS）。⚠ 但 m13/m14 的**层边界契约**（入/出口语义、`HIDDEN` 口径）按 docs/14 §8.2 已失效 —— 它们的**内部全链判据仍然有效**（锚在 device 字节上），需要改的是**边界行**，见 §8。
5. **decode / prefill 共用裁决 = 方案 B（docs/15 §5.2，tower 采纳）**：**单 TU、两个入口符号**（`*_decode_kernel()` / `*_prefill_kernel()`）、**一套编译期静态资源表**（BufferID/flagId/UB 窗/L1 区/GM 偏移按 mode 分节登记，两 mode 互斥执行 ⇒ id 可复用）；**host 按 m 选符号，kernel 内没有任何按 m 的运行时分叉**。附加硬约束：每节自带 `static_assert(footprint ≤ 248KB)`（UB）与 L1 断言。方案 A（段内按 m 分支）在 UB 叠放/同步表拓扑/icache 三个维度都不可行（docs/15 §5.2 五维对比）。

## 2. 算件接口盘点（要点）

| 算件 | 启动 | 输入 → 输出 |
|---|---|---|
| **hc_combine_norm**（新，docs/14 §9.1 L16） | `__vector__` | `[m,10240]` state + `[m,2560]` block + `[m,4]` inj → `[m,10240]` 物化态 + `[m,10240]` 归一化态；bf16，`(1+w)` 分组 4×2560 Gemma-RMSNorm |
| **hc_gate_mix**（新，同上） | `__vector__` | `[m,10240]` xn × `[m,10240]` gate → `[m,2560]`（4 流门控加权均值 + `/4`） |
| mixer 的两条 GEMM（新，占用第二组 GEMM ping/pong） | `__mix__(1,2)` | down+inject 合并 `[m,10240]×[10240,324]`（→320+4，vLLM 补 pad 到 336）→ silu(`/4`) → up `[m,320]×[320,10240]` |
| m6 rmsnorm | `__vector__` blk=1 | x/res/gamma bf16 → y bf16 + resOut **fp32**。**降级**：不再是层边界 op，只是"分组 RMSNorm 的归约数学（二分折叠求和 + NR rsqrt）与 `(x·rstd)·gamma` 可复用为 mixer 内的子步骤"（docs/14 §8.2 #2、§8.3） |
| m8#1 permute | `__vector__` blk≤28 | x + perm_src_token + counts → x_sorted |
| m3 grouped GEMM | `__mix__(1,2)` blk=28 | 槽位 A(qx+scale) + MXFP4 权重 + counts → GU bf16 →（AIV 量化）H qx+scale → Y bf16 |
| m5 swiglu_quant | `__vector__` blk=1 | gate\|up bf16 → qx+scale（全 VEC，无 S pipe） |
| m8#2 unpermute | `__vector__` blk≤28 | y_sorted + inv + w_tk_packed → out |
| m9 gdn prolog | `__vector__` blk=56 | qkvzba bf16 + conv_state → q/k/v/g/β（m4 契约逐字段对齐） |
| m4 recurrence | `__vector__` blk=56 | q/k/v/g/β + state[48,128,128]f32 → o |

pipeline 依赖（生产→消费）：GEMM 链 MTE2→MTE1(0,1 A/2,3 B)→M(4,5 L0)→FIXP(6 L0C)；AIV 通用 MTE2→V(0)、V→MTE3(1 drain)；m4 MTE2→V(0/1 slab,2/3 IV,6 GB)、V→MTE3(4/5 OV)；m3-AIV 特例 V→S(scalar quantizer)→MTE3（PIPE_S 有值依赖锁，集成时消除）。**HC 段的 AIV 侧 set 挂 `PIPE_MTE3`、AIC 侧只能 `PIPE_S/MTE1/MTE2/FIX/M`**（docs/14 §9.3 #4），写同步表时逐点核对。

## 3. GM tensor 平面图

### 3.1 层界（hyper-connection，10240 宽，docs/14 §4.2 / §9.1 L28）

`state[10240]` + `pending block[2560]` + `inj[4]` → `hc_combine_norm` → `state'[10240]` + `xn[10240]` → merged down+inject GEMM(10240→324) → `silu(/4)` → up GEMM(320→10240)（即 4 流 `gate`）→ `hc_gate_mix(xn, gate)` → `block_input[2560]` → 子层 → 新的 `pending block`（= 本子层输出）+ `inj`。

层界携带 3 个张量，字节/token：state 20,480 B（双缓冲 40,960 B）+ pending block 5,120 B + inj 8 B ≈ 25.6 KB/token（docs/14 §4.2）。**L 层的 mlp 输出不立即加回，而是作为 L+1 mixer 的 `prev_block_output`，与 L+1 的输入 RMSNorm 融合（延迟 combine）**。

### 3.2 MoE 子层内部（2560 宽，链不变）

`x_norm → router(logits/topk_ids/topk_weights/perm_src_token/counts) → m8#1 → x_sorted → A_qx+A_scale（重量化）→ GMM#1 → GU → m5 → H_qx+H_scale → GMM#2 → Y → m8#2 → routed_out`；shared expert（1 槽位，sigmoid 门）并行 → `moe_out=routed+shared`。权重：512 专家 MXFP4 原始 layout 常驻 GM，层内活跃 top-10。**expert GEMM 的 K=2560 不变**（`m13_resources.h:237,241,249` 的 `SZ_AQ/SZ_GU/SZ_Y`），只有层界行改口径（docs/14 §8.3）。

### 3.3 GDN 层链（docs/14 §9.1 L30）

`(state[10240], pending[2560], inj[4]) → attn_hc.combine_and_mix → block_input[2560] → in_proj → m9 → m4 → m12 → out_proj → mlp_hc.combine_and_mix → block_input2[2560] → MoE(§3.2) → (state[10240], mlp_out[2560], inj2[4])`。

原"M6 实例间矛盾（resOut fp32 vs bf16）/ 小改 D"**删除**：HC 路径上残差是 bf16 `[10240]`，新的精度契约是"**combine 结果先舍回 bf16 再 RMSNorm**"（docs/14 §8.2 #9、§10 #3）。m9/m4 契约（g/β stride-8、{H,1,0,0}）不变。

### 3.4 QSA 段 / PLE 段（本方案范围外，仅记接入点）

- QSA：12 层（+MTP），每层额外 `index_qk_proj [640,2560]` + 2×`[128]` = 3.125 MiB，选择缓冲 `[T,2052]` int32，双侧 paged cache（docs/14 §7.3）。**判定层类型必须用"indexer 字段存在"**（docs/14 §11 #5，详见本文 §9 地雷 1）。**首期验收基准 = 自建稠密 causal 参考，必须处处标注它不是官方行为**（官方只 attend 约一半历史，稠密与官方输出固有不等价 —— `docs/17` §7 附则）；**cache 填充（raw key ring + compressed key cache，含 off-by-one 契约）现在就要做**（docs/15 §0 #4、§8.2）。
- PLE：仅 0-based layer 1，插在"层 0 出口 combine"与"层 1 mixer"之间，多一次独立 `combine` 全程 pass + 1 个同步点（docs/14 §9.3 #3）。

## 4. 全局资源表草案

**AIC BufferID**（docs/14 §9.1 L34）：0-1=A(+scale) L1 ping/pong；2-3=B ping/pong；4-5=L0A/L0B ping/pong；6=L0C；7-10=第二组 GEMM ping/pong（**跨段权重预取**，用途明确包含 **2 个 mixer 的 4 条 GEMM**：down+inject-merged 与 up，各 ×2 mixer）；11-13=FIXP→UB 直写交接预留；14-27=扩展。**L1 预取滚动窗的流量预算每层 +26.9 MB**（GDN 层 115.9 → 142.8 MB，+23%）。

**AIV BufferID**（docs/14 §9.1 L36）：0-7=通用行 ping-pong 窗 4 对（m2/m5/m6/m8#2/m9/m3-AIV 的局部 0-2 全叠放，段间 barrier 分隔）——**该窗的单行宽度从 2560 变 10240（4×）**；8-14=m4 递推专用（slab×2/IV/OV/GB 平移）；15-18=m8#1 四级流水；19-23=预取 staging + gamma fp32 预转 + rstd scratch —— **gamma 预转槽要覆盖 3 类新 gamma**（`attn_hc_norm[10240]`、`mlp_hc_norm[10240]`、最终 mixer `[10240]`），建议每层只预转当前需要的 1 个（共 20 KB，bf16 直读即可）；24-27=保留。

**CrossCore flagId（旋转槽位制）**（docs/14 §9.1 L38）：0-3=MoE 段边界（与 m3 兼容）；4-7=GDN/其它段边界；8-11=段内 mode-2 流式对；**12-15=改为 HC 段专用**（原"保留"）。同步点数从"每层 ~8"涨到"Moe 8 + GDN 7 + HC 10"≈ **25/层**；旋转槽复核："每 id 每层 ≈2 次 ≪ 15 上限"在新计数下变为 ≈6-7 次，**仍合规但要重算**。

**UB/AIV（字节偏移、32B 对齐、GetCoreMemSize 获取）**（docs/14 §9.1 L40）：PERSIST @0 ~12KB；SEG-VEC 窗 @16384 ~112KB（unpermute 双 stage 110KB 最大消费者，其余 VEC op 叠放）；SEG-GDN @16384 ~138KB（m4 双 64KB slab + q/k/v/o+g/β，与 SEG-VEC barrier 分隔同址叠放）；**新增 SEG-HC 窗**：裸算 `[10240]` bf16 的 combine_norm 需 res+out+y+w ≈ **85 KB**；但 **`hc_norm` 的分组边界（4×2560）恰好等于 combine 的 stream 结构，可按 stream 窜流处理**，每 stream 5 张 `[2560]`（res/block/out/y/w）+ 双缓冲 ≈ **50 KB**（强烈建议按 stream 切，同时把 L0C 的 `[m,10240]` fp32 压力消掉）。另需 `[10240]` bf16 scratch（gate 与 xn）双缓冲 = **40 KB**。
> **预算记法**：SEG-HC 要与 SEG-VEC / SEG-GDN **同址叠放**（@16384，段间 barrier 分隔 ⇒ 编译期 linear allocator 允许）；**不是三段相加**，UB 峰值仍由 SEG-GDN ≈138 KB 决定（12 + 138 ≈ 150 KB，余 ~40 KB 机动）。若某段要与 HC 段并发（例如跨段预取 staging 与 SEG-HC 同时活），必须重新算峰值。

**L1/AIC**（docs/14 §9.1 L42）：A 区 @0 ×2 组、B 区 @262144 ×2 组；@~330KB 起 ~180KB=**权重预取滚动窗**。新增两个 mixer GEMM 的 tile 规划：**down+inject**：A `[m,10240]`（m=1 时 20 KB/行），B `[324,10240]` = **6.34 MB**（不补 pad；vLLM 补到 336=6.88 MB）→ B 必须分 K/N tile 流送（如 N=168 × K=512 bf16 = 172 KB）；**up**：A `[m,320]`，B `[10240,320]` = **6.55 MB** → K 只有 320，按 N（输出 10240 维）tile；**L0C 对 up 的输出 `[m,10240]` fp32 在 m=16 时 = 640 KB 超 L0C ⇒ 必须 N-tile**。

### 4.5 新增权重槽 / UB / L1 预算（抄 docs/14 §9.2）

| 项 | 值 | 说明 |
|---|---|---|
| 层界残差流 | `[m, 10240]` bf16 = **20,480 B/token** | 双缓冲 = 40,960 B/token |
| 层界附带张量 | pending block `[m,2560]` bf16 = 5,120 B + inj `[m,4]` | 双缓冲各一份 |
| 每层 HC 权重 | **25.20 MiB**（= 2 × 12.60 MiB，checkpoint 原始 324 行）；执行时按 vLLM 补 pad 为 25.66 MiB | 4 张量 × 2 mixer = 8 个权重张量/层 |
| HC 权重槽总数 | 48 层 × 8 + 最终 mixer 3 = **387** | 形状只有 4 种（`[10240]`/`[320,10240]`/`[10240,320]`/`[4,10240]`）⇒ icache 与描述符可复用 |
| 每层权重合计 | GDN 层 **1415.75 MiB = 1.3826 GiB** / QSA 层 **1403.33 MiB = 1.3695 GiB**（HDR；含 MoE 1280.0 + shared/gate 5.0） | HC 是**净增量**：+25.20 MiB/层（checkpoint 口径）/ +25.66 MiB/层（补 pad 口径），× 48 层 = +1.18/1.20 GiB |
| PLE 权重槽（仅 layer 1） | 表 **95.37 GiB**（磁盘/host 驻留，不入 HBM 常驻预算）+ **62.64 MiB** 小张量（进 HBM） | 需 16 行 × 320 B 的 gather 路径 |
| PLE 状态 | `[10240, 9]` bf16 = **180 KiB / 序列**，TP-replicated | 独立于 GDN 的 MambaSpec |
| QSA indexer 权重 | 3.125 MiB/层 × 12 | 进 HBM |
| QSA 压缩 KV | **768 B / token-of-context**（12 层合计，bf16） | 262144 ctx ⇒ 201.3 MB |
| QSA raw ring | **1,120 B / 请求 / 层**（num_spec=0） | 1 物理块/请求终生 |
| QSA 选择缓冲 | **8,208 B / token-of-batch-capacity / 层**（`[T,2052]` int32） | 8192 批 ⇒ 67.2 MiB/层 ⇒ 12 层 806 MiB |
| MTP 预留 | MTP 层权重 **4.856 GiB**（含 HC 8 件 + mixer 3 件，形状同主干；expert 为 **bf16 未量化**） | 暂不实现但资源表需留位；`mtp.pre_fc_norm_hidden.weight [10240]` 说明 MTP 消费 10240 宽多流态 |

## 5. 同步图（记号：BB=drain BufferID / PB=PipeBarrier / CC0=mode0 / CC2=mode2 / S=PIPE_S 仅链式）

- MoE 段序 S1-S8 与 GDN 段序详见 M18 mission notes（任务4 全表）；要点：段内跨 pipe 一律 BB；**行循环 BUF_X 复用一律 PB(PIPE_MTE2)**；段边界跨核走标准 mix 序列（AIV 侧 CC0 对齐 → CC2 反向通知 → AIC 侧 CC0 → CC2）；set 挂生产 pipe（FIX/MTE3），wait 窄 pipe。
- **HC 段（docs/14 §9.3）**：每层 +10 个 HC 段内边界（2 mixer × 5 op）；flagId 压力 ~25/层（§4）；`pending block_output` 跨层界存活到下一层 mixer —— **不破坏"段边界严格串行"前提**（仍逐层串行），但层 kernel 入口要等上一层 3 个出口；`combine_and_mix` 把"上一段收尾"与"本段开头"融进同一个核 ⇒ 可把两层间同步点从 2 个降到 1 个（**延迟 combine 的真实收益，mega kernel 应照搬而不是拆开**）；PLE 特例（层 1 的独立 `combine`）单独一组 flagId。
- 规则核对清单：set 禁 PIPE_S/ALL；相邻同步点 flagId 不同（旋转槽）；wait 窄 pipe、链式 wait→set 才挂 S；mode 0 仅同类型；**AIV 侧 set 挂 `PIPE_MTE3`、AIC 侧只能 `PIPE_S/MTE1/MTE2/FIX/M`**（HC 段逐点核对）；**同 pipe 背靠背复用同一 buffer 必须 PB**（HC 的 `[10240]` scratch 复用点很多，最容易踩）。**decode / prefill 共用方案 B（§1 item 5）下**：flagId / BufferID 命名空间**按 mode 分节登记**，头文件显式声明"两 mode 互斥执行、id 可跨 mode 复用"；新增跨核交接一律在头文件登记 flagId 与 pipe，**禁用 `PIPE_ALL`**（docs/15 §5.3 第 8/14 条）。

## 6. 算件小改清单（集成前置）

- **A.【最高优先】M16 latent 全清**：元素计数 `DataCopy` 重载（`DataCopyParams` 栈垃圾）全部落在各 kernel 的 MTE2 搬入段 —— `m2_mxfp4_quant.asc` 的 `ProcessTile`、`m4_gdn_recurrent.asc` 的 `CopyInHead`、`m5_swiglu_quant.asc` 的 `ProcessTile`、`m6_rmsnorm.asc` 的 `Process`、`m8_permute.asc` 的 `CopyIn`、`m3_grouped_gemm.asc` 的 `ProcessAiv` → 全部改显式 `DataCopyParams{1,len,0,0}`（照 m9 写法）。**以符号为准**；M16 审计时的行号口径是 m2:154、m4:248/252-254、m5:174/175/189、m6:417/442-443/536-537、m8:183/191/267/270/324、m3:443，**已实测漂移 +2~+3，不得用行号定位**。
- **B.【同 pipe PipeBarrier】**：m2/m5/m6/m8#2/m3-AIV 行循环 MTE2 背靠背复用补 PB（m4/m9/m8#1 天然规避）。
- **C.** m3-AIV scalar quantizer → m5 全 VEC 路径替换（消 S pipe 阻塞 + 消 M15 标量跨 pipe 风险类）。
- **D.（已取消，docs/14 §9.1 L54）** 原"m6 残差输入 fp32 支持"在 HC 路径上**取消**（残差是 bf16 `[10240]`）；改为：
  - **D′**：实现 `hc_combine_norm` 的融合（combine + 分组 RMSNorm，**含 combine→RMSNorm 之间的 bf16 舍入点**，docs/14 §8.2 #9、§10 #3）；
  - **D″**：实现 `hc_gate_mix`（4 流门控均值 + `/4`）。
  - ⚠ 三处 `/HC` 语义**不可互换**（docs/14 §9.4）：`silu(x/4)`（实参内）/ `(Σσ(gate[s])·xn[s])/4`（求和后）/ `2·σ(inj/4)`（实参内且带系数 2）。
- **E.** m8 topK 固定 10 单实例化（缩 icache）；w_tk_packed 协议保留；Σt_e=640 标量索引读改 MTE2 批量。
- **F.** m1/m3 calcM≥2 quirk 契约化进 host 分配；m>64 m-tile 路径（prefill）未验证。
- **G.** m9↔m4 的 g/β stride-8 与 {H,1,0,0} 契约冻结。

### 6.1 层界口径表（凡把 `HIDDEN` 当层界宽度用的常量 / DMA Block1 长度，逐条改标）

| 位置 | 现值 | 新口径 |
|---|---|---|
| `m13_resources.h:42` `HIDDEN=2560`、`m14_resources.h:47` | 被当作层界宽度 | **不是层界宽度**：层界宽度是 10240（4×2560）；`HIDDEN` 只能当**子层内部宽度**/MoE-GDN 段宽度 |
| `m13_resources.h:237,241,249` `SZ_AQ/SZ_GU/SZ_Y` | K=2560 | **不变**（expert GEMM 的 K 本来就是 2560，与层界无关） |
| `m13_resources.h:118` `UB_ZEROS_B16 bf16[HIDDEN]` | 零残差占位 | 层界输入是**真实的 4 流残差**（入口才由 `embed_tokens(x).repeat(1,4)` 得到）；zero-residual 语义只在"首层 attn_hc 无 pending combine"这一处成立 |
| 层界 DMA Block1 长度（各 .asc 的层界搬运点） | 2560 元素 | 层界搬运的 Block1 长度为 **10240 元素 = 20,480 B/token**（若按 stream 切，则回到 2560/stream × 4 段） |
| 层界 gamma | host 合成的 `gamma[2560]` | checkpoint **不存在**任何 `[2560]` 层界 gamma；只有 `hc_norm[10240]`（4 组 × 2560，(1+w) Gemma 风格） |

> 上表涉及的文件（`m13_resources.h` / `m14_resources.h` / `*.asc`）**不在本 mission 的 `docs/**` scope 内**，M38 只做口径标注；落地修改需另立 mission（已由 TowerFinding 报给 tower）。

## 7. icache / 标量发射初评

AIV 16KB 对 ~8-10 个 op 标量代码贴边（估 8-15KB）→ op 级 outline + 循环化 + hot 段排前；AIC 32KB 可控（10-20KB，防模板实例膨胀）。flagId 旋转槽以"段边界严格串行"为前提，与跨 op 预取不冲突（预取走窄 pipe 旁路）；若未来要消 barrier 做真跨段流水需重规划 flag。**新增 387 个 HC 权重槽但形状只有 4 种**（docs/14 §9.2）⇒ 描述符与标量代码可复用，icache 压力主要看 mixer 的 5 个 op 模板。

## 8. 失效假设与仍然有效的资产

### 8.1 四条失效假设（硬结论，改动前必读；全部出自 docs/14 §8.2）

1. **层界残差流 `[m,2560]` → `[m,10240]`**（4 流，HC 外层 / HS 内层 layout，4 组各自独立归一化）。所有以 `HIDDEN` 为层界宽的常量、DMA Block1 长度、buffer 数量都要改（§6.1）。
2. **层边界 op 不再是 `Add+RMSNorm(gamma[2560])` ⇒ `m6` 不再是层边界 op**。替代物是两个 mixer 的 `hc_norm`：`[10240]` 分组 Gemma-RMSNorm（4×2560，(1+w)，eps 1e-6）；`input_layernorm`/`post_attention_layernorm`/全局 `model.norm` 全部不存在。
3. **层间不是 1 个张量而是 3 个**：`state [T,10240]` + `pending block_output [T,2560]` + `injection [T,4]`（延迟 combine）。层 kernel 的 I/O 签名、双缓冲数量、跨层依赖 DAG 全变；层 kernel 不能自闭合。
4. **子层输入不再是 `norm(h)`，而是 4 流门控加权均值**（`Σ_s σ(gate_s)·xn_s / 4`）⇒ 需要一个 **10240→2560 的归约新 op**（`hc_gate_mix`），不是简单 copy。

### 8.2 仍然有效的资产（不受 HC 影响，docs/14 §8.3）

- **MoE 内部全链**：m13 S2 router / S3 计数排序索引胶水 / S4 permute / S5/S7 MXFP4 VEC 量化 / S6/S8 分组 GEMM / S9 unpermute+combine —— 只依赖喂进来的 `[m,2560]`，与层界语义无关；682 条判据 + 118 条 numpy 校验锚在 device 字节上。expert GEMM 的 K=2560 不变。
- **GDN 链**：m14 的 in_proj（K=2560, N=16480）/ out_proj（K=6144, N=2560）形状与 checkpoint 一致；m9 prolog、m4 递推（in-place ssm_state）、m12 RMSNormGated（逐 head 128 维）全部与层界宽度无关。
- **m12 / m5 / m9 / m4**：与层界无关（m12 的 norm 维是 128、宽度 6144；m5 是 `1280→640→320/20` 的 MoE 内部）。
- **m6 内部**：RMSNorm 归约（二分折叠求和 + NR rsqrt）与 `(x·rstd)·gamma` 数学可复用为 mixer 内的子步骤；只有 `[m,2560]` 的 IO 契约失效。
- **docs/05 全局设计规则**：单一编译期资源表"模块只声明 footprint"、CrossCore flagId/pipe 规则、同 pipe PipeBarrier 规则、mode-0/2 语义。
- **48 层 host 循环骨架**（m15 结构）：host 48 次启动、3:1 层派发骨架、`LayerSlot()` 权重/状态槽位、层间零 CrossCore —— 要改的是残差流平面与 kernel 出口语义，不是循环骨架。

## 9. 风险与遗留

- **两处地雷（来自 docs/14 §11，务必写进层派发）**：
  1. **`layer_types` 里 12 层标的是 `full_attention`**（没有 `qwen_sparse_attention`），但设置了全套 indexer 字段。官方的判定是 `layer_type == "qwen_sparse_attention"` **或** `indexer_n_heads is not None`。只按字符串识别的实现会把这 12 层误判为稠密 GQA ⇒ **Ascend 层派发必须用"indexer 字段存在"作判据**。
  2. **`split_ngram_parts` 不是 `Qwen4ExpTextConfig` 的声明字段**，代码用 `getattr(config, ..., 512)` 兜底；checkpoint/config 必须给出 **128**。若回落到 512，分片行数从 2,500,012 变成 625,003 并**直接加载失败** ⇒ Ascend 侧应把它当**必需字段校验**。
- **PLE 95.37 GiB 表的落地已由用户裁决解除阻塞**（裁决 ④，见 §10.1）：表放 **host 侧**，用 **PCIe-through 方式 MTE2 直接读 host 内存**、**可异步读**，**现在不实现**。⇒ 不再需要在 HBM / host 常驻 95.37 GiB 连续空间 —— docs/14 §6.3/§10 #1 记的"cgroup 32 GB 与 `/dev/shm` 16 GB 都装不下"从**立项阻塞**变成**实现形态问题**。仍需注意：`EngramConfig.verify_model_config` 要求 `is_cuda_alike()`，Ascend 上引用它会直接抛异常（要自己起一套 host 表）。PLE 挂在 **0-based layer 1**（已裁决，`ple_layer_ids=[2]` 是 1-based）。
- **HC 段带宽效率存疑**（docs/14 §10 #2）：0.816 TB/s 是用宽 GEMM（K=2560, N=16480）反推的；HC 的 down 是 N=336、up 是 K=320 的**极瘦 GEMM**，tile 利用率低，实际可能慢于 1.52 ms。按裁决 ② 现在**不做调优**；立项前只做一条"结构能不能装下"的单 mixer micro-benchmark（**设计输入，不是性能优化**）。
- **bf16 舍入点待定**（docs/14 §10 #3）：combine→RMSNorm 之间的 bf16 舍入是否必须复刻 —— 按 `docs/17-verification-standard.md` 的 L0 口径（逐位，或数学上不可能时 ≤1 ulp）**大概率必须复刻**，需数值实验确认并写明"为什么不可能逐位"。
- **prefill / decode 档位（裁决 ③）**：decode = **m=1、context 4096**；prefill = **单序列 m=4097**（MoE Σt≈40970 需 m-tile 分段，每段重复全链同步）；**无 batch>1 / 多并发需求**。m4/m9 目前仅覆盖 decode。PLE 在 prefill（`16T` 行随机 gather）代价结构完全不同（docs/14 §10 #8）。
- **prefill 与 decode 的瓶颈结构相反（docs/15 §0 #1、§6.3/§6.4）**：decode 是权重带宽支配；prefill m=4097 下**除 MoE 外全部翻转为算力支配**（每层：GDN 1.37ms 算力 vs 0.70ms 带宽；attention 1.72 vs 0.27；HC 0.30 vs 0.23；LM head 14.2 vs 2.08），**只有 MoE 仍是带宽支配**（1.28ms 带宽 vs 0.21ms 算力，48 层小计 61.4ms，是最大单项；要翻转需 m≈20480）。全 pass 下限 ≈**160ms**（≈25.6k tok/s；按实测 55% MTE2 效率折算 ≈210ms）。⇒ 同一层 kernel 里两条路径资源瓶颈相反，这是"共用 kernel"（方案 B）的关键输入；**GDN 的 chunk 扫描是 prefill 唯一串行临界路径**，必须实测。
- **QSA 验收口径（tower 裁决 2026-09-26，原文 `docs/17` §7 附则）**：官方 `indexer_budget=2048` 是 **token 预算** ⇒ block_topk = 2048/4 = **512**，而 4097 上下文有 **1025 个可见块** ⇒ 官方**只 attend 约一半历史**，**稠密 causal 与官方输出不可能一致**（gather 集合与 softmax 分母都不同；decode ctx=4096 同样成立）。裁决三条：① 首期基准 = **自建稠密 causal 参考**，但**必须处处标注它不是官方行为**，**任何地方都不许再写"m=4097 对齐官方输出"**；② **QSA 的 cache 填充必须现在做**（raw key ring + compressed key cache，含 pos 4096 属"开放组"不写 compressed cache 的 off-by-one 契约）；③ **打分 / topk / expand 是正确性必做项、不是性能选项**（prefill 下只省 ~1.1% 算力，但不做就与官方 token 不一致 —— 排期靠后，不可砍）。
- 权重流 L2 hint CACHE_MODE_DISABLE（decode，docs/11 §3.2）；**48 层 host 循环留在框架侧**（裁决 ①，见 §10.1）——我们要交的是 **per-layer 算子**，其 I/O 签名按 §8.1 #3 是 3 张量，验收前仍先跑通单层（docs/14 §9.1 L67）。
- 降本选项（立项时取舍，docs/14 §9.5；按裁决 ② 现在不投入）：HC 权重 bf16 → FP8/MXFP4（每 token 省 0.65–0.9 GB ⇒ 0.8–1.1 ms）；不补 `_input_mix_padding`（324 行而非 336，每 token 省 23.6 MB ⇒ 29 µs）；down/inject 拆两条 GEMM（4 行 GEMV 可并入 `hc_combine` 尾段）；**HC 权重常驻 L2 无收益**（算术强度 ≈1 FLOP/B）。
- 下一步立项：layer kernel 骨架 mission（**per-layer 算子，层内核含 hc mix —— 裁决 ⑦**；op 序列框架 + 全局资源表头文件 + 旋转 flag 管理器 + 小改清单 A/B/C/D′/D″ 落实），并把 §6.1 的口径修改落到 m13/m14 的资源表与层界搬运点。

## 10. 接口约束与验收口径（用户裁决 2026-09-26，立即生效，最高优先）

### 10.1 用户裁决五条（tower inbox 广播 `20260926-tower-all-7-kernel-m-ple-host-pcie-hc.md`）

| # | 裁决 | 对本方案的约束 |
|---|---|---|
| ① | **每个 kernel 只跑一层；48 层循环留在框架侧** | 我们的产物是 **per-layer 算子**，由 vLLM `qwen4_exp` 的 host 循环调用（vLLM / vllm-ascend 的 host 循环**不变**，"对 vLLM 的修改要小"）。算子的输入输出必须能对上 `qwen4_exp` **层模块的既有张量契约**（hidden in/out、KV cache、state、hc 的 3 张量层界）；**不要**设计成"接管整个 forward"的形态，也**不要**发明框架里不存在的私有中间表示 |
| ② | **现在不关心性能** | 一切以**正确性打通**为准：不调 tile 形状 / 带宽利用率 / 时延，也不为性能做取舍。**但"设计输入类估算"仍要做**（这个权重每 token 读多少字节、放不放得下、UB 够不够）——本文 §4/§4.5 的预算表就是这类，属**定结构**，不是调优 |
| ③ | **m 的语义** | `m=4097` = **prefill**（单序列 4097 token，需分块）；**decode 是 m=1，且 context 长度 = 4096**；**没有 batch>1 / 多并发请求** |
| ④ | **PLE / ngram 表放 host 侧** | 用 **PCIe-through 方式 MTE2 直接读 host 内存**，**可以异步读**；**现在不实现**，先做前面的。其余权重 **device 常驻（74.3 GiB）**（单层 ~1.4 GiB 可按层流式处理） |
| ⑦ | **层 kernel 要包含 hc mix** | `hc_norm` + 低秩 up/down mix + block inject **都在层内核里** ⇒ §2 的 `hc_combine_norm` / `hc_gate_mix` / 两条 mixer GEMM 是**层内核的段**，不是独立外设算子 |

> ⑥（事实类问题由 tower 查）已定：**PLE 挂载层 = 0-based layer 1**。

### 10.2 vllm-ascend 接入点（长期方向约束）

- 最终目标：**把 `qwen4_exp` 支持做进 vllm-ascend**。我们的 kernel 作为 **vllm-ascend 的自定义算子**，被官方 vLLM 的 `qwen4_exp` 模型实现调用；**官方 `qwen4_exp` 实现是唯一语义权威**（本机 `/workspace/vllm/vllm/models/qwen4_exp/`），数学/形状/语义一律以它为准，不自造一套。
- 算子边界尽量对应 `qwen4_exp` 的**一个模块级算子**（一个 attention / 一段 GDN / 一个 MoE block / 一次 hc mix），别做成只有我们内部才认识的私有接口。
- 权重布局沿用 checkpoint / vLLM 原始排布；dtype / 量化契约与 config 一致：**MoE 专家 MXFP4 e2m1 + e8m0 group32；attention / ngram embedding / MTP / HC / indexer 保持 bf16**；`quant_method="ascend"`。
- **逐段接入位置**（谁是调用方 / 入出参 / 有无现成槽位：GDN prefill·decode、QSA indexer 与稀疏核心、MoE、HC、PLE）见 `docs/15` §8.2。**三条硬要求**见 §8.3：① 权重按 checkpoint 原始排布消费（唯一建议偏离：HC 的 `336 = 324 + 12 pad` 按 324 解）；② **in-place state 必须申报**（`mutates_args`：ssm_state / conv_state / 主 KV / raw key ring / compressed key cache / PLE conv state），否则 vLLM 的 graph capture 与别名分析会出错；③ **kernel 只处理"一个 chunk"**（vLLM 是 chunked prefill，m 可为 1..M_MAX，**m=4097 是验收口径、不是接口假设**）。

### 10.3 验收口径（全仓库强制）

统一数值验收口径见 `docs/17-verification-standard.md`：

- **L0 单 kernel**：独立参考（numpy/double 或官方源码逐句复刻）+ **逐位一致**（数学上不可能时 ≤1 ulp 并写明原因）+ 非空洞性检验。
- **L1 段级**（MoE 段 / GDN 段 / **hc mixer** / attention 段）：double 参考链**分段抽点**，容差按该段**量化预算**推导，给"最差容差占用"。
- **L2 层 / 整网**：对官方 vLLM `qwen4_exp`，指标 = logits 相对误差 + argmax/top-k 一致率 + 生成序列一致率；量化模型以 **top-1 一致率**为主判据。
- 判定项与报告项**必须分栏**（guard 不许混进 PASS 计数）；判据要能由 reviewer 用归档数据 + 一条命令复现，dump 的 `sha256` 入库。
- **QSA 的固有验收偏差见 docs/17 §7 附则**（与 §9 的 QSA 裁决条目同源）：attention 段的稠密判据**不是**官方行为的判据，L2 对齐时必须显式说明差异来源。
- 本文 §8.2 引用的 m13「682 判据 + 118 numpy」、m14「114 + 103」是 **L1 口径**；其判据计数与 guard 分栏按 docs/17 §2 统一后再引用。
- **prefill 侧的契约、共用方案 B 与 14 条不返工清单**见 `docs/15` §5（方案 B）/§5.3（清单）/§6.3（瓶颈账）/§8（接入点）。
