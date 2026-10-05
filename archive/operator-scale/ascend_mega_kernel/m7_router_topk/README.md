# M7 Router softmax topk kernel

单 AIV 核完成 MoE 路由：`bf16 x[m,2560] × router_w[512,2560]^T → logits → 行 softmax
→ top-10 降序 → 按 norm_topk_prob 归一`，m 运行时 1..64。

路由语义与 CPU golden `tools/golden/moe_block_ref.py::router_topk` 完全一致
（M4 已定）：softmax over 512 专家（max-shift），top-k=10 降序（并列取小 id），
topk weights 归一到行和为 1。

## 输出

| 输出 | 形状 | 类型 | 说明 |
|---|---|---|---|
| logits | [m, 512] | fp32 | **max-shifted**（`l - rowmax`，与 moe_block_ref 返回一致） |
| topk_ids | [m, 10] | int32 | 降序；真实数据下与 golden 精确相等 |
| topk_weights | [m, 10] | bf16 | RNE；与 golden 容差 ≤ 1 ulp（实测 0~1 ulp） |

## 文件

| 文件 | 说明 |
|---|---|
| `m7_router_topk.asc` | kernel + host（确定输入生成、运行、读回、自检、dump） |
| `check_ref.py` | 对 `moe_block_ref.py` 的权威交叉校验（ids 精确 / weights bf16 网格 / logits） |
| `extract_router_weight.py` | 从 checkpoint 抽 [512,2560] bf16 router 权重到 `data/` |
| `data/router_weight.bin` | layer0 `mlp.gate.weight`，真实 checkpoint 逐字节切片（2.5MB） |
| `tools/archive_evidence.sh` | （M56 新增）一条命令做「构建 + 设备自检 + `check_ref` + 写 sha256 清单」到 `evidence/`，rc 三态 0/1/2 |
| `evidence/` | （M56 新增）`run.log` / `check_ref.log` / `dump_sha256.txt`（50 个 dump，M56 重跑）/ `dump_sha256_m48_after.txt`（M48 改造后，50 文件）/ `dump_sha256_m48_before.txt`（M48 改造前基线，m1/m2/m16 共 30 文件）—— 后两者用于位级前后对照 |

`tools/weights/data/layer0_e0-8/router_weight.bin` 只切了前 8 个专家行（该切片按 8
专家裁剪），不够 top-10 语义用；`extract_router_weight.py` 用同一工具链
（`tools/weights/safetensors_reader.py`）抽完整 512 行，byte-verbatim。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m7_router_topk/build -S m7_router_topk -DCMAKE_BUILD_TYPE=Release
cmake --build m7_router_topk/build -j4
./m7_router_topk/build/m7_router_topk            # 设备自检：m=1/2/16/33/64

# 权威校验（golden 交叉验证）：
M7_OUT=./m7_out ./m7_router_topk/build/m7_router_topk
/usr/local/python3.12.13/bin/python3.12 m7_router_topk/check_ref.py ./m7_out/m1
/usr/local/python3.12.13/bin/python3.12 m7_router_topk/check_ref.py ./m7_out/m33

# 或者一条命令把上面全做掉并把读数/清单归档到 m7_router_topk/evidence/（M56）：
m7_router_topk/tools/archive_evidence.sh            # --no-build 可复用已有 build/；rc 三态 0/1/2
```

## 校验结果（2026-09-26，Ascend950PR 真机；M48 复跑，M56 重跑归档）

kernel 内自检 + `check_ref.py` 对 `moe_block_ref.py`，m=1/2/16/33/64（seed 7）。
下面两段**原样取自归档** `evidence/run.log` 与 `evidence/check_ref.log`（后者只摘 `m=1` 那一档，
其余四档同形、数值见紧随其后的表；完整 20 行在归档里）：

```
# evidence/run.log
[M7] m=1   seed=7 : PASS | logits maxRel=7.83e-05 ids exact weights maxUlp=0 bf16-grid-ok
[M7] m=2   seed=7 : PASS | logits maxRel=7.83e-05 ids exact weights maxUlp=0 bf16-grid-ok
[M7] m=16  seed=7 : PASS | logits maxRel=1.80e-04 ids exact weights maxUlp=1 bf16-grid-ok
[M7] m=33  seed=7 : PASS | logits maxRel=1.80e-04 ids exact weights maxUlp=1 bf16-grid-ok
[M7] m=64  seed=7 : PASS | logits maxRel=1.80e-04 ids exact weights maxUlp=1 bf16-grid-ok
[M7] ===== ALL PASS =====

# evidence/check_ref.log（m=1 档，逐字）
===== check_ref  m7_router_topk/m7_out/m1
topk_ids exact:      True
topk_weights bf16-grid (<=1 ulp, denormal-flush allowed): True (max ulp 0)
router_logits (max-shifted) abs diff <= 2e-2: True (max 3.815e-05)
[check_ref] m=1 : PASS
```

`check_ref.py` 逐档结果（三条判据全 True；`logits max abs` 取自归档）：

| case | topk_ids exact | weights max ulp | logits max abs | verdict |
|---|---|---|---|---|
| m=1 | True | 0 | 3.815e-05 | PASS |
| m=2 | True | 0 | 2.289e-05 | PASS |
| m=16 | True | 0 | 4.578e-05 | PASS |
| m=33 | True | 0 | 4.578e-05 | PASS |
| m=64 | True | 0 | 6.866e-05 | PASS |

判据数：kernel 内建 **5** 条（m=1/2/16/33/64，逐条 `PASS`）+ `check_ref` **5 case × 3 项 = 15** 条
（ids 精确 / weights ≤1 ulp / logits ≤2e-2），**20/20 PASS**（读数逐条归档在 `evidence/run.log`、
`evidence/check_ref.log`）。M48 向量 API 改造后复跑，判据数与数值与本表**逐项一致**（零回归）。

## 证据级别（M56 诚实标注）

本模块的位级声明（"M48 改造前后 5 个 case 的 50 个 dump 逐文件 sha256 完全相同"）在此前的状态是：

| 项 | M56 之前 | **现在（M56 已补齐）** |
|---|---|---|
| `evidence/` 目录 | **不存在** | `evidence/` 已建：`run.log`、`check_ref.log`、`dump_sha256.txt`、`dump_sha256_m48_after.txt`、`dump_sha256_m48_before.txt` |
| 50 个 dump 的 sha256 清单 | **未归档**（只在作者本机 `/tmp` 工作副本里） | **已入库**：`evidence/dump_sha256.txt`（50 行 + 出处/复算注释头） |
| "改造前后逐字节一致"这条位级声明 | **只能靠源码级等价**（改造是机械替换：`Extract` → VF 内联、手写 RNE+`DeInterleave` → `Cast`+`DIST_PACK_B32`） | **有归档支撑（前后两侧都在）**：`dump_sha256_m48_after.txt`（M48 改造后，50 文件）与 `dump_sha256_m48_before.txt`（M48 改造前基线，m1/m2/m16 共 30 文件）均原样转存；两者在重叠的 30 个文件上**逐行相同**（`comm -23` = 0 行），且 after 侧与 M56 用当前代码重跑得到的清单也**逐行相同**（50/50） |
| 复算方式 | 无 | `m7_router_topk/tools/archive_evidence.sh`（构建 + 跑 + 校验 + 写清单一气呵成，退出码三态 0/1/2） |

每份 manifest 的头部都写了**哪一轮 / 哪个 commit / 哪个临时路径**（tower 要求）。其中一处如实标注：
**生成 M48 dump 时那个工作副本没有记录自身 commit**，故 after 侧写的是「内容等价的落地 commit
`2a07218`（merge `4ba5484`）」，而不是假装有版本戳；before 侧只有 mtime 与来源路径。

**现在能证明 / 不能证明**：
- **能**：当前 commit 的代码在设备上重跑，5 档全部 PASS；**M48 的前/后两份清单在重叠的 30 个文件上逐行相同**，
  且 after 侧与本次重跑清单**逐行相同（50/50）** ⇒ 「M48 改造未改变任何落盘字节」这条声明
  **可离线对读 + 可复算**（入库的两份清单 + 一条命令）。
- **不能**：仓库里**没有 dump 本身**（50 个文件约 10MB+，与其它模块同口径只入 sha256 清单），
  所以谁都**无法从零**重算「改造前」那一侧 —— 要复现它得 checkout M48 之前的 commit 再跑一次；
  入库的 before 清单是**当时的读数**，不是可再生的产物。另：before 侧只覆盖 m1/m2/m16 三档。
- **档位**：`topk_ids` 属 T1（整数/索引域，逐位）；`topk_weights` 按 bf16 网格 ≤1 ulp；
  `router_logits` 用绝对 ≤2e-2（max-shifted 后近零值的 fp32 累加次序噪声为主）。

## 设计

### UB 静态布局与余量（**230.5KB / 248KB，余量 17.5KB**）

全部偏移是编译期常量（`m7_router_topk.asc` 顶部），末址由
`static_assert(UB_WS + RB*128*2 <= 248*1024)` 把守。逐块字节数（RB=16、HIDDEN=2560、
E=512、EGRP=8）：

| 块 | 内容 | 字节 |
|---|---|---|
| `UB_XB` | x row-block bf16 `[16][2560]` | 81920 |
| `UB_WB` | w 行 ping-pong bf16 `[2][2560]` | 10240 |
| `UB_WF` | 权重 fp32 预转 `[8][2560]` | 81920 |
| `UB_LOG` | logits fp32 `[16][512]` | 32768 |
| `UB_VAL` | e_i fp32 `[512]` | 2048 |
| `UB_IDX` | arange 模板 u32 `[512]` | 2048 |
| `UB_TMP` | Sort32 对 fp32 `[1024]` | 4096 |
| `UB_MA` / `UB_MB` | 归并缓冲 A/B fp32 `[1024]` ×2 | 8192 |
| `UB_OUTV` / `UB_OUTI` | 前 64 对 value/idx | 512 |
| `UB_IDS` | ids 行 staging i32 `[16][128]` | 8192 |
| `UB_WS` | weights 行 staging bf16 `[16][128]` | 4096 |
| **合计** | `UB_END = UB_WS + 16*128*2` | **236032 = 230.5KB** |

**余量 17.5KB**（253952 − 236032 = 17920B）。⚠ 边界（finding
`20260926-agent-router512-improve-m7-router-topk-ub-80kb-150-5kb-230-5kb-5-5`）：本 README
与 `.asc` 旧文写「总 150.5KB」时漏算了 `UB_XB`（80KB），把余量高估成 97.5KB——任何想在
该布局上加 buffer 的后续工作（如 M42 的契约输出段）**必须按 17.5KB 余量估算**。

### GEMV：向量 MAC（任务书允许自选，未选 cube）

m≤64 时 router 是带宽受限 GEMV（权重 2.6MB 流式读一次），单 AIV 内 fp32 FMA
（`MulAddDst`）完成，免 AIC 参与及跨核同步；与 layer kernel 融合时保持 AIV 域内
自治。流程：x row-block（16 行）常驻 UB → 权重按行流式预转 fp32（ping-pong
BufferID 掩盖 MTE2）→ 8 专家一组共享 x 载入（40 步 `DIST_UNPACK_B16` 载入 +
Cast + MAC）→ 8 路归约后 Interleave 树拼成 8 连续 lane，`mask=8` 对齐
StoreAlign 写 logits（e0 步长 8 → 32B 对齐）。

### topk：donor Sort32 路线的 k=10 精确版

`Sort32`（16 个 32 块内降序，带 (value, expert_id) 对）→ **2 路归并树**
（`MrgSort validBit=0b0011`，8+4+2+1 次，elementLengths=32，输出 64 对/次）→
全序 512 对 → 前 64 对拆分（`Extract` 的 VF 内联，见下）→ top-10 求和归一。

donor（moe_gating_top_k_softmax_v2 perf 档）的 4 路 egreaterge 树对 k≤8 精确；
k=10 需要每级保留更多元素。本实现统一取"各级输入前 32 对、输出 64 对"：
`top-10(并集) ⊆ top-32(各输入)` 归纳成立，对 top-10 严格精确，且只用实测
可靠的 2 路 egreaterge（见 quirk）。

### 向量 API 合规（docs/05 §6.1；M48 裁定落地）

计算路径**全部**在 `__VEC_SCOPE__` 内、操作 `RegTensor`，经 `AscendC::Reg::` 重载：
gemv（`LoadAlign<DIST_UNPACK_B16>`/`Cast`/`MulAddDst`/`Reduce`/`Interleave`）、softmax
（`Max`/`Sub`/`Exp`）、renorm（`Reduce`/`Div`）、前 64 对拆分（`LoadAlign<DIST_DINTLV_B32>`）、
bf16 落盘（`Cast`/`StoreAlign<DIST_PACK_B32>`）。零经典 memory-based vector API、
零 `GetValue`/`SetValue` 型标量 lane 提取；`DataCopy`/`DataCopyPad`/BufferID 属搬运/同步类
（标准明文允许）。

**裁定例外**：`Sort32`（`SoftmaxTopkRenormRow` 步骤 3）与 `MrgSort`（`Merge2`/`MergeTree`）
保持 memory-based。依据 tower 裁定② = `docs/05` §6.1 **规则 ⓒ**「无寄存器等价物的原语例外」
（现行例外清单已收窄为只有这两个；取证见 `docs/18-vector-api-audit.md` §6.1/§9.2）：两者在 Reg 侧**无**等价物（CANN 9.1.0
`reg_compute/**` 穷举），且**官方 donor 同样是 memory-based**
（`ops-transformer/moe/moe_gating_top_k_softmax_v2/.../perf_arch35.h:187` = `Sort32`、
`:225`/`:243` = `MrgSort`）⇒ 属"行业惯例级例外"，**仅限本 topk 排序段、不得扩散**。
`Extract` **不在**例外内（它有 VF 形态，已改造，见下）。

### softmax：softmax_v2 的 RegBase 序列（禁 SoftMax<> 高阶 API）

行 max（Reg Max + Reduce）→ 减 max 写回 logits（对齐 golden 的 max-shift 返回）
→ `Reg::Exp` 得 e_i 写 UB。整行除 sum 与 renorm 抵消（`s_i/Σs = e_i/Σe`），
少一轮除法。

### renorm / bf16

top-10 e_i 求和（mask=10 `Reduce`）→ 除法 → `Reg::Cast<bfloat16_t, float, CAST_RINT>`
（ZEROING）→ `StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>` 一条落盘。这是
`docs/05` §6.1 位宽转换行（M35 确认）点名的规范写法，范本 `m8_permute.asc` 的
`MoeUnpermuteKernel::ComputeRow`（`Cast` + `StoreAlign<..., DIST_PACK_B32>` 那两条，行号提示 :313-314）。
**原实现**（手写整数 RNE `(bits+0x7FFF+((bits>>16)&1))>>16` + `DeInterleave` 奇偶打包）
正是同处点名的"多余的自造动作"（M35 曾因此把值写成 NaN），M48 已删除。

### 前 64 对拆分：`Extract` 的 VF 内联

原为经典 `Extract(vT, iT, mT, 2)`。现按厂商 dav_3510 的 `ExtractVf`
（`asc/impl/basic_api/dav_3510/kernel_operator_vec_gather_mask_impl.h:425-450`）的
`repeatTime=2` 路径内联：一条 `LoadAlign<float, LoadDist::DIST_DINTLV_B32>` 载入 128 float
（值/索引交错），偶 lane = value、奇 lane = idx，两条 `StoreAlign` 分别写
`UB_OUTV`/`UB_OUTI`。纯 pair 拆分、无算术 ⇒ 与经典调用逐位等价。
（官方 donor 亦已弃用经典 `Extract`，手写了同形的 `ExtractKFP32Perf`：
`…perf_arch35.h:309-343`。）

### 同步：只用 BufferID

`BUF_X`（x 块 MTE2→V）、`BUF_W0/W1`（w 行 ping-pong MTE2→V）、`BUF_LOG`
（logits V→MTE3）、`BUF_TOP`（ids/weights V→MTE3）。release 一律阻塞释放（`mode=false` = CANN `ASC_LOCK_BLOCK` 默认；两种模式都等本 pipe 已发射指令落地）。
无 set_flag/wait_flag、无 TPipe/TBuf/TQue/AllocTensor，UB 偏移/BufferID 全部
编译期静态。

## 3510 / CANN 9.1.0 实测 quirk（本 kernel 全部规避）

1. **fp32/int32 `GlobalTensor` 切片取址错误**（bf16 正常）：GM 一律按
   `bfloat16_t` 位视图寻址（bf16 元素偏移 = 字节/2）。
2. **同一 `GlobalTensor` 重复 `SetGlobalBuffer` 失效**：每对象只 Set 一次。
3. **`LocalMemBar` 必须在 `__VEC_SCOPE__` 内**，否则 507035。
4. **非 32B 对齐 `StoreAlign`（含 mask=1/小 mask）507035**：短向量写用
   32B 对齐的 mask=8 / 整寄存器写 + 行级紧凑拷出。
5. **`StoreUnAlign(dst, reg, ureg, n)` 是整寄存器写**（n 只是指针步进）：
   会踩踏邻区，不能当短向量写用。
6. **`elementLengths=256` 触发 507035**（≤64 实测可用，本 kernel 统一用 32）。
   本 kernel 只用 **2 路** `MrgSort`（validBit=0b0011）。旧条目「4 路 `MrgSort`
   （validBit=15）丢 src3/src4 胜出者的 index」**已由 `docs/05` §6.3 #19 取代**：
   M23 探针归因为当年调的是 `AscendC::MrgSort4`——该 API 在 dav-3510 上是
   `[[deprecated]]` 空函数体（静默 no-op），同参数改走 `MrgSort(dst, srcList, params)`
   结果正确。本仓库**生产模块**无 `MrgSort4(` 调用（只剩 `MrgSort4Info` 参数结构体）；
   唯一调用点 `probe_sync_quirks/probe_a2_mrgsort4_api.asc:124` 是该 API deprecated/no-op
   的取证反例、故意保留（tower 裁定④「探针豁免」），不在本次 scope、不应删。
7. **`Reg::Arange` 只填 64 lane**（fp32 VL）：int32 128-lane 寄存器高 64 为 0，
   索引模板按 64-lane 分块建，否则 idx%128∈[64,128) 的 id 全变 0。
8. **普通 128-bf16 寄存器 `Cast<float>` 只取偶数元素**；全量转换用
   `LoadAlign<bfloat16_t, DIST_UNPACK_B16>` + Cast（每步 64 连续元素）。
9. **fp32→bf16 cast 的寄存器结果落在 32bit lane 低半**（偶位）：用
   `StoreDist::DIST_NORM_B16` 直接 store 会产生空洞；规范做法是 store 的 **pack
   模式**——`StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>`（`docs/05` §6.1
   位宽转换行，M35 确认；范本 `m8_permute.asc` 的 `MoeUnpermuteKernel::ComputeRow`）。M48 已按此改造，替代
   原先手写的整数 RNE + `DeInterleave` 打包（改前/改后 dump 逐字节一致）。
10. `UpdateMask` 需 uint32_t 左值；`__VEC_SCOPE__` 内循环变量须 uint16_t；
    AIV 循环内不支持 scalar float 算术。

## 已知边界

* **并列语义**：golden 对完全相等的 score 按小 id 排序（stable argsort）；
  硬件 Sort32/MrgSort 对同值对不做 index tie-break（与 CANN donor op 一致）。
  真实路由数据（logits ~±20，score 连续分布）不存在并列，ids 与 golden 精确
  相等（已验证）。若将来需要严格并列语义，按白名单裁决的退路是
  moe_compute_expert_tokens 的 Cmp+Select+Reduce 幂等方案做二级排序。
* 单 AIV 串行专家循环（实测 kernel 时间：m=1 → 0.066ms，m=64 → 1.47ms；
  权重 2.6MB 流式读为带宽大头）。非性能优化目标；融合进 layer kernel 时可
  按专家槽位切多 AIV 并行。
* softmax 分数 fp32 下溢（logit 差 > ~88 时 e_i=0）与 golden 行为一致；
  此时多个 0 分并列次序不保证（见上）。

## 本次改动的位级影响面（M48 向量 API 合规 sweep）

| # | 改动 | 性质 | 位级影响 |
|---|---|---|---|
| 1 | 经典 `Extract(vT, iT, mT, 2)` → VF 内联（`LoadAlign<LoadDist::DIST_DINTLV_B32>` + 2×`StoreAlign`） | **纯机械替换**：厂商 `ExtractVf` 的 float 分支在 `repeatTime=2` 时就是这一串（loopTimes=1/tail=0），无算术 | **无**（实测逐字节一致） |
| 2 | 手写整数 RNE + `DeInterleave` 打包 → `Reg::Cast<bfloat16_t,float,CAST_RINT>` + `StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>` | **动了舍入实现**：手写式 `(bits+0x7FFF+((bits>>16)&1))>>16` 与 `CAST_RINT` 同为"就近取偶" | **无**（实测逐字节一致；数值域 e_i∈(0,1]，无 NaN/±Inf/溢出） |
| 3 | 撤回「4 路 MrgSort 丢 src3/src4 index」表述（`docs/05` §6.3 #19），补 `Sort32`/`MrgSort` 例外依据注释 + README §"向量 API 合规" | 注释/文档 | 无（代码未动） |
| 4 | README + `.asc` 的 UB 预算订正：「总 150.5KB」→ **230.5KB / 余量 17.5KB**（漏算了 `UB_XB` 80KB；finding `…router512-improve-m7…ub-80kb…`） | 注释/文档 | 无 |

**未触及 UB 布局与 BufferID 分配**（本次只改注释与算法实现，偏移常量、`BUF_*` 编号、
`static_assert` 一字未动）⇒ README 的 UB 表与「同步：只用 BufferID」表无需因本次改造同步订正
（UB 表按上一条 finding 单独订正了**预算数字**）；既有判据不受影响。

**需要同步更新参考的地方：没有。** 两条改造都落在同一条数值通路上且逐位不变：
`tools/golden/moe_block_ref.py`（golden）与 `check_ref.py` **均未改动**，判据口径
（ids 精确 / weights bf16 网格 ≤1 ulp / logits abs ≤2e-2）沿用。

**位级证据**（M48 实测，**M56 已归档**）：改造前/后各跑一次 m=1/2/16/33/64，`M7_OUT` 下 5 个
case 共 **50 个文件**（`x`/`router_weight`/`router_logits`/`topk_ids`/`topk_weights`
+ 各自 `.json`）逐文件 sha256 完全相同（`diff` 无输出）。

* **M48 当时**：该结论只在作者本机 `/tmp` 工作副本里比对，**仓库内没有 sha256 清单**
  ⇒ 声明只能靠源码级等价（上表两条改造都是机械替换）支撑。
* **M56 补齐**：`evidence/dump_sha256.txt` 是**用当前代码在本仓重跑**得到的 50 文件清单；
  `evidence/dump_sha256_m48_after.txt`（M48 改造后，50 文件）与 `evidence/dump_sha256_m48_before.txt`
  （M48 改造前基线，m1/m2/m16 共 30 文件）都是原样转存（只加注释头）。
  after 侧与本次重跑清单**逐行相同**，且前/后两清单在重叠的 30 个文件上**逐行相同** ⇒ 这条位级声明
  现在**可离线对读 + 可一条命令复算**（`tools/archive_evidence.sh`）。每份清单头部写明**哪一轮 /
  哪个 commit / 哪个临时路径**；「生成时未记录自身 commit」这一点如实标注（见 §"证据级别"）。

**落盘字节数变化（无功能影响）**：weights staging 原先按 64×int32 写 256B/行，现按
pack 模式写 128B/行（64 个 bf16）；被拷出的只有前 `KTOPK=10` 个（20B），行内其余字节
不参与任何判据，也不会被 `CopyOut` 读到（`DataCopyPad` 的 `blockLen=KTOPK*2`）。

## M56 修订记录（按 commit，不按轮次）

| # | 评审项 | 改动 |
|---|---|---|
| 1 | **观察 4**：`dump_sha256_m48_after.txt` 头部写了轮次 / 临时路径 / 转存方式，但**缺「哪个 commit」** | 补上：M48 分支 `feat/m48-m7-m4-m14-vector-api-compliance-swee`，落地点 `2a07218`（merge `4ba5484`）；并**如实标注**「生成 dump 时该工作副本未记录自身 commit，故这是内容等价的落地 commit，不是当时的明文版本戳」 |
| 2 | 同上（顺带加强） | 把 M48 **改造前**的基线清单也转存为 `evidence/dump_sha256_m48_before.txt`（m1/m2/m16 共 30 文件）⇒ 前/后两侧都在仓内，位级声明可**离线对读**（`comm -23` = 0 行），不必只靠 after 一侧 |
