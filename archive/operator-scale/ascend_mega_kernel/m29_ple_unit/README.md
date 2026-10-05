# m29_ple_unit —— PLE 单元级链路（M161）

> **写窄（先读这段）**：本目录是 **PLE 段体的单元级验证**，只跑 PLE 自身的
> `ids → gather → cube gemv → gate → conv/state` 这条链。
> **未接进 `m15_layer_loop/**`**、**不是生产路径**、**不算 prefill / T>1 已完成**。
> 它存在的目的（人类 Q3 逐字：「先另外开一个文件，先把单元功能打通再后续合入」）是把
> M158 survey 认定的 PLE 缺口里的两件**设备级**事做通：**真实表的多槽滑窗 + 跨分片取模**
> 与**跨 step 的 short-conv 状态搬运**；并顺手把段内同步换成合规形态。
> 合入层 kernel 需要做的事见 §8。

## 1. 交付物

| 文件 | 作用 |
| --- | --- |
| `m29_ple_unit.asc` | 单元 kernel（① ids / ② gather / ③ cube mmad / ④ gate / ⑤ conv+state）+ host harness（多 step + 表 staging） |
| `m29_ple_support.h` | 自包含 helper（BufferID/搬运/barrier 封装 + NormDonor + 形状·tile 常量），**不 include `m15_layer_loop/**`** |
| `gen_unit_data.py` | 数据生成：真实 checkpoint 权重 + id 常量 + 每 step 输入 + 表（合成 / 真实多槽 / 越窗档） |
| `m29_common.py` | 共享：checkpoint/分片读取 + **独立重写的 float64 参考数学** |
| `check_ref.py` | 判据：T1（ids/emb）+ T3（kv/gated/normed/out/state），三态退出码 |
| `reproduce.sh` | 一条命令复现（构建→数据→设备→判据→负向对照；锁纪律见 §7） |
| `CMakeLists.txt` | 独立 CMake 工程（不碰 `m15_layer_loop/CMakeLists.txt`） |
| `evidence/` | 逐档日志 + 锁内 `npu-smi` 快照 + 判据输出 |

## 2. 盘点：lift 了什么、新写了什么、为什么不改现有段体（任务 1）

### 2.1 lift（抄改来源，带 文件:行）

| 件 | 来源 | 处理 |
| --- | --- | --- |
| int64 取模 `UMod64` / `FloorMod64` | `m15_layer_loop/m15_ple.asc:277-299` | 逐字 lift |
| ① n-gram id `IdsOneToken` | `m15_ple.asc:334-408` | 逐字 lift（落盘已是 UB→MTE3 的合规形态，M111 修） |
| ③ cube 投影 `PleCubeGemv` | `m15_ple.asc:576-708` | 逐字 lift（M124 的 mmad 形态） |
| ③ 跨核握手 `PleGemv` | `m15_ple.asc:724-763` | lift（AIV/AIC 两臂的 mode-2 / mode-0 组合） |
| ④ 门控 + 分组归一 | `m15_ple.asc:766-994` | lift 数学；同步从 `PipeBarrier<PIPE_ALL>` 改成 BufferID + `PipeBarrier<PIPE_V>` |
| ⑤ 膨胀卷积 + SiLU + 残差 + 状态 | `m15_ple.asc:997-1156` | lift 数学；状态移位从 UB→UB `DataCopy` 改成 VF；同步改 BufferID |
| BufferID/搬运/barrier 封装 | `m15_layer_loop/m15_hc_layer.h:57-114` | 逐字复制进 `m29_ple_support.h` |
| NormDonor（rstd / sigmoid / cast traits） | `m15_hc_layer.h:119-215` | 逐字复制 |
| 形状·tile·L0/L1 常量 | `m15_layer_loop/m15_hc_resources.h:39-153` | 逐字复制 |
| 合规同步的**正对照** | `m15_hc_layer.h:653-703`（`CombineStage`） | 只作范式抄改（acquire→搬运→阻塞释放→对侧 acquire） |

### 2.2 新写（本单元独有的增量）

| 件 | 位置 | 说明 |
| --- | --- | --- |
| **设备端 64 位 `UDivMod64`** | `m29_ple_unit.asc:179-190` | 移位-减法一次出商+余（设备端回避 int64 除法） |
| **多槽 + 跨分片 gather** | `m29_ple_unit.asc:305-360`（`SlotRowOf` + `PleGather`） | `shard = id/2,500,012`、`local = id%2,500,012`，在槽表里找覆盖 `(shard,local)` 的槽 |
| **跨 step 状态搬运** | `m29_ple_unit.asc:1480-1490` | step k 的 `state_out` D2D 搬成 step k+1 的 `state_in` |
| **槽表 / 窗口 staging** | `m29_ple_unit.asc:1360-1420`（host） | 从真实分片抽行 → 匿名窗口 → 一次性 `aclrtHostRegisterV2(MAPPED)`；分片定位 `shard_map` 抄改自 `m15_layer_loop/ple/real_table_probe.py:54-76`（M92 口径） |
| 判定链 | `m29_common.py` / `check_ref.py` / `gen_unit_data.py` | 见 §5 |

### 2.3 为什么**新建单元目录**而不是改现有段体

- **撞车规避**：现有 PLE 段体在 `m15_layer_loop/m15_ple.asc` 与机械生成物
  `m15_layer_loop/m15_ple_wire.h`（改前者必须重跑 `evidence/ple_wire/lift_ple_device_segment.py`，
  `m15_layer_kernel.h:190-193`）。该目录正被在飞任务（M151）触碰的风险最高；单元验证放独立目录
  可以把「验证单元功能」与「合入层 kernel」两件事解耦。
- **缺口头号在 host 侧与设备侧同时**：单窗口装不下真实 token 的 16 个 id
  （`m15_layer_loop/evidence/ple_wire/INTERFACE.md:217` 的 U-E；`ple/REAL_TABLE.md:513-515` 的
  U-A/U-B）既改设备 gather、又改 host staging。把它独立成一个可跑单元，证据自洽、不牵动层路径。
- **同步改造的爆破半径**：段内 `PipeBarrier<PIPE_ALL>` 改 BufferID 会触及段体的每一处交接
  （`m15_ple.asc` 计 ~38 处）；在独立目录先把合规形态验证过（且用 `CombineStage` 作正对照），
  再把改造建议带回段体，是低风险顺序。

## 3. 设计要点

- **表建模 = 槽表**：host 把若干「行段」摆进一块注册窗口；每个槽记 `(shard, local0, rows, win)`。
  设备对每个 id 先做 64 位除/模，再线性扫槽找覆盖者，`row = 槽内窗口行 + (local - local0)`。
  跨越分片、或落在别的分片的 id，只要该分片有槽就能取到；没有槽才计 `miss`（设备侧计数落盘）。
- **合规同步**：核内跨 pipe 交接一律 `BufAcquire/BufRelease`（阻塞释放，`m29_ple_support.h:96-105`）；
  同 pipe 背靠背用 `PipeBarrier<PIPE_V>`；核间用 `CrossCoreSetFlag/WaitFlag`。
  **UB 按段分域**（gather / gate / conv 三段地址不相交），所以段与段之间不需要任何 UB 复用屏障 —— 这是把
  「必须用 PIPE_ALL 兜底」这条理由拆掉的构造性做法。
- **计算路径**：矩阵乘（③）走 cube `Mmad`；④⑤ 全部 VF；① 的 id 算术与 ② 的 `shard/local` 换算是
  **gather 索引/地址**（`docs/21-scalar-computation-sweep.md` §3.2 的 I 类），scalar 只承担它们与循环控制。
- **③ 的 M 维**：本单元 `n_tok = 2`（`gen_unit_data.py:44`），③ 走 `BASE_M` tile、`calcM = max(curM,2)`
  （3510 契约：Nd2Nz 行数为 1 时退化，`m29_ple_unit.asc:379`）。

## 4. 档位与规模

| 档 | 表 | 说明 |
| --- | --- | --- |
| `synth`（`data_synth`） | 合成缩减表 1470 行（16 素数）| flat 模式，全 32 个 id 命中；验证整链数值 |
| `real`（`data_real`） | **真实分片抽行**：64 槽 / 256 行 / **54 个分片** | 多槽 + 跨分片全覆盖（`miss=0`） |
| `real_miss`（`data_real_miss`） | 同真实表，**只 stage 1 个分片** | 越窗档：其余分片 id 记 `miss`，covered 行仍 T1 正确 |

每档 `n_tok=2`（2 请求各 1 token）、`n_steps=2`（跨 step 状态）。真实 ids 的 `shard` 分布见
`gen_real.log`；`real` 档的槽覆盖 54 个分片（`evidence/` 保留读数）。

> **跨 step 状态的可观测性（写窄）**：`n_steps=2` 且卷积抽头是 lag 9/6/3/0（用状态行 0/3/6）时，
> step0 只把新样本写进状态行 8 ⇒ step1 的 `out` **不依赖**被搬运的状态行。因此跨 step 状态目前
> **只由 `B5.state.*` 判据（及 `neg_state` 变红）见证，`out` 端不咬合**；要让 `out` 也依赖，需
> ≥3~4 step 才轮转到被搬运的行。

## 5. 参考与判据（任务 5）

- **参考的来源与独立性（写准）**：`m29_common.py` 的 `ref_ids` / `ref_gate` / `ref_conv_out` 是
  `m15_layer_loop/m15_ple_check.py` 对应函数的**同源转录（correlated transcription）** —— 逐字相同
  （仅注释/docstring 有差），规则本身钉在仓外 `qwen4_exp/nvidia/ops/ple.py` 的三个 kernel 符号上。
  **风险**：同一处转写错误会两边一起错。**覆盖它的独立证据**（不复用这两份的对拍）：① 真实档 `emb`
  走 `direct_row` **直读分片文件**（不经窗口、不经 m15）；② `docs/17` §1.2 的「参考完美输出喂进
  判据」非空洞见证（`evidence/r1/` 与复审自跑）；③ 出处符号级对照（`m29_common.py` 抬头）。
- **T1**：`ids`（逐位）、`emb`（逐字节；真实档与**直读分片文件**比对，独立于窗口）。
- **T3**：`|out−ref| ≤ ε·grid + 1.0·ulp(out)`（参考已量化到 bf16 格点，取 1.0·ulp）。`grid` 分两种：
  - `B3.kv` 用 **`Σ|terms| = |wcat|·|emb|`**（`docs/17` §1.1 的 T3 原式；与
    `m15_ple_check.py:492-503` 的 `cmp_t3_masked(extra_terms=Σ|terms|)` 同口径）—— 真实表行 +
    真实权重下 GEMV 会相消（`|Σterms| ≪ Σ|terms|`），代理会偏紧。实测
    `Σ|terms|/max(|out|,|ref|)` 的中位/max 随判据打印在 `evidence/r1/*/check.log` 的 `B3.kv` 行
    （`Sum|terms|(median/max ratio=...)`）。
  - `gated/normed/out` 用**代理** `max(|out|,|ref|)`（`m15_ple_check.py:500-502` 已就该代理在
    真实 GEMV 上的脆弱性留过警告；这三档无 GEMV 相消）。`check_ref.py` 把所用 grid 标在
    `grid=proxy=...` / `grid=Sum|terms|(...)` 里，口径不再与实现脱节。
- **ε 来源**：`kv` 的 `ε_MMAD = 2560·2^-24`（K=2560 的 mmad 累加，`docs/17` §1.1 触发条件 ③）；
  `gated/normed/out` 的 `ε = ε_RSQRT + ε_SIGMOID`（NR rsqrt 2 步 + Exp 官方精度，`m29_common.py`）。
  逐元素检查，报 `max(d/bound)` 作报告项。
- **负向对照（必须变红，任务 5 的 ≥2 条；r1 起含多槽路径一条）**：
  - `M29_NEG_TABLE=1`：表基址偏 1 行 ⇒ `B2.emb` 必红（实测 5116/5120 元素不符）。
  - `M29_NEG_STATE=1`：不搬跨 step 状态 ⇒ step 1 的 `B5.state` 必红（实测 20480/184320 不符）。
  - `M29_MUT=16384`（kernel bit14，**多槽/跨分片路径**）：真实表 + 槽内行偏移 +1 环绕
    ⇒ `B2.emb`/`B3.kv` 必红、`B2.miss` **不变**（证明红的是「行映射」而不是「覆盖」）——
    见 `evidence/r1/neg_slot_real/check.log`。

## 6. 设备验证与读数（任务 6）

每档各自进一次 `flock -w 300 /tmp/npu0.lock`，进锁先 `npu-smi`（快照落盘），`timeout 180` 在锁内；
失败则 `reproduce.sh` 以非 0 退出。逐字命令与读数见 `evidence/COMMANDS.md`；**r1 定稿批次**（含 P2-1
的判据式修正与多槽负向对照）`reproduce.sh` 退出码 **0**，编排 stdout 与逐档日志在 `evidence/r1/`
（`evidence/r1/reproduce_stdout.log`）；被审 tip `f7f9096` 的原批次读数保留在 `evidence/{synth,real,...}`。

| 档 | 设备读数 | 判据 |
| --- | --- | --- |
| `synth` | `dev_oob=0 dev_miss=0`（2 step 各 2 tok） | 16 条全 PASS，`VERDICT=OK` |
| `real` | 64 槽 / 256 行 / 54 分片；`register rc=0`，`dev_miss=0` | 16 条全 PASS，`VERDICT=OK`（`evidence/r1/real/check.log`） |
| `real_miss` | 只 1 槽；`step0 miss=31`、`step1 miss=32` | covered 行 `B2.emb` 逐字节过 + miss 计数相符，`VERDICT=OK` |
| `neg_table`（表基址偏 1 行） | `dev rc=0` | `B2.emb.s0` 5116/5120 不符 ⇒ `VERDICT=FAILED`，check rc=1（须红） |
| `neg_state`（不搬状态） | `dev rc=0` | `B5.state.s1` 20480/184320 不符 ⇒ `VERDICT=FAILED`，check rc=1（须红） |
| `neg_slot_real`（**多槽行映射错**，`M29_MUT=16384`，真实表） | `dev rc=0` | `B2.emb` 5116&5117/5120、`B3.kv` 25399&25480/25600 不符，`B2.miss` 仍 0/0 ⇒ `VERDICT=FAILED`，check rc=1（须红） |

`real` 档 `B3.kv` 的 `max(d/bound)`：**s0=0.033、s1=0.410**，`grid=Sum|terms|`、中位/max ratio
s0=32/1.34e6、s1=35.6/4.42e5（改前用 `max` 代理时是 0.974/0.973 —— 贴红界；见 `evidence/R1_FIXES.md`）。

**两件证据分开**（任务 3 要求）：①「单元能取到正确行」= `real` 档的 `B2.emb` 与**直读分片文件**
逐字节一致（5120/5120），与窗口无关；②「能覆盖任意 id」= `real` 档 `miss=0`（64 槽覆盖真实 token
的 16 个 id，落在 54 个不同分片上）；越窗行为由 `real_miss` 档的 `dev_miss` 计数证实（它正是旧单窗口
路径会出现的现象 —— `m15_layer_loop/evidence/ple_wire/INTERFACE.md:217` 的 U-E：真实词表下 `miss=16/16`）。
多槽路径的**红档**由 `neg_slot_real` 提供（`B2.miss` 不变 ⇒ 咬的是行映射不是覆盖）。

环境：Ascend950PR；`npu-smi 25.7.rc1.10`；aic=28 / aiv=56；CANN 9.1.0（快照见各档 `npu_smi.txt`）。

## 7. 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash m29_ple_unit/reproduce.sh /tmp/m29_repro      # 退出码 0 = 全部符合预期
```

## 8. 合入 `m15_layer_loop` 需要做的事（后续，不在本单元）

1. **设备 gather 换成多槽版**：把 `m29_ple_unit.asc` 的 `SlotRowOf`/`PleGather` 形态带回
   `m15_ple.asc:441-569`，并重跑 `evidence/ple_wire/lift_ple_device_segment.py` 更新
   `m15_ple_wire.h`（`m15_layer_kernel.h:190-193` 的生成纪律）。
2. **host staging 做多槽/驻留**：`m15_layer_loop.asc::H_PleWindowSetup`（`:2696`）当前只做
   单窗口单分片（`:2762` 附近）；需换成槽池 + 按 id 集合预取（`docs/14 §10` 第 1 条阻塞项，属
   `REAL_TABLE.md §9 U-A` 的 host 侧管理）。本单元只演示「槽表 + 跨分片取数」，**不做 LRU/驻留策略**。
3. **跨 step 状态接生产**：`m15_layer_loop.asc` 需把 `S_STOUT→S_STIN` 逐 step 搬运（G2）。
4. **同步合规回灌段体**：把 `m15_ple.asc` 段内 `PipeBarrier<PIPE_ALL>`（~38 处）按本单元的
   三段分域 + BufferID 形态改造，并做 `docs/20` 的 B10 扫尾。
5. **prefill / T>1**：本单元 `n_tok=2` 是「2 请求各 1 token」，**不是** T>1 的 prefill；prefill 的
   short-conv 写回（上游 `ops/ple.py:489-569`）仍未做。

## 9. 合规自查

- 段内无 PIPE_ALL 档屏障（`PipeBarrier<PIPE_V>` 仅用于同 pipe 背靠背）；无 `SetFlag/WaitFlag` 系列；
  核间用 `CrossCoreSetFlag/WaitFlag`（`m29_ple_unit.asc` 的 §FLAG）。
- 矩阵乘走 cube；④⑤ 走 VF；scalar 只做控制流与 gather 索引。
- 结论带 文件:行；新增文字不含禁用断言词。
