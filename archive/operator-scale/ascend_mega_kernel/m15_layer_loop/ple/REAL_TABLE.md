# M92 — 真实 ngram 表的落地形态（规格钉死 + 真实 checkpoint 取证 + 设备侧 host-mapped gather）

> 分支 `feat/m92-ple-ngram-real-table-host-mapped-gat`（**本文写作时未合入 `main`**；按 `docs/17 §9.6`
> 引用本文件时必须标注分支与状态）。
>
> 本文是 **task 1 的交付物**（落地形态规格 + 取证），也是 task 2/3 的实现说明与判据读数。
> 前置事实来自 **M86**（`probe_host_dma/**`，分支 `feat/m86-ngram-mmap-and-host-memory-device-re`，
> 复审 `p2-2items / tip c126bb6`；**本文写作时未合入 `main`，其后已随 `950a85a` 合入
> `main`** —— 本 branch 现已 rebase 到该 main 上，故 `probe_host_dma/**` 的引用路径在 main 上可直接核）：
> file-backed mmap 直接注册被拒 `rc=507899`、host-mapped 窗口设备可逐字节读、注册代价随 size 增长且方差极大。
>
> **立项依据（U2 的阻塞已解除）**：M85 §7 记「95.37 GiB 真实表卡在 `docs/14 §10` 第 1 条（阻塞，未取证）」；
> 那条阻塞正是 M86 解掉的（能力已由 M86 实测证实）⇒ 本 mission 把 U2 从「阻塞」推进到「有落地形态 +
> 真实规模已跑过一次」。

---

## 0. TL;DR

| 项 | 结论 | 读在哪 |
|---|---|---|
| 表几何 | 128 分片 × 2,500,012 行 × 160 列 × 2 B = 320,001,536 行 = 102,400,491,520 B = 95.37 GiB | §2 |
| 行 id → 文件位置 | `shard = id // 2500012`、`local = id % 2500012`、`file_off = data_off[shard] + local*320`，`data_off` 由各分片 safetensors 头解析 | §2 + §3 |
| 逐字节取证（真实 checkpoint） | **8 个真实行 id**（含 0/1/2/12345/片边界 2500011→2500012/末行 320001535）的 320 B 与**直读分片文件**逐字节一致：0 不符 | §3 |
| 真实规模读数 | **整片 shard_0 的完整行集合**（2,500,012 行 = 762.94 MiB）：一次搬运 800,003,840 B，抽样 4104 行逐字节 0 不符；host 墙钟单次 1.693 s（**非带宽证据**） | §4.1 |
| 落地形态 | file `mmap`（按需页）→ **匿名 staging 窗口**（`mmap(ANON)`）→ `aclrtHostRegisterV2(ACL_HOST_REG_MAPPED)` **一次** → `aclrtHostGetDevicePointer` → 设备 `DataCopy` 读窗口 | §4.2 |
| 窗口大小/驻留 | 规格按 M86 的 4 × 256 MiB = 1 GiB 池；本 mission 实跑用 **1 槽 × 256 MiB**（理由见 §4.3） | §4.3 |
| 设备侧接入 | ② 的 `PleGather` 支持窗口模式：`row = id - winBase`；入口符号仍是 M85 的两个；`M15_PLE_HM=1` 打开（默认关 ⇒ 既有 13 条判据路径一字不改） | §5 |
| 设备侧判据（补 M85 U6） | ① `Hd.row_fail`：kernel 自己把「刚取到的行前 64 元素」与 host 直读期望比、**错行计数**落 GM；② `Hd.miss`：越窗行计数落 GM；③ `Hd.cores`/`Hd.nonvac`：**非空洞守卫**（56/56 核自报参与） | §6 |
| 判据现状 | host 侧 **10 条 / 0 FAIL**（`ple/logs/hm_run.log`），其中 T1 逐字节 1 条（`H2.emb`，163,520 元素）、T3 4 条、结构性 5 条 | §6 |
| 负向对照 | `M15_PLE_MUT=16384`（id→窗口行号**方向反转**）：`H2.emb` **163373/163520 FAIL**、**设备侧 `Hd.row_fail` = 1022 FAIL**；共 6 条 FAIL（`ple/logs/hm_neg.log`） | §7 |

---

## 1. 为什么必须「mmap 按需页 + 一次性注册的 staging 窗口」

三条硬约束（都是实测，不是估计）：

| # | 事实 | 读数 / 出处 |
|---|---|---|
| C1 | 表**装不进 host 内存** | 表 102.40 GB；本容器 cgroup `memory.max = 34359738368`（32 GiB）⇒ 差 **2.98×**（`cat /sys/fs/cgroup/memory.max`） |
| C2 | 表**装不进 HBM 常驻预算** | HBM 131072 MB，M88/M86/M91 等并行进程已占 ~13 GB；表 = 102.4 GB > 128 GiB 的绝大部分 |
| C3 | 磁盘装得下 | `/` 可用 406 GB（`df -h`），表占 **25%** |

⇒ 只能**磁盘驻留 + 稀疏行查找**。而设备**不能直接读 file-backed mmap**：
M86 在 `file` / `filerd` / `filepriv` 三种 mapmode 上 `aclrtHostRegisterV2` 全部被拒 **`rc=507899`**
（M86 README §4.2 + `evidence/logs/filemmap_*.log`，跨 3 批稳定）。

⇒ 必须中转：**file mmap 出按需页 → 把需要的行搬进一块匿名内存 → 那块匿名内存一次性注册给设备**。
设备只读注册过的匿名窗口（M86 实测三种匿名来源 `mmap(ANON)` / `posix_memalign` / `aclrtMallocHost`
都 `rc=0` 且逐字节读对，AIV `DataCopy` 5 次独立进程 21.73–22.52 GB/s）。

---

## 2. 落地形态规格（task 1 的"钉死"部分）

### 2.1 分片与文件布局

checkpoint 张量名：`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`。
每个分片是一个独立张量，**BF16 `[2500012, 160]`**，落在 131 个 `model-*.safetensors` 里
（不是一片一文件：实测 `shard_0`/`shard_1` 同在 `model-00005-of-00131.safetensors`，
**`shard_2..5` 同在 `model-00006`**，`shard_127` 在 `model-00037`；**128 片的完整映射由
`real_table_probe.py` 的 `shard_map()` 逐片解析并断言 128/128 全在**，
本文只举这几例作说明）。

safetensors 布局 = `8 B 小端头长` + `对齐后的 JSON 头` + 数据段：

```
base      = 8 + len(JSON header)                     # 数据段起点
data_off[shard] = base + header[tensor]["data_offsets"][0]    # 该分片数据在**文件**里的绝对偏移
```

分片行数 `2,500,012` 的解释：`ple/PLE_SPEC.md §2.1` 的 padding 公式
`padded = ceil(total / divisor) * divisor`，`total = Σ size[h] = 320,001,446`、
`divisor = make_ngram_vocab_size_divisible_by = 128` ⇒ `padded = 320,001,536` ⇒
每片 `320001536 / 128 = 2,500,012`（与 checkpoint 分片形状**逐值相符**）。

### 2.2 行 id → 文件偏移（本文的核心换算）

```
shard    = id // 2,500,012          # 0..127
local    = id %  2,500,012          # 片内行号
file_off = data_off[shard] + local * 320      # 320 = 160 列 × 2 B(bf16)
```

`id` 是**全局行 id**（`ids[t,g] = (mixed mod size[g]) + offset[g]`，`PLE_SPEC.md §4.1 ①`），
不按片索引。128 片按 shard id 顺序拼成一条 `[320001536,160]`。

本文件的 `file_off` 值**全部由 `ple/real_table_probe.py` 在真实文件上算出并读回验证**（§3）。

### 2.3 staging 窗口的形态（设备看到的东西）

窗口对设备呈现为一个**平坦的 `[winRows, 160]` bf16 行数组**：

* `winBase` = 窗口第 0 行对应的**全局行 id**（本 mission 取 `winBase = shard * 2500012`，即窗口落在**一个分片内部**）；
* 设备侧行号 `row = id - winBase`，`id ∉ [winBase, winBase+winRows)` ⇒ **越窗**（设备侧计数，不读内存）；
* 窗口内容由 host 按 §2.2 的换算从分片文件搬入 —— 换算本身在 **host**，设备只做 `id - winBase`。

> 为什么设备不做 `id // 2500012`：窗口是**连续行区间**，`id - winBase` 与 `id % 2500012` 在
> 单分片窗口内等价，且省掉设备端 64 位除法（本文件另有一处 64 位取模 `UMod64`，是移位-减法实现，
> 成本高）。跨分片的一致性由 host 搬运时保证，并由 §3 的逐字节判据覆盖。

### 2.4 行宽与对齐

* 行宽 320 B = **32 B 的整数倍（10×）** ⇒ 行首天然满足 `DataCopy` 的 32 B 粒度要求，
  既不需要 `DataCopyPad`、也不需要在窗口里插 padding；
* 每行 160 bf16 = **2.5 个 64-lane 寄存器**（VL=64）⇒ 设备侧「整行取」用
  `DataCopy(rowL, win[row*160], Block1(320))`；「行首 64 元素」用 `Block1(128)`（整寄存器，无尾部掩码问题）。

---

## 3. 真实 checkpoint 取证：行 id → 偏移 → 逐字节

命令（本 tip 实跑，完整转录 `ple/logs/`；本节的输出是 `--evidence` 的 stdout 转录）：

```
/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/real_table_probe.py --evidence
```

`--evidence` 的判定量：对每个真实行 id，**窗口 gather**（file mmap → 匿名窗口 → 从窗口取 320 B）
与**直读该分片文件**（`seek(data_off + local*320); read(320)`）逐字节比。

```
global_id      shard  local      file(切片)                          file_off      结果
0              0      0          model-00005-of-00131.safetensors    68329314      byte-identical
1              0      1          model-00005-of-00131.safetensors    68329634      byte-identical
2              0      2          model-00005-of-00131.safetensors    68329954      byte-identical
12345          0      12345      model-00005-of-00131.safetensors    72279714      byte-identical
2500011        0      2500011    model-00005-of-00131.safetensors    868332834     byte-identical
2500012        1      0          model-00005-of-00131.safetensors    868333154     byte-identical
2500015        1      3          model-00005-of-00131.safetensors    868334114     byte-identical
320001535      127    2500011    model-00037-of-00131.safetensors    1600092312    byte-identical
结论：PASS —— 上述 8 个真实行 id 的 320 B 与直读分片文件逐字节一致
```

逐行 sha256（320 B）也在同一 stdout 里（`ple/logs/real_table_probe_evidence.log`）：
`id=0 → 67354d4be4c2d2d8ed2f6003c5e98e56…`、`id=320001535 → 5abe266cc3652a4944a86cc91dae0209…`。

**为什么这 8 个 id 够用（覆盖了换算的每个分支）**：

| id | 覆盖的分支 |
|---|---|
| `0` / `1` / `2` | 片 0 的头部（分片起点） |
| `12345` | 片内任意行 |
| `2500011` | **片 0 的最后一行**（`local = rows_per_shard - 1`，行尾贴数据段末端） |
| `2500012` | **跨片边界**（shard 0 → 1，且 `local` 回绕到 0） |
| `2500015` | 片 1 片内（跨文件内偏移正确） |
| `320001535` | **全表末行**（shard 127，`local = rows_per_shard - 1`） |

**未物化整表**：全程只有「file mmap 的按需页 + 4 MiB 匿名窗口」两个映射；
`--evidence` 的 `bytes_moved = 2560`（8 行 × 320 B）—— 这正是"按需"的意思。

---

## 4. 真实规模读数与窗口策略

### 4.1 真实规模读数（整片 shard_0 的完整行集合）

命令（本 tip 实跑，转录 `ple/logs/real_scale.log`）：

```
/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/real_table_probe.py --real-scale --shard 0 --samples 4096
```

| 项 | 读数 |
|---|---|
| 行数 | 2,500,012（= 该分片的**完整行集合**） |
| 数据段 | 800,003,840 B = 762.94 MiB = 0.745 GiB |
| 搬运字节 | **800,003,840 B**（file mmap 按需页 → 匿名 staging 窗口，一次 `memcpy`） |
| 耗时口径 | **host 墙钟、单次进程、单次搬运**：**1.693 s** ⇒ 0.47 GB/s。⚠ **不是带宽证据**（`docs/17 §9.1`：host 墙钟不构成证据）；此处只作"真实规模真的跑过一遍"的证据 |
| 采样次数 | 抽样 **4104** 行（含边界 0/1/2/3/2500011 + 4096 随机）与直读文件逐字节比 |
| 不符行数 | **0** |

> 抽取口径：`--samples 4096`、`np.random.default_rng(20260927)`；边界行固定加入。
> 抽样比对耗时 0.024 s（host 墙钟）。

### 4.2 设备侧实测：注册一次 + 设备读窗口

| 步骤 | 读数 | 出处 |
|---|---|---|
| file `mmap` → 匿名窗口 `memcpy`（256 MiB） | **76.0 / 80.9 ms**（两次独立进程，host 墙钟单次） | `ple/logs/hm_run.log` / `ple/logs/hm_neg.log` |
| `aclrtHostRegisterV2(win, 268435456, ACL_HOST_REG_MAPPED)` | **`rc=0`**；耗时 **8350.1 / 4132.7 ms**（两次独立进程，host 墙钟单次） | 同上 |
| `aclrtHostGetDevicePointer` | `rc=0`，`win=0x7f5950000000 → dev=0x40000000000` | 同上 |
| 设备 `DataCopy` 从该窗口读行 | 逐字节与 host 直读一致（`H2.emb` 163,520 元素 bad=0） | §6 |

**注册代价的方差**（本处两次 4.1–8.4 s，M86 的 regbench 256 MB 五次独立进程为
2.2 / 23.9 / 23.9 / 38.3 / 41.0 s）
与 M86 §3.6 的结论一致：**方差盖过尺寸效应 ⇒ 池大小必须在目标机上实测确定**。
这条读数进一步支持"**一次注册、反复复用，绝不按次注册**"。

### 4.3 窗口大小与驻留策略

**规格（按 M86 §4.3 的结论）**：窗口池起点 **4 × 256 MiB = 1 GiB**，`aclrtHostRegisterV2(MAPPED)`
**只在启动时注册一次**，之后反复复用；staging 以「文件内对齐的块」（256 MiB）搬，行落在块内偏移。
理由（M86 实测）：① 注册不额外增常驻（RSS delta = 0），池的常驻成本 ≈ 池大小，1 GiB 只占 cgroup 3.1%；
② 注册代价随 size 增长且方差极大 ⇒ 不能按次注册；③ 设备单侧读带宽约 20–22.5 GB/s，
搬 256 MiB 约 11.4 ms 量级。

**本 mission 实跑用 1 槽 × 256 MiB**，理由是**采集期的时间预算**：
本次两次 256 MiB 注册耗时 4.1 s / 8.4 s，而 M86 观测到的上界是 41 s/次 ⇒ 4 槽的最坏情形
可能吃掉 2–3 分钟纯注册时间，而单卡上还有 M88/M91/M86 在并行（并发背景见 §8）。
代码本身把窗口数做成参数（`ple/real_table_probe.py --win-mib`、harness 读的 `win_bytes`），
**槽数不是本 mission 要钉的结论**；本 mission 钉的是"**一次注册 + 设备只读窗口**"这条链路与判据。

**驻留策略（本 mission 实际状态，如实写）**：本 mission **未实现 LRU/多槽滑窗**——
窗口是一次填满、整段跑完的（窗口内 `winRows = 838,860` 行，全部 1024 个 item 都在窗内）。
多槽/滑窗属显式未完成项，见 §9 U-A。

---

## 5. 设备侧接入（task 2）

### 5.1 接在哪、怎么接

* **入口符号不变**：仍是 M85 的 `m15_ple_ids_kernel`（①）与 `m15_ple_body_kernel`（②③④⑤）；
  变的是 `m15_ple_body_kernel` 的**参数表**（新增 `exp64In` / `hmBadOut` / `winBase` / `winRows` /
  `stateSlots`）。
* **② 的 gather 扩展**（`m15_ple.asc` 的 `PleGather`）：`G.winRows != 0` 时走窗口模式
  `row = id - winBase`（+ 越窗判定）；`G.winRows == 0` 时**走原来的平坦 GM 表**，
  于是 M85 的 13 条判据行为**一字不改**（本 tip 复跑 13/13 PASS，`ple/logs/run_all.log`）。
* **表基址 = 注册窗口的 device 指针**：host 把 `aclrtHostGetDevicePointer` 拿到的指针直接当
  `tableIn` 传给同一个 kernel（`SetGlobalBuffer`）；**没有**任何 AscendC 资源管理对象参与。
* **同步约定沿用 M85**：核间仍是 `M15H::BarrierAiv<PIPE_MTE2, FLAG_B2/B3/B4>` 的
  `CrossCoreSetFlag/WaitFlag` mode-0；核内仍是 `PipeBarrier<PIPE_ALL>`。
  **没有** `SetFlag/WaitFlag` 系列、**没有** `TPipe/TBuf/TQue/AllocTensor`。
* **buffer 自管**：窗口地址由 host 管理（注册/解注册），UB 地址仍是编译期常量
  （新增 `UB_HMCHK=52224` / `UB_HMEXP=52352` / `UB_HMACCL=52480`，`UB_END=52512` < 248 KB），
  cross-core flag id 仍是本文件自定义的 `FLAG_B2/B3/B4 = 12/13/14`。

### 5.2 host 侧链路（`RunBodyMapped`）

```
1) 读 ple/data_hm/hm_meta.txt（shard / rows_per_shard / r0 / win_rows / win_bytes / file / data_off）
2) ::open(file) + ::mmap(MAP_PRIVATE, PROT_READ, page-aligned)        # 按需页
3) ::mmap(nullptr, win_bytes, MAP_PRIVATE|MAP_ANONYMOUS)              # 匿名 staging 窗口
4) memcpy(win, fmap + skew, win_rows*320)                             # 按 §2.2 的偏移搬行
5) aclrtHostRegisterV2(win, win_bytes, ACL_HOST_REG_MAPPED)           # ★ 只做一次
6) aclrtHostGetDevicePointer(win, &dev, 0)                            # ★ 设备地址
7) m15_ple_body_kernel<<<aic,0,stream>>>(..., (uint8_t*)dev, ..., winBase=r0, winRows=win_rows)
8) sync → 回读 H_*.bin（判据用）+ H_hm_dev.bin（设备侧计数原始落盘）
9) aclrtHostUnregister(win); ::munmap(win, win_bytes)
```

### 5.3 harness 接口

```
python3.12 m15_layer_loop/ple/real_table_probe.py --emit --shard 0 --win-mib 256 --tokens 64   # 生成 ple/data_hm/
M15_PLE_HM=1 M15_PLE_OUT=<outdir> ./m15_layer_loop/build/m15_ple                              # 跑设备（默认 data 目录切到 ple/data_hm）
python3.12 m15_layer_loop/m15_ple_check.py <outdir>                                            # 判据
env: M15_PLE_HM_IDS=<name>（默认 ids_hm.bin）换 ids 文件（越窗对照用 ids_hm_miss.bin）
```

`ple/data_hm/` 的生成物**不入库**（`ple/.gitignore`）；其中 `w_*` 六个 PLE 真实小张量与
`ple/data/` 同源（`gen_ple_data.py` 是它们的唯一来源，`--emit` 只复制）。

---

## 6. 判据（task 3 的设备侧判据 + task 4 的分档）

### 6.1 判据表（`ple/logs/hm_run.log`，基线 **10 条 / 0 FAIL**）

| # | 判据 | 档 | 量（元素） | 基线读数 | 咬谁 |
|---|---|---|---|---|---|
| 1 | `H2.emb.nonvac` **非空洞守卫** | T1-struct | 163520 | 对照行非零元素 163520/163520 | 整片为 0 ⇒ `H2.emb` 是 0 vs 0 空过 |
| 2 | `H2.emb` **真实表行 gather 逐字节** | **T1** | 163520 | **bad=0**（对照 = host 直读分片文件） | id→窗口行号换算（方向/基址）、head 落点、行宽 |
| 3 | `H3.kv` | T3（界 = **Σ\|terms\|**） | 806400 | bad=0 | key/value 顺序、GEMM 转置；同时是 ② 的下游证据 |
| 4 | `H4.gated` | T3（ulp×2） | 645120 | bad=0 | query 用错、4 流共享 value、`(1+w)` 漏 1 |
| 5 | `H4.normed` | T3（ulp×2） | 645120 | bad=0 | 同 + 分组归约 |
| 6 | `H5.out` | T3（ulp×2） | 645120 | bad=0 | taps lag、SiLU、残差顺序 |
| 7 | `Hd.cores` **设备侧非空洞守卫** | T1-struct | 56 | 设备自报参与核 **56/56**；设备侧读到的 `winRows=838860` | 核没跑 ② / `winRows` 没传到核里 |
| 8 | `Hd.row_fail` **设备侧错行计数** | T1-struct | 1022 | 设备侧计数 **0**（1022 个窗口内 item 都真的比过） | 设备取到的行 ≠ 该 id 应有的行 |
| 9 | `Hd.miss` **设备侧越窗计数** | T1-struct | 1024 | 设备侧 **2** = host 独立算出的越窗 item **2** | 窗口范围参数（`winBase`/`winRows`）与注册长度不符 |
| 10 | `Hd.nonvac` | T1-struct | 1024 | 窗口内 item 1022（>0）；参与核 56（=2×AIC） | 窗口内没 item 或没核跑 ② ⇒ #8 是空过 |

**task 3 的落地（覆盖 M85 §7 U6）**：M85 的设备侧判定量只有 `dev_range_fail`（① 的值域）与
`dev_fail`（只数 null 行）⇒ ②③④⑤ 全靠 host 参考咬。本 mission 新增：
**⑦⑧⑩**（设备侧自报参与核 + **错行计数** + 其非空洞守卫）与 **⑨**（越窗行计数）。

设备侧判据的实现（`m15_ple.asc` 的 `PleGather`）：

```
对每个 item（= t*16+g）：
  id = ids[item]
  越窗（id ∉ [winBase, winBase+winRows)） ⇒ miss++
  否则 row = id - winBase，从**注册窗口**取 320 B 行 → 写 emb
  再把「刚取到的窗口行的前 64 个 bf16」与 `exp64[item]`（host **直读分片文件**得到的同一行）
  逐元素相减、平方、Reduce→SUM，非零则累计器 +1
段末：累计器（错行数）与 miss 经 UB → GM 的 **DataCopy** 落盘（每核 1 槽 8×fp32）
```

**`exp64` 是按 item 索引的（不是按窗口行号索引）** —— 这是刻意的：设备用自己的算式
（`id - winBase`）选行，host 用**另一条路径**（`id // rows_per_shard` + 文件偏移）给出该 item
应有的行。两条路径不共用中间量 ⇒ 设备行选择一错，计数必然响（§7 的负向对照证明它确实响）。

### 6.2 判据本身的两处口径修正（真实数据暴露）

> 这两条只在**真实表行 + 真实权重**上暴露；M85 的合成数据上 `max(d/bound)` 都是 0.000。
> 本节（与 §6.3）**每一个数字**都由 `ple/recount_hm.py` 重算给出 —— 它是**只读的诊断脚本**
> （不下 PASS/FAIL、不产出 `RESULT|` 行），口径与 `cmp_t3_masked` 同（同样的 mask、同样的 `bf_ulp`）。
> 命令（仓库根目录，基线 / 负向各跑一次），转录 = `ple/logs/hm_recount.log`：
>
> ```
> python3.12 m15_layer_loop/ple/recount_hm.py m15_layer_loop/ple/out_hm
> python3.12 m15_layer_loop/ple/recount_hm.py m15_layer_loop/ple/out_hm_neg
> ```

1. **`H3.kv` 的 T3 界必须用 Σ|terms|，不能用 `max(|out|,|ref|)` 代理**。
   实测（806,400 元素）：代理界下 **6 个越界、`maxRel = 2.821`、`max(d/bound) = 94.43`**；
   这 6 个元素处在**相消区** —— `|out| / Σ|terms| ∈ [4.85e-09, 7.05e-07]`（比尺度小 6–8 个数量级），
   `d / (ε·Σ|terms|) ≤ 1.04e-03`，即比教科书前向界**小三个数量级** ⇒ 被打红纯粹因为代理界
   拿"相消后的量级"当尺度。换成 `Σ|terms|`（`|wcat| @ |emb|` 逐 token 算）后
   **bad = 0、`max(d/bound) = 0.993`**，与 `docs/17 §1.1` 的 T3 原式
   （`|out−ref| ≤ ε·Σ|terms| + 1.0·ulp`）一致。
   M85 的 `cmp_t3` 里那行注释自己写了"Σ|terms| 的**代理**上界：结果量级"，故这是代理失效而非实现错。
   **本 mission 不改 M85 的 `cmp_t3`**（改它会动 M85 那 13 条判据的口径）——
   HM 案用自己的 `cmp_t3_masked(extra_terms=...)`。

2. **输出是某个 bf16 中间量的乘/加时，格点项要取 `2.0·ulp`**。
   实测（`ulp` = `bf_ulp(out)`，逐元素）：

   | 判据 | 与参考逐位不同 | `d/ulp` 分布 | `1.0·ulp` 界：bad / `max(d/bound)` | `2.0·ulp` 界：bad / `max(d/bound)` |
   |---|---|---|---|---|
   | `H4.gated` | 62 | `1.0×50, 2.0×12` | **12** / 1.9978 | 0 / 0.9994 |
   | `H4.normed` | 85 | `1.0×66, 2.0×18, 1.5×1` | **19** / 1.9979 | 0 / 0.9995 |
   | `H5.out` | 2 | `1.0×2` | 0 / 0.9991 | 0 / 0.4998 |

   ⇒ **`19` 是 `H4.normed` 在 `1.0·ulp` 界下越界的元素数**，其中 **18 个恰好差 2 个格点、
   1 个差 1.5**；`H4.gated` 在 `1.0·ulp` 界下越界 **12 个**，**全部恰好差 2 个格点**。
   取 `2.0·ulp` 后两条都 bad=0。
   推导：`out = bf16(g·v)`；若门控标量 `g` 被允许差 1 个格点（`ulp(g)`）—— ④ 的两个输入
   （参考链的 `kv_ref` 与设备的 `kv_dev`）本身相差 ≤ 几个格点，`g` 的偏差可达这一档 ——
   则 `g·v` 与 `g_ref·v` 相差 ≈ `ulp(out)`，再经输出自己的 bf16 量化最坏到 **2 个格点**。
   ⇒ `ulp_mult=2.0`，推导写进 `cmp_t3_masked` 的 docstring。
   **系数是统一的、不是逐条往上抬**：同一个 `2.0` 用在 `H5.out` 上只吃到 `0.4998`（没吃满）。
   **残余（非阻塞）**：`H4.gated` / `H4.normed` 在本数据上裕度只剩 **0.06% / 0.05%**
   （`2.0·ulp` 界下 `max(d/bound) = 0.999441 / 0.999480`）⇒ 换输入时要重核这一档，
   **不能**当成"恒定安全"（机理与 §6.3 的端到端口径同源）。

两条都只影响 **HM 案**新写的判据；`B1..B5` 的 13 条口径与读数不动。

### 6.3 判据的输入三分（`docs/17 §1.3`）

`check_hm` 的**参考链是纯 host 侧的**：真实分片文件（直读 320 B）→ 行 → `emb_ref` → `kv_ref`
→ `gated_ref`/`normed_ref`/`out_ref`。设备产物**只**出现在比较的**左侧**（`H_*.bin`），
以及 `H_hm_dev.bin`（设备侧计数器**自身**的落盘，那是被判的量）。

| 参考（判据） | 吃的输入 | 类别 | 依据 |
|---|---|---|---|
| `H2.emb` | `ids_hm.bin` + **直读分片文件**的 320 B | ① | 输入目录 `ple/data_hm/`；ids 是**真实行 id**（0/1/2/3/4/5/12345/999983/2500011 + 随机，见 `hm_meta.json#ids_fixed`） |
| `H2.emb.nonvac` | 同 `H2.emb` 的对照侧（不读设备产物） | ① | 断言对照非零（163520/163520） |
| `H3.kv` | `w_key_proj`/`w_value_proj`（checkpoint 字节）+ 由 `rows_ref` 拼出的 `emb_ref` | ① | **不读 `H_emb.bin`**：参考链自己从分片文件重建 emb |
| `H4.gated` / `H4.normed` | 3×norm 权重 + `hidden_hm.bin` + 参考链自身的 `kv_ref` | ① | idem |
| `H5.out` | `w_conv1d_sq` + `conv_state_hm` + `state_idx_hm` + 参考链自身的 ④ 输出 | ① | idem（参考链的中间量不是设备产物） |
| `Hd.*` | `H_hm_dev.bin`（**设备侧判据自身的落盘**：每核 8×fp32） | **③** | 它是**设备产物**，而且正是被判的量；`Hd.cores`/`Hd.nonvac` 就是判"这份产物是否非空"的守卫 |
| **②（上游输出当参考输入）** | **没有** | — | 按 §1.3 第 1 步把 `check_hm` 里所有 `read_bytes`/`fromfile` 逐个归类 |

**口径选择：④⑤ 的参考吃 `kv_ref` 而**不**吃设备的 `H_kv.bin` —— 这是一条实测过的选择。**
理由：参考若吃设备自己的上游产物，**上游错误就变成共模**、判据失去咬合力。实测（同一 tip；
基线 `ple/out_hm`、负向 `ple/out_hm_neg`；纯 host 侧重算，两条参考链只差"喂给 ④ 的 kv 来源"）：

| 数据 | 参考链吃 `kv_ref`（= **现行判据**的输入） | 参考链吃设备 `H_kv.bin`（**诊断**口径） |
|---|---|---|
| 基线 `H4.gated` | 逐位不同 **62**；`1.0·ulp` 界下 bad **12**；`max(d/bound)=1.9978` | 逐位不同 **0**；bad 0；`max=0.0000` |
| 基线 `H4.normed` | 逐位不同 **85**；bad **19**；`max=1.9979` | 逐位不同 **4**；bad 0；`max=0.9991` |
| 基线 `H5.out` | 逐位不同 **2**；bad **0**；`max=0.9991` | 逐位不同 **0**；bad 0；`max=0.0000` |
| bit14 负向 `H4.gated` | 逐位不同 **644607**；bad **643574**；`max=1.64e5` | **逐位不同 0；bad 0 ⇒ 判据会 PASS** |
| bit14 负向 `H4.normed` | 逐位不同 **644518**；bad **643333**；`max=1.64e5` | 逐位不同 8；bad 0 ⇒ 会 PASS |
| bit14 负向 `H5.out` | 逐位不同 **515354**；bad **382488**；`max=1.64e5` | **逐位不同 0；bad 0 ⇒ 判据会 PASS** |

> **本表的读数口径**（避免与判据的 RESULT 行混读）：本表 `bad` / `max(d/bound)` 是
> **`1.0·ulp` 界下**的诊断量（`ple/recount_hm.py` 固定打 `ulp_mult ∈ {1.0, 2.0}` 两档）。
> **判据实际跑的是 `2.0·ulp`**（§6.2 第 2 条），所以 bit14 的 `RESULT` 行是
> `H4.gated 642546 / H4.normed 642175 / H5.out 312619`（= 本表 `2.0·ulp` 档），
> 与 `ple/logs/hm_neg.log` 一致；两档都判 FAIL，结论不变。

⇒ 若参考链吃设备自己的 kv，**bit14 这种"整行取错"的灾难性上游错误在 ④⑤ 上是隐形的**
（上游错与实现错同源、一起错）—— 正是 `20260927-tower-all-m85-pass` 点名的形态。
现行口径（吃 `kv_ref`）让 ④⑤ 是**端到端**判据：上游错必然在它们身上显形。
代价是它们的分辨力被上游的格点差占用（这正对应 §6.2 第 2 条那需要 `2.0·ulp` 的 12/19 个元素）；
而**同一输入下的 ④⑤ 自身算术**由"吃 `kv_dev`"那一列证明与参考一致到 ≤1 格点
（基线 0 / 4 个元素差 1.0 ulp、`max(d/bound)=0.999`；负向 0 / 8）。
**"吃 `kv_dev`"那两列是诊断，不是判据** —— 它不参与任何 PASS/FAIL。

（与 M85 `check_body` 的口径差别，写明以免混用：那里 ④⑤ 的参考吃**设备** kv，
于是 ③ 只由 `B3.kv` 单独判、④⑤ 只判**自身算术**。两种口径各自自洽，
但**不能混用**——混用会让"某条错到底归谁咬"说不清。HM 案明确采用**端到端**口径。）

上面这张表的两列由 `ple/recount_hm.py` 一次跑出（转录 `ple/logs/hm_recount.log` 的
`§6.3` 段；脚本同时给出 §6.2 的全部数字，**它不是判据**）。

### 6.4 一条实现注记：设备侧计数为什么走 `DataCopy` 而不是 GM 标量 `SetValue`

本 mission 第一版把设备侧计数写成 `G.hmMiss.SetValue(bid, miss)`（与 M85 的
`failG.SetValue(bid, bad)` 同形）。**实测落不下来**：加临时探针 `SetValue(bid, miss+100)` 后，
读回 128 个槽位里**只有 4 个核的值可见**（且这 4 个的 `miss` 都是 0）；
换成「累计器经 UB → GM 的 `DataCopy`」后，**56/56 核全部可见**（`Hd.cores` 判据就是这条的守卫）。
（探针已删除；见 §6.1 的 `Hd.cores` 读数。）

**正对照（原稿的判读，M111 已更正 —— 见下面「更正」段）**：同一个 `GlobalTensor::SetValue` 写法在
**① 的 kernel**（`m15_ple_ids_kernel`）里**曾被认为是可用的** —— 用 bit10（`id = mixed`，全部越界）
跑一次，`A_ids_meta.txt` 的 `dev_range_fail = 16`（`B_ids.*` 同）。命令与输出：

```
$ M15_PLE_MUT=1024 M15_PLE_OUT=/tmp/m92_setvalue ./m15_layer_loop/build/m15_ple
[m15ple] A    ① ids: 10 tok × 16 head, dev_range_fail=16 -> A_ids.bin
$ grep dev_range_fail /tmp/m92_setvalue/A_ids_meta.txt
dev_range_fail 16
```

**更正（M111 实测，命令与全文见 `m15_layer_loop/evidence/ple_wire/M111_LANDING_PATH.md`）**：
上面那个 `16` **不能**读成「① 的标量落盘可用」。该档下 A 的 10 个 token 由 **10 个不同的核**
各处理一个，每核数出 **16** 个越界 id ⇒ 若 10 个核的槽位都落下来，和应该是 **160**。
读回 **16** 说明**只剩 1 个核的值**（B 档同形：2 核 × 16 = 32，读回也是 16）。
⇒ 这一条与 body kernel 的「4/56 可见」是**同一类**现象（同一 cache line 内多个核的 4 B 标量写互相覆盖），
**不是**正对照。**连带的更正**：`dev_range_fail`（① 的 host 侧读数）在被修之前**只反映 1 个核**，
于是它也**不能**当作「① 的值域没有错」的证据。

⇒ 现象**不限于** body kernel；但当时的做法（设备侧计数一律经 `DataCopy` 落盘）**方向是对的**，
M111 把它推到 ① 的 ids 落点与全部三处标量落 GM 上（同一处置）。

**由此带出的一条既有状态**（顺手核过，**M111 已收尾**）：M85 body 里的 `G.fail.SetValue(bid, bad)`
从来没有被 increment 过（`PleGateItem`/`PleConvItem` 里是 `(void)bad`）⇒ 该槽位**没人写**，
读 `dev_fail = 0` 时**不能**当作"设备侧确认无错"（见 §9 U-K）。M111 把计数搬进 `PleGather`
（真的数「词表外的 id」）并改走 UB → MTE3 `DataCopyPad`，同时给 `dev_fail` 加了一条判据
（`B_dev.fail`，咬合力由变异 bit15 演示）。

---

## 7. 负向对照（task 4 的"判别力"要求）

**注入**：`m15_ple.asc` 的变异位 **bit14** —— 把「id → 窗口行号」的换算按**方向**改错：

```
正常： row = id - winBase
bit14：row = winRows - 1 - (id - winBase)      # 行号反转（在窗口内，不会越界读）
```

**为什么这个注入一定流过被判路径**：数据流是
`ids[item] → row → DataCopy(win[row*160]) → emb → 所有下游`，
而 `exp64[item]`（设备侧对照）与 `H2.emb` 的对照（host 直读文件）都**不经过** `row` 的计算
（前者按 item 索引、后者按 `id // rows_per_shard`）⇒ 注入一定让两边不一致。

命令与读数（`ple/logs/hm_neg.log`，本 tip 实跑）：

```
$ M15_PLE_MUT=16384 M15_PLE_HM=1 M15_PLE_OUT=m15_layer_loop/ple/out_hm_neg ./m15_layer_loop/build/m15_ple
[m15ple] hm dev-side raw: 参与核=56（应 56） winRows(设备侧读到)=838860 hmBad/槽0 合计=1022.0 越窗=2.0
[m15ple] hm ②③④⑤(host-mapped 真实表): n_tok=64 items=1024 in_win=1022 dev_row_fail=1022.0 dev_miss=2

$ python3.12 m15_layer_loop/m15_ple_check.py m15_layer_loop/ple/out_hm_neg    # 期望 FAIL
RESULT|H2.emb       |T1|163520|163373|FAIL      ← host 侧 T1 逐字节
RESULT|H3.kv        |T3|806400|804162|FAIL
RESULT|H4.gated     |T3|645120|642546|FAIL
RESULT|H4.normed    |T3|645120|642175|FAIL
RESULT|H5.out       |T3|645120|312619|FAIL
RESULT|Hd.row_fail  |T1-struct|1022|1|FAIL      ← ★ **设备侧**计数器（1022 / 1022 全中）
RESULT|Hd.miss      |T1-struct|1024|0|PASS      ← 越窗判定不受该注入影响（对照成立，不误报）
RESULT|Hd.cores     |T1-struct|56|0|PASS
===== 判据合计 10 条 = 判定项 10 + 对照项 0；FAIL 6 条 =====
===== FAILURES PRESENT =====   check rc=1
```

**读数怎么读**：

* `H2.emb` 163373/163520 不符 —— 只有约 147 个元素**碰巧**相同（反转后两行的对应元素偶然相等），
  这正是"逐行反转"应有的形态；
* **设备侧 `Hd.row_fail` = 1022** —— 与窗口内 item 数完全一致（**每个**窗口内 item 都被设备自己抓到），
  这是"设备侧判据不是装饰"的直接证据；
* `Hd.miss` 仍 PASS —— 注入不碰窗口范围判定 ⇒ **该判据没有被"顺手"带红**，说明两条设备侧判据
  各咬各的（不是一条计数器冒充两条）。

### 7.1 变异矩阵（`m15_ple_mutants.py`，一键复现）

`ple/logs/mutants.log`（本 tip 实跑，`rc=0`）新增了 M92 的一段：

```
[mut] M92 基线判据：10 条，FAIL 0 条
RESULT|mut_hm|14|id→偏移换算的方向|H2.emb|H2.emb,H3.kv,H4.gated,H4.normed,H5.out,Hd.row_fail|OK|...
RESULT|mut_hm|6|D7（gather 到 head 槽的映射）|H2.emb|H2.emb,H3.kv,H4.gated,H4.normed,H5.out|OK|...
RESULT|coverage_hm|6|10|H2.emb.nonvac,Hd.cores,Hd.miss,Hd.nonvac|NG
[mut] M92 咬合力被演示 = 6 / 10；未被演示 = ['H2.emb.nonvac', 'Hd.cores', 'Hd.miss', 'Hd.nonvac']
```

* bit14 的期望集合包含 **`Hd.row_fail`**（设备侧），脚本要求它必须在 FAIL 集合里 ⇒ **设备侧判据被强制**
  （哪条设备侧判据没被咬到，`mut_hm` 那行就是 `NG`）；
* `mut_hm|6` 是**同一注入在窗口模式下的复算**（窗口模式没有绕过 `gUse` 的分支），
  它**不**期望带红设备侧计数（head 落点错不改"取哪一行"，设备侧比的是行首 64 元素）
  —— 实测 `Hd.row_fail` 也**确实没**FAIL，与预期一致（脚本写 `None` = 不要求）；
* **`H2.emb.nonvac` / `Hd.cores` / `Hd.miss` / `Hd.nonvac` 未被任何 M92 变异咬到，这是预期的**：
  它们是**守卫类/范围类**判据（判"有没有东西可比"与"窗口范围参数对不对"），
  不该被"取错行"类注入带红。**不强制 HM 覆盖 10/10**，就是为了不诱导为凑覆盖写假变异
  （M85 §6 第 7 条的同族教训）。M85 的 13/13 覆盖仍然强制、且本 tip 复跑仍 **13/13**。

---

## 8. 并发背景（读数归因用）

采集时 `npu-smi info` 显示 NPU 0 上同时有：

* `probe_host_dma`（M86，`regbench` 模式，正在反复注册 4/64/256 MB 内存）；
* `m15_layer_loop ... prolog`（M88 的 worktree）。

采集时 NPU Util = 0%、HBM 12991 MB / 131072 MB。
M92 自己的读数因此**可能被拖慢**（尤其是注册耗时 4.1 s / 8.4 s 与 §4.1 的 1.693 s）——
本文所有 host 墙钟读数都**只作"跑过一遍"的证据**，不作带宽/性能结论。

---

## 9. 未完成 / 未覆盖（逐条，不声称完整）

| # | 项 | 卡在哪 |
|---|---|---|
| **U-A** | **多槽窗口池 + LRU 滑窗**（M86 §4.3 的 4×256 MiB 形态） | 本 mission 只实现"**一个窗口一次填满、整段跑完**"；`win_rows=838,860`。多槽需要 host 侧驻留管理 + 设备侧槽表，且注册 4 槽的最坏耗时（M86 观测到 41 s/次）会吃掉本 mission 的时间预算；代码里窗口大小/槽数已是参数，**策略本身未做**。**M171 已补多槽窗口池 + 设备侧槽选择**（`ple/MULTISLOT.md §2`，`m15_ple.asc:526-560`）；**LRU/滑窗驻留策略仍未做** |
| **U-B** | **跨分片窗口**（窗口横跨 shard 边界） | 本 mission 的窗口落在单分片内 ⇒ 设备侧 `row = id - winBase` 与 `id % rows_per_shard` 等价。跨分片需要设备端 64 位除法/取模（`UMod64` 已有，除以 2,500,012 需另实现），**未做**。**M171 已补**：槽 0 横跨 shard 0→1，设备侧 `UDivMod64(id, rows_per_shard)` 分解 `(shard,local)` 后按 `delta` 定位（`ple/MULTISLOT.md §2.2`） |
| **U-C** | **真实 ①→② 端到端**（用 ① 在**真实全词表**上产出的 id 直接喂 ②） | 全词表的 16 个 head 的 id 区间**互不重叠**（`offset[g]` 相邻差约 2×10^7 行 ⇒ 每个 head 落在约 8 个分片上），任何单个 256 MiB 窗口只能服务其中一小片 ⇒ 该场景下绝大多数 item 会落"越窗"。本 mission 的做法是：**① 由 `B1.ids.A/B`（真实常量、T1 逐位）单独验证**，② 的窗口模式用**真实行 id 文件**（`ids_hm.bin`，全部真行、含片边界）驱动。`ple/logs/hm_neg.log` 里的 `dev_miss=2` 就来自 ids 里的 2 个越窗真行 ⇒ ①→② 的**数据通路**是通的、**越窗计数**也是真的，但"① 的真实输出直接喂满 ②"未做 |
| **U-D** | **性能**：设备侧 gather 吞吐、窗口命中率、注册代价的端到端账 | 未做设备侧计时（host 墙钟不构成证据，`docs/17 §9.1`）；本文 §4.2 的注册/搬运耗时是 host 墙钟、单次、且采集期有并发（§8） |
| **U-E** | **prefill / spec 的 short-conv 独立 writeback**（M85 §7 U3） | **不在本 mission**（mission 明示） |
| **U-F** | **与 `m15_layer_kernel.h` §3b 挂载点的实际接线**（M85 §7 U1） | **不在本 mission**（归后续 mission；依赖 M88 的 host/`.asc`/README） |
| **U-G** | `has_initial_states` / ETP-DP gather+reduce / dequantize / prefetch（M85 §7 U7） | **不在本 mission**（mission 明示） |
| **U-H** | 真实表在 **128 片全量**上的抽样 | 只跑了 shard_0 的完整行集合（§4.1）与 shard 1/127 的单行（§3）。**未**对 128 片各抽样 |
| **U-I** | 窗口内容的**一致性**（登记期 vs 使用期） | 本 mission 是"填完就跑"，没有并发写者 ⇒ 不需要；多槽/异步预取（真有并发写者时）才需要，属 U-A |
| **U-J** | 设备侧行校验**只覆盖行首 64/160 列** | `UB_HMCHK`/`UB_HMEXP` = `VL` = 64 个 bf16、`exp64.bin` = 128 B/item ⇒ `Hd.row_fail` 只比一行的前 64 列。**列 64..159 的"错列"目前只能由 host 的 `H2.emb`（T1 逐字节、全 160 列）咬住**，设备侧对它们无判定量。task 3 的"至少一条设备侧计数"已满足；若要设备侧覆盖全行，需把 `exp64` 扩到 160 列（UB 与 GM 各多 192 B/item）——**未做** |
| **U-K** | **M85 body 的 `dev_fail` 是结构性空判据**（跨 mission 缺陷） | ★ **M111 已收尾**：`PleGateItem`/`PleConvItem` 里 `(void)bad;`、`bad` 从不自增 ⇒ 原来的 `if (bad > 0u) { G.fail.SetValue(bid, bad); }` 是**死分支**，host 求和写进 body meta 的 `dev_fail` **恒 0** ⇒ 读 `dev_fail = 0` 不构成任何设备侧证据。M111 把计数搬进 `PleGather`（数「词表外的 id」，与 ① 同口径同常量）、落盘改 UB → MTE3 `DataCopyPad`，并把 `dev_fail` 加进判据（`m15_ple_check.py` 的 `B_dev.fail`，咬合力由变异 **bit15** 演示）；同时更正了 §6.4 那条被误读的「正对照」。读数与命令见 `evidence/ple_wire/M111_LANDING_PATH.md` |

### 9.1 复审提出的**非阻塞观察**（本轮按塔的指令只动文档，代码冻结；登记给后续 mission）

M92 的复审意见（`.tower/comms/reviews/`，**不在仓库跟踪范围内**；对本文的裁决落在 `814bf89`）
除 1 条 p2 外还提了几条非阻塞观察。
自 `ca347b2` 起的文档改动**只动文档 + 新增只读重算脚本**，设备代码与判据脚本**零改动**（塔明示"不必重跑设备"），
故下面几条**有意留到后续 mission**（那条小 mission 本来就要动 `m15_ple.asc` + `ple/**`）：

| 观察 | 现状 / 处置 |
|---|---|
| `m15_ple_mutants.py:214-215` 有重复的 `ok_all = ok_all and ok_hm` | 幂等、无行为差异；**未改**（改它会动判据驱动、使已归档的 `ple/logs/mutants.log` 与脚本版本不再同一份）。后续 mission 顺手删 |
| `coverage_hm` 打 `NG` 与末行"每条基线判据都被变异演示过"措辞不自洽 | HM 段**刻意不强制全覆盖**（守卫类判据不该被"错行"类注入带红），脚本里已写明理由；末行措辞宜点明"（M85 段强制覆盖 / HM 段只强制期望集合）"。**未改**（同因）；后续 mission 改措辞 |
| `ids_hm_flat.bin` / `exp_rows.bin` 无任何读数使用 | `ids_hm_miss.bin` **保留**（`M15_PLE_HM_IDS` 的越窗对照入口）；另两个确属多余。`exp_rows.bin`（全 160 列期望行）与 U-J 的收尾天然配套，故**不删**；`ids_hm_flat.bin` 待后续 mission 删 |
| `ple/logs/emit.log` 第 1 行是绝对路径（其余日志是相对路径） | 该行由 `real_table_probe.py --emit` 打印自身解析出的 `data_hm` 绝对路径；**未改生成器**（改了要重生成 `emit.log`）；后续 mission 顺手统一为相对路径 |
| **`ple/README.md:18`（§0 TL;DR 的"变异矩阵"行）有同类的**渲染吞字**缺陷**：该行内联代码里 5 个**未转义的管道符**使它成为 7 个单元（表头 2 列），而 GFM 会**忽略超出表头的单元** ⇒ 渲染后 `RESULT` 之后的 `coverage…OK`，rc=0）**整段看不见** | **已修**：把这 5 个管道符改写成反斜杠转义形式（纯文本改动，不动任何读数）。它是 **M85 的原文**（`git show 0bd80c6:m15_layer_loop/ple/README.md` 第 18 行逐字相同；本 branch 的 README diff 里不出现该内容 ⇒ 非本 mission 引入），由"逐表核对单元数"扫描顺带发现。该扫描在 `REAL_TABLE.md` 与 `README.md` 上现均为 **0 行不符**（`LC_ALL=C` 与 `LC_ALL=C.UTF-8` 同） |

---

## 10. 文件清单（本 mission 触碰的）

| 路径 | 角色 |
|---|---|
| `m15_layer_loop/ple/REAL_TABLE.md` | **本文**：落地形态规格 + 取证 + 判据 + 对照 |
| `m15_layer_loop/ple/real_table_probe.py` | 换算取证 / 真实规模搬运 / harness 输入生成（`--evidence` / `--real-scale` / `--emit`） |
| `m15_layer_loop/ple/recount_hm.py` | **§6.2 / §6.3 读数的重算工具**（只读；与 `check_hm` 同口径；**不下 PASS/FAIL**） |
| `m15_layer_loop/m15_ple.asc` | ② 的窗口模式 + 设备侧计数 + `RunBodyMapped`（file mmap + ACL 注册）；变异位 bit14 |
| `m15_layer_loop/m15_ple_check.py` | `check_hm`：`H2.*` / `H3.*` / `H4.*` / `H5.*` / `Hd.*`；`cmp_t3_masked`（Σ\|terms\| 界、格点系数可调） |
| `m15_layer_loop/m15_ple_mutants.py` | 新增 M92 数据面矩阵（`HM_MUTANTS`：bit14 强制带红设备侧判据） |
| `m15_layer_loop/ple/.gitignore` | 增 `data_hm/`（HM 输入同属生成物） |
| `m15_layer_loop/ple/logs/real_table_probe_evidence.log` | §3 的完整 stdout |
| `m15_layer_loop/ple/logs/real_scale.log` | §4.1 的完整 stdout |
| `m15_layer_loop/ple/logs/emit.log` | harness 输入生成清单 |
| `m15_layer_loop/ple/logs/hm_run.log` | §6.1 基线：10 条 / 0 FAIL |
| `m15_layer_loop/ple/logs/hm_neg.log` | §7 负向对照：6 条 FAIL |
| `m15_layer_loop/ple/logs/mutants.log` | §7.1 变异矩阵（含 M92 段） |
| `m15_layer_loop/ple/logs/hm_recount.log` | §6.2 / §6.3 全部数字的重算转录（基线 + 负向各一段） |
| `m15_layer_loop/ple/logs/run_all.log` | M85 既有 13 条判据在**本 tip** 的复跑（13/13 PASS，回归证据） |

**未触碰**（mission 边界）：`m15_layer_kernel.h` / `m15_layer_resources.h` / `m15_layer_loop.asc` /
`m15_attn*`（M88）/ `m15_moe_*.h` / `lift_moe_segment.py` / `moe_relift/**`（M91）/
`probe_host_dma/**`（M86）/ `docs/**` / `m15_layer_loop/README.md` /
`m15_hc_host.h` / `m15_chain_host.h` / `slice_layer_manifest.py` / `weights_manifest.txt`。
