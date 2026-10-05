# M117 / Wave-B3 证据（`m25_attn_fa_core/evidence/`）

> **口径**：本目录里的数字全部是**实跑读数**（命令与输出在下面逐条给出）。
> 未取到的读数**不写成结论**。本文里没有「0 命中」这类绝对断言 —— 需要时写成
> 「在该 pattern 与该范围内未命中」。

## 0. 结论速览（先读这 4 条）

> **最新（M196，2026-10-05）**：两个**生产形态**验收通过 —— `core` **m=4097**
> `over=0/25171968 maxratio=0.2962`、**m=1** `over=0/6144 maxratio=0.0262`；两档的 `nTiles`
> 上界都 = **33**（公式与自算命令见 `M196_readings.txt` §2）；同档四跑 sha256 一致；
> 三个负向对照在 m=4097 均 FAIL、在 m=1 不咬（如实登记）。`m=4097` 设备核墙钟 ≈ 3.4 ms。
> 逐档读数与原始日志见 `M196_readings.txt` 与 `m196_logs/`；复现 `reproduce_accept.sh`。
> 口径：**core 级 / 合成 bf16 / 对自建 fp64 dense-causal 参考**（非真实权重/KV parity）。

> **M189（2026-10-05）**：B3 数值已收敛 —— `m=16/64/96/128` 四档 `check_ref.py`
> **core 全 PASS**（`over=0`、`maxratio=0.2962`），三个负向对照全 FAIL，同档三跑 sha256 一致。
> 根因 = `L1 -> L0B` 的生产侧（MTE1）`Acq/Rls` 被放到 BMM1 的 `ks` / BMM2 的 `nh` 循环之外，
> M 侧两条 `Mmad` 都读到循环最后一次装载。逐档读数与定位见 `M189_readings.txt`。
> 下面第 1–4 条保留为 **M117/M185 时代的历史读数**（挂死期与「数值 FAIL」期）。

1. **段体头与独立验证路都能编译**（dav-3510，bisheng 15.0.5 / CANN 9.1.0）：`[100%] Built target
   m25_attn_fa_core`（AIC 与 AIV 两份镜像都过；唯一告警来自被 include 的 `m15_hc_layer.h:740/766`）。
2. **判据本身可用**（**离线自证**，不依赖设备）：`evidence/check_ref_selftest.txt` 里，把 fp64 参考
   **按 bf16 圆整后回灌**当输出 ⇒ `VERDICT=PASS`（`maxratio 0.2203`）；把同一份输出乘 1.05 ⇒
   `VERDICT=FAIL`（`over=117926/227328`，`maxratio 3.0760`）。⇒ 判据**非空洞**、且**把被测对象弄坏会变红**
   （M189 起另有 **设备档** 的负向对照读数）。
3. **（历史 M117）端到端设备读数未取到**：`m=64`（最小档）曾**挂死**、无输出文件。
   M185 已解挂死（四档 EXIT=0），M189 已修数值（四档 PASS）。
4. **（历史 M117）挂死曾定位到「第 3 个工作项（wi=2）的 AIV 收尾」**；M185 改跨核 mode 2 后不再挂死。

## 1. 复现命令（逐字）

```bash
cd /workspace/ascend_mega_kernel/.tower/worktrees/wt-117
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m25_attn_fa_core/build -S m25_attn_fa_core -DCMAKE_BUILD_TYPE=Release
cmake --build m25_attn_fa_core/build -j4               # -> [100%] Built target m25_attn_fa_core
df -h /                                                # 写文件前看盘
flock -w 300 /tmp/npu0.lock bash -c '
  npu-smi info -t common -i 0 | sed -n "3,5p"          # 进锁后复查
  M25FA_M=64 M25FA_BLOCKS=1 M25FA_OUT=m25_attn_fa_core/evidence/out \
    stdbuf -oL timeout 60 ./m25_attn_fa_core/build/m25_attn_fa_core core; echo "EXIT_C=$?"'
```

`evidence/run_m64_clean.log`（**无探针的干净版**，逐字）：

```
	Aicore Usage Rate(%)           : 0
	Aicore Freq(MHZ)               : 1650
[M25FA] aicNum=28 nBlk=1 mode=core
[M25FA] pre-launch m=64 posBase=0 ctx=64 nBlk=1
[M25FA] launched, waiting sync
EXIT_C=124
```

`EXIT_C=124` = `timeout` 到 60s 杀进程；`m25_attn_fa_core/evidence/out/` 里**没有** `m64_core/`
子目录 ⇒ 落盘段一次都没走到。

## 2. 挂死的定位（探针口径，`M25FA_PROBE` 编译期开关）

`probe_logs/` 里是带探针的实跑日志。探针量级：每个核在**第一个工作项**（`run_m64d/e/f`）
或**每个工作项**（`run_m64i/j`）打印若干行。逐条读数：

| 日志 | 配置 | 读到的最后几步 |
|---|---|---|
| `run_m64d.log` | m=64, nBlk=4 | `AIC t=0 PV_RDY set`；`AIV t=0 PV_RDY ok` —— **tile 0 全阶段过** |
| `run_m64e.log` | m=64, nBlk=4 | AIC0 `WI 0 done` → WI 4 的 tile 过；AIV0 `WI 0 done`、`WI 4 done`（**含 tail**） |
| `run_m64f.log` | m=64, nBlk=4（收尾改成复用 `B_PC` 之后） | AIV0/AIV1 各 `WI 0 done`、`WI 4 done`；随后停 |
| `run_m64i.log` | m=64, nBlk=1，**逐工作项**探针 | AIV0/AIV1：`WI 0 done`、`WI 1 done`；wi=2 的 `softmax+S_FREE / P0 set / PV_RDY ok` 都有，**`datacopy done` 没有** ⇒ 停在收尾 |
| `run_m64j.log` | 同上 + 行状态初值改 RegBase | **与 `run_m64i` 同一个点位** |

⇒ 两个读数：① 单工作项的 AIC↔AIV 握手**逐阶段是通的**（含两次 mmad、Fixpipe dualDstCtl、
P 经 GM 往返、softmax、acc 累加、落盘）；② **从第 3 个工作项起在 AIV 收尾挂死**。

## 3. 试过、且**没有**改变挂死点位的改动（都实跑过）

按时间顺序（每条都重新编译 + 重跑）：

1. `Acq/Rls` 由 `GetBuffImpl/ReleaseBuffImpl` 换成仓库既有封装 `GetBufInternal<p,false>` /
   `RlsBufInternal<p,true>`（同 `m15_gdn_layer.h` 的 `BufAcquire/BufRelease`）⇒ 挂死点位不变。
2. 收尾从「32 行逐行 DataCopy」改成「一次带 `dstStride` 的 32 行 DataCopy」⇒ 点位前移（能过 2 项）。
3. 去掉 AIV 上那个「先持有 `B_ACC` 再在 `Acq<PIPE_MTE3>(B_OUT)` 之前 drain-release」的额外令牌对
   ⇒ `Acq<PIPE_MTE3>` 那一处通了（`MTE3 acq` 打印出现）。
4. 核间旗标由「同一 id 双向令牌」改成**单写方**（`CC_S_RDY/CC_S_FREE`、`CC_PV_RDY/CC_PV_FREE`）
   ⇒ 挂死点位不变（这一步同时消掉了一个**真竞态**：旧写法下 AIC 可能消费掉自己留下的令牌，
   从而在 AIV 读完 `S[q]` 之前覆写它）。
5. 收尾改成**复用 `B_PC`（UB_PC）** + 分两个 16 行半块（复刻 P 落盘那条**已跑通**的路）⇒ 点位不变。
6. 行状态初值（m/sum/acc）由经典 API `Duplicate` 改成 RegBase VF（`StoreAlign`）⇒ 点位不变。

## 4. 判据的离线自证（`check_ref_selftest.txt`）

自造 q/k/v（bf16），用 `check_ref.py` 的 fp64 参考算出真值，再：

* 输出 = **按 bf16 圆整后的参考** ⇒ `VERDICT=PASS dir=/tmp/selfcheck1 over=0/227328 maxratio=0.2203`
  （`maxratio` = `max(|got−ref| / tol)`，<1 即全部覆盖）。
* 输出 = 同一份参考 **× 1.05** ⇒ `VERDICT=FAIL ... over=117926/227328 maxratio=3.0760`。

⇒ 判据在「正确」与「坏掉」两侧都有咬合力。

## 5. 未取到的读数（**显式写窄**）

1. **`m=64 / m=1(ctx=4097) / m=4097` 的 PASS/FAIL 数值读数**：一个都没取到（设备挂死）。
2. **三个负向对照（`negmask` / `negshift` / `negstart`）的读数**：没取到（要求先有一个能跑完的 `core` 档）。
3. **确定性（连跑多次比 sha256）**：没取到。
4. **`m=4097` 的运行时长**：没取到。
5. **性能**：没取到（本来也未做深流水）。
6. **融合清单里「每 id 的 set 次数 ≤ 15」的按档断言**：只给了 id 清单与相邻性，未按 `m` 档实测。

## 6. 本档没有做的事

* 没有拿官方 QSA 的稀疏输出当判据（`docs/17` §7）；本档判据全部是**自建稠密 causal fp64 参考**。
* 没有改 `m15_layer_loop/CMakeLists.txt`（归 M101），也没有改 M101 的 `m15_attn_core.*`。
* 没有改 `m15_attn_kv.h`（KV 布局唯一权威，只读）。

## 7. M126 之后的第二轮（2026-10-04，本档新增读数）

M126（独立探针，塔派）把「AIV 收尾的 `Acq<PIPE_V> → 写 UB → Rls<PIPE_V>`(drain) → `Acq<PIPE_MTE3>` →
`DataCopy` → `Rls<PIPE_MTE3>`」这一形态在 N=1..64 逐档跑过，**没有挂死**，并给出三条硬事实；
本档按其 §4「唯一新增规则」照抄并**真机重跑**：

### 7.1 已照抄的规则（在段体里逐处加，未全铺）
* AIV：`B_PC` 的每次 `Acq<PIPE_V>` / `Acq<PIPE_MTE3>` 之前各加 `PipeBarrier<PIPE_V>` / `<PIPE_MTE3>`。
* AIC：MTE2 的 `B_Q/B_K/B_PL1/B_V`、MTE1 的 `B_L0/B_K/B_PL1/B_V`、M 侧 ks 与 nh 两个循环的
  `B_L0/B_K/B_V/bC`、FIX 的 `bC` → 各在同 pipe 背靠背复用点加 `PipeBarrier<该 pipe>`。
* 段体全文共 16 处 `PipeBarrier<PIPE…>`（`grep -c "PipeBarrier<PIPE" m15_layer_loop/m15_attn_fa_core.h`）。

### 7.2 读数：**加了屏障，`m=64` 仍然挂死**
| 档 | 命令要点 | 读数 |
|---|---|---|
| 屏障版 | `M25FA_M=64 M25FA_BLOCKS=1 timeout 60` | `EXIT_PB=124`，无输出文件（`run_m64_pb.log`） |
| 收尾改「一轮 32 行」（UB_PC 8 KB→16 KB） | `M25FA_M=64 M25FA_BLOCKS=1 timeout 60` | `EXIT_1R=124`，无输出文件（`run_m64_1r.log`） |
| **每块恰好 1 个工作项** | `M25FA_M=64 M25FA_BLOCKS=28 timeout 45` | `EXIT28=124`（`run_m64_28.log`） |
| 同上（24） | `M25FA_M=64 M25FA_BLOCKS=24 timeout 45` | `EXIT24=124` |
| `PipeBarrier<PIPE_ALL>` 诊断（S 消费前 / PV 消费前 / 收尾 MTE3 取前各一条） | `M25FA_M=64 M25FA_BLOCKS=28 / 1 timeout 45` | `EXIT_DIAG28=124`、`EXIT_DIAG1=124`（`run_m64_diag.log`） |

⇒ **两个直接结论**（都有读数支持）：
1. **不能**把本档挂死归因于「同 pipe 背靠背复用同一 buffer」—— 与 M126 §4 的判断一致。
2. `M25FA_BLOCKS=28`（24 个工作项 ⇒ **每块恰好 1 个工作项**）**也挂死** ⇒ 挂死**不需要跨工作项轮转**，
   它发生在**单个工作项内部**、且是**概率性**的（`nBlk=1` 时曾连过 2 个工作项才挂，
   并发块多时命中更快）—— 这条把「per-iteration 计数/槽轮转耗尽」这一类解释**排除掉**。
3. 诊断用的全 pipe 排空（S/PV 消费点 + 收尾取令牌前）**也没让它过去** ⇒ 至少不是
   「AIC Fixpipe 写进本核 UB 的可见性」或「收尾取令牌前的 pipe 未排空」单独造成的。

### 7.3 探针定位到的两个阻塞点（`evidence/run_m64_fg.log`，探针构建）
* **AIV**：卡在 `Acq<PIPE_MTE3>(B_PC)`。逐轮打印显示同一轮的 `Acq<PIPE_V>(B_PC)` 通过、
  `FacNormalizeVf` 完成、`Rls<PIPE_V>(B_PC)`（drain）完成（`norm done` 打印），
  紧随的 `Acq<PIPE_MTE3>(B_PC)` 之后的 `dma done` 没有出现。
  （「第 9 次」这个数字随轮次数变化而变化：把收尾从 2 轮压到 1 轮后仍然挂死
  ⇒ **它不是硬上限，只是当时那一档的命中点**。）
* **AIC**：卡在 `(B)` 段的 **P 装载组**里某个 `Acq<PIPE_MTE2>(B_PL1)` / `Acq<PIPE_MTE1>(B_L0|B_PL1)`
  之前（最后一个打印是 `AIC B begin`，下一个 `AIC P in L0A` 没有出现）。
  **这个位置受设备 printf 每核缓冲滞后影响（本次每核 17~20 行即打住），所以只写成「疑似位置」，
  不写成已定位。**
* 静态对账（`Acq/Rls` 逐 (pipe,id) 计数）**没有发现不平衡**；`FacAic`/`FacAiv` 两侧、
  含循环体内的 `bC` 在内都配平。

### 7.4 仍然**未取到**（与第 5 节相同，未因本轮改变）
`m=64 / m=1(ctx=4097) / m=4097` 的 PASS/FAIL 数值读数、三个负向对照的设备读数、
确定性读数、运行时长 —— **一个都没取到**（所有档都 `timeout` 124、无输出文件）。

### 7.5 本轮**未做**的（留给下一步，不得当结论）
* 跨核最小复现（AIC Fixpipe 直写 AIV UB + 6 个 CrossCore flagId 的记账）—— M126 §4 也列为未覆盖；
  **这是本档现在最可疑的方向**（因为「单工作项内」+「概率性」+「不在同 pipe 序上」三条读数都指向它）。
* AIC 侧 `bC`（L0C 槽）与 AIV 侧 `B_PC` 是否在某个硬件资源上互相影响 —— **纯猜测，未取读数**。

## 8. 第三组诊断（同日）：Fixpipe `dualDstCtl` 的 `mSize` 也排除了

M126 的探针只覆盖了 AIV 内部（V 写 UB → drain → MTE3 搬），**没有**覆盖
「AIC 的 `Fixpipe(dualDstCtl=1)` 直写对侧 AIV 的 UB」这条跨核数据面（它自己在 §4 列为未覆盖）。
本档把 **`FAC_P` 从 64 改成 32**（⇒ Fixpipe 的 `mSize` 从 64 变 32、每个 AIV 分得 16 行），
其余一字未改，重跑：

| 档 | 命令 | 读数 |
|---|---|---|
| `FAC_P=32` | `M25FA_M=64 M25FA_BLOCKS=1 timeout 45` | `EXIT_P32_1=124`，无输出文件（`run_m64_p32.log`） |
| `FAC_P=32` | `M25FA_M=64 M25FA_BLOCKS=28 timeout 45` | `EXIT_P32_28=124`，无输出文件 |

⇒ **`mSize=64` 与 `dualDstCtl=1` 的组合不是原因**（`mSize=32` 同样挂死）。
（该档是诊断档；段体已恢复 `FAC_P=64` = donor config4 的 sOuter。）

### 8.1 到此为止，**有读数支持**的排除项汇总
| # | 排除了什么 | 读数出处 |
|---|---|---|
| 1 | 同 pipe 背靠背复用同一 buffer（M126 §3 的规则） | `run_m64_pb.log`（屏障加了仍挂） |
| 2 | 收尾 DataCopy 的形态（逐行 / 单次带 stride / 一轮 32 行 / 两个 16 行半块） | `run_m64*.log` 各档 |
| 3 | 跨工作项轮转（`BLOCKS=28` ⇒ 每块恰好 1 个工作项，也挂） | `run_m64_28.log` |
| 4 | 全 pipe 排空（3 处 `PipeBarrier<PIPE_ALL>` 诊断） | `run_m64_diag.log` |
| 5 | Fixpipe `dualDstCtl` 的 `mSize`（64 vs 32） | `run_m64_p32.log` |
| 6 | BufferID 封装（`GetBuffImpl/ReleaseBuffImpl` vs 仓库既有 `GetBufInternal/RlsBufInternal`） | `probe_logs/run_m64d.log`（该轮另有一份 `run_m64b.log` 已不在库中 —— 见 `exclusions.md` 第 6 行的说明） |
| 7 | 核间旗标的「同 id 双向令牌」（已改单写方） | 各档均仍挂 |
| 8 | 行状态初值用经典 API `Duplicate` vs RegBase VF | `run_m64j.log` |
| 9 | 静态逐 `(pipe,id)` 的 `Acq/Rls` 不平衡 | 逐 (pipe,id) 对账脚本（`evidence` 里给了方法） |

### 8.2 仍然**没有**被任何读数触及的一侧（下一步的唯一方向）
**AIC ↔ AIV 的跨核记账**：`Fixpipe(dualDstCtl=1)` 直写对侧 UB + 6 个 mode-4 CrossCore flagId
（`CC_S_RDY/CC_S_FREE/CC_PV_RDY/CC_PV_FREE/CC_P0/CC_P1`）+ `+16` 通道偏移 的成对记账，
在「单工作项内、概率性、不在同 pipe 序上」这三条读数下是唯一还没被排除的方向。
**未取读数 ⇒ 不得写成已定位。**

## 9. 第四组：把「Fixpipe 直写对侧 AIV 的 UB」整条换掉（塔指定的判别实验 1）—— **仍然挂死**

塔（M134 派单那条）要求的最便宜判别实验：把「AIC 的 `Fixpipe(dualDstCtl=1)` 直写对侧 AIV 的 UB」
换成「**AIC 写 GM → 显式 CrossCore 通知 → AIV 用 MTE2 读回来**」。本档照做了一版（只改这一条通路，
其余一字未动）：

* AIC：`FacFixpToGm(S0/C0...)`（新增 helper，`dualDstCtl = 0`、`Fixpipe<..., kFixGm>`，L0C→GM 写 fp32），
  S 写到 `spGM[aic*2P*SIN + q*P*SIN]`（行距 SIN）、PV 写到 `spGM[spS + aic*2P*DV + q*P*DV + nh*128]`（行距 DV）。
* AIV：等旗标改成 **`PIPE_MTE2`** 侧（消费者 pipe），`Acq<PIPE_MTE2>(B_S0+q)` → `DataCopy` 本核那 32 行
  （S：`{32, 16, 0, 0}`；PV：`{32, 32, 0, 0}`）→ `Rls<PIPE_MTE2>` → `PipeBarrier<PIPE_V>` →
  `Acq<PIPE_V>` → 软最大 / acc 更新。**`B_S0/B_S1/B_PV0/B_PV1` 这四个原本"声明但未用"的 id 正好用上。**
* host：多分配 `nBlk*2*(P*SIN + P*DV)*4 B` 的 fp32 scratch；`.asc` 入口壳多两个参数
  （`sp` 指针 + `spSBytes`）。段体签名相应变为
  `FaCoreBody(q, qStride, kv, out, pScratch, spScratch, spSBytes, m, posBase, ctx, lane, nBlk)`。

**读数**（`run_m64_gm.log`）：

| 档 | 命令 | 读数 |
|---|---|---|
| GM 往返版 | `M25FA_M=64 M25FA_BLOCKS=1 timeout 45` | `EXIT_GM1=124`，无输出文件 |
| GM 往返版 | `M25FA_M=64 M25FA_BLOCKS=28 timeout 45` | `EXIT_GM28=124`，无输出文件 |

⇒ **「Fixpipe 直写对侧 UB」这条跨核数据面也排除掉了**（换掉它以后挂死照样出现）。
因为换掉后本档**更差**（多 11 MB fp32 scratch、多一趟 DMA），且它不是根因，
**本档已把这一版回退**，恢复 M101 那条已被本仓验证过的 `dualDstCtl` 直写对侧 UB 写法。

### 9.1 排除清单（第四轮后，全部有读数）
第 §8.1 的 9 条 **加上**：
10. **Fixpipe 直写对侧 AIV 的 UB**（换成「AIC→GM→CrossCore→AIV MTE2 读」后仍挂）—— `run_m64_gm.log`

### 9.2 到第四轮为止的盘点
* 已被读数**排除**的：同 pipe 背靠背复用、收尾 DataCopy 形态、跨工作项轮转、全 pipe 排空、
  Fixpipe `mSize`、BufferID 封装、旗标单写方、行状态初值 RegBase、静态 (pipe,id) 不平衡、
  **Fixpipe 直写对侧 UB 这条跨核数据面**。
* **仍未取到**：`m=64` / `m=1(ctx=4097)` / `m=4097` 的任何 PASS/FAIL 数值读数、三个负向对照、
  确定性、运行时长。**离线自证仍在**（`check_ref_selftest.txt`）。
* 变量维度里**还剩没有被单独动过的**：`m`/KV 的**具体数据**（到目前为止只跑过 `m=64,ctx=64` 一档数据）、
  L1/L0 的**装载参数**（`FacGmToL1Nz` 的 `ndNum`/stride、`FacLoadL0_2D` 的字段）、
  `FacSoftmaxTileVf` 的**掩码阈值**（`thrBase/colHiF`）、以及 `m` 档位本身。
  **这些都没有读数，不得当成结论。**

## 10. 第六轮读数（VF 从严上机 + 装载几何对照）
* VF 从严版（`63a37b6`）：`m=64 ctx=64` **3/3 挂**（`run_vf64.log`）。
* 装载几何改成逐矩阵 `Nd2Nz`（`ndNum=1`，M101 形态）：**3/3 挂**（`run_geo.log`）。
* 逐条表与目标修正见 `exclusions.md` §6。
