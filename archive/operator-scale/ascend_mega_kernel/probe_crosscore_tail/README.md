# M134 —— 跨核一侧：AIC Fixpipe L0C→对侧 AIV UB + CrossCore flagId 记账 的最小 mix 复现

> 独立工程（自带 `CMakeLists.txt` + 含 `main()` 的 `probe_crosscore_tail.asc`，不碰仓库任何既有 CMakeLists）。
> 目标：把 M117/B3 `m=64` 挂死里**未被 M126 覆盖**的那一侧——「AIC 算 → Fixpipe（`dualDstCtl` 直写**对侧 AIV 的 UB**）
> → CrossCore 通知 → AIV 在对侧 UB 上消费（配 `Acq/Rls`）→ 回告 → 下一轮」——缩成一个**不含算法**的最小 mix 循环，
> 逐档只动一个维度，直接读「返回 / 不返回」与「消费到的内容对不对」。

> ⚠ 未定界纪律（同 M126）：本 README 只写设备读数直接支持的东西。**合成档与 M117 的业务核有差异**
> （shape、tiling、层数、参与核数都不同），所以**不写**「M117 挂死的根因就是这个」。
> 另：本 README 的一切挂死/内容口径都**以 `evidence/logs/` 里的逐次日志为准**（不再引用任何未随分支提交的旧冒烟读数）。

---

## 0. 结论速览（TL;DR）

| # | 结论 | 判据（可复现） |
|---|---|---|
| 1 | **本档复现了「跨核一侧」的稳定挂死，且挂死位置随变体不同**：`N=16/32/64` 各 3 次独立进程，`base` **全挂（3/3、3/3、3/3）**，N=2/4 亦挂 ⇒ **确定性挂死，不是概率性**（与 M117 `m=64` 的「概率性」不同）。**`base` 的挂点在 AIC 的循环内**：AIC 最后一条打印是 `r=1 waitFREE`，停在第一次轮间 `CrossCoreWaitFlag<0x2, PIPE_S>(F_FREE0)`（`evidence/logs/base_N2_trace.log`、`base_N4_trace.log`、`base_N16_trace.log`；`loop done` 计数 = 0） | §2 表；§3.1 |
| 2 | **挂点位置不唯一**：`base`/`c_m2mte3` 是 **AIC 卡在循环内**（AIV 跑完全部轮次后卡在退出前）；`a_nowait` 是 AIC 卡循环内、**两个 AIV 打印了 `loop done`**；`c_m2setonly` 是**三个核都打印 `loop done`**、仍 `rc=124`（卡在**退出**） | §3.1 表（逐核最后打印 + `loop done` 计数，均给日志路径） |
| 3 | 挂死**与数据面无关**：`dualDstCtl` 直写对侧 UB 换成「AIC Fixpipe→GM + AIV DMA GM→UB」（`b_via_gm`）同样挂；去掉 AIV 侧 `Acq/Rls`（`d_no_acquire`）同样挂；去掉末尾 `PipeBarrier<PIPE_ALL>`（`dbg_nobar`）同样挂 | §2 表 |
| 4 | **挂死与「mode 2 回告方向的形态」强相关**（与塔广播 `20260927-tower-all-mode-2-n-set-1-wait-mode.md` 的实测口径一致）：准确条件 = **两个 AIV 的 set 归并到同一个共享 id 上，且 AIC 每轮 wait 恰好 1 次**（规范 `2 set(AIV) ↔ 1 wait(AIC)`）。偏离即挂：`c_broadcast`/`c_m2sameid`（共享 id 但 AIC **wait 2 次**）、`c_m2setonly`（**wait 0 次**）、`base`（两个 set 落在**不同 id** 且 wait 2 次）、`c_wait1`（两个 set 落在**不同 id**、AIC 只 wait 其中一个 **1 次**）**都挂**；**规范配对 `c_m2canon`（共享 id + wait 恰好 1 次）在 `N=4/16/32/64` 都不挂** | §2 表；§3.2 |
| 5 | **把回告方向换成 mode 4**（M117 的 `id + 16*channel` 口径，`c_mode4free`）同样**不挂**（N=4/16/32/64） | §2 表；§3.2 |
| 6 | **「不挂」≠「门控正确」**：本档**所有返回型**变体（`c_m2canon`/`c_mode4free`/`c_mode4full`/`dbg_rdyonly`/`dbg_nosync`）在**非 `--trace`** 运行里的消费内容都有差异（`evidence/logs/matrix_N*.txt`）。**没有任何一个变体在非 `--trace` 运行里 `rc0`**；唯一的 `rc0` 出现在隔离阶段的 **`--trace`（循环内 printf）** 运行（N=4，见 §3.3）⇒ 内容对错**对 printf/时序敏感**。内容判据**有牙**：`dbg_nodata`（空循环）mismatched=65536×3 | §2 表；§3.3 |
| 7 | **对 M117 的含义（写窄）**：M117 现用 **mode 4**（`CC_* + 16*c`），而本档说明 **mode 2 回告方向一旦偏离规范形态（两个 AIV 的 set 未归并到同一 id，或 AIC 每轮 wait≠1）就会挂**、规范形态才不挂。**但**本档**没有**把 M117 `m=64` 的挂点（AIV 收尾的本地 `Acq<PIPE_V>`）搬进来：本档的挂点是一个**跨核 `CrossCoreWaitFlag<PIPE_S>`**（不是本地 BufferID），二者的可观测点不同 ⇒ **不主张** M117 根因已定位 | §4 |

---

## 1. 被测形态与判据

### 1.1 结构（`probe_crosscore_tail.asc`）

`__global__ __mix__(1, 2)`，`blockDim = AIC 核数`，**只让 AICore 0 的 1 AIC + 配对 2 AIV 干活**（其余核立即返回）。
shape：`M=64, N=64, K=128`，bf16 操作数、fp32 累加。

**AIC（每轮 `r`，parity `q=r&1`）** —— `probe_crosscore_tail.asc:335`–`:393`：

```
（prologue）A0=1s / A1=2s / B=1s → L1 → L0；两次 Mmad 得 L0C slot0=128、slot1=256   // :305–:327
for r in 0..N-1:
    [若非首轮] mode2/mode4 wait(FREE)（回告方向；等待数见变体）                      // :366（c_m2canon）/:370（base）
    FixpToUb(ubDst, l0c slot q, nSize=64, dstStride=64)  // dualDstCtl=1 直写两个 AIV 的 UB  // :378 -> :239
    mode2/mode4 set(RDY)                                                            // :387（c_m2canon）/:392（base）
```

**AIV（每个 AIV `half`）** —— `probe_crosscore_tail.asc:499`–`:530`：

```
（循环前）FillVf(UB_SRC, sentinel)；FillVf(UB_STG, 0)                                 // :425/:426
for r in 0..N-1:
    mode2/mode4 wait(RDY, PIPE_V)                                                    // :458（c_m2canon）/:462（base）
    Acq<PIPE_V>(B_UB)                                                                // :499
    CopyVf(UB_STG, UB_SRC)        // VF 消费 AIC 直写的对侧 UB                        // :501
    Rls<PIPE_V>(B_UB)                                                                // :503
    [若 r<N-1] mode2/mode4 set(FREE, PIPE_V)                                          // :513（c_m2canon）/:523（base）
    SetFlag<V_MTE3>；Acq<PIPE_MTE3>(B_LOG)；DataCopy(logGM[r], UB_STG)；Rls         // :527–:530
```

- **`--trace`**（`TRACE=1 bash run_probe.sh`）：每轮在 AIC/AIV 各打一条 `[PCT]` 行 ⇒ 能直接读**逐核停在哪一行**。
  ⚠ 该 printf 会改变时序（§3.3）；**内容/返回判据一律以非 `--trace` 运行为准**，`--trace` 只用于定位挂点。
- **判据**：host 把每个 `(half, r)` 的 32×64 fp32 日志与期望值逐元素比。`q=0` 期望 `128`（=`K×1×1`）、`q=1` 期望 `256`（=`K×2×1`）。
  返回且逐元素相等 = `rc=0`；返回但有差异 = `rc=1`；`timeout` 未返回 = `exit=124`（记「未取得内容读数」）。
- **自管静态**：BufferID（`B_UB/B_LOG/B_L0/B_L1/B_C`）、CrossCore flagId（0–9，mode4 用 `+16`）、UB/L1/L0 地址全部编译期常量；
  核内用 `GetBufInternal/RlsBufInternal`（均 `false` = CANN `ASC_LOCK_BLOCK` 默认、阻塞），核间用 `CrossCoreSetFlag/WaitFlag`；落 GM 只走 DMA。
- **VF 消费用 RegBase**（`__VEC_SCOPE__`，整 64 lane），不用经典 `Duplicate`/标量 UB 交接。

### 1.2 变体清单（`./build/probe_crosscore_tail list`，20 个）

| 变体 | id | 轴 | 构造 |
|---|---|---|---|
| `base` | 0 | 基线 | mode2：每 AIV 独立 RDY/FREE id；AIC 每轮 wait 2 个 FREE |
| `a_nowait` | 1 | (a) 去 wait | 去掉 RDY 方向；保留 mode2 回告 |
| `b_via_gm` | 2 | (b) 数据面 | AIC `Fixpipe`→GM，AIV `DMA GM→UB` |
| `c_rotate` | 3 | (c) id 轮转 | RDY 的 id 按 parity 两组交替 |
| `c_broadcast` | 4 | (c) 共享 id | 共享 id：AIC set 一次；两 AIV set，AIC **wait 2 次** |
| `c_wait1` | 5 | flag 记账 | AIC 每轮只 wait 1 个 FREE（另一 id 不 wait） |
| `d_no_acquire` | 6 | (d) BufferID | AIV 去掉目标 UB 的 `Acq/Rls` |
| `neg_wrongval` | 7 | 负向 | host 期望值整体错位 |
| `neg_nofree` | 8 | 负向 | AIV0 不 set FREE |
| `c_preset` | 9 | flag 记账 | 复刻 M117「循环前预置 FREE」形态 |
| `dbg_rdyonly` | 10 | 隔离 | 只留 RDY 方向，无 AIV→AIC set |
| `dbg_nosync` | 11 | 隔离 | 完全无跨核 flag |
| `dbg_nodata` | 12 | 隔离 | 循环体空 + 末尾 `PIPE_ALL` |
| `dbg_nobar` | 13 | 隔离 | base 去掉末尾 `PIPE_ALL` |
| `c_mode4free` | 14 | 候选替代 | RDY mode2 + FREE mode4 |
| `c_m2sameid` | 15 | 隔离 | mode2 FREE，两 AIV 共用同一 id，AIC wait 2 次 |
| `c_m2setonly` | 16 | 隔离 | AIV 发 mode2 set，AIC **不 wait** |
| `c_m2mte3` | 17 | 隔离 | mode2 set 挂 `PIPE_MTE3` |
| `c_mode4full` | 18 | 候选替代 | RDY 与 FREE 都用 mode4 |
| `c_m2canon` | 19 | **规范配对** | mode2：`1 set(AIC) ↔ 2 wait(AIV)`、`2 set(AIV) ↔ 1 wait(AIC)` |

---

## 2. 变量矩阵与设备读数

命令（本目录）：`N=16 REPS=3 bash run_probe.sh`（脚本**每档各自进一次锁** `flock -w 300 /tmp/npu0.lock`，
进锁先 `npu-smi`，每条变体每次运行 `timeout 20s`）。证据按 `N` 打标：逐次日志 `evidence/logs/<变体>_N<N>.log`、
汇总 `evidence/logs/matrix_N<N>.txt`（由 `tools/summarize.sh` 从逐次日志的 AGG 行机械重算）。

<!-- MATRIX_TABLE_START -->
**N=16，每档 3 次独立进程**（`evidence/logs/matrix_N16.txt`，20 个变体全覆盖）。`hang` = `rc=124`（锁内超时未返回）；
`rc1` = 返回但内容有差异；`rc0` = 返回且内容逐元素相等。**本表没有任何 `rc0` 行**（见 §3.3）。

| 变体 | 读数（N=16，3 次） | 逐次内容读数（mismatched / 65536） | 结论 |
|---|---|---|---|
| `base` | **hang 3/3** | — | mode2、AIC wait 2× ⇒ 挂（AIC 卡循环内，见 §3.1） |
| `a_nowait` | **hang 3/3** | — | 去 RDY、保留 mode2 回告（wait 2×）⇒ 仍挂 |
| `b_via_gm` | **hang 3/3** | — | 经 GM 中转 ⇒ 仍挂（与数据面无关） |
| `c_rotate` | **hang 3/3** | — | RDY id 轮转 ⇒ 仍挂 |
| `c_wait1` | **hang 3/3** | — | 两个 set 落在不同 id（且另一 id 无人 wait）⇒ 仍挂 |
| `c_broadcast` | **hang 3/3** | — | 共享 id、AIC wait 2× ⇒ 仍挂 |
| `d_no_acquire` | **hang 3/3** | — | 去 `Acq/Rls` ⇒ 仍挂 |
| `c_m2setonly` | **hang 3/3** | — | AIC wait 0 次 ⇒ 仍挂（三核都 `loop done`，卡退出） |
| `c_m2sameid` | **hang 3/3** | — | 共享 id、AIC wait 2× ⇒ 仍挂 |
| `c_m2mte3` | **hang 3/3** | — | mode2 set 挂 MTE3、AIC wait 2× ⇒ 仍挂 |
| `c_preset` | **hang 3/3** | — | 循环前预置 FREE ⇒ 仍挂 |
| `neg_nofree` | **hang 3/3** | — | 可控挂死（挂死判据有牙） |
| `neg_wrongval` | **hang 3/3** | — | 坐在 mode2 失配形状上 ⇒ 先挂，读不到内容（见 §6） |
| `dbg_nobar` | **hang 3/3** | — | 去掉末尾 `PIPE_ALL` ⇒ 仍挂 |
| `dbg_rdyonly` | rc1 **3/3** | 31264 / 30944 / 30880 | 只留 RDY ⇒ 返回；缺 FREE 门控 ⇒ 内容 race |
| `c_m2canon` | rc1 **3/3** | 31904 / 28256 / 31168 | **规范 mode2 配对 ⇒ 不挂**；内容 race |
| `c_mode4free` | rc1 **3/3** | 27712 / 28384 / 28992 | mode4 回告 ⇒ 不挂；内容 race |
| `c_mode4full` | rc1 **3/3** | 28384 / 28160 / 30432 | 全 mode4 ⇒ 不挂；内容 race |
| `dbg_nosync` | rc1 **3/3** | 34240 / 65536 / 62176 | 无 flag ⇒ 返回；race |
| `dbg_nodata` | rc1 **3/3** | 65536 / 65536 / 65536 | 空循环 ⇒ 返回、日志全 0（判据有牙） |
<!-- MATRIX_TABLE_END -->

### 2.0b N 扫描（关键变体，各 3 次独立进程）

| 变体 | N=4 | N=16 | N=32 | N=64 |
|---|---|---|---|---|
| `base` | **hang 3/3** | **hang 3/3** | **hang 3/3** | **hang 3/3** |
| `c_m2canon`（规范 mode2） | rc1 3/3 | rc1 3/3 | rc1 3/3 | rc1 3/3 |
| `c_mode4free`（mode4 回告） | rc1 3/3 | rc1 3/3 | rc1 3/3 | rc1 3/3 |
| `dbg_nosync` | — | rc1 3/3 | rc1 3/3 | rc1 3/3 |
| `dbg_rdyonly` | rc1 3/3 | rc1 3/3 | rc1 3/3 | **hang 3/3** |

（矩阵文件：`evidence/logs/matrix_N4.txt`、`matrix_N16.txt`、`matrix_N32.txt`、`matrix_N64.txt`。）
⇒ `base` 从 `N=2` 起**每次运行都挂**（确定性）；`c_m2canon`/`c_mode4free` 四个 N 都**不挂**；
`dbg_rdyonly`（无 AIV→AIC set）在 `N=64` **也挂** ⇒ 除「回告方向形态」外还有一条未定界的挂死现象（§6）。

### 2.1 逐条对账（mission 的 (a)–(e)）

- **(a) 去掉 CrossCore wait**：`a_nowait`（去 RDY、保留 mode2 回告）3/3 挂；**内容判据的返回型对照**用
  `dbg_nosync` / `dbg_rdyonly`（返回、见 §2 表计数）。
- **(b) `dualDstCtl` 直写对侧 UB vs 经 GM 中转**：`b_via_gm` 3/3 挂 ⇒ 与数据面无关。
- **(c) flagId 复用/轮转**：`c_rotate` 挂、`c_broadcast` 挂、`c_wait1` 挂、`c_preset` 挂、`c_m2sameid` 挂；
  `c_m2canon` 不挂。
- **(d) AIV 侧 `Acq/Rls` 有/无**：`d_no_acquire` 3/3 挂 ⇒ 与 `Acq/Rls` 无关。
- **(e) N 扫描 + 重复**：`N=4/16/32/64` 关键变体各 3 次，见 §2.0b。

---

## 3. 挂死定位

### 3.1 逐核最后打印（`--trace`，已提交源码重跑）

每档 `--trace` 跑 N=2/N=4（`TRACE=1 TAG=_trace ... bash run_probe.sh`）。下表的「最后打印」与「`loop done` 计数」
都可从对应日志逐字复核：

| 档 | AIC 最后一条 `[PCT][AIC]` | `loop done` 计数 | 挂点判读 | 日志 |
|---|---|---|---|---|
| `base` N=2 | `r=1 waitFREE` | **0** | **AIC 卡在循环内**：首次轮间 `CrossCoreWaitFlag<0x2, PIPE_S>(F_FREE0)`；AIV 跑完全部轮次后卡在退出前 | `evidence/logs/base_N2_trace.log` |
| `base` N=4 | `r=1 waitFREE` | **0** | 同上 | `evidence/logs/base_N4_trace.log` |
| `base` N=16 | `r=1 waitFREE` | **0** | 同上 | `evidence/logs/base_N16_trace.log` |
| `c_m2mte3` N=4 | `r=1 waitFREE` | **0** | 同 base（AIC 卡循环内） | `evidence/logs/c_m2mte3_N4_trace.log` |
| `a_nowait` N=4 | `r=1 waitFREE` | 4（=2 核 × 2 次运行） | AIC 卡循环内；**两个 AIV 打印了 `loop done`**（AIV 退出前卡住） | `evidence/logs/a_nowait_N4_trace.log` |
| `c_m2setonly` N=4 | `loop done` | 6（=3 核 × 2 次运行） | **三核都跑完循环**，仍 `rc=124` ⇒ 卡在**退出** | `evidence/logs/c_m2setonly_N4_trace.log` |

⇒ **挂死位置随变体不同**：`base`/`c_m2mte3`/`a_nowait` 的 AIC 停在**循环内**的轮间 mode2 等待；
`c_m2setonly` 的**全部核**跑完循环、卡在退出。**不能用一句「循环跑完但内核不返回」概括所有挂死档。**
（上一版 README 曾用该表述，已被本条替换。）

### 3.2 收窄到「mode 2 回告方向的形态」

规范形态 = **两个 AIV 的 set 归并到同一个共享 id 上，且 AIC 每轮 wait 恰好 1 次**（`2 set(AIV) ↔ 1 wait(AIC)`）。
下表的「构造」按代码逐档写清（set 落在几个 id / AIC 每轮 wait 几次）：

| 档 | 构造 | 读数（N=16） | 说明 |
|---|---|---|---|
| `c_m2canon` | 两个 AIV set **同一共享 id**；AIC wait **恰好 1 次** | rc1 **3/3**（不挂） | **规范形态 ⇒ 不挂** |
| `c_broadcast` | 两个 AIV set 同一共享 id；AIC **wait 2 次** | **hang 3/3** | 共享 id 但多 wait 一次 ⇒ 挂 |
| `c_m2sameid` | 同上（共享 id；AIC wait 2 次） | **hang 3/3** | 同上 |
| `c_m2setonly` | 共享 id；AIC **wait 0 次** | **hang 3/3** | 少 wait 一次 ⇒ 也挂 |
| `base` | 两个 AIV set 落在**两个不同 id**（`F_FREE0`/`F_FREE1`）；AIC wait 2 次 | **hang 3/3** | **set 未归并到同一 id** ⇒ 挂 |
| `c_wait1` | 两个 AIV set 落在**两个不同 id**；AIC 只 wait **1 次**（另一个 id 的 set 无人 wait） | **hang 3/3** | **set 未归并**（且另一 id 无人 wait）⇒ 挂 |
| `c_m2mte3` | 两个 AIV set 落在两个不同 id；AIC wait 2 次；set 挂 `PIPE_MTE3` | **hang 3/3** | 与 set 挂哪条 pipe 无关 |
| `c_preset` | 循环前预置 FREE（复刻 M117 形态；两个 AIV 独立 id，AIC wait 2 次） | **hang 3/3** | 见 §1.2 |

⇒ 本档里**上面两个条件只要有一条不满足就挂**，满足的只有 `c_m2canon`：
- **id 未归并**：`base`/`c_wait1`/`c_m2mte3`/`c_preset` 的两个 AIV set 落在 `F_FREE0`/`F_FREE1` 两个 id 上
  ⇒ 挂（注意 `c_wait1` 的 AIC **只 wait 1 次**、`c_wait1` 仍挂 —— 所以「每轮恰好 1 次 wait」**单独并不充分**）；
- **id 已归并但 AIC 每轮 wait ≠ 1**：`c_broadcast`/`c_m2sameid`（2 次）、`c_m2setonly`（0 次）⇒ 挂。

这与塔广播 `20260927-tower-all-mode-2-n-set-1-wait-mode.md` 的实测口径**同向**（该广播测的是共享 id + `2 set ↔ 1 wait` 这一规范形态）。

### 3.3 「不挂」≠「门控正确」：内容 race，且对 `--trace` 敏感

- 非 `--trace` 运行里，**所有返回型变体**（`c_m2canon`/`c_mode4free`/`c_mode4full`/`dbg_rdyonly`/`dbg_nosync`）
  的 mismatched 都在 ~28k–65k / 65536（§2 表）；其中 `dbg_nodata`（空循环）为 65536×3，用于证明判据有牙。
- **唯一的 `rc0`（mismatched=0）出现在 `--trace`（循环内 printf）运行**（N=4）：
  `dbg_rdyonly`/`c_mode4free` 的 N=4 `--trace` 运行各 **`rc0` 3/3**（日志 `evidence/logs/dbg_rdyonly_N4_trace.log`、
  `c_mode4free_N4_trace.log`）；**同一档在非 `--trace` 下是 `rc1` 3/3**（`evidence/logs/dbg_rdyonly_N4.log`、
  `c_mode4free_N4.log`）。⇒ 内容对错**依赖循环内 printf 带来的时序**（与 M117「加 printf 能跑、去掉 printf 挂死」同族现象）。
  本 README**不引用**那些 `--trace` 的 `rc0` 作为「内容正确」的证据；它们只作挂点定位。
- ⇒ 本档**没有**验证出「既稳定返回、又内容正确」的跨核 flag 配置；门控正确性**未定界**（§6）。

### 3.4 候选替代形态（只主张「不挂」，不主张「门控对」）

```
// 约定：RDY_ID / FREE_ID 为 mode2 的共享 id；AIC 侧对 AIV1 通道不另开 id（mode2 由硬件按对折成一次）
AIC：CrossCoreWaitFlag<0x2, PIPE_S>(FREE_ID);                 // 每轮 1 次（不是 2 次！） // :366
      ... Fixpipe dualDstCtl -> 对侧 AIV UB ...
      CrossCoreSetFlag<0x2, PIPE_FIX>(RDY_ID);                // 每轮 1 次            // :387
AIV：CrossCoreWaitFlag<0x2, PIPE_V>(RDY_ID);                 // 两个 AIV 各等一次     // :458
      ... Acq<PIPE_V>/消费/Rls ...
      CrossCoreSetFlag<0x2, PIPE_V>(FREE_ID);                // 两个 AIV 各 set 一次   // :513
```

mode4 回告方向（M117 同族）见 `c_mode4free`（`FREE` 用 `id` 与 `id+16`）：同样**不挂**、内容同样 race。

---

## 4. 对 M117 的含义（写窄）

**设备读数直接支持（可引用）**：

1. 本档这套「AIC Fixpipe（`dualDstCtl` 直写对侧 AIV UB）→ 跨核通知 → AIV 消费 → 跨核回告 → 下一轮」的
   **合成形状**，在 **mode2 回告方向偏离规范形态（set 未归并到同一 id，或每轮 wait≠1）** 时**稳定挂死**（`N=4/16/32/64` 各 3/3）。
   **挂点位置随变体不同**：`base`/`c_m2mte3`/`a_nowait` 的 AIC 停在**循环内**的轮间 mode2 等待（§3.1），
   `c_m2setonly` 则三核跑完循环、卡在**退出**。
2. 规范配对（每轮 wait=1）与 mode4 回告方向**都不挂**（但内容 race）。
3. 该挂死**与本档的数据面、`Acq/Rls`、末尾屏障无关**。

**未定位（不得写成已定位）**：

- 本探针**没有**把 M117 `m=64` 的挂点（AIV 收尾段停在**本地 BufferID `Acq<PIPE_V>`**之前）搬进来：本档
  在 `base`/`c_m2mte3` 上卡住的是 AIC 的**跨核 `CrossCoreWaitFlag<0x2, PIPE_S>`**、在 `c_m2setonly` 上卡在**退出**，
  与 M117 记录的「本地 BufferID `Acq`」**不是同一个可观测点** ⇒ **M117 根因不在本 mission 的读数里**。
- 本档**所有返回型变体的内容仍有 race** ⇒ 「不挂」不等于「跨核门控正确」；门控正确性未定界。
- `N=64` 的 `dbg_rdyonly`（**无** AIV→AIC set）也挂 ⇒ 除「回告方向形态」外还有一条**未定界**的挂死现象。

---

## 5. 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd probe_crosscore_tail
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
./build/probe_crosscore_tail list                       # 变体清单
./build/probe_crosscore_tail base 16 1 out --trace      # 单变体单次（逐轮打印）
N=16 REPS=3 bash run_probe.sh                           # 全部 20 变体，每档各自进锁 + 锁内 timeout
N=16 REPS=3 VARIANTS="base c_m2canon c_mode4free" bash run_probe.sh
TRACE=1 TAG=_trace N=4 REPS=2 VARIANTS="base a_nowait c_m2mte3 c_m2setonly" bash run_probe.sh
bash tools/summarize.sh                                 # 从逐次日志机械重算 matrix_N<N>.txt
```

设备槽纪律：`run_probe.sh` **每档各自** `flock -w 300 /tmp/npu0.lock`，进锁先 `npu-smi`；每条变体每次运行
`timeout 20s`（在锁内）；`exit=124` 记成「HANG（未取得内容读数）」，锁未取得记「未取得读数」。

---

## 6. 完成度与未完成项（逐条）

- [x] 最小复现工程（独立 CMake + 含 `main()` 的 `.asc`，不碰既有 CMakeLists）
- [x] 1 AIC + 配对 2 AIV、N 轮、`dualDstCtl` 直写对侧 UB + CrossCore 通知 + AIV 消费（`Acq/Rls`）+ 回告
- [x] `N=16` 全 20 变体 ×3；`N=4/32/64` 关键变体 ×3；`--trace` 逐轮定位（N=2/4/16）；每档有 EXIT 码 + 内容读数 + 锁内 `npu-smi`
- [x] 变量矩阵 (a)–(e) 与隔离链（`dbg_rdyonly`/`dbg_nosync`/`dbg_nodata`/`dbg_nobar`）
- [x] 负向对照：挂死判据 `neg_nofree`（3/3 挂）；内容判据 `dbg_nodata`（65536×3 错）、`dbg_nosync`（返回、内容错）
- [x] 每个引用的读数都有随分支提交的日志（`evidence/logs/`；旧冒烟读数已删除）
- [ ] **未定界**：本档**所有返回型变体的内容都有 race** ⇒ 跨核 flag 的**门控正确性未验证**；
      `c_m2canon`/`c_mode4free` 只主张「不挂」，不主张「门控对」
- [ ] **未定界**：`N=64` 的 `dbg_rdyonly`（无 AIV→AIC set）也挂 ⇒ 除「mode2 回告方向形态」外另有一条挂死现象，未定界
- [ ] **未把 M117 收尾的本地 `Acq` 阻塞点搬进来**（本档挂点是跨核 `CrossCoreWaitFlag<PIPE_S>` 或退出，与 M117 收尾的本地 BufferID 阻塞点不同）
- [ ] **未做**：`neg_wrongval` 坐在 mode2 失配形状上 ⇒ 先挂死，读不到内容；本档没有「坐在不挂形状上、期望值错位」的返回型内容负向对照
- [ ] **未做**：`N>64` 的长稳、以及 56 核全参与（本档只让 AICore 0 的 1 AIC + 2 AIV 干活）
- [ ] **过程自陈**：内容对错对循环内 `printf`（`--trace`）敏感——N=4 的 `--trace` 运行是 `rc0`（`dbg_rdyonly`/`c_mode4free` 各 3/3，日志 `evidence/logs/dbg_rdyonly_N4_trace.log`、`c_mode4free_N4_trace.log`），
      非 `--trace` 运行是 `rc1`（`..._N4.log`）；本 README 已按「非 `--trace` 判内容、`--trace` 只定位挂点」的口径重写
- [ ] **过程自陈**：`run_probe.sh` 早期版本按 `<变体>.log` 命名导致 N=32 被 N=64 覆盖；
      现已按 `N`/`TAG` 打标，且矩阵由 `tools/summarize.sh` 从逐次日志机械重算
