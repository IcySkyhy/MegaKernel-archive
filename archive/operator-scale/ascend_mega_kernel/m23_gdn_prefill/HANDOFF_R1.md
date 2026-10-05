# M115 / B1 交办件 v1（**给人看**；不需要读代码）

**一句话**：GDN prefill 段的 mmad 版在 m=4097 上抛 `aicore error 507015`，已把范围从"整个段体"
收窄到"**Job2 里那三次 MTE2→L1 写入**"，但**没有找到根因**；下面是可直接复现的最小档、
已排除的假设、剩下的候选，以及**要请人回答的一个具体问题**。

---

## 1. 最小复现

```
# 构建（本目录下）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4     # 已验证 rc=0
# 逐档跑（每档一次独立 flock 会话；进锁后先 npu-smi 复查）
cd build && flock -w 900 /tmp/npu0.lock timeout 45 ./m23_gdn_prefill_j12        # J1+J2 档
```

| 档（编译期开关） | 期望 | 实际 |
|---|---|---|
| `m23_gdn_prefill`（全档 J1+J2+J3） | m=4097 算出 o/ht | **m=1 OK；m=4097 抛 507015** |
| `m23_gdn_prefill_j1`（只 J1） | — | **两档都 OK**（m=4097 1.930 ms） |
| `m23_gdn_prefill_j12`（J1+J2） | — | **m=1 OK；m=4097 抛 507015**（与全档**同签名**） |
| `m23_gdn_prefill_j12nz`（J1+J2，但 Job2 的三次 L1 写入**从第 1 个 chunk 起全跳**） | — | **两档都 OK** ← 定位到"Job2 的 L1 写入" |
| `m23_gdn_prefill_j12nz_{q,st,w}`（同上，**只跳其中一条**） | — | **三档都仍抛 507015**（**同签名**）⇒ 三条没有一条单独必要 |

设备错误报告（逐字，core 0）。**⚠ 签名只有 `mte error info` 一项逐次稳定**，`error code` 与
`l1 error info` **逐次不同** —— 逐次读数如下（全部为 core 0；出处见 `evidence/logs/`）：

| 出处（日志） | `error code` | `mte error info` | `l1 error info` |
|---|---|---|---|
| `j12_only_r1.log`（J1+J2 档，r2） | **0** | `0x13d10000000202ce` | `0x23b00001893` |
| `full_r2.log`（全档，r2） | **171** | `0x13d10000000202ce` | `0x29700001810` |
| `j12nz_q_r1.log`（只跳 q，r4） | 171 | `0x13d10000000202ce` | `0x15900001893` |
| `j12nz_st_r1.log`（只跳 ST，r4） | 171 | `0x13d10000000202ce` | `0x2090000e814` |
| `j12nz_w_r1.log`（只跳 W，r4） | 0 | `0x13d10000000202ce` | `0x2090000e814` |

逐字样例（`full_r2.log`，core 0）：
```
aicore error exception, core id is 0, error code = 171,
mte error info: 0x13d10000000202ce, vec error info: 0, cube error info: 0,
l1 error info: 0x29700001810, aic error mask: 0x395856
```
⇒ **可用的稳定判据只有**：`mte error info`（逐次逐位相同）+ `vec`/`cube` 全 0 + 是 AIC 侧；
`error code` 与 `l1 error info` **不能**当稳定签名用（同一档两次就不同）。
（m=1 档：o 有值但**没有合格判据读数**；`check_ref.py` 在 m=4097 档因异常拿不到 dump。）


### 1b. 同一份 `full_r2.log` 里另有三段我之前漏引的（逐字；超长行截断到 700 字符）

我此前只 grep 了 `aicore error exception` / `mte error info` 且只打印头几行，**漏掉了同一份日志里的
20 行 `multi-bit ECC` 与 48 行 `aivec error`**。逐字补引如下（**只陈述事实，不下结论**）：

* ① ECC 行（`full_r2.log:5`；同一份日志里共 20 行同文本）
```
The extend info: errcode:(171) errorStr: A multi-bit ECC error occurs when fixpipe reads L0C. See the RAS alarm handling. subErrType: 0x4.
```

* ② 异常上下文（`full_r2.log` 的 `dump info: …` 段，core 0）
```
EZ9999[PID: 466547] 2026-09-27-10:10:56.953.202 (EZ9999):  The error from device(chipId:0, dieId:0), serial number is 1206, there is an aicore error exception, core id is 0, error code = 171, dump info: pc start: 0x120041000000, current: 0x120041000a80, sc error info: 0xffffffffffff, su error info: 0xfbbddff31d34ffee,0xfd74f2afd800ffff, mte error info: 0x13d10000000202ce, vec error info: 0, cube error info: 0, l1 error info: 0x29700001810, aic error mask: 0x395856, para base: 0x120000200000, mte error: 0, aic cond: 0.
```

* ③ AIV 侧异常（`full_r2.log` 尾部，core 56/57；同份日志共 48 行 `aivec`）
```
The error from device(chipId:0, dieId:0), serial number is 1207, there is an aivec error exception, core id is 56, error code = 0, dump info: pc start: 0x120041000d00, current: 0x120041001c14, sc error info: 0xffffffffffff, su error info: 0xe4fa32553abe01de,0xde6515fec800bb1b, mte error info: 0x850a480000020043, vec error info: 0x410062180031126e, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x120000200000, mte error: 0, aic cond: 0.
```

* ④ 任务级收尾（`full_r2.log`）
```
An error occurred in the kernel task, retCode=0x26, [aicore exception].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1262]
```

**行数（同一份 `full_r2.log`）**：`multi-bit ECC` **20** 行、`aivec error exception` **48** 行、
`aicore error exception` **24** 行、`retCode=0x26` **2** 行。

**一条纯事实的观察（不下结论）**：`errStr` 把出事位置指向 **fixpipe 读 L0C**；而 §3 的四条候选
（同 chunk 二次写同一 L1 地址 / 64KB `dstNzC0Stride=128` 的 `Nd2Nz` / `TPosition::B1` 声明 /
MTE2 与紧邻 MTE1·FIXP 的令牌次序）**都在 MTE2→L1 的写侧**，与 `errStr` 指的位置**不是同一处** ——
请人类看日志时留意。另：**这行 `multi-bit ECC` 在不加 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 时就已存在**
（逐字可查）⇒ 该环境变量不是拿根因的关键。**本 mission 未据此改任何代码/同步/L1。**


### 1c. "两个 kernel task"这个事实 + "p4097 作为第一个且唯一一个 task"的对照档（人类自己问出来的）

**事实（本文档此前没写、由人类问出来后核实）**：本档的 `main()` 里 case 表有两项
（`m23_gdn_prefill.asc:274-275`：`{"m1",48,1}` 与 `{"p4097",48,4097}`），循环里每个 case **一次**
`<<<nblk,0,stream>>>`（`:198`）+ `aclrtSynchronizeStream`（`:209`）
⇒ **同一 stream 上串行两个 kernel task**（m=1 先、p4097 后，中间有同步）。
⇒"两个 task 并发互扰"**已被这一事实排除**；但此前**从未跑过"p4097 是第一个、也是唯一一个 task"**。

**新对照档**（运行期开关 `M15GP_ONLY_P4097=1`；默认不设时循环起止**逐字不变**）：
程序自打印 `[M23] M15GP_ONLY_P4097=1 ⇒ 本次跑 case [1, 2)`；结果**仍挂 507015**，
且 `mte error info`（core 0）`0x13d10000000202ce` 与两 case 档**逐位相同**；
`multi-bit ECC` 20 行 / `aicore error exception` 24 行 / `aivec error` 48 行（与两 case 档相同）；
`timeout or trap` 52 行。

**判读（对照预登记表 `evidence/logs/only_p4097_prereg.md`，规则先写后跑）**：
命中"**仍挂 + 同签名 ⇒ 与 m=1 那次留下的设备状态无关；故障属 p4097 这个 shape/这段代码自身**"。
**限度**：只跑 1 次；`timeout or trap` 行数在同故障上会变（12/20/24）⇒ **不是**稳定签名；
未做 SLOG 对照（塔已自行跑过）。逐字读数见 `evidence/logs/only_p4097_r1.md`。


### 1d. M 扫描：从 M=1 遍历找"最小复现的 M"（人类指令）—— **非单调、且非确定**

开关 `M15GP_ONLY_M=<int>`（默认不设时行为与改动前逐字一致；见证行打印本次跑的 M）。逐条读数见
`evidence/logs/m_scan_r1.md`（每个 M 一个日志 `mscan_m<M>.log`）。

| M | 判定 | M | 判定 | M | 判定 |
|---|---|---|---|---|---|
| 1,2,4,8,16 | **OK** | 24 | OK | 512,1024,2048,4096,4097 | **FAIL**（4/4） |
| 17,18,19,20 | **OK** | **26** | **FAIL**（且三次跑出 **FAIL/OK/FAIL**） | 64,65,128,256 | FAIL |
| 27,28,29 | OK | 30 | FAIL | 31 | OK |

* **最小 FAIL = 26、最大 OK = 31**；**非单调**（24 OK → 26 FAIL → 27/28/29 OK → 30 FAIL → 31 OK → 32 FAIL）。
* **临界不在 chunk 边界**：26~32 全在**同一个 chunk（BT=64）之内** ⇒ 按人类给的判据属"**指向单 chunk 内部**"那支。
* 失败时 `mte error info` 的**低 8 位与 4097 档相同**（`…0202ce`），第 5 位（核号字段）随运行变 ⇒ 同类故障、报错核不固定。
* **命中率随 M 上升**（事实）：`M≤24` 0/6 命中；`26~32` 4/8 命中；`M≥512` 4/4 命中，且 `multi-bit ECC` 行数随 M 增大（1→20）。
* **因"同一 M 三次两次不同"，本档不能给出"最小复现 M = 常数"**；能说的是上面那张命中率事实。
  `timeout or trap` 行数**同档也会变**（4096 档 48 行 vs 4097 档 6 行）⇒ 非稳定签名。
* 限度：每个 M 只跑 1 次（`26` 三次）；`25`、`33~63` 未测；未做 SLOG 对照与命中率矩阵；未改被测段体。


### 1e. M=1 的重复性统计（人类问"m=1 也是概率失败吗"）—— **50 次全 PASS、0 次 FAIL**

同二进制、同开关路径（`M15GP_ONLY_M`），**每次 run 一次独立 `flock`**；预登记与判据见
`evidence/logs/m1_repeat_prereg.md`，逐字读数 `evidence/logs/m1_repeat_r1.md`。

| 档 | N | FAIL | 命中率 | 95% CI（Wilson） | 逐次序列 |
|---|---|---|---|---|---|
| **M=1**（主档） | **50** | **0** | 0% | [0%, 6%] | 50 个 `P` 连续 |
| M=24（对照） | 20 | **3** | **15%** | [5%, 36%] | `PPPPPPPPF PPF PPPF PPPP`（F 在 i=9/12/16） |
| M=4097（对照） | 10 | **10** | **100%** | [72%, 100%] | `FFFFFFFFFF` |
| M=1（4097 之后，跨档残留探测） | 10 | 0 | 0% | [0%, 28%] | 10 个 `P` 连续 |

* **措辞纪律（预登记先定）**：M=1 的 50 次全 PASS ⇒ 只能写「**在 50 次里未观察到失败
  （命中率上界约 6%，95% CI）**」，**不得**写"m=1 不会失败"。
* **更正上一轮**：上一轮 M=24 只跑了 6 次（0/6）我就倾向"小 M 不挂"；20 次采样后命中率 **15%** ⇒
  **"M≤24 不挂"这个倾向不成立**（样本太少）。
* **跨档残留**：M=4097 批之后回到 M=1 跑 10 次 ⇒ 10/10 PASS ⇒ 未观察到"被 4097 批污染"。
* 失败签名：失败档 `mte error info` 取值去掉第 5 位（核号字段）后**低位一致（`…0202ce`）** ⇒ 与 4097 档同类。
* **限度**：本档 `npu-smi` 的 Health 字段**未取到**（我的抽取格式与实际列不匹配，采到空串）⇒ 原样记"未取到"，
  不据此下结论；每 run 的设备错误报告都在 `evidence/logs/rep/rep_*.log`；未做"更大 N 的上界"、未做 SLOG 对照。


### 1f. 最小 AIC 探针：L0A/L0B/L0C 的 16/17 tile 边界 —— **未复现**

人类改向"直接测试硬件"后新建的最小探针：`m23_gdn_prefill/probe_l0/probe_l0.asc`（目标 `probe_l0`，1 个 AIC，
`__global__ __cube__`、fp32、N=K=128、tile=16 行、iters=200）。两轴 = tiles{1,2} × arm{0=段体现用形态,
1=tile 之间加显式 `PipeBarrier`}。

| tiles | arm | rc | ECC 行 | `mte error info` |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | （无） |
| 2 | 0 | 0 | 0 | （无） |
| 2 | 1 | 0 | 0 | （无） |
| 1 | 1 | 0 | 0 | （无） |

⇒ 命中预登记第三支：「**两臂都不挂 ⇒ 探针没复现到触发条件**」。因此**既不能说**"软件同步缺陷"，
**也不能说**"硬件约束"；指纹比对**没有对象**。限度：未上 mix(1,2) 满核、未试更大 iters / 其它 dtype /
L1 多槽位形态；`tiles=2` 读回的 `C[0][0]=128.0` 与我打印的期望式不匹配（探针用途是"是否报错"，未追查）。
逐字见 `evidence/logs/probe_l0_r1.md`。M=16/17 的段体统计批按"先搁置"未跑完即停（M16 30/30、M17 26/30），未据此下结论。


### 1g. 【已修·人类批准】L0C 的 M↔FIXP 成对交接（独立复审的唯一 P1）

**diff**：`GdnPrefillAic::DoMmad` 净增两条 M 侧 BufferID —— `BufAcquire<PIPE_M>(GP_BUF_L0C)` 在 `MmadF32` 之前（WAR）、
`BufRelease<PIPE_M>(GP_BUF_L0C)` 在 `BufRelease<PIPE_M>(GP_BUF_L0)` 之后（RAW），并改正顶部注释。
只用 BufferID、**未用 set/wait flag**；未动 L1 布局/资源记账；未动 P2-2（只登记不修）。构建 rc=0。

| 项 | 档 | 改前 | 改后（逐次序列） |
|---|---|---|---|
| A | `M=4097` ×10 | 10/10 FAIL | **`PPPPPPPPPP`（10/10 OK）**；`multi-bit ECC` 0 行、无 `mte error info` |
| B | `M=1` ×10 | 50/50 OK | `PPPPPPPPPP` |
| C | `M=17` ×10 | 26/30 OK | `PPPPPPPPPP` |

**D：第一次拿到合格数值读数**（`check_ref.py`，rtol=2e-3）——
`m4097`：**`o` 超界 0/25,171,968，`max|Δ|=3.042e-08`**（改前该档**从未跑完**、从来没有可判对象）
⇒ **输出 `o` 已达 fp32 舍入水平**；但 **`ht` 超界 778,845/786,432、`max|Δ|=9.151e-01`** ⇒ **末态仍错**。
由于 `o` 正确说明跨 chunk 状态推进每一步都对，`ht` 的问题**不在这条 L0C 通路上**，嫌疑在**末态写回路径**
（`StoreState` 的列窗转置/`DataCopy`），**本轮未查**。

**口径警示**：`check_ref.py` 会把历次扫描留下的**陈旧 dump**（m26/m27/… 是改前 + 概率失败批）一并判 ⇒ 那些 FAIL 不代表改后状态。
逐字见 `evidence/logs/l0c_fix_r1.md`。


### 1h. 【已修】末态 `ht` —— **布局/转置 bug**（不是缺同步），修后 `o`/`ht` 双双 0 超界

**判定（零设备审计，逐字见 `evidence/logs/ht_audit_r1.md`）**：`StoreState` 的跨 pipe 交接**是齐的**
（V→MTE3 成对 `GP_BUF_BLK`、上传缓冲 `UP` 令牌、寄存器↔UB 有 `MemBarVL`）⇒ **不是缺同步**；
错在 `TransposeVF(stUb + cb * GP_VL * GP_DK, tmp, GP_DV, GP_VL)` 的 **R/C 与源 stride 不符**：
该函数要求源行 stride = C，而 ST 是 `[DV,DK]`（stride = DK=128），调用却传 `R=DV, C=VL=64`
⇒ 源被当成「128 行、stride 64」错位读取。**同文件自对照**：`LoadState` 的读入路径有 gap-`DataCopy` 打包、
R/C 自洽 ⇒ 本来是对的；**只有写回这条错** ⇒ 解释"只错 `ht`、`m=1` 就错、结构性 O(1)"。

**修**：`TransposeVF(…, GP_VL, M15G::GP_DK)`（R=64, C=128）；构建 rc=0；只改这一处。

**新鲜判据（陈旧 dump 已全部移入 `stale_dumps/` —— 即 `m23_gdn_prefill/stale_dumps/`，不在 `build/` 下；`check_ref.py` rc=0）**：
```
m1    : o 0/6144 超界 max|Δ|=7.7e-09 | ht 0/786432 超界 max|Δ|=3.3e-08 | PASS
m4097 : o 0/25171968 超界 max|Δ|=3.0e-08 | ht 0/786432 超界 max|Δ|=1.5e-07 | PASS
```
限度：各档只跑 1 次新鲜读数；未重跑满量级统计；未做 SLOG/性能；**不自行声称 clean**（按纪律交复审）。
逐字见 `evidence/logs/ht_fix_r1.md`。

## 2. 已排除的假设（每条一句话 + 支撑读数）
1. **核间同步死锁** —— 排除：只跑 J1 的档在真实 shape 满量级跑通（65 chunk × 48 head、每 head 195 次握手）。
2. **CrossCore mode 选错（该用 mode 0）** —— 排除：探针实测本段用的 mode 2 形态在 200 轮下正常。
3. **flagId 计数器每 id ≤15 未配平** —— 排除：同上（200 轮 ≫ 15）。
4. **set 挂的 pipe 非法（`PIPE_V` / `PIPE_FIX`）** —— 排除：两个 pipe 各单独实测正常。
5. **`Job2` 少配一次 BufferID acquire** —— 排除：`Job1`/`Job2` 的令牌用法逐项对照**完全相同**。
6. **"核内 BufferID 成对 acquire"是这里的原因** —— 排除（对 j1 档而言）：j1 档 65 chunk 满量级跑通。
7. **故障在 J3（三角/后段）** —— 排除：加回 J3 不改变错误签名。
8. **q/k 平面行 stride 不匹配（真 bug，已修）** —— 与本次故障**无关**：修后签名不变。
9. **具体某一条 L1 写入是必要条件** —— 排除：三条各跳一条都仍复现。
10. **同步配对必须严格 N set ↔ 1 wait**（本 mission 顺带实测的舰队口径）—— 另一支读数（非本条故障）。

## 3. 剩余候选（每条一句话 + 静态依据）
1. **同 chunk 内对同一 L1 地址写两次** —— `Job2` 的 q 重装写的 `L1Kq(h)+32KB`，本 chunk 内 **Job1 已经写过**；
   静态对照表里这是**唯一**这样的位置（其余两个 ND2NZ 目的区在本 chunk 内是首次写）。
2. **一次 64KB 的 L1 写入** —— `L1St(h)`：`rows=128, cols=128, dstNzC0Stride=128`，是段内**唯一** 128 行/64KB 的 `Nd2Nz`。
3. **`TPosition::B1` 的声明** —— `Job2` 里的 `l1w` 按 **B1** 声明；而 **`Job3` 的 `l1kt` 也是 B1**，
   两处都在"每次 job 一遍的 MTE2→L1 写"路径上；相对照，**`Job1` 的两个张量全是 A1**。
   ⇒ 影响面判断时请把"Job1 无 B1 / Job2 与 Job3 各有一处 B1"一起看（不是 Job2 独有）。
4. **这段 MTE2 与紧邻 MTE1/FIXP 之间的令牌次序** —— 若"只保留一条"三档仍都复现，就是它（静态上看不出差异，
   所以需要那组补集读数来三分）。

## 4. 要请人回答的问题（**一句话**）
> **在 dav-3510 上，同一 chunk 内对同一 L1 地址做两次 `Nd2Nz` 写入（或一次 64KB、`dstNzC0Stride=128` 的
> `Nd2Nz`，或把 `LocalTensor` 按 `TPosition::B1` 声明在 L1 的高半区），有没有已知限制/是否需要额外的
> pipe 间隔离？另外：**L1 的 A1 / B1 是不是各自 256KB 的独立地址窗**（跨窗声明是否合法）？**

（我们查到的官方文档只说 L1 是**一根扁平 512KB**、16 Bank×32KB、地址编码
`L1_ADDR[18:0]={BANK,BANK_DEPTH,BG,BANK_WIDTH}`；CANN/bisheng 头文件里**没有** A1/B1 的基址常量，
本仓的"[0,256KB)=A / [256KB,512KB)=B"只是约定 ⇒ **窗口语义未确定**。）

## 5. 边界（没试的 / 为什么）
* **补集实验（只保留一条：q / ST / W）已就绪但两轮都没跑完** —— 设备槽排队（B2–B5 并发）+ 工具时限；
  代码与 target 已入库（`M15GP_J2_NZ_WHICH=4/5/6`），逐档单独进锁即可。
* **L1 窗口最小探针没做** —— 排在补集之后（两者会咬合：若"只留 ST"出结果，直接拿 ST 那一档去做探针）。
* **设备档全程没拿到任何合格数值判据**（`check_ref.py` 从未有可判对象）⇒ 本段体**不是**已验收件。
* **未做硬件级取证**（无 ISA/微架构 dump）⇒ 上面所有判读**只针对本段的现象**，**不构成硬件结论**。
