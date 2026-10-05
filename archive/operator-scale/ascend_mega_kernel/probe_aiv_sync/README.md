# M64 —— AIV 核间屏障可靠性探针（`probe_aiv_sync`）

> **一句话结论（可复现，见 §3/§4）**：**AIV 间的 mode-0 会合本身是可靠的**；M53 读到的「残缺计数」
> 来自一个**协议缺陷：读窗口没有关**——发布屏障只保证"本轮所有写都落盘了"，它**不阻止跑在前面的核
> 把下一轮的写压进 MTE3 队列**，于是慢核读本轮表时，表已经被快核的下一轮值覆盖。
> 补上一次**收尾会合**（"我读完了才允许别人覆写"）之后，56 核 × 16 轮全核交换 **0 违规、9/9 次运行**
> 全绿；同一条件下现行写法（M53/m19/m15/m20 都用它）是 **75~117 处 (核,轮) 违规（9 次）、16 轮里只有 1~3 轮干净**。
> 另外：**跨核 GM 可见性既不依赖 flag，也不需要 DCCI**（纯轮询 56/56 收敛、最慢 16 次迭代，见 §7）——
> M53「纯轮询 0 核收敛」的读数同样是**读窗口没关**造成的（它轮询的是一个正在被覆写的目标）。
> 按用户铁律（"默认假设是我们用法不对"）裁定：**这是用法不对，不是平台不可靠。**

---

## 0. 结论速览（TL;DR）

| # | 结论 | 判据（可复现） |
|---|---|---|
| 1 | **mode-0 all-to-all 会合可用**：全体 AIV 到齐才放行，释放时刻跨核散布带 `wait<PIPE_S>` 时仅 **38 cycle**（≈20ns） | §3.2 表；设备侧 `SYS_CNT`（不靠 host 时钟） |
| 2 | 现行写法（只有发布屏障）**必然出现"读到下一轮的值"**：快照逐词归因显示 100% 的错词都是 **T+1 轮 ticket**，`past/zero/other` 全 0 | `evidence/diag_A2_open_only.txt` |
| 3 | **成功配方 R1 = 发布会合 + 收尾会合**：`set<MTE3>/wait<MTE2>` + `set<MTE2>/wait<MTE3>`；**9/9 次 0 违规** | §4.1；`evidence/logs/aiv_sync_A3_open_close.log` |
| 4 | **成功配方 R2 = 发布会合 + 每轮独立窗口**（用内存换第二次会合）；同条件 9/9 全绿 | §4.2；`aiv_sync_A20_sepwins.log` |
| 5 | **wait 挂哪条 pipe 是决定性的**：`wait<PIPE_V>` / `wait<PIPE_MTE3>` 都**挡不住后续 MTE2 读**（769~812 / 793~823 违规，9/9 次）；`wait<PIPE_MTE2>` / `wait<PIPE_S>` 才行（0 违规 9/9） | §3.3（A6/A7 vs A3/A5） |
| 6 | `set` 挂哪条 pipe、要不要 `drain`、要不要 DCCI、flagId 是否复用（16 轮固定一对 id）——**在本实验里都不是关键变量** | §3.4 |
| 7 | **跨核 GM 可见性不依赖 flag，也不需要 DCCI**：纯轮询 56/56 收敛（64B 槽 ≤5 次迭代、4B 密排 ≤16 次） | §7 `probe_visibility` v0~v6 |
| 8 | **写几何只放大"坏协议下的受害程度"**：M53/m19 的 `st=1,wl=4`（56 核 4B 挤在 224B）在现行（只有发布会合的）配方下 = **B4**，四个独立会话的 9 次区间分别是 **262~487 / 276~456**（本 mission 会话 1/2）、**229~296 / 339~477**（reviewer-m64 会话 2/3）；同条件 `st=16,wl=64` = **A2**，四会话 **75~117 / 79~129 / 74~110 / 73~100**（合并 **73~129**）⇒ **比值约 2~6.5×（按各会话自身区间两两组合）**，且 **B4 的绝对量比 A2 更吃会话环境**（B4 跨会话差近 2×，A2 四会话都落在 73~129）。**换成 R1/R2 配方后，四种几何全部 9/9 次 0 违规**（B0~B3） | §2（表下复核注）+ §4.4 |
| 9 | **mode 1 不能用于跨 AI Core 的 AIV 同步**（语义只覆盖同一 AI Core 内 2 个 AIV）：741~815/896 违规 | §3.4（A14） |
| 10 | **只让一半核参与会合 = 死锁**（9/9 次超时未返回） | §10（C0_halfset） |
| 11 | **mode2 是点对点**（跨 AI Core 的 set 满足不了别组的 wait：9/9 挂死）；**AIC 侧把 set 挂 PIPE_MTE3 = 静默空操作**（对侧永久挂死：9/9） | §7.5（`probe_aic_aiv` v2/v1）；机制见 `kernel_event.h:264` |
| 12 | **flagId 计数器是消费型（不是 sticky）**：同一个 id 既做发布又做收尾、16 轮反复用，**9/9 次 0 违规** ⇒「一个 id 就够」实测成立（预算：2 个/段 → 8 组；共用一个 id → 16 组） | §4.3（`A26`/`A16` vs 无会合对照 `A0`/`A1`） |

---

## 1. 读数方式与探针结构（为什么这样设计）

mission 硬要求"用一张可校验的计数/序号表直接读屏障正确性，不许靠下游数值反推；读数给分布而非单点"。
本探针据此把**观测**和**被测对象**分开，并用了**三件互相独立的仪器**：

1. **序号表（被测对象）**：GM 上 64 个槽位，槽跨度 `st`（int32 字数：1/8/16 ⇒ 4B/32B/64B），
   每核每轮只写自己的槽：`TICKET(T,rank) = (MAGIC16<<16) | ((T&0xFF)<<8) | (rank&0xFF)`，
   `T` = 轮次序号（1..rounds），未触碰位置期望 0（host 预清）。
   `st=1,wl=4` 是 **M53 `probe_aiv_barrier.asc` 与 m19 `WriteToken()` 的逐字复刻**（56 核 4B 挤在 224B 内）。
2. **设备侧自读数（仪器 A）**：屏障之后**每核自己**读回整张表，在 `__VEC_SCOPE__` 里数
   "head 命中本轮 ticket 的槽数"（`headOK`）与"已写词命中数"（`wordOK`）。分布按 (核,轮) 全员落盘。
3. **设备侧快照 + host 逐词归因（仪器 B）**：把每核读回的那 4KB 原样搬回 GM，host 对**每个 4B 词**
   给出归因：`match(本轮) / future(T' > 本轮) / past(T' < 本轮) / zero(从未写) / other`。
   `future>0` 就是"读的时候别人已经把**后面轮次**写进去了"的**直接证据**。
4. **设备侧时间戳（仪器 C）**：每核每轮记 `SYS_CNT`（`GetSystemCycle()`，设备侧计数器）在四个点的值：
   写完 / 排空后 / 会合返回后 / 读回后。用来做"慢核的读完成时刻 vs 快核的下一轮写时刻"的交叉验证。
   **host 时钟不参与任何判据**；`SYS_CNT` 的跨核可比性有自检（各核 kernel 起点 stamp 散布见 CYCLE 行）。

**变体轴**（每个变体一次独立进程、一次 kernel launch；同一份骨架代码由 host 侧表驱动 +
mode/pipe 编译期模板分发）：

| 轴 | 取值 |
|---|---|
| ① 会合模式 | 无 / mode0 / mode1 / mode2（+ 点对点语义对照见 §7、§7.5） |
| ② flagId | 固定一对（16 轮复用）/ 7 对轮换 / 16 个两两配对（`2*(r%8)`、`2*(r%8)+1`） |
| ③ SET/WAIT pipe | set ∈ {MTE3, V, MTE2}；wait ∈ {MTE2, PIPE_S, V, MTE3}（含 4 种错挂） |
| ④ drain | 无 / `PipeBarrier<MTE3>` / `MTE3_S` 事件 / `MTE3_S`+`DCCI(clean)`+`DSB` |
| ⑤ 写几何 | `st×wl` = 1×4（M53 形态）/ 8×4 / 8×32 / 16×64 |
| ⑥ 偏斜 | 有（每核每轮 (aiv%8)×8 次 32B 空转 DMA）/ 无 |
| ⑦ 轮询 | 单次读 / bounded 纯轮询(4096) / 发布会合+轮询混合 |
| ⑧ 核数/参与度 | blockDim=28(56 AIV) / 14(28 AIV) / 只让一半核 set |
| ⑨ **收尾会合** | 无 / `set<MTE2>+wait<MTE3>` / `set<MTE2>+wait<MTE2>` / `set<MTE3>+wait<MTE3>` |
| ⑩ **窗口复用** | 复用同一张表（M53/m19 形态）/ 每轮独立窗口 |
| ⑪ 负向对照 | 无屏障但只校验自己那一格（本方法的已知盲区） |

### 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
./build/probe_aiv_sync                                   # 打印全部变体清单
./build/probe_aiv_sync A3_open_close 1 out --snap --trace # 单变体单次（可带快照/时序）
bash run_probe.sh                                        # 全量：每变体 9 次独立进程 + 归档
```

**退出码（三态）**：`0` = 比过且 56 核 × 全部轮次精确匹配；`1` = 比过有差异（并给出分布）；
`2` = 没得比（launch 失败/输入缺失/读数行缺失）。`run_probe.sh` 另把 `124`（超时未返回）单列为 `HANG`。

---

## 2. 主表：配方 × 参数 × 收敛次数/9 × 失败形态

下表由 `tools/make_tables.py` 从 `evidence/logs/aiv_sync_*.log` 的**原始 SUMMARY 行**机械汇总
（逐变体：9 次独立进程；"收敛次数/9" = rc==0 的次数；"违规 (核,轮) 区间" = 9 次里 viol 的最小~最大；
"干净的轮数" = 9 次里 rounds_clean 的取值集合 / 总轮数）。命名：`A*` 屏障配方与参数轴、`B*` 写几何轴、
`C*` 参与度轴、`N*` 负向对照。**"失败形态"一列的归因来自 §3 的两件独立仪器**（逐词归因 + 设备侧时间戳），
不是从数字猜的。

**表的读数已被独立复现（reviewer-m64 第 1 轮复审）**：它从 **tip 源码**在 `/tmp` 重建、自己各跑 9 次独立进程
（设备当时无其它作业），逐条对上本表：`A0` 892~895、`A2` 74~110、`A3`/`A5`/`A16`/`A20`/`A26` 各 0 viol·9/9、
`A6` 764~807、`A7` 764~818、`A15`/`C0` rc=124、`N0` rc=0·viol=896、`probe_visibility` 56/56 收敛、
`probe_m53_reverse` 逐变体同向。
**唯一明显的量级偏差是 `B4`**：本 mission 会话 1 = 262~487、会话 2 = 276~456；复审会话 2 = 229~296、会话 3 = 339~477
（同条件 `A2` 四会话都在 73~129）——同一变体跨会话可差近 2×，
因此本报告对 B4 只主张**方向与比值**（比同会话 A2 约 **2~6.5×**），不主张绝对量（详见 §0 第 8 条与 §4.4）。
本 mission 的**同会话两会话复测**留在 `evidence/logs/session2_A2_B4.log`（会话 2：`A2` 79~129、`B4` 276~456）；
复审的两次会话读数见 `.tower/comms/reviews/review-feat-m64-…-reviewer-m64-r1.md` / `…-r2.md`。

| 变体 | 配方 / 参数 | 收敛次数/9 | 违规 (核,轮) 区间 | 干净的轮数 | 失败形态（§3 归因） |
|---|---|---|---|---|---|
| `A0_none_readonce` | 无屏障（写完即读） | **0/9** | 892~895 | 0 /16 | 无任何会合（对照：证明表灵敏） |
| `A1_poll_nobarrier` | 无屏障 + 纯轮询 4096 | **0/9** | 118~222 | 0 /4 | 轮询追一个正在被覆写的目标 ⇒ 0 核收敛 |
| `A2_open_only` | m0 型：set<MTE3>/wait<MTE2> 仅发布（**现行**） | **0/9** | 75~117 | 1~3 /16 | 读到 **T+1 轮** 的值（读窗口未关） |
| `A3_open_close` | 发布 + 收尾 `set<MTE2>/wait<MTE3>`（**候选 R1**） | **9/9** | 0 | 16 /16 | **无违规** |
| `A4_open_only_waits` | 仅发布，wait<PIPE_S> | **0/9** | 90~110 | 1~3 /16 | 同 A2（wait 挂 PIPE_S 不改变窗口未关这件事） |
| `A5_close_waits` | 发布(wait<S>) + 收尾（候选 R1'） | **9/9** | 0 | 16 /16 | **无违规** |
| `A6_open_close_waitV` | 发布 wait<PIPE_V> + 收尾（**错挂**） | **0/9** | 769~812 | 0 /16 | 读早于放行（wait<PIPE_V> 挡不住 MTE2 读） |
| `A7_open_close_waitE3` | 发布 wait<PIPE_MTE3> + 收尾（**错挂**） | **0/9** | 793~823 | 0 /16 | 读早于放行（wait<PIPE_MTE3> 挡不住 MTE2 读） |
| `A8_setV_close` | 发布 set<PIPE_V> + 收尾（**错挂**） | **9/9** | 0 | 16 /16 | **无违规** |
| `A9_setE2_close` | 发布 set<PIPE_MTE2> + 收尾（**错挂**） | **9/9** | 0 | 16 /16 | **无违规** |
| `A10_nodrain_close` | 发布（不排空）+ 收尾 | **9/9** | 0 | 16 /16 | **无违规** |
| `A11_pipebar_close` | 发布（PipeBarrier<MTE3>）+ 收尾 | **9/9** | 0 | 16 /16 | **无违规** |
| `A12_dcci_close` | 发布（MTE3_S+DCCI+DSB）+ 收尾 | **9/9** | 0 | 16 /16 | **无违规** |
| `A13_dcci_close_rd` | A12 + 读侧 DCCI | **9/9** | 0 | 16 /16 | **无违规** |
| `A14_m1_close` | **mode1** + 收尾 | **0/9** | 741~815 | 0/1 /16 | mode1 只同步同一 AI Core 内 2 个 AIV ⇒ 跨核不受保护 |
| `A15_m2_close` | **mode2（AIV 自 set/自 wait）** + 收尾 | **0/9**（挂死 9） | - | — | mode2 的 set 记在 AIC 侧 ⇒ AIV 自己的计数器永不满（死锁） |
| `A16_fid_fixed_close` | 发布+收尾，flagId 固定一对（16 轮复用） | **9/9** | 0 | 16 /16 | **无违规** |
| `A17_fid_rot7_close` | 发布+收尾，7 对 id 轮换 | **9/9** | 0 | 16 /16 | **无违规** |
| `A18_noskew_close` | 发布+收尾，无偏斜 | **9/9** | 0 | 16 /16 | **无违规** |
| `A19_poll_hybrid_close` | 发布+轮询混合+收尾 | **9/9** | 0 | 4 /4 | **无违规** |
| `A20_sepwins` | 仅发布 + **每轮独立窗口**（**候选 R2**） | **9/9** | 0 | 16 /16 | **无违规** |
| `A21_28aiv_close` | 发布+收尾，**28 AIV** | **9/9** | 0 | 16 /16 | **无违规** |
| `A22_close_closepipeE2` | 收尾 wait 挂 MTE2（**错挂**） | **0/9** | 253~380 | 1/2 /16 | 收尾 wait 挂 MTE2 ⇒ 挡不住下一轮 MTE3 写（读窗口仍开） |
| `A23_close_closesetE3` | 收尾 set 挂 MTE3（**错挂**） | **9/9** | 0 | 16 /16 | 收尾 set 挂 MTE3：本实验里也 0 违规（形式上侥幸，见 §8） |
| `A24_poll_sepwins` | 无会合 + 独立窗口 + 纯轮询 | **9/9** | 0 | 4 /4 | **无违规** |
| `A25_sepwins_dense4` | 仅发布 + 独立窗口 + **4B 密排几何** | **9/9** | 0 | 16 /16 | **无违规** |
| `A26_fid_same_id` | 发布+收尾**共用同一个 flagId（1）**，16 轮反复用 | **9/9** | 0 | 16 /16 | **无违规**（同一 id 兼做发布/收尾） |
| `A27_fid_same_id_sep` | 同 A26 + 每轮独立窗口 | **9/9** | 0 | 16 /16 | **无违规**（同一 id + 独立窗口） |
| `B0_geom_dense4` | 发布+收尾，几何 **st=1,wl=4（M53 形态）** | **9/9** | 0 | 16 /16 | **无违规** |
| `B1_geom_align4` | 发布+收尾，几何 st=8,wl=4 | **9/9** | 0 | 16 /16 | **无违规** |
| `B2_geom_align32` | 发布+收尾，几何 st=8,wl=32 | **9/9** | 0 | 16 /16 | **无违规** |
| `B3_geom_align64` | 发布+收尾，几何 st=16,wl=64 | **9/9** | 0 | 16 /16 | **无违规** |
| `B4_dense4_openonly` | **仅发布**，几何 st=1,wl=4（M53 逐项复刻） | **0/9** | 262~487 | 1 /16 | M53/m19 形态：4B 密排 + 只有发布会合（最差组合） |
| `B5_dense4_28aiv` | 发布+收尾，几何 st=1,wl=4，28 AIV | **9/9** | 0 | 16 /16 | **无违规** |
| `C0_halfset` | 发布+收尾，**只让一半核参与** | **0/9**（挂死 9） | - | — | 只让一半核 set ⇒ 会合永不满（死锁） |
| `N0_negctrl_selfonly` | 无屏障，**只校验自己那格**（负向对照） | **9/9** | 896 | 0 /16 | **0 违规（假 PASS）**：只校验自己那格 ⇒ 无同步也过判 |


---

## 3. 机理证据：为什么是"读窗口没关"而不是"屏障不可靠"

### 3.1 逐词归因（仪器 B）：错词 100% 是**下一轮**的值

`A2_open_only`（= 现行写法，16 轮复用同一张表）run 1 的快照逐词归因（`evidence/diag_A2_open_only.txt`）：

| round | 读到全量的核 | headOK_min | word_match | word_future | word_past | word_zero | word_other | 观测到的轮次 |
|---|---|---|---|---|---|---|---|---|
| 0 | 33/56 | 15 | 38400 | **11776** | 0 | 0 | 0 | T2:11776 |
| 1 | 51/56 | 7 | 46960 | **3216** | 0 | 0 | 0 | T3:3216 |
| … | … | … | … | … | 0 | 0 | 0 | … |
| 14 | 52/56 | 7 | 47536 | **2640** | 0 | 0 | 0 | T16:2640 |
| 15（末轮）| **56/56** | **56** | 50176 | **0** | 0 | 0 | 0 | - |

三条读数的含义，逐条钉死：

* `word_past = word_zero = word_other = 0`（**全部 16 轮、全部 56 核**）：没有任何一个词是
  "陈旧的上一轮值 / 从未被写 / 无法解释"。⇒ **发布屏障确实保证"本轮所有写都落盘了"**：
  如果会合没等齐，慢核一定会读到"某些槽还没写（zero）或还是上一轮（past）"——一次都没有。
* `word_future > 0` 且**只出现 T+1**：慢核读到的是**下一轮**的 ticket。⇒ 覆盖发生在**读之后**、
  由**跑在前面的核**执行。末轮没有"下一轮"可用，于是**末轮 56/56 全干净**——这是同一机理的反证。
* `headOK_min ≈ 7~15`：慢核读的时刻，已有 7~15 个快核把下一轮的值写进去了。

### 3.1b 双源交叉核对（设备侧计数 vs host 逐词归因）——两条不同代码路径、同一份数据

主读数有**两个独立来源**，本报告要求它们逐轮相等（这就是"参考与实现不同源"的落点）：

* **源①（设备侧）**：每核在 `__VEC_SCOPE__` 里用 `Compare/Reduce` 数出「head 命中本轮 ticket 的槽数」→ 写进 obs 行 → `ROUND ... cores_clean=K/56`；
* **源②（host 侧）**：把设备**读回的那张表**整块搬回 GM，host 用**逐词整数比较**再数一遍（完全不同的代码路径）。

工具：`python3 tools/crosscheck_diag.py <diag_run.log> <snap_*.bin>`（`run_probe.sh` 的诊断阶段已把它接在 `evidence/diag_*.txt` 里）。
在 `A2_open_only` 的快照/日志上实跑：**16/16 轮 `cores_clean` 逐一相等**（例：round 0 = 33/56、round 11 = 55/56、round 15 = 56/56；5 个诊断变体全部 `不一致轮次：0`）
⇒ 两条路径的计数一致，读数不是"某一条路径自己说了算"。

### 3.2 时间戳交叉验证（仪器 C，设备侧）

同一份 `A2` run 1（`--trace`，`evidence/diag_A2_open_only.txt` 后半）：

| round | 最慢读者的读完成时刻(相对本轮基准) | 它读到的 headOK | 最快核写下一轮的时刻 | 慢核读 **晚于** 快核下一轮写？ |
|---|---|---|---|---|
| 0 | core47 +11243 cycle | 15 | +785 | **YES** |
| 1 | core47 +11058 | 7 | +335 | **YES** |
| … | … | … | … | YES（15/16 轮） |
| 15（末轮）| core39 +1105 | **56** | 无下一轮 | n/a |

⇒ 设备侧时间戳独立复现了逐词归因的结论：**慢核的读完成时刻落在快核的下一轮写之后**。
两条独立仪器一致（一条看"读到什么值"，一条看"什么时候读的"）。

### 3.3 会合本身没问题的正面证据 + wait pipe 的决定性

* `A5_close_waits`（`set<MTE3>` / **`wait<PIPE_S>`** + 收尾会合）：**0 违规 9/9 次**，
  且"会合返回时刻"的跨核散布只有 **38 cycle**（`A3` 的 wait<MTE2> 是 ~1.5~2.1 万 cycle——
  那不是释放散布，而是 scalar 没被挡住、测得的是 scalar 进度差）。**38 cycle 就是"全体同时放行"的实测形态。**
* `A6`（`wait<PIPE_V>`）/ `A7`（`wait<PIPE_MTE3>`）：**加收尾会合后仍然 769~812 / 793~823（/896）违规，9/9 次**。
  原因：`CrossCoreWaitFlag<0,pipe>` 在 3510 上**只阻塞该 pipe 的后续指令下发**（`kernel_operator_sync_impl.h:818`
  → `wait_flag_dev(pipe, flagId)`），挂 `PIPE_V`/`PIPE_MTE3` **挡不住后面那条 MTE2 读**：
  读在会合放行**之前**就发出了，读到的是放行前的表。⇒ "wait 必须挂**你要保护的那条 pipe**（或挂 PIPE_S 全挡）"。
* ⇒ **我们现行写法用的是 `wait<PIPE_MTE2>`：这条恰好是对的**（挡住了读），所以问题不在 wait pipe 上，
  而在"读完之后没有任何东西挡住别人覆写"。

### 3.4 其它被排除的候选（逐条给读数）

| 候选原因 | 变体 | 读数（9 次中 rc==0 的次数见 §2） | 判定 |
|---|---|---|---|
| 缺排空（`MTE3_S` drain 没做） | `A10_nodrain_close`（完全不排空） | 0 违规 | **不是**：`CrossCoreSetFlag<…,PIPE_MTE3>` 本身就是"等前置 MTE3 完成后再通知"，额外 drain 是冗余的 |
| `PipeBarrier<MTE3>` 形态不对 | `A11_pipebar_close` | 0 违规 | **不是** |
| 跨核 GM 可见性需要 cache flush | `A12_dcci_close`（写侧 `DCCI(clean)+DSB`）、`A13`（+读侧 DCCI） | 0 违规 | **不需要**（做与不做读数相同；见 §7 更直接的可见性实验） |
| flagId 复用 / 4-bit 计数器饱和 | `A16_fid_fixed_close`（16 轮固定同一对 id）、`A17`（7 对轮换） | 0 违规 | **不是**：配对的 set/wait 使计数器回到 0，复用 16 次无碍 |
| SET 挂错 pipe | `A8_setV_close`（set 挂 PIPE_V）、`A9_setE2_close`（set 挂 PIPE_MTE2） | 0 违规（本实验里） | 本实验未打出差异；**但语义上仍是错的**（通知不覆盖 MTE3 落盘），见 §8 存疑栏 |
| 到达时间偏斜（skew） | `A18_noskew_close`（无偏斜） | 0 违规 | **不是**，只是把竞态概率压低（现行配方无偏斜时 22/896 vs 有偏斜 90~106/896） |
| mode 选错 | `A14_m1_close`（mode1 + 收尾） | 741~815/896 违规（9/9 次） | **是**（语义错）：mode1 只同步**同一 AI Core 内** 2 个 AIV，跨 AI Core 不受保护 |
| 核数（28 vs 56） | `A21_28aiv_close` / 现行配方 28 AIV（`c00` 等价） | 收尾后 0 违规；只有发布屏障时 28 AIV 也违规 | 与核数无关，是协议问题 |
| 轮询能不能替代 flag | `A19_poll_hybrid_close`、`A24_poll_sepwins` | 0 违规 | 能（前提同 §7：窗口不复用） |
| **4B 写且 GM 地址非 32B 对齐（`g[aiv*2]`）会静默丢弃** | 所有变体的 host 终态判据 `final_head_ok` | **9/9 次、全体变体（含 `st=1,wl=4` 的 4B 密排）都是 56/56** | **不是**：窄写/非对齐写**照样落盘**（`DataCopyPad` 的 4B 写是有效的）；密排几何的问题只在"被覆写时损失更大"，不是"写不进去" |

---

## 4. 成功配方（稳定收敛，9/9 次）

### 4.1 配方 R1：**发布会合 + 收尾会合**（推荐；flagId 预算 2 个/轮）

```cpp
// 每轮（"本轮产物写 GM → 全核各自读全量 → 下一轮覆写"）：
WriteMySlot(round);                          // MTE3 DataCopyPad
SetFlag<HardEvent::MTE3_S>(EV); WaitFlag<...>(EV);      // 让本核 GM 写落盘（可省，见 §3.4）
CrossCoreSetFlag<0x0, PIPE_MTE3>(fidOpen);              // ① 发布：等本核前置 MTE3 完成
CrossCoreWaitFlag<0x0, PIPE_MTE2>(fidOpen);             //    挡住本核后续 MTE2 读
ReadWholeTable();                                        // 读别人写的东西
UseIt();                                                 // 消费（含 VF 计算）
CrossCoreSetFlag<0x0, PIPE_MTE2>(fidClose);              // ② 收尾：等本核**读**做完
CrossCoreWaitFlag<0x0, PIPE_MTE3>(fidClose);             //    挡住本核**下一轮的写**
```

* **为什么这样挂 pipe**：SET 挂"我已做完的那条 production pipe"（发布写=MTE3，读完成=MTE2）；
  WAIT 挂"我想挡住的**下一条**指令所在 pipe"（`fidOpen` 挡读=MTE2，`fidClose` 挡写=MTE3）。
  这与官方 `SyncAll<isAIVOnly=true>`（`ffts_cross_core_sync(triggerPipe)` + `wait_flag_dev(waitPipe)`）
  的形态一致（`impl/basic_api/dav_3510/kernel_operator_sync_impl.h:145-160`），只是我们把"读"也当成一段要发布的工作。
* **flagId 预算**：**每个"轮次"用 2 个 id**（`fidOpen`/`fidClose`）——**实测也可以用 1 个 id**
  （`A26` 9/9，见 §4.3 对用户猜想的实测回答）。每核共 16 个可用 id（`docs/05 §6.1`；
  源码 `GetffstMsg` 把 flagId 截到低 4bit：`kernel_operator_sync_impl.h:107`）⇒ **同屏最多 8 组流水线**
  （16 个 id 两两配对，每 8 轮复用一次，实测 16 轮复用 `A16/A17` 全绿）。
* **是否"需要多次位次"**：**是**。同一片 GM 被反复覆写的场景里，"发布"与"读完"是**两次不同的会合**；
  只做第一次就是本探针复现出来的那个坑。
* 可选优化：把 `wait<PIPE_MTE2>` 换成 `wait<PIPE_S>`（`A5`）：更保守（整条 scalar 被挡），
  换来更紧的会合（散布 38 cycle），代价是牺牲"scalar 跑在指令前面"的跨 op 预取重叠
  （`docs/05 §2` 的 PIPE_S 禁令）——**只在必须消除 run-ahead 时用**。

### 4.2 配方 R2：**发布会合 + 每轮独立窗口**（用内存换第二次会合）

`A20_sepwins`：同一张表不再复用，轮 r 的读写都落在 `tab + r*regionWords`，**只做发布会合**，9/9 全绿。
代价是 GM 容量 = 轮数 × 每轮地区；收益是每轮少一次会合（少 2 个 flagId、少一段串行）。
`A25_sepwins_dense4` 证明：**换成独立窗口后，M53 那个 4B 密排几何也不再违规**。

### 4.3 flagId 计数器语义：**sticky 还是消费型**——实测回答用户猜想

**用户的猜想（逐字）**：「其实一个 flag id 就行了，前后都放一个，只不过方向相反；
**前提是两个同步 id 的生命周期没有交集**。」

**实测（三个变体，各 9 次独立进程；读数 = `viol`（56 核 × 16 轮的违规数），rc==0 次数）**

| 变体 | flagId 用法 | 9 次读数 | 结论 |
|---|---|---|---|
| `A3_open_close` | 每轮**两个不同 id**（open=`2*(r%8)`、close=`2*(r%8)+1`，16 轮复用 16 个 id） | `viol=0`，rc=0 **9/9** | 基线配方 |
| `A16_fid_fixed_close` | 固定**一对** id（1=open、2=close）**反复用 16 轮**（每核 32 次 set + 32 次 wait） | `viol=0`，rc=0 **9/9** | 复用不饱和（非 4bit 溢出问题） |
| `A26_fid_same_id` | **同一个 id（1）既做 open 又做 close**，每轮 set/wait ×2、16 轮 | `viol=0`，rc=0 **9/9**（896/896 (核,轮) 干净） | **「一个 id 就够」实测成立** |
| `A27_fid_same_id_sep` | 同 A26 + 每轮独立窗口 | `viol=0`，rc=0 **9/9** | 与 A26 一致 |

**判定：消费型（consuming），不是 sticky（写一次可多次消费）。** 推理链（可复核）：

1. 若是 sticky（一次 set 之后该 id 的"到齐"状态**一直有效**），那么每一轮的**第一个** wait 都会
   被上一轮残留的 set 立刻满足 ⇒ **发布会合会退化成 no-op** ⇒ 读在写齐之前发生 ⇒
   逐词归因必然出现 `past`（读到上一轮值）或 `zero`（没写）> 0，形态与 §3.1 的 `A2` 同族。
2. 实测 `A26` 9/9 次 `viol=0`（`A16` 9/9 次 `viol=0`）；而同期**真正没有会合**的对照（`A0` 9/9 次
   `viol=892~895`、`A1` 118~222）证明本表的灵敏度足以抓住 no-op 会合。
   ⇒ 同一个 id 在 16 轮里被反复 set/wait **每次都重新武装**，说明一次 `wait` 消费一次「到齐」事件，
   计数器归零后才接受下一次 set（与厂商文档一致：`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md:149`「每一个计数器计数范围为0-15…将对应计数器的值减去1进行还原」，以及同目录 `key_features.md:135/166/213/260`「…计数器值减去1」；两处均在 §9 的 `#doc_counter_tpl`/`#doc_counter_keyfeat` 段里可复跑）。
3. **flagId 预算**：R1 配方的每个"复用段"需要 **2 个位次**（发布 + 收尾）；每核 16 个 id
   ⇒ 两两配对后同屏可支持 **8 组**（`A17` 7 对轮换 9/9 ✓）。若按用户猜想**共用一个 id**，
   则同样 16 个 id 可支持 **16 组**（`A26` 9/9 ✓）——这是"能用但要自己论证"的优化。

**如实标注两条边界（用户铁律"未隔离的观测不得写成规则"）**：

* `A26` 只隔离了「**同一个 id 兼做前后两个会合**」这一个轴。用户猜想里那句
  「**前提是两个同步 id 的生命周期没有交集**」在本协议里**并未严格成立**：快核的下一轮 open-set
  可以与慢核本轮的 close-set 同时在飞。所以 A26 的 9/9 只能说明"这点交叠在本窗口/本几何下没打出来"，
  **不能**据此断言"生命周期交叠也无害"。
* 因此**生产建议**：默认仍用 R1 的两个独立 id（`A3`，语义最干净）；
  要用一个 id 就按 `A26` 的形态复测自己的窗口（本探针一条命令：
  `./build/probe_aiv_sync A26_fid_same_id 1 out`）。

### 4.4 反面清单（复现失败形态用）

| 写法 | 实测 |
|---|---|
| 只有发布会合（现行） | 75~117/896 违规（9 次）；16 轮里 1~3 轮干净 |
| 只有发布会合 + 4B 密排几何（M53/m19 形态） | 262~487/896 违规（会话 1 的 9 次；四会话区间与比值见 §0 第 8 条）；56 核里只有 1~6 核全程干净 |
| `wait<PIPE_V>` / `wait<PIPE_MTE3>` | 769~823/896（读早于放行） |
| mode1（跨 AI Core） | 741~815/896 |
| 无任何会合 | 893/896（对照，说明表本身灵敏） |
| 一半核参与会合 | 9/9 超时死锁（`C0_halfset`） |
| mode2 用于 AIV↔AIV（自 set/自 wait） | **9/9 挂死（`A15_m2_close`）**：mode2 的 set 记在 AIC 侧，AIV 自己的计数器永不满 |

---

## 5. M53 反向验证（逐条对照，含条件差异）

被验证对象：`m19_qsa_indexer/probe_aiv_barrier.asc`（commit `149b920`，sha256
`db1fcf50501d5ef4103054404aaf0fe6d97e0eaba5956a0bb9ba79d5db9a36bf`）。
我把它**逐字复制**到 `probe_m53_reverse.asc`（同 sha256）并用同一构建链、9 次独立进程重跑
（`evidence/m53_reverse.log`）。

| M53 README 的读数 | 我的复现（9 次独立进程，`evidence/logs/m53_reverse.log`） | 条件差异 | 我的判断 |
|---|---|---|---|
| 无屏障对照：56/56 核违规 | **9/9 次都是违规核 56/56**（合计 220~223，最少见槽数 1~2） | 无 | 一致：探针灵敏 |
| `Set<0,MTE3>+Wait<0,MTE2>`（现行）：**19~40/56 核读到残缺计数**、最少见 2/56 槽 | **违规核 16~51/56**（逐次：16,20,25,28,35,36,38,51…），最少见槽数 1~22 | 我的范围更宽；我把表拆成了"逐轮 × 逐词"分布（M53 只有 4 行汇总） | **现象一致，解释不同**：残缺值 100% 是**下一轮**的 ticket（§3.1）⇒ 不是"读不到"，是"读到被覆写的" |
| "无模板 `Wait` 形态仍违规" | **无法逐字复现**：该变体**不在交付的 4 变体文件里**（只在 README 表里；`v1/v2/v3` 之外的 2 条是手工改代码跑的，日志与文件都未归档） | **条件差异：证据不可追** | 记为"M53 声称但未归档"；我用 `wait<PIPE_V>`/`wait<PIPE_MTE3>` 做同类验证：**确实违规**（769~823/896，§3.3），机理是"wait pipe 挡不住 MTE2 读" |
| "16 个不同 flagId 仍违规" | 同上：**不在交付文件里**；我用 `A15`（16 个两两配对 id）/`A17`（7 对轮换）验证：**配合收尾会合时 0 违规**；只有发布会合时同样违规 | 同上 | flagId 复用**不是**原因 |
| 纯轮询：**0 核收敛**（4096 上界） | **9/9 次都 0/56 收敛**（违规核 56/56，合计 147~194） | 无 | **现象逐字复现，归因被否证**：把窗口改成不复用后，纯轮询 **56/56 收敛、1~2 次迭代**（§7）。⇒ 不是"可见性依赖 flag" |
| flag + 轮询混合：约 2/3 的核不收敛（≈18/56 收敛） | **收敛核 26~51/56**（逐次 26,33,40,41,42,49,49,51…） | 数值更高（方向一致：**不保证全收敛**） | 一致（同一形态：轮询在追一个会被覆写的目标）；数值差异记录在案 |

**总判断（逐字回答 mission 的提问）**：M53 的 6 条读数**现象基本可复现**，但把"屏障不可靠/可见性依赖 flag"
作为结论**不成立**。真因是**读窗口未闭合**（发布会合之后没有任何东西挡住下一轮写）：
* 我们是**用法不对**（用户铁律的默认假设成立）；
* 而且在这次探针里找到了 M53 没试过的**正确用法**（§4 两条配方），
  在同一规模（56 核 × 16 轮全核交换）上 9/9 次 0 违规。

**M53 的 2 条不可追证问题（过程性发现，供 review 参考）**：README 声称测了 6 个变体
（含"无模板 Wait"、"16 个不同 flagId"），但 `probe_aiv_barrier.asc` 只有 4 个变体、
`evidence/probe_aiv_barrier.log` 只有 4 行读数 ⇒ 那 2 条读数**没有可复现证据**。
本探针按同族写法补测（见上表）——**结论方向相反**。

---

## 6. per-AIV 随机分裂：来源判别（硬件随机 / 未初始化 / 未同步）

mission 的第三问："同一 case 连跑多次分裂结果是否真的不同，并给出三态判别实验"。

**判别实验与结论**（三个候选各自被单独排除）：

| 候选来源 | 判别实验（本探针内的等价复现） | 读数 | 判定 |
|---|---|---|---|
| **未初始化状态** | host 把表预清 0；快照逐词归因统计 `zero/other` | `word_zero = 0`、`word_other = 0`（全部轮次/核） | **排除**：没有任何一个值是"没写过的初始内容" |
| **硬件随机** | 同变体 9 次独立进程；再看"修好协议后是否还有随机性" | 有窗未关时违规数逐次波动（分布见 §2）；**加上收尾会合后 9/9 次全 0** | **排除**（作为"硬件随机故障"）：随机性来自 timing 竞态；协议修好后随机性消失 |
| **未同步（会合不成立）** | `wait<PIPE_S>` 时测"放行时刻"跨核散布 | **38 cycle**（≈20ns）；且 `past=0` | **排除**：会合是真会合 |
| **✅ 真因：协议缺口（读窗口未关）** | 逐词归因：错词 100% 是 T+1 轮；末轮 0 违规 | §3.1/§3.2 | **成立** |

**"随机分裂"的等价物在本探针里就是 `headOK` 的逐核分布**：读得早的核看到"全部是本轮值"，
读得晚的核看到的表**混了下一轮**（`headOK` 掉到 7~15，且每轮、每次运行落在不同的核上——
`COREVIOL` 行逐次不同）。映射到 m19 的 radix 计数交换上：每核读回的计数被污染成"本核 + 部分对端的下一轮计数"，
于是每核算出的前缀/阈值 `K` 各不相同，且**每次运行被污染的核不同** ⇒ 就是 M53 记录的"per-AIV 的 K 随机分裂"。
⇒ 分裂**不是硬件随机**，是**未闭合的读窗口**（同样的输入、不同的到达时序）。

---

## 7. 跨核 GM 可见性单独实验（`probe_visibility`）

把"可见性"从"会合"里拆出来（无 flag、纯轮询、上界 4096 次；只在相位起点用一次 mode0 会合对齐各核相位），
每变体 9 次独立进程（`evidence/logs/visibility_*.log`、`evidence/logs/visibility_matrix.txt`）：

| 变体 | 写侧 | 读侧 | 9 次读数 | 结论 |
|---|---|---|---|---|
| `v0_wall_M0` | 全体 56 核各写自家 64B + `MTE3_S` 排空 | 纯轮询 | rc=0 9/9，`poll_converged=56/56`，`iter_max≤5` | 全核互见，**不需要 flag** |
| `v1_wall_M1` | v0 + `DCCI(clean)+DSB` | 纯轮询 | rc=0 9/9（与 v0 **完全相同**） | **DCCI 无必要** |
| `v2_wall_M0_R1` | v0 | 读前 `DCCI` | rc=0 9/9 | 读侧也不需要 flush |
| `v3_wall_M2` | 标量直写 GM + DCCI | 纯轮询 | rc=0 9/9 | 官方 SuperKernel 形态同样可见 |
| `v4_wall_M3` | v0 但只 `PipeBarrier<MTE3>` | 纯轮询 | rc=0 9/9 | 排空形态不影响可见性 |
| `v5_wall_dense_M1` | **4B 密排（M53 几何）** | 纯轮询 | rc=0 9/9（`iter_max≤16`） | 几何不影响可见性（只影响"被覆写时"的损失量，§2） |
| `v6_wall_M0_tail0` | 无尾会合 | 纯轮询 | rc=0 9/9 | 尾会合不是可见性的必要条件 |
| `v7_w1_M0` | **只 rank0 写**（单写者对照） | 纯轮询 | rc=0 9/9，`host_head_ok=1`、`host_zero=55` | 判据自洽：单写者时**只有 1 槽**该有值（也是 §10.2 盲区的定量证明） |
| `v8_w1_M1_tail0` | 单写者 + DCCI + 无尾会合 | 纯轮询 | rc=0 9/9，`host_head_ok=1` | 同上 |
| `v9_wall_M0_28aiv` | 28 AIV 全体写 | 纯轮询 | rc=0 9/9 | 核数不影响可见性 |

⇒ **跨核 GM 写→读的可见性本身没有任何问题**（不需要 flag、不需要 DCCI、不需要尾会合、与几何无关）。
M53「纯轮询 0 核收敛」之所以成立，是因为它轮询的目标（"本轮全体到齐"）在下一轮写到来后
**永远不再成立**——轮询追的是一个正在被覆盖的表，而不是"看不见对端的写"。
（本节 `v0~v6/v9` 是"写一次、读多次"的**不复用**窗口；M53 的探针是"每轮复用同一片槽"。
只差这一点，读数就从 0/56 收敛变成 56/56 收敛。）

---

## 7.5 AIC↔AIV 对照（`probe_aic_aiv`，独立工程）

AIC 不能写 GM（只能用 `Fixpipe`，`docs/05` 硬规则），所以这一节**不测数据面**，只测
**跨类型会合的语义与陷阱**（这是 mission 矩阵点名的"mode 0 vs 点对点 mode 2"那一格）。
完整读数见 `evidence/logs/aic_aiv_*.log`、汇总见 `evidence/logs/aic_aiv_matrix.txt`。

| 变体 | 构造 | 9 次读数 | 结论 |
|---|---|---|---|
| `v0_chain_mix_standard` | AIV 写→**AIV mode0 全核会合**→`set<2,MTE3>` 通知配对 AIC→（AIC）`wait<2,PIPE_S>`→**AIC mode0 全核会合(`set<0,PIPE_FIX>`/`wait<0,PIPE_S>`)**→`set<2,PIPE_FIX>`→AIV 写第 2 段→AIV 再会合→读回 | rc=0 9/9（`hit_full=56/56`、`final_head_ok=56`） | **docs/05 §2 的"跨类型 all-to-all 标准组合"端到端可用**（不挂、不坏） |
| `v4_aic_only` | 只让 AIC 做 mode0 全核会合（`set<PIPE_FIX>`+`wait<PIPE_S>`）；AIV 不做跨类型 | rc=0 9/9 | **AIC 侧 mode0 会合本身可用**（`PIPE_FIX` 是 AIC 的合法 pipe） |
| `v5_aic_relay_nobar` | 与 v0 同，但 AIC **不做**会合、只把 mode2 直通中继 | rc=0 9/9 | 与 v0 读数相同 ⇒ **AIC 会合的门控强度在本探针里观测不到**（无 AIC 时间戳），如实记为"未测" |
| `v3_aiv_only` | AIV 只做自身 mode0 会合（隔离 AIV 侧） | rc=0 9/9 | AIV 侧独立可用；**附带复现了主探针的结论**：第 2 段写完后若不再来一次会合就直接读回，`hit_full` 掉到 12~56/56 —— 与 §3 的"读窗口未关"同源 |
| `v1_aic_set_wrong_pipe` | AIC 把 mode0 的 **set 挂 PIPE_MTE3**（AIC 无 MTE3 ⇒ 指令不发射） | **9/9 超时挂死**（rc=124） | **独立复现 docs/05 的硬规则**："AIC 侧同步不得挂 PIPE_MTE3，后果是静默空操作 + 对侧永久等待"（`impl/basic_api/kernel_event.h:264` 的 `IsSplitCubePipe` 不含 MTE3） |
| `v2_cross_group_mode2_wait` | 只有 AIC 0 `set<2,PIPE_FIX>(flag5)`，而**全体 56 个 AIV 都 `wait<2,MTE2>(flag5)`** | **9/9 超时挂死**（rc=124） | **mode2 是点对点**（AI Core 内 AIC↔其 2 个 AIV）：别组的 set **满足不了**我的 wait；若它返回就说明 mode2 有 all-to-all 语义 —— 二值可判 |

⇒ 对 mission"mode 0（all to all）vs 点对点 mode 2 的区别"的直接回答：
**跨核（跨 AI Core）的 all-to-all 只能用 mode0；mode2 只在 AI Core 内部有效（`v2` 挂死为证）；
AIC 侧的 pipe 合法性同样会静默失效（`v1` 挂死为证）。**

注：`v1/v2` 是"预期挂死"的变体，`run_aic_probe.sh` 用 20s 超时跑并单列 `HANG=`；
挂死进程用 `pkill -f "^<本 worktree>/build/probe_aic_aiv "` 精确清理（只杀本 worktree 的进程）。

---

## 8. 已确认 / 仍存疑

**已确认（有 9 次独立进程读数，可复现）**

1. AIV 间 mode-0 会合可靠、释放时刻跨核一致（`wait<PIPE_S>` 下散布 38 cycle；无 `past/zero` 错词）。
2. 复用窗口 + 只有发布会合 ⇒ 必然出现"读到下一轮值"的污染，污染量与几何/偏斜强相关。
3. 收尾会合（或独立窗口）能把它压到 0（9/9）。
4. wait pipe 必须覆盖"被保护的下一条指令"所在 pipe；`wait<PIPE_V>`/`wait<PIPE_MTE3>` 保护 MTE2 读是无效的。
5. mode1 不提供跨 AI Core 的 AIV 间同步；mode0 提供全核 all-to-all。
6. 跨核 GM 可见性不依赖 flag/DCCI；`DCCI(clean)+DSB` 在本场景无观测差异。
7. flagId 配对复用（同一对 id 用 16 轮）无饱和问题；flagId 截低 4bit 由源码确认（未做 17 以上的实测）。
8. 写几何（4B 密排 vs 32B/64B 独占）只影响"坏协议下的受害程度"（**比值 ~2~6.5×，跨会话绝对量散布较大**：B4 四会话区间见 §0 第 8 条）；**好协议下四种几何都 9/9 次 0 违规**。
9. 部分参与会合 = 死锁（9/9 超时）。

**仍存疑（本轮没测或读数不足以定论）**

* **SET 挂错 pipe 是否有害**：`A8/A9`（set<V>/set<MTE2>）在有收尾会合时 0 违规，与本探针的机理模型不符
  （模型预期"通知早于落盘"应出错）。可能是收尾会合把窗口收得太紧、把该错误掩盖了；
  也可能 `ffts_cross_core_sync` 的 pipe 参数在本平台并非"排空触发"。**待定：需要"无收尾会合 + set 错挂"的矩阵**。
* **`wait<PIPE_MTE2>` 下 1.5~2.1 万 cycle 的"释放散布"**：已解释为"scalar 未被挡住、量到的是 scalar 进度差"，
  但没有反证实验（如"堵住 scalar 但保留 wait<MTE2>"）。不影响配方结论。
* **AIC 侧会合的"门控强度"**：`probe_aic_aiv v0`（标准 mix 链，AIC 做 mode0 全核会合）与
  `v5`（AIC 不做会合、只中继 mode2）**都正常返回**；AIC 不能写 GM ⇒ 拿不到 AIC 侧时间戳，
  因此"这次会合到底门控住了什么"**没有直接读数**（只能说"没挂、没坏"，不能说"它门控了"）。
* **AIC↔AIV 的数据面**（`Fixpipe` L0C→GM 的跨核可见性）：需要 lift Mmad+Fixpipe 脚手架，**本轮未做**。
* **核数 > 64 / 满配 die（32 AIC）**：只测了 28 AIV 与 56 AIV。
* **16 个 flagId 之外的复用（>16 轮同 id）与"未配平 set 超 15 次即报错"**：未实测（会导致异常中断，
  属"预期中断"类，本轮不做）。
* **msprof 时间线**：设备侧证据用的是 `SYS_CNT` 时间戳 + 逐词归因，没有跑 msprof（无法给出算力/带宽侧的交叉证据）。

---

## 9. 对 `m15_layer_loop` / `m20_hyperconn` 现有 AIV barrier 用法的影响清单（**只报告不改**）

> **结论**：**没有任何一处现有用法命中"已证明不可靠"的形态** ⇒ 塔的问题「m15/m20 要不要改屏障用法」的回答是 **不需要**。

**本列口径（r2 复审后整列重做）**：下表"核验"列的每条命令都**实跑过**，**完整输出**落在
`evidence/section9_greps.txt`（由 `tools/dump_section9_evidence.sh` 生成，19 段，每段带 tree / commit / LC_ALL / 命令 / 完整 stdout / exit / 行数）。
每格统一写成 **`<命令>` → `#<tag>`（完整 N 行）→ 相关行**；凡「完整行数 > 表中列出行数」的格子**一律标注「仅列相关行（M 行中列 K 行）」并给出被略去行的行号**。
逐格机制化自检（**非空洞版**，三个独立来源互证）：`python3 tools/verify_section9_claims.py` 对每个 `#tag` 段断言
V1 **实际 stdout 行数 == dump 页脚**（抓"dump 里被增删一行、页脚不变"）、V2 **页脚 == README 声明**、
V3 **exit == 0**（抓"命令失败却当证据"）、V4 stdout 不含错误特征行（`No such file` 等，独立于页脚/exit 的**内容级**判据）、
V5 README「列出 K 行」的每个行号都在该块内且个数等于 K、V6 无未被引用的段；当前 **rc=0（19/19 段全过）**。
**咬合力由变异正对照证明**：`python3 tools/verify_controls.py` 在 `/tmp` 副本里注入 7 种变异（含 ★"删掉一行真实 stdout、页脚保留"）
并断言自检必须 FAIL、同时原样副本必须 PASS —— 当前 **不符合预期的对照数 = 0**（读数见 §11 与 `evidence/commands.txt`）。
复现：`bash tools/dump_section9_evidence.sh`；`LC_ALL=C` 与 `LC_ALL=C.UTF-8` 下输出逐字节一致（已验证）。

**判据**（两条**同时**成立才算命中）：(a) 同一段 GM 被**再次覆写**（且覆写者是**别的核**）；
(b) 覆写与"别人读它"之间只有**一次**会合。

**树的口径**：m15/m20 这几个被引用的文件本 mission **未改动**（`git diff --name-only 89ce319 HEAD -- m15_layer_loop m20_hyperconn` 输出为空）
⇒ 标 WT 的段 == 基快照 `89ce319` 内容（本仓 `docs/05-…md` 也用 WT 副本，避免 main 上别的 mission 改动引起行漂）；
`m15_chain_host.h` **不在基快照里**（M6x 之后新增），标 MAIN 的段是在 main checkout 上跑的 ⇒ **它的行号固定在 dump 头部记录的 main commit 上**
（`# main checkout : …（commit <sha>）`）—— 若 main 前移导致该文件行号变动，`tools/verify_section9_claims.py` 会以 V5 报错（**这条在 r3 修订过程中真的触发过一次**：
本仓 `docs/05` 的行号在 main 上被别的 mission 改动后，自检立刻报 `声明行号不在块内[110]`，我据此把该命令改为读 WT 副本 ⇒ 又回到 rc=0）。

**反事实留痕**：若将来把 48 层改成**同一个 kernel 内的层循环**、并复用同一 `ws`/`state` 段，则判据 (a) 会重新成立
⇒ **那时才需要**按 §4 给这些段边界加收尾会合（或改成每轮独立窗口）；当前形态（48 次 launch + launch 间整条 stream 同步）不满足 (a)。

| 位置 | 形态 | 判定 | 核验（`<命令>` → `#<tag>`（完整 N 行）→ 相关行） |
|---|---|---|---|
| m20 `BarrierAiv<FLAG_AV0/1/2>`（定义 `:120-124`；调用 `:490/498/506`） | 段间 Injw→Combine→barrier→Norm→barrier→Silu→barrier；每段只发布一次 | **未命中** | `grep -n -e "ProcessAiv()" -e "dWS" m20_hyperconn/m20_hyperconn.asc` → `#m20_processaiv_dws`（**完整 8 行**；列出 4 行：`:928`、`:1725`、`:1739`、`:1744`）；**仅列相关行（8 行中列 4 行）**：`:928 op.ProcessAiv();`（唯一调用点，`:476` 是定义）、`:1725/:1739/:1744`（`dWS` 分配→清零→单次 launch）；略去 `:1707` 声明、`:1716` 释放、`:1753` D2H 回读。`grep -n "BarrierAiv<"` → `#m20_barrierAiv_calls`（**完整 3 行**；列出 3 行：`:490`、`:498`、`:506`）（`:490/:498/:506`，全列）；定义体 `grep -n -A6 "inline void BarrierAiv"` → `#m20_barrierAiv_def`（**完整 7 行**；列出 3 行：`:120`、`:122`、`:123`） ⇒ 判据 (a) 不成立 |
| m15 `M15L_PhaseBoundaryAiv()`（`m15_layer_kernel.h`） | 相位 A→B 边界（单生产者/单消费者） | **未命中** | `grep -n "M15L_PhaseBoundaryAiv" m15_layer_loop/m15_layer_kernel.h` → `#m15_phase_boundary`（**完整 2 行**；列出 2 行：`:103`、`:185`）（定义 `:103`、调用点 `:185`，全列）；函数体 `grep -n -A8` → `#m15_phase_body`（**完整 9 行**）。层循环结构（MAIN 段）：`grep -n -e "for (uint32_t L = 0; L < nL" -e "H_LaunchChainLayer" m15_layer_loop/m15_chain_host.h` → `#m15_chain_loop`（**完整 8 行**；列出 2 行：`:348`、`:364`）；**仅列相关行（8 行中列 2 行）**：`:348 for (uint32_t L = 0; L < nL; ++L) {`、`:364 H_LaunchChainLayer(...)`，略去 `:206` 定义与 `:725/:736/:760/:794/:802/:830` 等其它循环。`grep -n "H_Sync(C" m15_layer_loop/m15_chain_host.h` → `#m15_chain_sync_calls`（**完整 19 行**；列出 1 行：`:366`）；**仅列相关行（19 行中列 1 行）**：`:366 if (!H_Sync(C, "ch_layer"))`（每层 launch 后立即整条 stream 同步）。层数 `grep -n "NL = 48" m15_layer_loop/m15_loop_layout.h` → `#m15_nl_def`（**完整 1 行**；列出 1 行：`:42`） ⇒ 层间没有 in-kernel 复用 |
| m15 `BarrierAiv<…SEG…>`（`:1699/1714/1720/1728`；S4 = 递推段，`ssm_state` in-place RMW） | S4 的 state 是**核私有、单次读写** | **未命中**（**修正上一版的"命中"**） | `grep -n "for (uint32_t h = bid" m15_layer_loop/m15_gdn_layer.h` → `#m15_head_loop`（**完整 2 行**；列出 2 行：`:1098`、`:1247`）（全列）：`:1098 for (uint32_t h = bid; h < numHeads_; h += nblk, ++slot)`（在 `Process()` 内，`Process` 始于 `:1070`）、`:1247 for (uint32_t h = bid; h < HEADS_; h += nAiv)`（属别的 stage）；±2 行上下文 `grep -n -B2 -A2` → `#m15_head_loop_ctx`（**完整 11 行**；列出 2 行：`:1098`、`:1247`） ⇒ head `h` 的唯一属主 = 一个 AIV。`grep -n "stateGm_" m15_layer_loop/m15_gdn_layer.h` → `#m15_state_access`（**完整 4 行**；列出 4 行：`:1066`、`:1123`、`:1170`、`:1185`）（全列）：`:1066 SetGlobalBuffer`、`:1123` 读、`:1170` 写、`:1185 GlobalTensor<float> stateGm_;`（成员声明，非访问）⇒ 读写只发生在该 head 属主的同一核内，判据 (a) 不成立 |
| m15 ws 分段偏移表（`m15_gdn_resources.h:284-297`） | 段间 ws 分段**按构造不重叠** | **未命中** | `sed -n '284,297p' m15_layer_loop/m15_gdn_resources.h` → `#m15_ws_segments`（**完整 14 行**）（全列）：每段都写成 `WS_X = WS_前一段 + SZ_前一段`（`WS_XNORM / WS_RES1 / WS_QKVZBA / … / WS_RES2`，末 `WS_BYTES = WS_RES2 + SZ_RES2`）⇒ 快核跑在前面写的是**另一个区段** |
| m15 AIC 侧 mode0 会合（**内联** `CrossCoreSetFlag/WaitFlag`；m15 里**没有** `BarrierAic` 包装） | 段边界 AIC 全核会合 | **未命中** | `grep -n -e "CrossCoreSetFlag<CC_MODE0" -e "CrossCoreWaitFlag<CC_MODE0" m15_layer_loop/m15_gdn_layer.h` → `#m15_aic_inline`（**完整 10 行**；列出 10 行：`:1751`、`:1752`、`:1758`、`:1759`、`:1768`、`:1769`、`:1775`、`:1776`、`:1785`、`:1786`）；**仅列相关行（10 行中列 10 行）**：AIC 侧 `:1751/:1752/:1758/:1759/:1768/:1769/:1775/:1776`（SET 挂 `PIPE_MTE2`/`PIPE_FIX`、WAIT 挂 `PIPE_S`，均在 AIC 合法 pipe 集内）、`:1785/:1786` 属 AIV 侧 `BarrierAiv` 模板体。形态同 `probe_aic_aiv v4`（**9/9 正常返回**）；AIC 侧 pipe 白名单 = CANN 头文件 `IsSplitCubePipe` 的 3510 分支 `{S,MTE1,MTE2,FIX,M}`（**不含 MTE3**）→ `grep -n -A8 "IsSplitCubePipe" asc/impl/basic_api/kernel_event.h` → `#aic_splitcube_pipe`（**完整 59 行**；列出 2 行：`:263`、`:268`） |
| m20 `BarrierAic`（包装；`set<SET_PIPE>` + `wait<WAIT_PIPE>`；调用 `BarrierAic<PIPE_S, PIPE_MTE2\|PIPE_FIX, …>`) | AIC 侧会合，pipe 均合法 | **未命中** | `grep -n -A5 "inline void BarrierAic" m20_hyperconn/m20_hyperconn.asc` → `#m20_barrierAic_def`（**完整 6 行**；列出 3 行：`:128`、`:130`、`:131`）（`:128-132` 定义，全列）；`grep -n "BarrierAic<" m20_hyperconn/m20_hyperconn.asc` → `#m20_barrierAic_calls`（**完整 4 行**；列出 4 行：`:527`、`:535`、`:540`、`:544`）（`:527/:535/:540/:544`，全列） |
| m15 段序 flagId 分表（`m15_layer_resources.h:197-200`，16/核） | flagId 预算 | —（不是风险项） | 若**将来**出现需要收尾会合的形态：每段 +1 个 id；`A16/A17` 证明 16 个 id 够 8 组轮换、`A26` 证明共用一个 id 也成立（§4.3）。消费型语义的第三方口径（本仓 + 厂商文档，见下方 tag；key_features.md 在正确路径下是 4 行）：`#doc_counter_repo`（**完整 1 行**；列出 1 行：`:110`） / `#doc_counter_tpl`（**完整 1 行**；列出 1 行：`:149`） / `#doc_counter_keyfeat`（**完整 4 行**；列出 4 行：`:135`、`:166`、`:213`、`:260`）（原文：每个 flagId 一个 0-15 计数器；wait 时"计数器的值减 1 进行还原"；先前一版把厂商文档路径写成 `ascend-devkit`（不存在）导致这两段是 `No such file` 报错 —— r3 复审 F2，已改 `asc-devkit`） |
| `m15_layer_kernel.h:101-105` 注释「这里的 wait 后面**不紧接** CrossCoreSetFlag，不触发 PIPE_S 例外」 | wait 挂 `PIPE_MTE2` | ✓ 与本探针结论一致 | `wait<MTE2>` 恰好挡住后续 MTE2 读（`A3` 9/9 次 0 违规）⇒ 无需改 |

**行号口径**：上表行号以**本分支基快照 `89ce319`** 为准；main 在其后由 M6x 系列改动位移过部分文件
（例：`M15L_PhaseBoundaryAiv` 定义在基快照 `m15_layer_kernel.h:103`、调用点 `:185`，在 main 已在 `:159`；
两条命令 `git show 89ce319:m15_layer_loop/m15_layer_kernel.h | grep -n M15L_PhaseBoundaryAiv`
与在 main checkout 上直接 grep 的结果不同，即可复核）⇒ **按符号名检索为先，行号只作定位提示**。

**给后续 mission 的最小建议（不在本 mission 范围，仅登记）**：

1. 本清单的结论是「**当前没有命中**」（前提见上"反事实留痕"）；若日后出现「同一 GM 段被**跨核**反复覆写 + 两次覆写间只有一次会合」
   的**新**形态，再按 §4 的两条配方加收尾会合（或改独立窗口）——别对现有的 m15/m20 段边界做提前改造。
2. 跨核交换的槽位**别做 4B 密排**（`st≥8`、每核独占 ≥32B）：坏协议下受害程度比 64B 独占高约 **2~6.5×**
   （四个会话各自的区间见 §0 第 8 条）；好协议下无差别，但留宽槽位几乎不要成本。
3. 复现/回归用本探针（`bash run_probe.sh`），不要用"下游数值看起来正常"当判据。

---

## 10. 覆盖范围三态 + 三态退出码 + 负向对照

### 10.1 覆盖三态

**已测（有 9 次独立进程读数）**：见 §2 主表与 §7 表，覆盖 mission 要求的 ①~⑦ 各轴：
会合模式（无/mode0/mode1/mode2）、flagId（固定/轮换/16 配对）、pipe 组合（set 3 种 × wait 4 种）、
drain（4 种）、几何（4 种）、偏斜（2 种）、轮询（3 种）、核数（28/56）、参与度（全体/一半）、
窗口复用（复用/独立）、收尾会合（4 种）+ 负向对照。

**AIC↔AIV 对照（`probe_aic_aiv`，独立工程，见 §7.5）**：
mode2 点对点语义（跨组 wait 不被满足 ⇒ 挂死）、AIC 侧 `set<PIPE_MTE3>` 静默空操作 ⇒ 对侧永久挂死
（`docs/05` 硬规则的独立复现）、AIC 侧 mode0 会合（`set<PIPE_FIX>`+`wait<PIPE_S>`）可正常完成、
"标准 mix 序列"（AIVn→AIC mode2 → AIC mode0 → AIC→AIV mode2）端到端可跑通；
**未覆盖 AIC 侧数据面（Fixpipe 写 GM）与 AIC 会合的门控强度**。

**未测（明确列缺）**：

* AIC 侧：Fixpipe 写 GM 的跨核可见性；AIC 会合的门控强度（无可观测手段，见 §8）。
* flagId 越界（>15）截断行为、未配平 set 的 4bit 计数器溢出报错（预期中断）。
* 满配 die（32 AIC / 64 AIV）；多流并发（`docs/05` 的 batchmode 死锁条件）。
* msprof 时间线（本报告的设备侧证据是 `SYS_CNT` + 逐词归因，没有 msprof）。
* 30 轮以上的长稳（本轮上限 16 轮）。

**不在覆盖内**：m19/m15/m20 的源码改动（本 mission 只报告）、radix 算法本身、性能/makespan、
AIC 侧数据面（Fixpipe 带宽、L0C 语义）。

### 10.2 负向对照：**已知会被漏掉**的那一条

`N0_negctrl_selfonly`：**完全没有屏障**，但校验时只统计"自己那一格"是否等于自己的 ticket。
读数 `headOK = 1`、`viol = 0` ⇒ **判 PASS**。

这正是本读数方法的**盲区**，写下来是为了防止后来者拿它当验收：
**"每核只校验自己写的那格"这种自证式判据在任何同步缺失下都会 PASS**（自己的写对自己可见，
`§7 v7/v8` 单写者对照也印证：只有 writer 那 1 槽可见）。⇒ 判据必须要求"每核校验**整张表**"，
且表里必须能区分"谁写的、第几轮"（本探针用 ticket 的 rank/T 字段做到）。

### 10.3 三态退出码

| 码 | 含义 | 本探针里的例子 |
|---|---|---|
| `0` | 比过且通过 | `A3/A5/A20`（0 违规） |
| `1` | 比过有差异 | `A2`（73~129 违规，四会话）、`B4`（229~487 违规，四会话） |
| `2` | 没得比/输入缺失 | launch 失败、读数行缺失（`missing_rows>0`）；变体名非法 |
| （`124`） | 超时未返回（`run_probe.sh` 单列 `HANG=`） | `C0_halfset`（预期） |

---

## 11. 证据归档

| 文件 | 内容 |
|---|---|
| `evidence/commands.txt` | 复现命令、环境版本、**源码 sha256** |
| `evidence/logs/aiv_sync_<variant>.log` | 每变体 9 次独立进程完整输出（SUMMARY/ROUND/COREVIOL/CYCLE[/TRACE]） |
| `evidence/logs/main_matrix.txt` | 主探针逐变体摘要（逐次一行） |
| `evidence/logs/visibility_<variant>.log` + `visibility_matrix.txt` | 可见性探针逐变体摘要（§7）｜**交付的 10 个变体各 9 次** |
| （汇总口径）| `tools/make_tables.py` 与 `tools/analyze_snap.py matrix` **按文件名跳过 `session*.log`** —— `session2_A2_B4.log` 是跨会话补充证据，不进变体汇总，避免同一变体出现两条相互冲突的区间（§2 表的 9 次口径因此与复审核对时一致） |
| `evidence/logs/superseded/` | **11 份早期单跑日志（已作废）**：它们来自一次被中止的冒烟运行，变体名用的是**修标签前的旧编号/旧名**（`v0_w1_*` vs 交付的 `v0_wall_*`、`v3_wall_M0_*` vs 交付的 `v3_wall_M2_*` 等），与交付的 9 次日志**同目录混放会误读**，故移入本子目录保留、不删（`run_probe.sh`/`analyze_snap.py`/`make_tables.py` 的 `*.log` 都是非递归匹配，不会读到它） |
| `evidence/logs/session2_A2_B4.log` | **会话 2 的 A2/B4 各 9 次复测**（用于判定 B4 的跨会话散布，见 §0 第 8 条 / §2 表下注） |
| `evidence/section9_greps.txt` | **§9 核验列的完整原始输出**（19 段；每段 = tree/commit/LC_ALL/命令/完整 stdout/exit/行数），由 `tools/dump_section9_evidence.sh` 生成，`LC_ALL=C` 与 `LC_ALL=C.UTF-8` 下逐字节一致 |
| `tools/dump_section9_evidence.sh` | 生成上一条的脚本（§9 每格的 `<#tag>` 都能在这里复现） |
| `tools/verify_section9_claims.py` | **§9 逐格自检（非空洞版）**：V1 stdout 行数==页脚、V2 页脚==README 声明、V3 exit==0、V4 无错误文本、V5 声明行号在块内且个数吻合、V6 无未引用段；当前 `rc=0`（19/19） |
| `tools/verify_controls.py` | **自检的变异正对照**（在 `/tmp` 副本里做）：N0 原样必须 PASS，M1~M6'（含 ★删一行真实 stdout、页脚保留）必须 FAIL；当前 `不符合预期的对照数 = 0` |
| `tools/crosscheck_diag.py` | **双源交叉核对**（§3.1b）：设备侧 VF 计数 vs host 对设备读回快照的逐词归因，逐轮比对 `cores_clean`；A2 快照上 16/16 轮相等 |
| `evidence/logs/aic_aiv_<variant>.log` + `aic_aiv_matrix.txt` | AIC↔AIV 对照（§7.5） |
| `evidence/logs/m53_reverse.log` | M53 原探针逐字副本 9 次读数（反向验证原始证据，§5） |
| `evidence/logs/matrix_summary.txt` | 由 `tools/analyze_snap.py matrix` 汇总的最终表（§2 的来源） |
| `evidence/diag_*.txt` | 设备侧快照逐词归因 + 时序交叉验证（窗口未闭合的直接证据，§3） |
| `evidence/tables/` | `tools/make_tables.py` 生成的 §7 / §7.5 表 |

环境与采集口径（如实声明）：

* **边界自查（命令 → 输出）**：
  * `git diff --name-only 89ce319 HEAD | grep -v '^probe_aiv_sync/'` → **空**
    （我的 4 个 commit 只动 `probe_aiv_sync/**`）
  * `git diff --stat 89ce319 HEAD -- m18_gdn_prefill/... m19_qsa_indexer/...` → **空**
    （三点 diff `main...HEAD` 里出现的 `m18_gdn_prefill/evidence/readme_number_audit.md`、
    `m19_qsa_indexer/evidence/probe_aiv_barrier.log` 两个文件来自**塔在 spawn 时打的 WIP 快照 `89ce319`**，
    **不是本 mission 的改动**；本 mission 未碰 m19/m15/m20/m10/`docs/**`/`tools/**`）

* 设备：Ascend950PR，`npu-smi` 25.7.rc1.6；CANN 9.1.0；bisheng 15.0.5；`dav-3510`。
* 核数由 `aclrtGetDeviceInfo(ACL_DEV_ATTR_AICORE_CORE_NUM/VECTOR_CORE_NUM)` 取（28/56），无硬编码。
* **采样期间设备上另有其它 agent 的 NPU 作业在跑**（`npu-smi` 可见进程，非本 worktree）。
  本报告的判定都是**结构性**的（0 违规 vs 90~106 违规；挂死 vs 返回；逐词归因 future>0 vs =0），
  并有 9 次独立进程的分布作抖动范围；**没有一条结论依赖单点计时**。
  已验证的干扰指纹：个别 `rel_spread_max` 出现离群（正常 15~25k cycle，个别 6.98M/21.3M/20.6M cycle，
  见 `evidence/logs/aiv_sync_A3_open_close.log` 等），但同批的 **违规数与 rc 判据全无变化**
  （A3 仍是 9/9 次 0 违规）——这正是"判定不依赖计时"的证据。
* 设备侧计时一律用 `SYS_CNT`（`GetSystemCycle()`，`MOV %0, SYS_CNT`）；**没有使用 host 时钟**，
  也没有跑 msprof（列入 §10 未测）。
* 清场：`timeout` + 只按**本 worktree 内确切二进制路径** pkill（`run_probe.sh`/`run_aic_probe.sh` 内可见），
  未触碰其它目录/其它 agent 的进程。
* **二进制版本与读数的一致性（与 sha256 一起读）**：`A*/B*/C0/N0` 三组读数由 `probe_aiv_sync.asc` 的
  **前一版**二进制产生（与当前版本差别只有两处：新增 `A26/A27` 变体、SUMMARY 末尾新增两个负向对照上报字段；
  既有变体的屏障配方/几何/判据代码位同）；`A26/A27` 与 `session2_A2_B4` 用当前版本跑。
  ⇒ **`evidence/commands.txt` 里的 sha256 是"交卷时源码"的指纹，不是"当时产出该读数的二进制"的指纹**；
  要逐位复现全部读数：按该 sha256 重跑 `bash run_probe.sh`（每变体 9 次，约 30~50 min）。
  `commands.txt` 的**第一个块**是运行期写入的（其中 `README.md`/`tools/make_tables.py` 的哈希在后续文档修订后已过期），
  以文件末尾「交卷前刷新」块的哈希为准。

**复审与修正记录（可追溯）**：本文件是**修正版**，经两轮复审：
* **r1**（对象 `7b0b77c`，`p2-2items / fix-then-merge`）两条 p2：
  (1) §9 影响清单的判定与行号口径（见 §9 开头）；
  (2) B4 跨会话口径（§2 表下复核注 / §0 第 8 条，新增 `evidence/logs/session2_A2_B4.log`）
  与证据目录卫生（11 份旧标签单跑日志 → `evidence/logs/superseded/`）。
* **r2**（对象 `baa964d`，`p2-1items / fix-then-merge`）一条 p2 + 两条 advisory：§9 的"核验"列被指
  **把手工子集标成命令输出**（其中 `stateGm_` 的计数在 `5cfb143` 上确为过期的"仅 3 处"，`baa964d` 已改）
  ⇒ 本轮**整列重做**：每条命令的**完整输出**落盘 `evidence/section9_greps.txt`（19 段，`#<tag>` 索引），
  表格只给"完整输出行数 + 相关行"，凡略去行号一律标注 **"仅列相关行"**；
  同时修掉一处**名称误引**（m15 的 AIC 会合是**内联** CrossCore 调用，`BarrierAic` 是 **m20** 的包装函数）
  并按实测补齐 `for (uint32_t h = bid`（2 行）、`stateGm_`（4 行）、`ProcessAiv()|dWS`（8 行）等计数；
  advisory (b) 已吸收（`A2` 合并区间改为 **73~129**、比值区间 **2~6.5×**，见 §0 第 8 条）；
  advisory (a) 的反事实句已显式写入 §9"反事实留痕"。
* **r3**（对象 `d337b44`，`p1-2items / fix-then-merge`）两条 p1 都在"r3 新引入的 §9 证据/自检机制"里：
  **F1** = `verify_section9_claims.py` **不满足非空洞**（只比 README 声明与 dump 自产页脚，从不数块内真实 stdout、也不读 exit
  ⇒ 变异 M2"删一行真实 stdout、页脚保留"竟 PASS）⇒ 本轮把自检改成**三源互证 + 内容级判据**（V1~V6，见 §9 前言），
  并新增 `tools/verify_controls.py` 把这 7 条变异（含 M2）变成**可复跑的对照**（当前 0 条不符合预期）；
  **F2** = `dump_section9_evidence.sh` 把厂商文档路径写成 `ascend-devkit`（不存在，正确是 `asc-devkit`）
  ⇒ 那两段"证据"其实是 `No such file` 报错、且 `key_features.md` 的真实行数是 **4**（不是 1）⇒ 已修路径、重新生成 dump、
  按实测更新 §9 该格（`#doc_counter_tpl` 1 行 / `#doc_counter_keyfeat` 4 行），并让生成脚本在**任何段 exit≠0 时非零退出**（失败命令不得当证据）；
  另吸收了 F3（dump 里显示的命令改用 `%q` 引用，可复制粘贴）。
  与此同时按塔 item-3 做了一次同类自查：主读数的两个来源（设备侧 VF 计数 vs host 逐词归因）现在由 `tools/crosscheck_diag.py`
  逐轮互证（§3.1b，A2 上 16/16 轮相等），并把该核对接进了 `evidence/diag_*.txt`。
复审文件在塔侧 `.tower/comms/reviews/review-feat-m64-…-reviewer-m64-r1.md` / `…-r2.md` / `…-r3.md`。

复现：

```bash
cd probe_aiv_sync
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
bash run_probe.sh            # REPS=9（默认）；REPS=3 冒烟
```

> 证据边界（如实声明）：快照 `.bin` 体积大，`run_probe.sh` 只把它的**逐词归因结果**（`diag_*.txt`）
> 入库、原始 `.bin` 留在 `out_tmp/`（不入库）；所有 log 都是原始 stdout 重定向，未经改写。
