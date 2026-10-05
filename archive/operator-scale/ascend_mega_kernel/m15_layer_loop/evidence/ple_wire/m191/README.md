# M191 — PLE 跨 step short-conv 状态承接（G2 / S2）

> 交付：让 ⑤（`M85P::PleConvItem`）写在**出口平面** `stOut` 的 per-step short-conv 状态，逐字节
> 搬回**入口平面** `stIn`，使下一 step 真的从上一 step 的状态续上；状态槽数按**活请求数**
> （`pleNReq`）而非分配上界（`PLEW::T_MAX`）处理。判据复用现成 `Pw.state.evolve` /
> `Pw.state.shift` 的同一段逻辑，落在**状态平面**上；负向对照 `M15_PLE_STATECARRY=0` 使新判据变红。
>
> **行号基准**：`main` 的 `7738e0669d7e271434d2feb4ff9fa68867e7365b`（M190 合入后）。
> 结论一律给内容锚点 + 行号；跨 mission 引用只钉这个不可变 rev。

## 0. 一句话结论

跨 step 状态**此前没有接线**：⑤ 读 `stIn`、写 `stOut`，两平面是 scratch slab 里相邻但**不同**的区间
（`S_STIN` / `S_STOUT`）；段体内没有任何动作把 `stOut` 搬回 `stIn`，host 也没有 ⇒ 每个 step 都从同一
初值出发，decode 的逐 token 状态演化没有序列承接。本 mission 在 host 的 step 边界加一步
device→device 拷贝（`H_PleCarryState`）把出口平面的**活槽**搬回入口平面，并把状态平面的定尺改成
`min(pleNReq, pleStateSlots)`。

## 1. 取证：跨 step 状态为什么没接（file:line）

| 事实 | 位置（base `7738e06`） |
|---|---|
| 两个平面是 scratch slab 里的不同区间 | `m15_layer_kernel.h:315`（`S_STIN = S_HMBAD + FAIL_SLOTS*8u*4u`）、`:316`（`S_STOUT = S_STIN + T_MAX*STLEN*HYPER*2u`）、`:317`（`S_BYTES`） |
| ⑤ 读**入口**平面 | `m15_ple.asc:1123`（`stBase = slot*STLEN*HYPER`）、`:1127`（`DataCopy(..., G.stIn[stBase + h*HYPER + c0], ...)`） |
| ⑤ 写**出口**平面（新状态） | `m15_ple.asc:1216`（`DataCopy(G.stOut[stBase + h*HYPER + c0], ...)`）；移位语义 new[i]=old[i+1]、new[8]=normed 见 `:1194-1219` |
| 段体内没有 `stOut→stIn` 搬运 | `m15_ple.asc:1065-1224`（`PleConvItem`）整段只有读 stIn / 写 stOut；调用者 `m15_layer_kernel.h:614-618`（`M15L_PleBody` 只调 `PleConvItem`），无第二个写入者 |
| host 的两趟之间也不搬 | `m15_layer_loop.asc::H_PleChainOnce`（一趟层链 = 一个 step，只 D2H 读回）；`H_PleWireRun` 里 A1/A2 两趟之间无搬运 |
| 槽数恒 `T_MAX` | `m15_hc_host.h:225`（`pa.stateSlots = M15L::PLEW::T_MAX`；本文件**不在本 mission scope**） |
| 段体旧按 `pleStateSlots` 给状态定尺 | `m15_layer_kernel.h:216`（字段 `pleStateSlots`）、旧 `:570/:572`（`SetGlobalBuffer(..., A.pleStateSlots*STLEN*HYPER)`） |
| `PleConvItem` 的 tap 只读状态行 0/3/6 | `m15_ple.asc:1142-1151`（for-k：`kt*3*VL`，lag 9/6/3；k=3 读当前输入 `cin`） |

**为什么没接**：M100 把 PLE 段体接进层 kernel 时，状态平面按「入口 / 出口」两个平面布置，⑤ 的语义
是「读旧状态、写新状态」；但「新状态回到下一 step 的入口」这一步**没有任何调用者** —— PLE 只在
`runs=plewire` 的**单 step 见证**里跑，而该见证一次只跑一趟层链（`H_PleChainOnce`），没有第二个
step 去承接。`ple/README.md §7 U3` 把 short-conv 的独立 writeback 记为「未做」，但 decode 的
**跨 step 承接**同样没有实现（这正是 M158 survey 的 G2/S2）。

**已核实 vs 待验证**
- 已核实（上表 file:line + §3 设备读数）：`S_STIN`/`S_STOUT` 不同区间；⑤ 只读 stIn、写 stOut；
  host 两趟之间不搬；`pleStateSlots=T_MAX`；不搬状态时新判据**变红**。
- 待验证（不在本 mission 范围）：production 的 48 层链 / step 循环把 PLE 打开（G4/S4）后本承接是否
  被调用；多请求（`pleNReq>1`）的槽映射（G1）；prefill（T>1）的独立 writeback（`ple/README.md §7 U3`）。

## 2. 最小改动

改动只落两处（`git diff --stat 7738e06 -- <两个源文件>`：+114 / −15）：

1. **搬运（跨 step 承接）** —— `m15_layer_loop.asc::H_PleCarryState`（新增）：device→device
   `aclrtMemcpy`，把 `PLEW::S_STOUT` 的**活槽**逐字节搬到 `PLEW::S_STIN`（每槽
   `STLEN` × `HYPER` × 2 B）。调用点 = `H_PleWireRun` 里 A2（第 1 步）之后、A3（第 2 步）之前；
   `M15_PLE_STATECARRY=0` 时跳过（负向对照）。
   - **不引入任何核内 / 核间同步原语**：搬运是 host 侧 DMA 拷贝，在同一 stream 上按序完成
     ⇒ 不新增 set_flag/wait_flag，也不新增 BufferID / CrossCore 号（人类口径「核内 BufferID、
     核间 CrossCore、不得 set_flag/wait_flag」在此档不触发）。
2. **活槽定尺** —— `m15_layer_kernel.h`：新增宏 `M15L_ACT_STATE_SLOTS(nreq, stateSlots)`
   （= `nreq`，且不超过分配上界 `stateSlots`；`nreq==0` 退回上界）。`M15L_PleBody` 的
   `SetGlobalBuffer` 与 host 的 `H_PleCarryState` **共用这一个表达式**（r1 复审 P2-2：此前
   host 写 `(pleNReq!=0)?pleNReq:1`、kernel 写 `min(pleNReq, pleStateSlots)`，两处边界行为分叉，
   在 `pleNReq==0` / `pleNReq>pleStateSlots` 上会不一致）。依据 `ple/PLE_SPEC.md §2 I6`
   （状态每请求一行 ⇒ 活槽数 = 活请求数 `pleNReq`）。
   - **为什么是宏不是函数**：bisheng 的设备 pass 不为头里的非 `__aicore__` 函数发射定义
     （`constexpr` 版本实测 ld.lld `undefined symbol: M15L::PLEW::ActStateSlots`），宏在
     host 与 device 两侧都展开。
3. **判据** —— `m15_layer_loop.asc::H_PleWireRun`：把既有移位判断提成 `stateShiftOk` lambda，
   单步（A1，既有读数不变）与跨 step（A2→A3）复用；新增第 2 步三条判据（§3）。
4. `m15_ple.asc` / `m15_ple_wire.h` **一字未动**（`git status` 只列两个源文件）：搬运在 host 侧，不改
   device 段；`lift_ple_device_segment.py --check` 复跑仍逐字节通过（60422 字节）。

## 3. 判据与设备读数

新增判据（净增 3 条；`Pw.state.evolve` / `Pw.state.shift` 的既有读数逐字不变）：

| 判据 | 含义 | 承接通（档 A） | 不搬（档 B，负向） |
|---|---|---|---|
| `Pw.step2.carry` | 第 2 步 stIn 逐字节 == 第 1 步 stOut | PASS（184320/184320 B） | **FAIL**（163882/184320 B） |
| `Pw.step2.evolve` | 第 2 步结束状态 ≠ 第 1 步结束状态 | PASS（差 20438/184320 B） | **FAIL**（差 0 B） |
| `Pw.step2.shift` | 第 2 步仍满足 new[i]=old[i+1] 且 new[8]=normed | PASS | PASS（stIn=0 时平凡成立） |

**为什么判据落在状态平面、不是 10240 宽的多流态激活**：⑤ 的膨胀卷积 tap 只读状态行 0/3/6
（lag 9/6/3，`m15_ple.asc:1144-1151`）；初始状态为零 ⇒ 第 1 个 tap 要到**第 4 步**才非零。所以几 step
连跑的跨度内激活平面**不变**（本读数里 `Pw.wired.delta` 两档相同），跨 step 要承接的量就是**状态
平面本身**，判据因此比较状态平面。

设备读数（二进制 sha256 见 `binary_sha256.txt`；npu-smi 快照见 `npu_smi_*.txt`）：

```
[档 A  carry-on ]  M15_PLE_WIRE=1 M15_PLE_STAGE=31 M15_PLE_STATECARRY=1  → rc=0 wall=29s
  Pw.state.evolve  PASS 状态输出 ≠ 输入（活跃行确实演化）
  Pw.state.shift   PASS new[i]=old[i+1] 且 new[8]=normed 逐字节
  Pw.step2.carry   PASS 第 2 步 stIn == 第 1 步 stOut：184320/184320 字节（活槽=1、M15_PLE_STATECARRY=1）
  Pw.step2.evolve  PASS 第 2 步结束状态 vs 第 1 步：差 20438/184320 字节
  Pw.step2.shift   PASS 第 2 步 new[i]=old[i+1] 且 new[8]=normed 逐字节
  验证 Pw：判定项 15 条 / FAIL 0 条（累计 checks 147 / fails 0） → ALL PASS
  Pw.wired.delta 读数：5636/10240 个 bf16 元素被 PLE 改动，max|Δ| = 0.427734（同 B_full_stage31）

[档 B  carry-off]  M15_PLE_WIRE=1 M15_PLE_STAGE=31 M15_PLE_STATECARRY=0  → rc=1 wall=27s
  Pw.step2.carry   FAIL 第 2 步 stIn == 第 1 步 stOut：163882/184320 字节（活槽=1、M15_PLE_STATECARRY=0）
  Pw.step2.evolve  FAIL 第 2 步结束状态 vs 第 1 步：差 0/184320 字节
  验证 Pw：判定项 15 条 / FAIL 2 条（累计 checks 147 / fails 2）
```

档 B 的两个数自洽：`184320 − 163882 = 20438`，正等于档 A 里第 2 步相对第 1 步的演化字节数 ——
即「不搬时，第 2 步 stIn 里有 20438 字节停在初值 0，与第 1 步的 stOut 不相等」。

## 4. 零回归

`M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 ... all` → `ALL PASS（checks=2095, guards=302,
fails=0）`，与合入前的 `evidence/ple_wire/runs_all.log` 的判定项 / guard 计数逐值一致。既有 Pw 判据
逐字不变：`diff` 在 `Pw.wired/ids/emb/win/fail/state/planes/det` 各线上无差异（仅本 mission 净增的
`Pw.step2.*` 与 M124 已存在的 `Pw.kv.*` 为多出的行）。

## 5. 复现

`commands.txt`（逐字命令）。设备槽：每档各自一次 `flock -w 300 /tmp/npu0.lock`，进锁先 `npu-smi`
（快照落盘到 `npu_smi_*.txt`），`timeout` 在锁内；单进程、不并发。

## 6. 限度 / 未做（不声称完整）

- **只在 `runs=plewire` 见证里承接**：production 的 48 层链 / step 循环尚未接 PLE（G4/S4 不在本
  mission scope），所以搬运的调用点落在见证里、不在生产 step 循环里。生产接线时要复用同一个
  `H_PleCarryState`（或把等价搬运放进链启动路径）。
- **两 step 用的是同一 token**：`H_PleChainOnce` 每趟固定 `h0[0]`。这足以证明「状态从上一 step
  续上」这一机制（不搬 ⇒ 判据红），但不是真实 token 序列；真 `input_ids` / 多请求管线是 G1。
- **激活平面在 2 步内不变**（lag 9 的卷积），故判据只覆盖状态平面；覆盖激活需要 ≥4 步（已在本档
  读数里说明机理，未铺该轴）。
- `ple/README.md §7 U3` 的**字面**（prefill/spec 的独立 writeback）仍**未做**；本 mission 只关掉
  decode 的 G2。
- `m15_hc_host.h:225` 的 `stateSlots = T_MAX`（分配上界）**不在 scope**：kernel 侧与 host 搬运
  均已按 `M15L_ACT_STATE_SLOTS(pleNReq, stateSlots)` 定尺（host 用 `PLEW::T_MAX` 作 `stateSlots`，
  与 `H_PleArgsOf` 传的值同源），但 host 的分配上界字段本身未动；已按纪律记 `TowerFinding`
  （相邻 mission 可收口）。
- **两处共用表达式的一致性**（r1 复审 P2-2）：`M15L_ACT_STATE_SLOTS` 覆盖 `nreq==0` 与
  `nreq>stateSlots` 两个边界；本档 `pleNReq=1`、`stateSlots=64` 只走 `nreq` 分支，两边界未在
  设备上被触发（属结构一致性收口，不是新读数）。

## 7. 文件清单

| 路径 | 角色 |
|---|---|
| `m15_layer_loop/m15_layer_kernel.h` | 活槽表达式宏 `M15L_ACT_STATE_SLOTS` + `M15L_PleBody` 定尺 |
| `m15_layer_loop/m15_layer_loop.asc` | `H_PleCarryState`（搬运）+ `M15_PLE_STATECARRY` 旋钮 + 第 2 步判据 |
| `m15_layer_loop/evidence/ple_wire/m191/README.md` | 本文件 |
| `.../m191/commands.txt` | 逐字复现命令 |
| `.../m191/carry_on.log` / `carry_off.log` | 档 A / 档 B 的完整设备转录 |
| `.../m191/runs_all.log` | 档 C 零回归转录 |
| `.../m191/npu_smi_*.txt` | 三档进锁时的 npu-smi 快照 |
| `.../m191/binary_sha256.txt` | 被测二进制 sha256 |
