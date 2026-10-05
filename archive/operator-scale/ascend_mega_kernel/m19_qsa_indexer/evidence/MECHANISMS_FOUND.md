# M19 选择段：机理归档（M35 的 6 条 + 1 条否证 + 1 条教训，**逐条本轮复核**）

每条给：**现象 / 读数 / 复现 / 状态**。判据口径见 `docs/17 §1.1`；契约与判据见 README §1/§3。
复现环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh && cmake --build m19_qsa_indexer/build -j4`。
**注意（M53 起）**：守卫已移除，运行**不再需要** `M19_ACK_INCOMPLETE_SELECTION=1`。
环境变量：`M19_V`（覆盖 V，pos 默认取 4V−1）、`M19_POS`（显式覆盖 pos ⇒ 造 tail>0 档）、`M19_BUDGET`（token 预算，blkTopk=budget/4）、
`M19_CORES`（打分列切分核数 1/2/4/8/56）、`M19_SINGLE`（**兼容性别名，不推荐**：语义 = `M19_CORES=1`，**仅当 `M19_CORES` 未设时**生效；推荐改用 `M19_CORES`）、`M19_OUT`（dump 目录）、`M19_PROBE_NVAR`（探针只跑前 N 个变体）。

| # | 机理 | 现象 / 读数 | 复现 | 本轮（M53）复核结论 |
|---|---|---|---|---|
| (A) | **子块计数越界 64-lane**：`CountSub`/`EmitSub` 用 `full`（64 lane）掩码，把**邻子块**的 score 也计入 | 每个 **16 列**子块数到 58~104 个（应 ≤16） | `M19_V=256 M19_BUDGET=8 M19_CORES=1` | **已修（M35 修在库）**；本轮把该掩码从 `UpdateMask(nv)` 换成「lane 序号 < nv」的比较掩码（同族更稳，见 (LANEF)） |
| (B) | **候选计数写错槽**：`CountCand(c,0)` 恒写槽 0，而 `WriteCountToGm(d-1,…)` 读槽 d-1 ⇒ `digit` 恒 3 ⇒ `base` 回绕 ⇒ `candGT=0/candEQ=-1` | 修复前 stats `candGT=0 candEQ=-1`（V=256>blkTopk=2 却像走了 select-all） | `M19_V=256 M19_BUDGET=8`，读 `*_stats.bin` 字段 11/12 | **已修（M35 修在库）**；M53 起这条交换整体**不再存在**（选择段单核，无跨核计数交换） |
| (LANEF) | **`UB_LANEF` 存成了 `lane%4`**：Cast 发生在 `And(idx,idx,3)` **之后** ⇒ 前缀掩码错 ⇒ 跨核前缀为负 | 修复前默认档 232/286/1004/**0** | 默认档读字段 5 | **已修（M35 修在库）**；M53 复核：`UB_LANEF` 是**选择段**所有动态掩码的唯一来源（数值前端 `ScanChunk` 里原有的 `UpdateMask(nc)` 未动），并且 UB 布局里不再与 `UB_TAIL` 重叠（原 `UB_SELS` 与 `UB_TAIL` 后半同址，是潜在踩踏源，已删） |
| (C) | **越界守卫应"截断"不该"整块丢弃"** | 修复前最小档 `clipped = localCgt`（全被丢 ⇒ 0 token） | `M19_V=256 M19_BUDGET=8` | **已修（M35 修在库）**；M53 起 rank 由单核按列序单调分配，正常路径 `truncEmits` 恒 0（守卫留作安全网） |
| (ceq) | ~~`ceq` 读回虚高（幻影前缀）~~：M35 认为"GM 里 `ceq` 合计 2 而 kernel 读回 56 ⇒ 错在读回路径" | M35：小档 GM `ceq=[1,1,0,…]` 合计 2，kernel 上报 `eqTotal=56`；`myEqBase=[0,1,2,2]` 被判为"幻影" | 见下方"复核读数" | **原结论被本轮否证**（根因是**口径**不是读回）：<br>① `myEqBase=[0,1,2,2]` **与 GM 逐核一致**——本轮默认档实测 `myGtBase/myEqBase` 与 `*_cnt.bin` 的累积前缀**逐核完全相等**（如 A 档 AIV0..15：`0,16,33,47,60,74,88,109,128,143,164,181,195,213,227,240`），而 `myEqBase−myGtBase` 在**第一个 tie 所在核之前恒为 0**、之后恒为 1（A 档 tie 在 AIV22）⇒ 读回路径**没有**虚高；<br>② 两路求和（`laneF<NAIV` 与 `UpdateMask(NAIV)`）读数**相同**（默认档都是 512 = `count(score≥K)` 的真值）⇒ M35 的 `eqTotal=56` **未能复现**（其探针后被 M35 自己标记"不可信"）；<br>③ 真正的缺陷是**语义**：`CountCand(candEQ,…)` 计的是 **`score ≥ K` = GT ∪ tie**，不是 tie 本身。旧代码把它当 tie 前缀用（`need − myEqBase`）⇒ 只要 GT 前缀超过 `need`，配额就恒 0，而配额只发给 `myEqBase` 最小的核（往往没有 tie）⇒ 边界块永不写出；<br>④ M53 修法：**设计简化**（选择段单核 ⇒ 无跨核前缀、无配额），并把该口径写进代码注释；`tieAll` 仍作为诊断量落 trace（`tieAll` vs `need`） |
| (tail) | ~~tail 收尾少写 2 个 token~~ | M35：B 档"块集合完全正确、只丢 tail 的 2 个" | 默认档 B | **本轮复核：现象不成立（属算术推断，未重放修改前二进制；标注为推断）**。B 档实测少的是 **4 个 token（1 个 block）**，与 A/D 同源（tie 边界块未写出）；tail 两列**始终正确写出**（`check_ref` 的 tail 判定项 4/4 档 ✅）。M35 的"少 2"是把 `tokens=2046` 与 `count=2050` 的差拆成"1 块 + 2 tail"的误读 |
| — | **已否证假设**：缺 MTE2→V 事件（`docs/06 §4/§5`） | M35 在 8 处补 `SetFlag/WaitFlag<MTE2_V>` 后 `eqTotal` 仍是 56 | — | **否证维持**（那是另一个量）。但 M53 **在本模块自己的新代码里撞到同一个坑**并修掉了：`LoadStageOnly` 的「MTE2 搬 GM→UB_CROW」与 V 读之间缺交接时，症状是"子块里只有零星几列被选中、输出少块，且加一条无关 DMA 就消失"的 heisenbug ⇒ 照 `ScanChunk` 的 `MTE2 acq/rel → PipeBarrier<MTE2> → V acq/rel` 写，并额外补 `MTE2_V` 事件 |
| — | **方法教训**：内核侧对拍探针必须先自证 | M35：同一 `v` 的归约 4.0 而上报 `gtTotal=1` ⇒ 探针自相矛盾 | — | **教训成立并被本轮采纳**：`probe_aiv_barrier.asc` 自带对照（无屏障变体必须违规）、自带上界（不会挂死）、token 语义使"没写"与"写了上一轮的值"可区分 |

## M53 新增机理（选择段为什么必须是单核）

| # | 机理 | 现象 / 读数 | 复现 | 状态 |
|---|---|---|---|---|
| (DEV) | **per-AIV 的 `K` 随机分裂**：16 轮 radix 每轮都做「56 核写计数 → 全核读回求和」——形态是**同一片槽多轮复用、每轮只挂一道「写完」屏障**（读回后到下一轮写之前无屏障），而该形态下 AIV 间的可见性**不可证** | 同一 case 连跑 3 次：多数派 `K` 都在变，每次有 1~5 个核与多数派不同（如 `1097757488` vs `1097695024/1098731312/1097790256`）；B 档选中集合与自身 logits 的理想 top-512 差 **84 个 block**，而 token 计数仍"看起来对" | `M19_CORES=56`，读 `*_stats.bin` 字段 12（M53 前的字段表） | **已消除**（选择段改单核；不再有跨核计数交换） |
| (BAR) | **AIV 间计数交换的适用边界：单发可证、同片槽多轮复用不可证**（`149b920` 曾把它写成「没有可靠原语」，属过度概括；`7edd74d`/`c641b5a` 按 `docs/06 §5.2 规则 6 / §5.3` 收窄） | `probe_aiv_barrier.log`（6 变体）：**单发** v1（`Set<0,MTE3>(1)+Wait<0,MTE2>`，本 kernel 交接段用法）与 v4（官方 `SyncAll<true>`，wait 挂 `PIPE_S`）都 **0/56 违规**；**同一片槽 4 轮复用** v2（同款 flag）**每次运行都违规**（本仓已归档 5 次运行里违规核 18~52/56、单次最少见到 1 个槽；只覆盖已归档运行、不作为上下界）；对照 v0 无屏障 56/56 违规 | `./m19_qsa_indexer/build/probe_aiv_barrier`（`M19_PROBE_NVAR=4` 只跑前 4 个） | **已规避**：选择段改成「只需一次交接」的用法（v1/v4 那种）且带 AIV0 记号校验；**并行化前置条件 = 带重校验/复位语义的多轮交换协议**（换屏障原语不够，v5 已证）。**适用范围前提**：上述"不可证"是在**每轮只挂一道「写完」屏障**（读回后 → 下一轮写之前无屏障）的形态下观测的；本行**不**据此下"就是缺一道屏障"的结论（另由 probe mission 实测中）。另：v4/v5（官方 `SyncAll<true>` = mode0+flagId14+`wait_flag_dev(PIPE_S,14)`）排除了"wait 挂 MTE2 才是根因"的归因 |
| (COMP) | **ASC 编译器对 AIV 循环体敏感** | 循环体多一条比较/多一个动态掩码即报 `Unsupported scalar instruction in AIV loop`（`-O0..-O3` 同样）；缩短循环体即过 | 见 README §3.6 | **已规避**（1 阈值/轮、循环内只用满掩码、阈值循环外算好） |

## 复核读数（(ceq) 的能量化证据）

```bash
# ① 默认档：每核前缀 == GM 里计数数组的累积前缀（逐核相等 ⇒ 读回没有虚高）
./m19_qsa_indexer/build/m19_qsa_indexer 0        # 只跑 A 档
python3.12 - <<'PY'
import numpy as np, pathlib
d=pathlib.Path("m19_out"); st=np.fromfile(d/"A_close_2048_stats.bin",dtype=np.int32)[:32*32].reshape(32,32)
cnt=np.fromfile(d/"A_close_2048_cnt.bin",dtype=np.uint8)[:2048]
gt=cnt[1536:1536+224].view(np.float32); eq=cnt[1792:1792+224].view(np.float32)
print("GM gt 累积:", np.cumsum(gt)[:16].astype(int))
print("kernel myGtBase:", st[:16,5]); print("kernel myEqBase:", st[:16,6])
print("GM eq 累积:", np.cumsum(eq)[:16].astype(int), " gtTotal", st[0,13], " eqTotal", st[0,16])
PY
# ② 小档（M35 报 eqTotal=56 的场景）：两路求和读数相同
M19_V=256 M19_BUDGET=8 M19_OUT=./m19_out_small ./m19_qsa_indexer/build/m19_qsa_indexer 2
```
（M53 起 `myGtBase/myEqBase/eqTotal` 这些跨核前缀量已随单核化删除；上面的复现指向 M53 前的字段表，
即 `git show 175a488:m19_qsa_indexer/m19_qsa_indexer.asc` 时代的 stats 布局。）
