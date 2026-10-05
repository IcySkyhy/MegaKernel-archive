# M105 · 机制口径与塔的更正（M107 r1 的 `MutexUnlock` mode-0 发现）

**本文是两轮记录**：① 塔 2026-09-27 08:21 的 M107-r1 更正（`MutexUnlock` 是 mode-0、不 drain）；
② 本轮 r1 复审的 P1 与塔裁 —— 「只把写挪进一个 mode-0 release 的区域」**不算修好**（`docs/20 §3.2 WO-B1`
选项 3 / `docs/05 §6.1 ⓔ`），要求把**写侧释放改成 drain 模式**。

**一句话结论（本轮）**：更正的事实成立；塔据此维持原裁正确；本轮的交付形态因此是**两处一起** ——
写提前到写侧 acquire 之前（位置必要性，§5 的对照档证明）+ 写侧释放 `MutexUnlock<PIPE_S>`（mode 0）
→ **`BufRelease<PIPE_S>`（drain）**（可见性由 `docs/05 §6.1 ⓔ` 的成对形态**建立**：写 → 写侧 acquire →
写侧 drain release → 搬侧 acquire → 搬；本轮的写落在写侧 acquire **之前**）。
另外：本目录**从未**声称同函数"另外 10 处有序"，本轮也没有把它们升级成"已坏"（§2），不过本轮的 drain
release 顺带把它们的可见性也纳入了同一条覆盖。

## 1. 塔要我核的三处：逐条读数与判断

| 我核的对象 | 内容（逐字/逐行） | 我的判断 |
|---|---|---|
| CANN `basic_api/kernel_common.h:138-160`（`/usr/local/Ascend/cann-9.1.0/x86_64-linux/asc/include/`） | `class Mutex { Lock<pipe>(id){ …GetBufInternal<pipe, 0>(id); } Unlock<pipe>(id){ …RlsBufInternal<pipe, 0>(id); } }`（3510 分支内） | **更正的事实成立**：`Mutex::Unlock` 的 mode 硬编码为 `0` |
| `m15_moe_layer.h`（生成物）的包装 | `BufAcquire<pipe>(id)` = `GetBufInternal<pipe, false>(id)`（注释「acquire 立即获取」）；`BufRelease<pipe>(id)` = `RlsBufInternal<pipe, true>(id)`（注释「release 延迟到 pipe drain」）；`MutexLock/MutexUnlock` = `AscendC::Mutex::Lock/Unlock<pipe>` | ⇒ S3 那条 token 的**释放端是 `MutexUnlock<PIPE_S>`（mode 0，无 drain）**，消费端是 `BufAcquire<PIPE_MTE3>`（mode false）。更正里"本仓 `BufRelease` 传 true、`MutexUnlock` 传 0，二者不同"属实 |
| 仓内规则 `docs/05-…md`「同步约束（硬性要求）」 | 逐字：「跨 pipe 数据交接（如 MTE2→V）用 **drain 模式 BufferID release**（`RlsBufInternal<pipe,true>` + 对侧 get）即可承载数据可见性（10/10 PASS 实证）—— **但公开 `Mutex::Lock/Unlock` 把 mode 硬编码为 0，必须用 Internal 原语**」；`docs/05` §6 的 BufferID 行：「`mode=false` rls 立即生效（互斥交接），`mode=true` rls 延迟到 pipe drain（同步点）」 | **仓内规则确实把 mode-0 的 release 排除在"承载数据可见性"之外** ⇒ 更正成立 |
| 仓内规则 `docs/06-…md:76`（§5.2 第 2 条） | 「**跨 pipe 的 UB 数据交接必须走 `SetFlag/WaitFlag` 事件**（MTE2_V、V_MTE3、…）**或 drain BufferID**…`PipeBarrier<PIPE_X>` 只能阻塞标量等 pipe 指令退休，**不能保证 UB 数据路径对另一 pipe 可见**」 | 同向：可见性要么 set/wait 事件、要么 drain BufferID |

**结论**：我没有理由反驳复审 —— 它引的 CANN 事实与仓内规则都成立。**因此产物里我原先那句注释
"MTE3 读 UB 的次序只由「S 侧 unlock（drain）→ MTE3 acquire」保证"是不准确的**（那里的 unlock 是 mode 0，
不是 drain）。我已把它改成不含未证机制的口径（§4）。

## 2. 我此前**没有**做过的声称（更正第 1 条的前提，对我这份文档不适用）

塔的更正针对「只有 `bUb[32]` 无序、同函数另外 10 处有序」这条结论。**这条结论不在我的证据里**。

**可复跑的非自指检查**（把范围限定到**程序输出**，不把本说明文档自己算进去 —— r1 复审 P2-2 指出上一版那条
命令是自指的：该短语本身就出现在这些说明句里，所以"无输出"不成立）：

```
$ cd m15_layer_loop/evidence/moe_race && grep -rn "另外 10\|另 10\|其余 10\|10 处有序" run_*.log; echo rc=$?
rc=1      # 该命令的输出为空、grep 以 rc=1 表示无匹配（命令可复跑）
```

⇒ 前面这条命令在**程序输出**（`run_*.log`，冻结入库）里无匹配；它**不覆盖**说明文档自身，本目录也不据此
推断任何"全仓/全部"式的结论。手写说明里出现该短语的地方只有两类：**下面的登记表**，以及**引用该声称以否定它**
的句子（例如本节第一句）—— 该短语出现在说明句里，所以"命中计数"本身没有意义，本目录不给这类断言。

本 mission 的 README / 各日志**从未**对 `IndexGenStage::Run` 里其余标量写（`cntUb/offUb/curUb/srcUb/
expUb/invUb/wtkUb`）作任何顺序性或可见性声明；只声明了 `bUb[32]` 那一处（有 36 条红与 A/B 位置对照支撑）。
⇒ 按塔的选项 (b)，我现在把这些位置显式登记为 **顺序性未确认**：

| 位置（base rev `20bd20d` → 本 tip 的实测行号；**以内容锚点为准**） | 相对次序 token（本 tip） | 顺序性状态 |
|---|---|---|
| `cntUb[e]` / `offUb[e]`（计数与前缀和；锚点 `cntUb[e] = cntUb[e] + 1;`：base `:1320` → tip `:1324`） | 在写侧 acquire `MutexLock<PIPE_S>(BUF_AIV_IDX)` 之前 | **未确认**（本 mission 未测） |
| `curUb[e]` / `srcUb[pos]` / `expUb[pos]` / `invUb[s]` / `wtkUb[...]`（落位循环；锚点 `wtkUb[t * 16 + k] = …`：base `:1357` → tip `:1375`） | 在写侧 acquire 与**写侧 drain release**（`BufRelease<PIPE_S>`，本 tip `:1385`）之间 | **未确认**（本 mission 未测；但见 §3 第 3 条的大面积旁证） |
| `bUb[32]`（诊断槽） | 改前在 `BufAcquire<PIPE_MTE3>` **之后** | **已证无序**（36 条红 / 位置 A/B，见 README §4①⑤） |

## 3. 机制：上一轮形态为什么"不算修好"（塔裁的落点）

上一轮我只做了"把写提前"（释放仍是 `MutexUnlock<PIPE_S>` = `RlsBufInternal<PIPE_S,0>`，mode 0）。
按 `docs/05` §6「`mode=false` rls 立即生效（互斥交接），`mode=true` rls 延迟到 pipe drain（同步点）」，
那次写只被一个**不做排空**的 release 覆盖：

- 消费端 `BufAcquire<PIPE_MTE3>` 的 get **不保证**生产 pipe（S）已排空 ⇒ "MTE3 那次读一定在标量写落地之后"
  **没有被建立**；读数只能支持"这次写现在提前了一整段落位循环（数千条 S pipe 指令）= 提前量"。
- 本缺陷本身是**调度依赖**的（同源码 `c9f97e3` 36 红 / main 绿；两基座只差 attn kv/cache 一族）⇒
  "提前量"这种解释恰好最难外推。
- 而我当时的读数**区分不了**"次序已建立"与"提前得足够远"（都表现为 0 条红）⇒ 正如塔裁：不能以"已修"入库。

**所以本轮把 release 换成 drain**：`BufRelease<PIPE_S>(BUF_AIV_IDX)` = `RlsBufInternal<PIPE_S,true>`
—— S pipe 排空后才释放 token，**覆盖它之前的全部标量写**（含诊断槽那次）。成对形态 = `docs/05 §6.1 ⓔ`
「写 → 写侧 acquire → 写侧 drain release → 搬侧 acquire → 搬」（本轮的写落在写侧 acquire **之前**；
ⓔ 的覆盖面前提只要求写在该 drain release 之前）。同款的**已受审**站点 = `m15_ple.asc`（IDS/BODY 三处，
内容锚点 `BufAcquire<PIPE_S>(id)`（写侧成对）→ 标量写 UB → `BufRelease<PIPE_S>(id)`，见其文件头 §BUF）。
（r1 复审信里另提的 `m15_hc_layer.h::CombineStage` 我在 `main` 上没核到 PIPE_S 的 drain 站点 —— 该文件的
drain 站点是 `PIPE_MTE2`/`PIPE_V`；本条**不据信件转述**，只记我自己核到的。）

## 3b. 位置必要性（即使有 drain，也不能不挪写）

`c9_racy2`（只把那次写搬回 `BufAcquire<PIPE_MTE3>` 之后、**保留** drain release，其余一字不改；二进制
`1c0d3a6961e7`，与 r1 复审自己跑的 D2 同一二进制）⇒ rc=1 `fails=36`，`first off 988224`。
⇒ 任何 release 都排在那次写**之前**，drain 也覆盖不到它 ⇒ **挪位与换 drain 缺一不可**（README §4⑥ 的四行表）。

## 4. 本轮改了什么（两处一起）

**上一轮**（只改注释）已经不足以回应塔裁；**本轮**在生成器里做了三件事（都经 `--check` 复核）：

1. **保留**：诊断槽那次写提前到写侧 acquire（`MutexLock<PIPE_S>`）**之前**（位置必要性见 §3b）；
2. **新增**：写侧释放 `MutexUnlock<PIPE_S>(BUF_AIV_IDX);` → **`BufRelease<PIPE_S>(BUF_AIV_IDX);`（drain）**
   —— 这一处就是 r1 复审 P1 的落点（它的提案原文："把写仍留在它之前，只换释放"）；
3. 生成器断言从三条扩到**四条**：写唯一 / **写侧 drain release 存在且唯一、且不得残留 `MutexUnlock`** /
   次序（写 → `MutexLock<PIPE_S>` → `BufRelease<PIPE_S>` → `BufAcquire<PIPE_MTE3>`）/ 改前特征串消失且上游确是抢跑+mode-0（防空转）。

**注释改动的二进制身份**（可复跑，沿用上一轮的同一现象）：

```
$ cmake --build build -j32 --target m15_layer_loop     # 同一棵树，替换 artifact 前后各一次
注释改动前（artifact 81b4c76bb5b8…） → build/m15_layer_loop sha256 8310e71df5d83f21…    # 与 D1 同一二进制
注释改动后（artifact ee12ba635c95…） → build/m15_layer_loop sha256 8310e71df5d83f21…    ← 完全相同
```

⇒ 本目录里本轮形态的全部读数（20 次确定性、9 层落盘、拨值探针）所用的二进制 `8310e71df5d8…` 就是
**改注释后的 tip 二进制**，无需重跑。

## 5. 对照档：drain release 能救什么、不能救什么（本轮全部重取）

| 档 | 写的位置 | 写侧释放 | 二进制 sha256（前 12） | `runs=all` |
|---|---|---|---|---|
| **交付形态** | 提前到写侧 acquire 之前 | **`BufRelease<PIPE_S>`（drain）** | `8310e71df5d8` | rc=0 `ALL PASS（checks=2068, guards=290, fails=0）`；+20 次确定性（§4④）|
| **反例**（`c9_racy2`） | 搬回搬侧 acquire **之后** | `BufRelease<PIPE_S>`（drain） | `1c0d3a6961e7` | rc=1 `FAILURES PRESENT（checks=2068, guards=290, fails=36）`，`first off 988224` |
| **上一轮形态** | 提前 | `MutexUnlock<PIPE_S>`（mode 0） | `2d279dc01fa5` | rc=0 全绿（但按塔裁**不算修好**：ⓔ 选项 3） |

读法：

- **反例仍红**：那次写发生在消费端 `BufAcquire<PIPE_MTE3>` **之后**，任何 release 都在它之前 ⇒ drain 也
  覆盖不到它 ⇒ **"挪进次序 token 之内"是本缺陷的必要动作**（这一档与 r1 复审自己跑的 D2 是同一二进制）。
- **上一轮形态也绿**：所以症状面**区分不了**"次序建立"与"提前得足够远" —— 这正是塔裁"不能以已修入库"的理由；
  本轮把它变成**规则保证**（写落在 drain release 覆盖的 span 内）。
- **交付形态绿**：写提前 + drain release ⇒ 次序由 `docs/05 §6.1 ⓔ` 的成对形态建立。

## 6. 代价与后续（如实写）

1. **性能**：drain release 让 S pipe 排空后才释放 token（消费端也跟着等）⇒ 削弱 "scalar 发射在指令前面" 的
   重叠；这是 `docs/05`「PIPE_S 同步限制」担心的那条。本 mission **未做**段级计时（不给耗时读数），
   README §5 已写明这条 trade-off。
2. **"另外 10 处"**：本轮 drain release 顺带覆盖了它们（它们本来就在写侧 acquire 之后、释放之前），
   但**它们的顺序性仍登记为未确认**（本 mission 没有对它们做受控 A/B，也没有声称它们有序）。
3. **m13 上游同款**（README §7 第 1 条）仍应另立 mission。

## 7. 复跑本文读数所需的最小命令

```
# 1) 产物行为不变（注释改动不改二进制）
cd <wt-105>/m15_layer_loop && python3 lift_moe_segment.py --check; echo rc=$?    # 期望 rc=0
bash evidence/moe_race/reproduce_host.sh /tmp/m105_repro                        # 四档二进制 sha256 见 README §1

# 2) 两个 /tmp 对照档（D1/D2）的构建 + 跑
#    D1/D2 的 artifact 由「库内 artifact + 两处定点替换」得到（替换命令见 §5 的说明），
#    构建口径与 reproduce_host.sh 相同，跑法 = flock -w 900 /tmp/npu0.lock env M15_LAYERS=48 \
#    M15_STEPS=3 M15_HC_LAYERS=0,1,3 <bin> <manifest> all
```
