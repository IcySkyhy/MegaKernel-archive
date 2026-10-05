# M111 —— PLE 的「标量写 GM」落盘通路修复：消费者盘点 + 读数

> 立项理由（人类原话，逐字）：「**本来就不应该用 scalar pipe 写 GM，为什么要这么做**」。
> 本文件是 M111 的交付物：三处标量落 GM 的**消费者盘点**、改动清单、改前/改后读数与命令转录。
> 上游两份证据必须连带读：`WITNESS.md §4`（①→② 交接 = U-A）、`ple/REAL_TABLE.md §6.4`（设备侧计数落盘）。
>
> **本文件里的每条读数都是同一份二进制**：`binary_sha256.txt` 记的 sha256 与
> 目录下全部 `*.log` 同批（本批由 `run_wire_evidence.sh` 一次采集）。
>
> **r2（2026-09-27，按复审 P1-1 收口）**：三处落盘通路的**核内次序**已从「`PipeBarrier<PIPE_ALL>`」
> 改成 `docs/05 §6.1 ⓔ` 的**写侧成对 drain release** 形态（§2.4 有形态、先例与一个会挂死的陷阱）。
> r1 把 `m15_attn_cache.h:452-462` 当"本仓正确写法"引用过的那几处说法**在 r2 里都改掉了** ——
> ⓔ 明写那处「既不是反例也不是先例，**不得**当正确形态引用」。功能读数在 r2 与原实现逐项相同（§3）。
>
> **计数口径（M179 标注）**：本文件记的是 **M111 时点**的 `Pw` 判据 = **10 条**。M124（`6a8fee3`）之后
> 为 **12 条**（新增 `Pw.kv.T3` / `Pw.kv.nonvac`）—— 见 `WITNESS.md §2` 的计数口径注；本文件的历史读数不改。

---

## 0. 一句话结论

① 的 ids 落点、①② 的设备侧计数落点，原先走 **GM 标量 `SetValue`**；实测这条路有两个彼此独立的
失效机制（§1.4）。改成 **UB → MTE3 `DataCopy/DataCopyPad`**、核内次序按 **`docs/05 §6.1 ⓔ`
的标量条款**（写侧 `BufAcquire<PIPE_S>` → 标量写 → `BufRelease<PIPE_S>` 的 **drain release** →
才搬；**正对照** = `m15_hc_layer.h::CombineStage`；不靠 `PipeBarrier`）之后：

- **已接线路径的默认档（`M15_PLE_WIRE=1 M15_PLE_STAGE=31`，① 也跑）由 7/10 转成 10/10**，
  `Pw.emb.T1` 由 `bad=15/16` 变成 `bad=0`，且 `Pw.wired.delta` 与「②③④⑤ 独跑」档一致（都是 5636/10240）；
- `dev_range_fail` 在 bit10 档由 **16** 变成 **160**（10 个核各 16 个越界 id，全部读回）；
- ② 的设备侧计数从「恒 0 的空判据」变成真的计数，并有了判据（`B_dev.fail`）与咬合力见证（变异 bit15）。

**没有**改动 `PleGemv → mmad`、没有改 `m15_ple.asc` 以外的段体、没有做性能优化（显式不做，见 §5）。

---

## 1. 三处标量落 GM 的消费者盘点（先盘后改）

### 1.1 站点 ①：`IdsOneToken` 的 ids 落点

| 项 | 内容 |
|---|---|
| 位置（改前） | `m15_ple.asc` 的 `IdsOneToken` 内 `outG.SetValue(t*NG + g, id)`（M85 时代的 `:249`） |
| 写入者 | ① 的核 `bid`：循环 `for (t = bid; t < nTok; t += nblk)` ⇒ **每个 token 由一个核独占写出**（16 个 int64 = 128 B，独占一条 128 B 线） |
| 消费者（已接线路径） | **② 的别的核**。`PleGather` 按 **item** 轮转分核（`for (idx = bid; idx < n; idx += nblk)`，`idx = t*NG+g`），token 的核分配与 item 的核分配**不同** ⇒ 除「写 ids 的那个核恰好处理该 token 的 g=0」外，读到的都是别的核写的内容 |
| 消费者（独立档） | host（下载 `A_ids.bin` / `B_ids.bin`）；以及**下游 kernel**：`RunBody` 把 ① 的输出当 body 的 `idsIn`（① 的核 0..1 写、body 的核 0..31 读） |
| 是否跨核读 | **是**（两条路都是） |
| 顺序由谁保证（改后） | **核内（本轮 write→read）**：`docs/05 §6.1 ⓔ` 的标量条款 —— 写侧 `BufAcquire<PIPE_S>(20)` → 标量写 UB → **`BufRelease<PIPE_S>(20)`（drain release：S pipe 排空，覆盖它之前的标量写）** → `BufAcquire<PIPE_MTE3>(20)` → `DataCopy` → `BufRelease<PIPE_MTE3>(20)`。**同一块 UB 被逐 token 复用**时的跨轮次序也由这套 token 承担（写侧 acquire 等上一轮 MTE3 的 drain release）—— 见 §2.1 / §2.4。**跨核**：本条是**读数**不是论证 —— 实测「同一 base（main 源码自建二进制 `1596284e…`）B 档 `Pw.emb.T1 bad=15/16` → 本分支 `bad=0/16`」，即 ② 的 16 个 item（其中 15 个由**别的核**处理）都读到了本核正确写出的 ids；`M15L_PhaseBoundaryAiv` 的 set 挂 `PIPE_MTE3` 是**它为什么成立**的一种解释，但本 mission 没有独立探到「先到的核一定等不到数据」的反向读数，故按事实记成读数（复审 r1 的观察项，§2.4 末）。**独立档**：kernel 结束 + `aclrtSynchronizeStream`（两次独立启动） |

### 1.2 站点 ②：① 的 `failG.SetValue(bid, bad)`

| 项 | 内容 |
|---|---|
| 位置（改前） | `m15_ple_ids_kernel` 内 `if (bad > 0u) failG.SetValue(bid, bad)`（M85 时代的 `:891`） |
| 写入者 | ① 的每个核（每核 4 B，落在 `fail[bid]`，与相邻核的槽位在同一 cache line 内） |
| 消费者 | **host**：`RunIds` 求和 128 个槽位 → `*_ids_meta.txt` 的 `dev_range_fail`（层循环侧的对应物是 `Pw.fail` 的 `si`） |
| 是否跨核读 | **否**（host 在 kernel 结束后读）—— **但这一处仍然是 bug**，见 §1.4 的机制 (ii)：它丢数据的原因与「跨不跨核」无关 |
| 顺序由谁保证（改后） | 该落盘一次调用一次（不在循环里），但仍按 ⓔ 的**写侧成对**形态：`BufAcquire<PIPE_S>(21)` → 标量写 UB → `BufRelease<PIPE_S>(21)`（drain release）→ `BufAcquire<PIPE_MTE3>(21)` → `DataCopyPad` → `BufRelease<PIPE_MTE3>(21)`。**不用 `PipeBarrier` 承担可见性** |

### 1.3 站点 ③：body kernel 的 `G.fail.SetValue(bid, bad)`（**空判据**）

| 项 | 内容 |
|---|---|
| 位置（改前） | `m15_ple_body_kernel` 内 `if (bad > 0u) G.fail.SetValue(bid, bad)`（M85 时代的 `:971`） |
| 写入者 | ④⑤ 的 `bad` —— 它**从不自增**（`PleGateItem`/`PleConvItem` 里两处都只有 `(void)bad;`）⇒ 该分支是**死分支** ⇒ 槽位一直没人写 ⇒ `dev_fail` 恒 0 |
| 消费者 | host（独立档 `B_body_meta.txt` 的 `dev_fail`）；层循环的 `Pw.fail` 读 `S_FAILB`（`sb`） |
| 是否跨核读 | 否（host 侧读） |
| 处置 | **不留这条永不可能失败的判据**：删掉死分支，把 ② 真正的计数搬进 `PleGather`（数「词表外的 id」，与 ① 同口径同常量），落 `G.fail[bid]`（UB → MTE3 `DataCopyPad`），并**给它一条判据**（`B_dev.fail`）与**咬合力见证**（变异 bit15）。读数见 §3.2 / §3.3 |

### 1.4 实测到的两个**彼此独立**的失效机制（不要把两者混为一谈）

- **(i) 同一次启动内、跨核可见性**：GM 标量走 scalar pipe，不在相位边界（set 挂 `PIPE_MTE3`）的
  drain 覆盖内 ⇒ **除「自己写自己读」的那个核，其余核读到旧值**。证据 = `WITNESS.md §4` 的逐 item
  转录（`g=0` 对、`g=1..15` 全是同一个旧值）。**修前 B 档红、修后绿**（§3.5）。
- **(ii) 同一 cache line 内、多核小写互相覆盖**：`M15_PLE_MUT=1024` 下 A 档有 **10 个核**各数出 16 个
  越界 id，改前 `dev_range_fail` 读回 **16**（应为 160）⇒ 只剩 1 个核的值；**同一个 kernel 里**
  每核独占 128 B 线的 ids 落点读回 **160/160 全在**。⇒ 机制 (ii) 与「谁来读」（host 还是别的核）无关。
  读数与命令见 §3.3。这一条也**更正**了 `ple/REAL_TABLE.md §6.4` 原先那条被误读的「正对照」。

---

## 2. 改了什么（`文件:符号`）

### 2.1 `m15_layer_loop/m15_ple.asc`

| 符号 | 改动 |
|---|---|
| `§BUF`（新增） | 本段自己的两个 buffer id：`BUF_PLE_IDS = 20`、`BUF_PLE_SINK = 21`。**依据**：CANN 9.1.0 的 `Mutex::Lock/Unlock<pipe>(id)` 就是 `GetBufInternal/RlsBufInternal<pipe,0>(id)`（`kernel_common.h:137-160`）⇒ `MutexLock<PIPE_S>(X)` 与 `BufAcquire<PIPE_MTE3>(X)` 是同一 id 上的同一套 token；`MAX_MUTEXID = 27`（`kernel_event.h:695`），20/21 避开 hc 段的 0–16 与 attn 段的 10–12 |
| `VOCAB_ROWS`（新增） | 词表总行数 `320001536`，① 的值域判据与 ② 的输入值域守卫共用（原先 ① 里是字面量） |
| UB 布局 | 新增 `UB_IDS = 52512`（`int64[16]` = 128 B）、`UB_SINK = 52640`（`uint32[8]` = 32 B）；`UB_END` 52512 → 52672（`m15_layer_kernel.h:244` 的 `static_assert(UB_END <= UB_PEAK_FUSED)` 仍成立，UB_PEAK_FUSED ≥ 89472） |
| `IdsOneToken` | 落点 `outG.SetValue(t*NG+g, id)` → `idsL.SetValue(g, id)`（UB）+ ⓔ 三元组（见 §2.4）+ `DataCopy(outG[t*NG], idsL, Block1(128))` |
| `PleGather` | 新增「词表外 id」守卫与计数 `oob`：`(id < 0) || (id >= VOCAB_ROWS)` ⇒ 计数并 `continue`（**不拿它当行号**，避免越界读表）；段末把 `oob` 经 UB → MTE3 `DataCopyPad(G.fail[bid], …)` **无条件**落盘（0 也写：「0」是设备真的核过一遍的读数）。变异 bit15 也在此处（`id += VOCAB_ROWS`） |
| `m15_ple_ids_kernel` | `failG.SetValue(bid, bad)` → UB → ⓔ 三元组 + `DataCopyPad(failG[bid], …, ExtBlock1(4))` |
| `m15_ple_body_kernel` | 删掉④⑤ 的死分支 `if (bad > 0u) G.fail.SetValue(bid, bad)`；`bad` 形参保留（层循环 TU 按该签名调用 `PleGateItem`/`PleConvItem`，那两个文件不在本 mission scope） |
| 文件头注释 / `§BUF` / UB 注释 | 改成如实描述；`§BUF` 里补**成对纪律与挂死陷阱**（§2.4）；把 `m15_attn_cache.h` 从「正确写法」改成 ⓔ 明禁引用的**不作先例**（正对照改指 `m15_hc_layer.h::CombineStage`）；bit15 的说明 |

### 2.2 `m15_layer_loop/m15_ple_wire.h`（**机械生成物，不是手改**）

`evidence/ple_wire/lift_ple_device_segment.py` 从 `m15_ple.asc` 逐字抽出 `namespace M85P { … }`。
证据（§4.1）：改 `m15_ple.asc` 后**先跑 `--check` 得 rc=1（漂移被抓到）**，再跑生成器，再跑
`--check` 得 rc=0。三个时点的字节数：main = 40723 B、M111 r1 = 44047 B、M111 r2（本 tip，含 ⓔ 形态）= 45471 B。

### 2.3 判据的两处改动（逐条列理由 + 咬合力见证）

| 改动 | 理由 | 「把被测对象弄坏必须变红」的见证 |
|---|---|---|
| `m15_ple_check.py` 新增第 14 条 `B_dev.fail`（读 `B_body_meta.txt` 的 `dev_fail`） | 原来 `dev_fail` 恒 0 **且没有任何判据读它** ⇒ 「修好了计数」如果不判，等于没修（§1.3） | 变异 **bit15**（`id += VOCAB_ROWS`）：`dev_fail` 32 → `B_dev.fail` FAIL（§3.2） |
| `m15_ple_mutants.py` 新增变异位 **bit15** | 新判据必须有咬合力见证；这一位**不假借任何上游分歧**，它就是把被测对象（② 的输入值域守卫）弄坏 | 见上；位 15 的 `FAIL 集合 = {B2.emb, B_dev.fail}` |

**既有 13 条判据与既有 14 个变异位：口径一条都没放宽**，读数见 §3.1 / §3.2（改前基线 13 条全 PASS
＝ 改后基线 13 条仍全 PASS，新增的第 14 条也 PASS）。

### 2.4 落盘次序改成 `docs/05 §6.1 ⓔ` 的形态（r2 按复审 P1-1 收口）

**r1 的形态不合格在哪**（复审 P1-1 逐条核过）：三处落盘的 release 都**在搬运之后**、且挂在
**MTE3（消费侧）** ⇒ 它只挡住「下一轮标量写压到上一轮没读完的 UB」（= 逐 token 复用的循环保护），
**挡不住本轮「S 写 → MTE3 读」**；本轮 write→read 之间唯一的原语是 `PipeBarrier<PIPE_ALL>`，
而 `docs/06-m0-bringup.md` §5.2/§5.3 对它逐字判「不能保证 UB 数据路径对另一 pipe 可见 / **不可靠**」，
ⓔ 亦明写「**不得**只靠 `PipeBarrier`」。此外 r1 把它写成「抄自本仓正确写法 `m15_attn_cache.h:452-462`」，
而 ⓔ 明写那两处「既不是反例也不是先例、**不得**当正确形态引用」。

**r2 的形态**（三处一律，`BufRelease` 用 `M15H::BufRelease` = `RlsBufInternal<pipe,true>` 的 drain 模式）：

```
BufAcquire<PIPE_S>(id);        // 写侧成对：S pipe 取得槽
  ... 标量写 UB ...
BufRelease<PIPE_S>(id);        // drain release：覆盖它「之前」的全部标量写
BufAcquire<PIPE_MTE3>(id);     // 搬侧等写侧的 drain release
  DataCopy / DataCopyPad
BufRelease<PIPE_MTE3>(id);     // 延迟到 MTE3 drain（跨 pipe 交接）
```

三处对应的 id 与站点：`BUF_PLE_IDS(20)` = `IdsOneToken` 的 ids；`BUF_PLE_SINK(21)` =
`PleGather` 段末的计数与 `m15_ple_ids_kernel` 的 fail 槽。**正对照**（塔已升格为舰队级）=
`m15_layer_loop/m15_hc_layer.h` 的 `HyperConnOp::CombineStage`（纯 BufferID 的 V→MTE3 握手，
每一步 release 都是 drain 模式；本文件是同一形态的 S→MTE3 版本）。

> ⚠ **会挂死的陷阱（复审实测，写在这里给后来人）**：**只**把 `PipeBarrier<PIPE_ALL>` 换成
> `BufRelease<PIPE_S>(id)`、**不给写侧配 `BufAcquire<PIPE_S>(id)`** ⇒ **挂死**
> （`timeout 60` rc=124；同条件对照 = 原二进制 8 s rc=0）。原因：token 是**跨 pipe 共享**的一套计数，
> 只 release 不 acquire 会把它推成不平衡态，下一次 acquire 永远等不到。⇒ 写侧
> `BufAcquire` / `BufRelease` **必须成对**（本文件三处都是完整三元组；注释也留在 `§BUF` 里）。
> 本 tip 实跑佐证：独立档 `time timeout 120 … timeout 120 env M15_PLE_OUT=…` → `real 0m6.807s` rc=0。

**没有一并收的同形站点（登记，不在本 mission 内）**：本文件里还有若干「写 UB → `PipeBarrier<PIPE_ALL>`
→ MTE3 搬」的交接（例如 `PleGather` 里 M92 遗留的 `G.hmBad` 计数落盘，以及 ③④⑤ 各段的
`G.kv`/`G.gated`/`G.normed`/`G.out` 数据面落盘）。它们的定性是 `docs/20` 的 **B10「未建立 / 待确认」**，
按 **WO-B4**（work order，M111 之后另行派）做全文件排查 —— M111 只收 ⓔ 的**标量条款**点名的三处；
把这批一起扫会超出一个复审轮次的范围，故明确留白、不留"已经合规"的错觉。

---

## 3. 读数（改前 → 改后）

判据分档按 `docs/17 §1.1`；「改前」＝ 库内 main（`d348547`）的二进制，「改后」＝ 本分支的二进制。

> **r2 复跑口径**：落盘通路的核内次序改成 ⓔ 形态（§2.4）之后，本节全部读数**整批重采**了一遍：
> 独立档 14 条 / FAIL 0、变异 15 位 + `coverage 14|14`、M92 真实分片档（`Hd.row_fail=0` n=1022、
> `Hd.miss=2`、`Hd.cores=56/56`）+ bit14 负例（`Hd.row_fail=1022`）、接线 A..F2 七档、`runs=all`。
> 与 r1（`e5971447…`）**逐项相同**：七档的 `Pw.*` 行 `diff` 为空、`runs=all` 的判据行与末行相同
> （只差时序毫秒）。新二进制 `bin sha256 aa3652f1…`，与本目录全部 `.log` 同批。
> 附带一个独立性强的对证：**复审自己按 `CombineStage` 造的合规版探针二进制
> `m15_layer_loop = aa3652f1…` / `m15_ple = 40e66c7d…`，与本 tip 重建的这两个哈希逐字节相同**
> ⇒ r2 的代码路径与复审已实测的那一版是同一份。

### 3.1 独立档：`m15_ple` 的 13 条判据（+ 新增第 14 条）

| 档 | 改前 | 改后 |
|---|---|---|
| 13 条判据 | **13 条全 PASS**，rc=0（`ple/logs/run_all.log` 同族读数） | **13 条全 PASS**，rc=0；新增 `B_dev.fail` 也 PASS ⇒ 本脚本报 **14 条 / FAIL 0** |
| 关键量 | `B2.emb bad=0`、`B3.kv max(d/bound)=0.999`、`B5.out bad=0` | 同上（数值逐条相同；`B2.emb bad=0`、`B3.kv max(d/bound)=0.999`、`B5.out bad=0`） |

\* 该档跑的是「②③④⑤ 内部以 `negMask=0` 复算的 ids」，因此**看不到** ①→② 的交接问题 ——
交接问题只在已接线路径上暴露（§3.5）。

### 3.2 变异矩阵（`m15_ple_mutants.py`，15 位 + M92 2 位）

| 项 | 改前 | 改后 |
|---|---|---|
| 基线 | 13 条 / FAIL 0 | **14 条 / FAIL 0** |
| 位 0..13 | 14/14 `OK`（每条期望的判据都 FAIL） | 14/14 `OK`（FAIL 集合逐位相同） |
| 位 15（新增） | —— | `OK`：`RESULT\|mut\|15\|…\|B_dev.fail\|B2.emb,B_dev.fail\|OK` |
| 覆盖 | `RESULT\|coverage\|13\|13\|(none)\|OK` | `RESULT\|coverage\|14\|14\|(none)\|OK` |
| M92 档 | 基线 10 条 / FAIL 0；位 14 与位 6 `OK` | 基线 10 条 / FAIL 0；位 14（`H2.emb … Hd.row_fail`）与位 6 `OK` |

转录：`m15_layer_loop/ple/logs/mutants.log`（本批）。

### 3.3 「修好的通路上，别的核真的读到了」——两处落盘的前后对比

**（a）① 的 ids：跨核读回一致（这是 U-A 的直接反面）**

`IdsOneToken` 由**核 `t`** 写 token `t` 的 16 个 id；而 `PleGather` 按 **item** 分核
（`idx = t*NG + g` ⇒ 由核 `idx` 处理）。⇒ 除 `idx = t*16 + 0`（核 `t` 读自己写的）以外，
**其余 15 个 item 都是「一个核读另一个核写的内容」**。

| 档 | 改前 | 改后 |
|---|---|---|
| B（`STAGE=31`） | `Pw.emb.T1 bad=15/16`（只有 g=0 对，其余 15 个读到旧值） | **`Pw.emb.T1 bad=0 miss=0/16`** |

⇒ 16/16 个 item 的**行内容**都与 host 独立参考逐字节一致（`Pw.win.dev row_fail=0` 是同一件事的
设备侧读法：设备自己把行首 64 元素与 host 直读分片文件的期望比过，0 条错）。

**（b）①② 的设备侧计数槽：改前只有 1 个核的值读得回来**

`M15_PLE_MUT=1024`（bit10）让**每个** id 都越出词表 ⇒ 每个处理过 token 的核都数出 16
（A 档 10 个核、B 档 2 个核）：

| 档 | 改前读回 | 改后读回 | 每核都落下来的话应为 |
|---|---|---|---|
| A（10 tok） | `dev_range_fail = 16` | **`dev_range_fail = 160`** | 10 × 16 = 160 |
| B（2 tok） | `dev_range_fail = 16` | **`dev_range_fail = 32`** | 2 × 16 = 32 |

同一次运行里，① 的 **ids** 落点（每核独占 128 B 线）改前读回就是 160/160（`A_ids.bin` 里
越界元素 = 160）⇒ 改前的丢失**只**发生在 4 B 相邻槽位上，与「谁来读」是 host 还是别的核无关
（§1.4 机制 (ii)）。**注意**：这条的读回方是 host，它证明的是「槽位真的落下来且读得回」，
**不是**核间可见性 —— 核间可见性由 (a) 的已接线档证明（同一个 kernel、只隔一条屏障）。

### 3.4 M92 设备侧判据（`Hd.row_fail` / `Hd.miss`，真实分片 shard_0 的窗口）

命令见 §4.3。改后读数（同一份二进制；`m15_ple_check.py` 在该档判 **10 条 / FAIL 0**，rc=0）：

| 量 | 读数 |
|---|---|
| 窗口 | shard=0、`rows_per_shard=2500012`、`r0=0`、256 MiB = **838,860 行**；`aclrtHostRegisterV2(MAPPED)` rc=0；`getDevPtr` rc=0 |
| 规模 | n_tok=64、items=1024、**窗口内 item = 1022**（host 独立算出的越窗 = 2） |
| `Hd.cores` | 设备侧自报参与核 = **56 / 56**；设备侧读到的 `winRows` = 838860 |
| `Hd.row_fail` | **0**（设备把窗口内 1022 个 item 的行首 64 元素与 host 直读分片文件的期望逐元素比过） |
| `Hd.miss` | **2**，与 host 独立算出的越窗 item 数 **相等** |
| `Hd.nonvac` / `H2.emb.nonvac` | PASS（窗口内 item 1022 > 0；对照行非零元素 163520/163520） |
| `H2.emb`（T1 逐字节，窗口内） | n=163520、bad=0 |
| `H3.kv` / `H4.gated` / `H4.normed` / `H5.out`（T3） | bad=0（`H3.kv`：`max(d/bound)=0.993`，界尺度 = Σ\|terms\|；`H4.*`/`H5.out` 用 2.0·ulp） |

**负向对照（bit14：`row = winRows-1-(id-winBase)`）** —— 同一份二进制、同一份数据，
`M15_PLE_MUT=16384`：

```
[m15ple] hm dev-side raw: 参与核=56（应 56） winRows(设备侧读到)=838860 hmBad/槽0 合计=1022.0 越窗=2.0
[m15ple] hm ②③④⑤(host-mapped 真实表): n_tok=64 items=1024 in_win=1022 dev_row_fail=1022.0 dev_miss=2
⇒ RESULT|Hd.row_fail|T1-struct|1022|1|FAIL    （Hd.miss / Hd.nonvac / Hd.cores 仍 PASS）
   RESULT|H2.emb|163520|163373|FAIL、H3.kv|806400|804162|FAIL、H4.gated|642546、H4.normed|642175、H5.out|312619
```

⇒ 设备侧计数在**真实规模窗口**上是「真的在数」：把行号换算弄坏 ⇒ 1022/1022 窗口内 item 全部被判错。
（`H4.*`/`H5.out` 的 fail 数与 `ple/REAL_TABLE.md §7` 表里 M92 记的三个数一致 —— 说明本 mission 的
改动**没有动**这条数据面的判读。）

改前该档的基线同样是 10/10 PASS（这一档的 ids 由 host 灌入，不受 U-A 影响）。

### 3.5 已接线路径（`M15_PLE_WIRE=1`，`m15_layer_loop` 的 `plewire` 档）

| 档 | 改前 | 改后 |
|---|---|---|
| **A** `STAGE=15`（②③④⑤；ids 由 host 灌入） | 10/10 PASS，rc=0 | 10/10 PASS，rc=0 |
| **B** `STAGE=31`（**①+②③④⑤**，默认档） | 7/10，**FAIL 3**，rc=1：`Pw.emb.T1 bad=15/16`、`Pw.win.dev row_fail=15`、`Pw.det` | **10/10 PASS，rc=0**：`Pw.emb.T1 bad=0 miss=0/16`、`Pw.win.dev row_fail=0 miss=0 非空洞核=56 见窗核=56` |
| B 的 `Pw.wired.delta` | 6322/10240（② 读到旧 ids 时的产物） | **5636/10240**（= A 档同一个数） |
| **C** 接线关 | 5 FAIL（负向对照） | 5 FAIL（不变） |
| **D** 表基址错位 | 5 FAIL | 5 FAIL（不变） |
| **E** 行基址偏 1 | 2 FAIL（`Pw.emb.T1`、`Pw.win.dev`），`row_fail=16` | 2 FAIL，`row_fail=16`（不变） |
| **F** 真实词表 + `STAGE=31` | 3 FAIL（`miss` 被 U-A 污染） | 5 FAIL、`miss=16/16` ⇒ 与 F2 一致（单窗口装不下真实 token 的 16 个 id，见 §5 的 U-E；这是**已知未做**的多槽滑窗，不是回归） |
| **F2** 真实词表 + `STAGE=15` | 5 FAIL、`miss=16/16` | 5 FAIL、`miss=16/16`（不变） |

### 3.6 零回归：`runs=all`

命令（接线默认关）：
`M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all`

| 二进制 | 日志末行 | rc |
|---|---|---|
| **改后 r1**（`bin sha256 e5971447…`） | `===== ALL PASS（checks=2095, guards=302, fails=0）=====` | 0 |
| **改后 r2**（本 tip，`bin sha256 aa3652f1…`；落盘通路改 ⓔ 形态之后重采） | `===== ALL PASS（checks=2095, guards=302, fails=0）=====` | 0 |
| **改前**（main 源码；`git stash` 回到改前源 → 重建（`bin sha256 1596284e…`）→ 同命令同 env） | `===== ALL PASS（checks=2095, guards=302, fails=0）=====` | 0 |

⇒ 三者的 `判据分账` 行与 `guard` 行**逐字相同**；本改动（r1 + r2）**没有**移动这两个计数。

**与 M100 那批记的 `2068 + 290` 不同 —— 原因是基线不同，不是本改动**：那份 `runs_all.log`
出自提交 `3f62351`，而 `3f62351` **不含** M98 的 attention-KV 判据
（`git merge-base --is-ancestor fc83496 3f62351` 返回假；`fc83496`/`4829927` 只随 main 的合并进来）。
计数差落在 **`Kv 56 → 83`**（+27）与 guards 的 **`其余 130 → 142`**（+12）两组，两组都在
attention-KV 段（本 mission 一字未动）。⇒ **当前 main 上这条命令的基线是 2095 + 302**，
不是 2068 + 290。

### 3.7 r2 后与 r1 的逐行对比（落盘通路改 ⓔ 形态的回归证据）

```
$ for f in A_body_stage15 B_full_stage31 C_wire_off D_miswire_table E_miswire_winbase F_real_vocab_full F2_real_vocab_body; do
    diff <(git show HEAD:<r1 提交>:m15_layer_loop/evidence/ple_wire/$f.log | grep -E "Pw\.") \
         <(grep -E "Pw\." m15_layer_loop/evidence/ple_wire/$f.log) ; done
（七个档都是空输出）
$ diff <(git show HEAD:…/runs_all.log) …/runs_all.log
（差异只在「权重装载/来源校验耗时」「step 时序百分比」这类毫秒行）
```

⇒ 接线档的判定行与 `runs=all` 的判据行**逐行相同**，只差时序毫秒。

---

## 4. 命令 → 输出（本批实跑）

设备段一律在进程锁里：`flock -w 900 /tmp/npu0.lock <命令>`，**进锁后**再 `npu-smi` 复查。

### 4.1 生成物三件套（改前一致 → 改后一致）

```
$ python3.12 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py --check        # 改代码之前（库内 main 状态）
[ok] m15_layer_loop/m15_ple_wire.h == 从 m15_layer_loop/m15_ple.asc 重新抽取的结果（40723 字节，逐字节）
rc=0

$ python3.12 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py --check        # 改了 asc、还没重生成
[FAIL] m15_layer_loop/m15_ple_wire.h 与 `m15_ple.asc` 抽出的片段不一致（漂移）—— 重新生成
rc=1

$ python3.12 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py               # 重生成（r1）
[ok] 写出 m15_layer_loop/m15_ple_wire.h（44047 字节，抽自 m15_ple.asc 的 namespace M85P）

$ python3.12 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py --check
[ok] m15_layer_loop/m15_ple_wire.h == 从 m15_ple.asc 重新抽取的结果（44047 字节，逐字节）
rc=0

# ---- r2（落盘通路改 ⓔ 形态）：改 asc → 重生成 → check；再放回 r1 的头做漂移探针 ----
$ python3.12 …/lift_ple_device_segment.py                                              # 重生成（r2）
[ok] 写出 m15_layer_loop/m15_ple_wire.h（45471 字节，抽自 m15_ple.asc 的 namespace M85P）
$ python3.12 …/lift_ple_device_segment.py --check
[ok] …重新抽取的结果（45471 字节，逐字节）
rc=0
$ git show HEAD:m15_layer_loop/m15_ple_wire.h > m15_layer_loop/m15_ple_wire.h && python3.12 …/lift_ple_device_segment.py --check
[FAIL] m15_layer_loop/m15_ple_wire.h 与 `m15_ple.asc` 抽出的片段不一致（漂移）—— 重新生成
rc=1                       # ← 把 r1 的头放回去，`--check` 立刻红（asc→头 的耦合是活的）
$ cp /tmp/wt111_r2_wire.h m15_layer_loop/m15_ple_wire.h && python3.12 …/lift_ple_device_segment.py --check
[ok] …（45471 字节，逐字节）  rc=0
```

三个时点的字节数：**main = 40723 B / M111 r1 = 44047 B / M111 r2 = 45471 B**；`m15_ple_wire.h` 自始至终**没有手改**。

### 4.2 独立档

```
$ flock -w 900 /tmp/npu0.lock env M15_PLE_OUT=/tmp/wt111_final ./m15_layer_loop/build/m15_ple
[m15ple] M85 PLE 独立 kernel | aic=28 aiv=56 | out=/tmp/wt111_final data=m15_layer_loop/ple/data neg=0
[m15ple] A    ① ids: 10 tok × 16 head, dev_range_fail=0 -> A_ids.bin
[m15ple] B    ① ids: 2 tok × 16 head, dev_range_fail=0 -> B_ids.bin
[m15ple] body ②③④⑤: stage_mask=15 n_tok=2 table_rows=1470 dev_fail=0
[m15ple] ===== kernel-side OK =====
```
（紧随其后的 `m15_ple_check.py` 转录见 §3.1。）

### 4.3 M92 真实分片窗口档

```
$ flock -w 900 /tmp/npu0.lock env M15_PLE_HM=1 M15_PLE_OUT=/tmp/wt111_final_hm ./m15_layer_loop/build/m15_ple
[m15ple] hm: shard=0 rows_per_shard=2500012 r0=0 win=268435456 B (838860 rows) | file mmap->anon staging =
         268435200 B（75.3 ms，host 墙钟单次）| aclrtHostRegisterV2(MAPPED) rc=0（32097.9 ms，host 墙钟单次）|
         getDevPtr rc=0 win=0x7fc594000000 dev=0x40000000000
[m15ple] hm dev-side raw: 参与核=56（应 56） winRows(设备侧读到)=838860 hmBad/槽0 合计=0.0 越窗=2.0
[m15ple] hm ②③④⑤(host-mapped 真实表): n_tok=64 items=1024 in_win=1022 dev_row_fail=0.0 dev_miss=2
[m15ple] ===== kernel-side OK =====
rc=0
（紧随其后的 m15_ple_check.py：判据 10 条 / FAIL 0；Hd.cores 56/56、Hd.row_fail 0（n=1022）、Hd.miss 2）
```

### 4.4 接线档与零回归

```
$ flock -w 900 /tmp/npu0.lock bash m15_layer_loop/evidence/ple_wire/run_wire_evidence.sh
…（A..F2 逐档转录落在 evidence/ple_wire/*.log，每份末尾一行 rc=）…
=== … runs=all（接线未开）===
…（见 evidence/ple_wire/runs_all.log 末行）…
```

### 4.5 M92 的负向对照（bit14）

```
$ flock -w 900 /tmp/npu0.lock env M15_PLE_HM=1 M15_PLE_MUT=16384 M15_PLE_OUT=/tmp/wt111_hm_neg14 ./m15_layer_loop/build/m15_ple
[m15ple] M85 PLE 独立 kernel | aic=28 aiv=56 | out=/tmp/wt111_hm_neg14 data=m15_layer_loop/ple/data_hm neg=16384
[m15ple] hm: shard=0 rows_per_shard=2500012 r0=0 win=268435456 B (838860 rows) | … rc=0 | getDevPtr rc=0
[m15ple] hm dev-side raw: 参与核=56（应 56） winRows(设备侧读到)=838860 hmBad/槽0 合计=1022.0 越窗=2.0
[m15ple] hm ②③④⑤(host-mapped 真实表): n_tok=64 items=1024 in_win=1022 dev_row_fail=1022.0 dev_miss=2
[m15ple] ===== kernel-side OK =====
rc=0
$ python3.12 m15_layer_loop/m15_ple_check.py /tmp/wt111_hm_neg14 --brief
RESULT|Hd.row_fail|T1-struct|1022|1|FAIL     ← 唯一被这个变异咬住的设备侧判据
RESULT|Hd.miss|T1-struct|1024|0|PASS
RESULT|Hd.cores|T1-struct|56|0|PASS
（`H2.emb` / `H3.kv` / `H4.gated` / `H4.normed` / `H5.out` 同时 FAIL）
```

---

## 5. 边界与显式未完成项

- **不做** `PleGemv → mmad`（另一条任务），**不做**性能优化，**不碰** `m15_ple.asc` 以外的段体；
  逐条对应 mission 的「显式不做」。
- **落盘通路的形态范围**：本 mission 只收 `docs/05 §6.1 ⓔ` **标量条款**点名的三处（+ 删掉那条死判据）。
  同一文件里 M92 遗留的 `G.hmBad` 计数落盘与 ③④⑤ 各段的数据面 MTE3 落盘仍是
  「写 UB → `PipeBarrier<PIPE_ALL>` → MTE3 搬」，其定性是 `docs/20` 的 **B10「未建立 / 待确认」**，
  按 **WO-B4** 另行派（§2.4 末）。⇒ 本文件**不声称** `m15_ple.asc` 已整体符合 ⓔ。
- **`m15_layer_kernel.h` 里有两处同族残留**（`M15L_PleIds` 的 `failG.SetValue`、
  `M15L_PleBody` 的死分支 `if (bad > 0u) G.fail.SetValue`）**外加一段已经为假的注释**
  （它仍写着「① 的 ids 是标量写 GM… `Pw.emb.T1` 仍是 `bad=15/16`」）—— 该文件不在本 mission scope，
  已按纪律走 `TowerFinding`（`…bug-m15-layer-kernel-h-gm-ple-u-a-m111-scope.md` 与
  `…improve-m15-layer-kernel-h-m111-gm-pw-emb-t1-bad-15-16.md`），**未改**。注意：层循环的 `PleGather`
  已由本 mission 修好，`S_FAILB` 的读数是真的；但 `M15L_PleIds` 那份 `S_FAILI` 仍是标量写。
- **`docs/20` §3.4 的 `m15_ple.asc` 行**（「没有 acquire 调用 … 不属本族」）在 M111 后失真，
  连带 C15 的 `PipeBarrier<PIPE_ALL>` 计数 —— 走 `TowerFinding`，`docs/` 不在本 scope。
- **单窗口装不下真实 token 的 16 个 id**（`INTERFACE.md §7 U-E` / `REAL_TABLE.md §9 U-A`）：
  本 mission 未做多槽滑窗 ⇒ 接线档的 F / F2 两档仍是红的（它们的红不是 U-A）。
- **`Pw.det` 在 `E_miswire_winbase` 档的 run-to-run 跳变**（M100 r2 的 p2）：本批该档一次观察是 PASS；
  本 mission 没有结论（`E_det_probe_summary.txt` 的结论仍适用）。
- **`PleGather` 的守卫是「词表值域」**：id 在词表内但落在窗口外仍走 `miss`（那是合法读数，不是坏值）。
  设备侧对「行内列 64..159 的错列」仍只由 host 的 `H2.emb`/`B2.emb` 咬（`REAL_TABLE.md §9 U-J`）。
- **跨核那一跳按读数记，不按论证记**：`M15L_PhaseBoundaryAiv` 的 set 挂 `PIPE_MTE3` 是
  「为什么可能成立」的一种解释（`docs/06 §5.2` 第 6 条对同一形态的结论相反）；本 mission 给的是
  实测（B 档 `bad=15/16 → 0/16`）与一条负向对照（main 二进制上仍是 15/16），**不是**一个证明。
