# M95 · MoE top-k 归并树（`#4`）的落地与独立锤点

本文件是 mission **M95**（branch `feat/m95-moe-topk-merge-tree-at-large-expert`）的证据与口径说明。
改动只落在 4 个路径内：`m15_layer_loop/m15_moe_layer.h`（生成物）、`m15_layer_loop/m15_moe_resources.h`、
`m15_layer_loop/lift_moe_segment.py`（规则 9）、`m15_layer_loop/moe_relift/**`（本文件 +
`check_stream.py` + `m95_e512/**`）。M84/M91 的证据（`README.md` / `check_prep.py` / `check_rescale.py` /
`m84_*.log` / `m91_*.log`）原样保留，除 `check_stream.py` 按本节 §4 同步改（#4 落地必须改它）。

---

## 1. 这一步解决的是哪堵墙（改前的事实）

`m15_moe_resources.h` §4 的 router 行距写死 **64 lane**、`SoftmaxTopkRow` 用 **单块** `Sort32(pairT, valT, idxT, 1)`
（32 对）+ `Reduce<MAX>` 覆盖 32 lane ⇒ **top-k 的正确性上界是 `NUM_EXPERTS ≤ 32`**
（M91 已把这条分档写进 `m15_moe_resources.h` 与 `m91_README.md:159`；复审核过）。
`E = 33..64` 时 logits 行装得下但候选集已被截断（结果错）；`E ≥ 65` 连行都放不下。

**M95 的任务就是拆这堵墙**：16 块 `Sort32` + 4 级二路 `MrgSort` 归并树 → 根列表 → `Extract` 前 64 对
→ 前 `TOPK` 对 renorm。

## 2. 改了什么（一句话形态）

| 件 | 改前 | 改后 |
|---|---|---|
| 行距 | 对数行写死 64 lane | `RT_ROWL = 512` lane（= 16 块 × 32 对；`RT_ROWL == RT_SORTLN`） |
| 排序 | `Sort32(..., 1)` | `Sort32(..., RT_SORT_NBLK)` + `MergeTree()`（8/4/2/1 级，`MrgSort` validBit=0b0011、每级 `elementLengths=[32,32,0,0]`） |
| 拆分 | VF 内联 Extract 1 遍（32 对） | 同一 VF 内联重复 `RT_EXTRACT_REP = 2` 遍（64 对） |
| top-k staging 行距 | `r * 64 * 4`（IDS/WS） | `r * RT_WROW * 4`（`RT_WROW = 64`，**不随 E 变**：只放 ≤ 64 个结果） |
| 行内 max/exp | 单寄存器 32 lane 一次 `Reduce` | 按 `RT_NCHUNK_W = 8` 个 64-lane chunk 做 `Max` 树 + 一次 `Reduce` |
| 索引模板 | 铺 0..31 | 铺 0..`RT_ROWL-1`（否则第 2..16 块的 expert id 错） |

形状来源 = `m17_moe_real/m17_moe_layer.asc` 的 `SoftmaxTopkRenormRow` / `MergeTree` / `Merge2`
（E=512/top-10 已验收形态；`m17_moe_real/README.md:33`）。`MrgSort` 的用法照抄 m17/m22（**不用**
`MrgSort4`：它在 dav-3510 上是 deprecated 空函数体）。`Extract` 沿用**本段已有的 VF 内联形态**
（`LoadAlign<DIST_DINTLV_B32>` + 两条掩码 `StoreAlign`，M50 落地），**不引入**经典 memory-based
`Extract`（不在 M44 白名单内）。`Sort32`/`MrgSort` 的裁定例外依据注释随代码落盘（docs/05 §6.1 规则 ⓒ）。

### 为什么行宽写成固定的 16 块 / 512 lane（而不是随 E 缩到 `ceil(E/32)`）

1. m17 的已验收形态就是硬编码 16 块（E=512）；
2. `E=4` 下这棵树是**逐字节退化档**：行内 lane ≥ `NUM_EXPERTS` 全是 `RT_NEG_BIG` ⇒ `exp` 得 0、
   排序/归并落尾部（索引模板更大 ⇒ 并列时排在真实专家之后）⇒ top-`TOPK` 与改前单块 Sort32
   **同集合、同次序**；排序 / 归并 / pair 拆分**无算术** ⇒ 逐字节相同（§3 的 A/B dump 见证）；
3. 固定行宽让 `UB_RT_*` **整体与 E 解耦**：M91-#3 买到的「权重窗与 E 无关」不被 #4 退回去，两档
   UB 读数完全相同（`UB_RT_END = 222752`，E=4 与 E=512 同值；GDN 相位峰值 227328 ⇒ 融合 UB 峰值
   不被抬高，`m15_layer_resources.h` 的 `UB_PEAK_FUSED` 一字未改）。

代价：E=4 缩形档多排 15 次空归并（**性能**代价，不是正确性；人类原话「先把这 1 个层把总体功能
打通了再细扣性能」）。

### correct-by-construction 的口径（**不要读成「全局 top-64」**）

每级归并各取两条输入列表的**前 32 对**、输出 64 对。归纳可证：**根列表的前 32 对 = 全体候选的
top-32（精确）**（证明写在 `m15_moe_layer.h` 的 `SoftmaxTopkRow` 头注），`Extract` 取前 64 对
⊇ top-32 ⇒ top-`TOPK`（判据 `TOPK_MAX ≤ 32`）精确。
**但根的 64 对不是全局 top-64**：第 2 级起每路只取前 32 对，名次 33..64 不再上行。m17 的注释
「各级 top-64」指的是「该级输出的 64 对」。本 mission 的验收声明**只到 top-32**（mission 文本里
「→ 全局 top-64」这一句与 m17 的实现形态不符，这里如实标出）。

## 3. 验收 ①：E=4 零回归（两个二进制各自的读数）

| 档 | 改前二进制（`2d42ba86…`） | 改后二进制（`958c6f2a…`） |
|---|---|---|
| `runs=all`（**纯跑，无 dump**） | `checks=2068, guards=290, fails=0` | `checks=2068, guards=290, fails=0`（r2 复跑） |
| `runs=all`（**`M15_DUMP=1`**） | `checks=2068, guards=291, fails=0` | `checks=2068, guards=291, fails=0`（r2 复跑） |
| `runs=chain`（纯跑） | `checks=910, guards=52, fails=0` | `checks=910, guards=52, fails=0`（r2 复跑） |

**口径（r1 复审 P2-2）**：同一个二进制，`runs=all` 的 guard 条数取决于**是否带 dump** ——
**纯跑 = `guards=290`**（与 mission 文本、以及仓内既有日志 `moe_relift/m91_accept_run_all.log:2327`
一致）；**`M15_DUMP=1` 时 = `guards=291`**（多一条条件 guard `H_Guard(C, nDump > 0)`，
`m15_chain_host.h:840`）。两者都不是笔误 ⇒ 上表按跑法分列。

改前二进制在本 worktree 上当场重建并跑（基线读数）、改后同款：日志 `/tmp/m95_base_*dup.log`（r1）、
`/tmp/m95_r2_all_pure.log`、`/tmp/m95_r2_chain_pure.log`、`/tmp/m95_r2_all_dump.log`（r2 复跑）
（**未入库**：`moe_relift/` 里已有 M91 的同类日志；本节给出的是实跑读数）。

**A/B dump 逐字节对拍**：两个二进制各 `M15_DUMP=1 M15_STEPS=3 … all` 一次，各落 **1307** 个文件，
`sha256sum * | sort -k2` 清单 `diff` **rc=0**（r1：`/tmp/m95_dump_base` vs `/tmp/m95_dump_new`；
**r2 在新 tip 上复跑**：dump 再次 1307 文件、与基线清单 `diff` rc=0，见 `moe_relift/m95_r2_fixes.log`）。
⇒ **E=4 下 #4 退化为 no-op** 这条由逐字节相同见证（这正是 mission 允许把 A/B 当证据的唯一前提）。

> ⚠ **这条是回归证据，不是新实现正确的证据**。M91 的 r1 复审实测过：把归并树末级换序的变异
> **仍能过全部 2068 判定项 + 291 guard 与三条静态判据**。⇒ 见 §5 的独立锤点。

## 4. 验收 ③④：静态判据（两档 locale）

| 判据 | `LC_ALL=C` | `LC_ALL=C.UTF-8` |
|---|---|---|
| `check_prep.py` | 16 条 0 FAIL | 16 条 0 FAIL |
| `check_rescale.py` | 30 条 0 FAIL | 30 条 0 FAIL |
| `check_stream.py` | **22 条 0 FAIL** | **22 条 0 FAIL** |
| `lift_moe_segment.py --check` | rc=0 | rc=0 |

`check_stream.py` 的同步改（复审判据早已预告「落地 #4 时 A10/A2 必须同步改，否则合法落地会报
FAIL」）：

- **A2**：去掉「落地 #4 时本判据与 A10 都须同步改」这句提示（已兑现），判据本体不变（旧形态串仍
  必须不出现）；
- **A10 语义反转**：M91 时它钉「#4 未落地（`Sort32 == 1 && MrgSort == 0`）」，现在钉**已落地**：
  `Sort32` 块数实参 = `RT_SORT_NBLK`、`MrgSort` 恰 1 处、`MergeTree` **定义 1 + 调用 1**、
  `Merge2` **定义 1 + 4 调用点**、经典 `Extract` 0 处、`MrgSort4` 0 处；
- **新增 A11**：行距重排的机器可核形态（对数行 4 处 `RT_ROWL`、0 处写死 64；IDS/WS 4 处 `RT_WROW`、
  0 处写死 64）；
- **新增 A12**：索引模板按 `RT_NCHUNK_W` 铺满行宽（1 处）+ `Extract` 用 `RT_EXTRACT_REP` 重复（1 处）；
- **新增 A13**：归并树 4 级结构齐备（8 / 4 / 2 / 1 各 1 处）；
- **新增 B5/B6（两档各一条）**：`RT_ROWL == RT_SORTLN == RT_SORT_NBLK × 32 ≥ NUM_EXPERTS`
  （**覆盖性**判据：候选集不得被截断）+ 该档 `UB_RT_END ≤ UB_RC_END`。

B1（每一个 `UB_RT_*` 都与 E 无关）**未改也仍然通过**：固定 16 块后行宽读数不随档漂移，两档
`UB_RT_*` 逐项相同。

**r2 修（r1 复审 P2-3：「注释绕过」洞的深一层）**：A 组判据原先只有部分是判「剥注释后的代码视图」，
其余（A11/A12/A10 的定义子计数、A1/A3..A9）判的是**含注释的原文** ⇒ **把实现弄坏、再用一条注释
把被匹配的文本补回去**，正向计数就恢复了（r1 复审实测三例）。r2 的两处改动：

1. `check_stream.py`：`A1..A13` **全部**判据改判 `strip_comments`（剥注释后的代码视图；
   `m95_negctl.sh` 的 `nc9a/nc9b/nc9c` 三条注释绕过对照现在都按**正确理由**变红 —— 代码计数真的为 0）；
2. `check_prep.py` 的 `strip_comments`：**顺序改成「先按行去 `//`，再去 `/*...*/`」**。旧顺序下，
   生成物里那行行注释中的 `reg_compute/**` 会被当成块注释起点，此后文件里只要再出现一个 `*/`
   （例如有人追加一条块注释），正则就从那一行**一路吞到那个 `*/`** ⇒ `collect()` 缺符号、`evaluate()`
   直接 `KeyError`（`m95_negctl.sh` 的 `nc10_benign` 就是这个崩溃的可复现证据）。

**r3 修（r2 复审 P2-1：换序引入了**反方向**的洞）**：两趟换序后，若块注释的 `*/` 与 `//` **同行**
（`close // */`），行注释那一趟会把 `*/` 一起吃掉 ⇒ 留下**未闭合的 `/*`**，其后若无 `*/`，块注释
那一趟**什么都不删** ⇒ **注释文本残留进代码视图**，于是「破坏实现 + 追加一条注释」又能 PASS。
r3 把 `check_prep.strip_comments` 换成**单趟字符扫描器**（一次遍历按**最先出现的**开符号处理
`/*` 与 `//`，并跳过字符串/字符字面量、保留换行），两个方向的洞同时消失；另加**后置自检**：剥完
若仍有未闭合的 `/*` / 字面量，或代码视图里残留 `/*` / `*/` ⇒ **直接 `raise`**（拒绝给假绿）。
常备对照补了两条**反向**用例 `nc11_survivor` / `nc11b_survivor`（在 r2 的两趟扫描器下 rc=0 假绿，
单趟扫描器下 rc=1）；另用 **GNU `cpp -fpreprocessed -dD -P`** 作独立工具逐字符比对两个头文件的
剥注释结果（非空白字符序列完全相同）。

常备对照脚本 **`moe_relift/m95_negctl.sh`**（**13 条 = 12 条破坏对照 + 1 条健全性正对照**；全部只落
`/tmp` 副本；`rc=0` 表示「12 条破坏对照全红 + 正对照绿」。**计数以脚本自报为准**：脚本结尾打印
`# 对照总数 = N（破坏 M 条，健全性正对照 K 条）`，另可 `grep -c '^run ' m95_negctl.sh` 复核；
各轮条数的变化记在 `m95_r2_fixes.log` / `m95_r3_fixes.log`）：

| 对照 | 注入（1 处） | 应红的是 | r3 实测 |
|---|---|---|---|
| `nc_row` | `RT_ROWL = RT_SORTLN` → `64` | B5（两档） | FAIL ×2 |
| `nc_b6` | `RT_WROW` `64` → `4096` | B2/B3/B6 | FAIL ×4 |
| `nc_tree` | 删 `MergeTree();` 调用 | A10 | FAIL |
| `nc_a11` | 1 处对数行取址 → `r * 512 * 4` | A11 | FAIL |
| `nc_a12` | chunk 上界 `RT_NCHUNK_W` → `1` | A12 | FAIL |
| `nc_a13` | level2 循环上界 `4` → `0` | A13 | FAIL |
| `nc_a2` | 把 `PrecastWeights();` 写回代码（旧形态回归） | A2 | FAIL（`仍出现的串 = ['PrecastWeights']`） |
| **`nc9a_com`** | `nc_a12` + 追加**块注释**补回循环文本 | A12 | FAIL（`模板 chunk 循环 = 0`，按**正确理由**红） |
| **`nc9b_com`** | `nc_a11` + 追加**行注释**补回 `UB_RT_LOG + r * RT_ROWL * 4` | A11 | FAIL（`RT_ROWL = 3`）—— **修复前这条是 rc=0（假绿）** |
| **`nc9c_com`** | 删 `MergeTree()` 真定义、签名留在**块注释**里 | A10 | FAIL（`MergeTree 定义/调用 = 0/1`）—— 修复前注释会被计入 |
| **`nc11_survivor`** | `nc_a11` + 追加**块注释**（末尾 `close // */` ⇒ `*/` 与 `//` 同行） | A11 | FAIL（`RT_ROWL = 3`）—— **r2 的两趟扫描器下这条是 rc=0（假绿）**，单趟扫描器修复 |
| **`nc11b_survivor`** | 删 `MergeTree()` 真定义 + 同类 `close // */` 注释 | A10 | FAIL（`MergeTree 定义/调用 = 0/1`）—— 同上，r2 下假绿 |
| `nc10_benign` | **只**追加一条无害块注释（含被匹配文本） | 读数不许变 | rc=**0**（正对照）—— 修复前这条会崩（`KeyError`），即「无害注释把所有 B 栏判据打崩」 |

（上表读数取自 `moe_relift/m95_r2_fixes.log`；该文件同时留了「修复前 checker」在同一套对照上的
逐条读数，其中 `nc9b_com rc=0`、`nc10_benign rc=1` 两条是 r1 复审与本轮发现的假绿/假红的复现。）

## 5. 验收 ①②（本 mission 的关键）：E=512 的**独立数值锤点**

**为什么必须有它**：本 mission 改的是**实现**，「与改前逐字节相同」这条证据在 E>32 下不成立
（改前在 E>32 下算错）。若只交 dump 对拍，交付物就**没有任何独立证据证明它是对的**。

**锤点 = `moe_relift/m95_e512/`**（新目录，独立 CMake 工程）：

- **被测代码**：仓库的 `m15_moe_layer.h` **逐字节副本**（构建期 `file(COPY)` + sha256 对照，
  见 `build/m95_tier.txt`）+ `m15_moe_resources.h` 的 **E=512/top-10 档**（构建期只按字面替换
  `NUM_EXPERTS = 4` / `TOPK_MAX = 4` 两个字面值，副本与源的 `diff` 只有这两行）；
- **输入**：`--w <文件>`（真实 checkpoint `layers.0.mlp.gate.weight` bf16[512,2560] 的 2.5MB 切片
  = `m22_router512/data/router_weight.bin`，**只读引用**）或 `--w onehot`（合成/缩形权重）；
  x 由确定性 Hash3 生成并落盘 `x.bin`（**硬闸门 X1**：必须与 `--x-seed` 重建的逐字节相同）；
- **参考**：`tools/golden/moe_block_ref.py::router_topk`（M4/M26 独立编写、**已建模设备 FTZ** 的
  CPU 银点）—— 不是 dump 对拍，也不依赖 m17/m22 的设备结果。

**按 `docs/17 §1.3` 重列「入参来源三分 / 反自指」（r1 复审 P2-4 修）**：§1.3 规定的三类是
**①声明输入 / ②上游输出 / ③被判量自身的产物**，逐条落到本锤点：

| 类 | 本锤点里的对应物 | 用途 / 标注 |
|---|---|---|
| **① 声明输入** | `x.bin`（确定性 Hash3 生成，硬闸门 X1 要求与 `--x-seed` 重建的逐字节相同）+ `w.bin`（`--w <文件>` 的真实切片，X2 要求与源文件逐字节相同；或 `--w onehot` 的合成构造） | **J1（ids 逐元素相等）吃 ①** ⇒ 它是**独立性判据**（判据的输入与被判量无因果关系之外的联系） |
| **② 上游输出** | 银点自己算出的 `log_ref` / `ids_ref` / `w_ref`（`router_topk` 的输出，与设备无关） | J3 = 设备 logits vs `log_ref`；`Extract`/归并的**正确性**由 J1 落在 ① 上 |
| **③ 被判量自身的产物** | **设备的 `router_logits.bin`** —— J2 用它按银点规则复算权重期望值 | ⇒ **J2 是「非独立性判据」**：它只在 J3 已把设备 logits 钉在银点上的前提下，才能把「权重分母/renorm」与「GEMV 链」分开归因；**单独看 J2 不能证明 logits 对** |

（`m95_e512.asc` 头注里的另一套「被测代码 / 输入 / 参考」分法是工程口径的说明，不是 §1.3 的分类；
按 §1.3 的分类以上表为准。）

**读数**（完整 stdout 见 `m95_e512/run_commands.log`）：

| 档 | J1 ids | J2 weights | J3 logits | rc |
|---|---|---|---|---|
| 真实权重 m=33 seed=7 | PASS（330 项 0 差异） | PASS（max abs 1.19e-7，max 3 ulp） | PASS（max abs 4.58e-5，界占用 0.03%） | 0 |
| 合成 one-hot m=33 seed=11 | PASS（330 项 0 差异） | PASS（max 2 ulp） | PASS（max abs 0） | 0 |

（J1 = ids 逐元素相等，是归并树的**主判据**：级序错 / 块数错 / Extract 起点错都会改 id。
J2 在**设备 logits** 上按银点规则复算权重 ⇒ 隔离掉 GEMV 链误差，只剩 Exp/Div/求和序的舍入。）

## 6. 验收 ⑤：有判别力的负向对照（变异只在 `/tmp` 副本）

按塔的**第五变体纪律**（「把**被测对象**弄坏，判据必须变红」）做：每一条对照注入的都是**被测对象**
（设备实现 / 生成物 / 资源表），不是参考侧；且判据被判的那份量确实由注入处产生（数据流：
`SoftmaxTopkRow`/`MergeTree` → `ids/weights/logits` GM → `check_ref`；`RT_ROWL/RT_WROW` →
UB 取址 → `check_stream` 的算术）。

| 对照 | 注入（都在 `/tmp` 副本，1 处） | 应红的是 | 实测 |
|---|---|---|---|
| `NC1` 设备 | `Extract` 读根缓冲 `UB_RT_MB` → `UB_RT_MA`（归并级序错） | J1/J2 | J1 **FAIL**（330 项），J2 **FAIL**，判定 1/3；rc=1。guard G1/G2 仍 ok ⇒ 只有数值判据能咬住形态正确的错实现 |
| `NC2` 设备 | `Sort32(..., RT_SORT_NBLK=16)` → `Sort32(..., 8)`（一半输入未排序） | J1 | J1 **FAIL**（286/330 项不同），rc=1 |
| 静态 12 条破坏 + 1 条正对照 | 见 §4 的表（`m95_negctl.sh` 常备脚本，一条命令可重跑） | A2/A10/A11/A12/A13/B2/B3/B5/B6 | 12 条破坏全红（rc=1）、`nc10_benign` 绿（rc=0）、BAD 0 条 |

**两个「打标即豁免」洞都是本轮自己踩到并修好的**（第五/第四变体纪律要求的「最小绕过对照」）：

1. **A10 只数定义不数调用** —— 删掉 `MergeTree()` 调用仍 PASS。修法 = 把**调用**与 4 级结构
   （8/4/2/1 各 1 处，新增 A13）一起计入判据；修后 `nc_tree` 立刻 FAIL。
2. **注释可以把坏代码「打回绿」**（r1 复审实测出深一层）—— A11/A12/A10 定义子计数等原先判的是
   **含注释的原文**。修法 = A 组判据全部改判剥注释后的代码视图，并修 `strip_comments` 的剥离顺序
   （见 §4 的 r2 段）。`nc9a/nc9b/nc9c` 三条注释绕过对照**现在按正确理由变红**（代码计数真的为 0）。

> **未做对照的地方如实标出**：E=4 的 2068/910 条判定项是 M84/M91 的既有链，本 mission 只用它做
> **两个二进制的读数对照**（§3），**没有**为它们逐条做「弄坏被测对象」对照 —— §3 的 A/B dump
> 逐字节相同是这条回归声明的全部证据，不是新实现正确的证据（后者见 §5）。

## 7. 复现（全部命令都在 tip 上实跑过；设备 = NPU 0）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <worktree>
# 静态
LC_ALL=C python3 m15_layer_loop/lift_moe_segment.py --check
LC_ALL=C python3 m15_layer_loop/moe_relift/check_prep.py
LC_ALL=C python3 m15_layer_loop/moe_relift/check_rescale.py
LC_ALL=C python3 m15_layer_loop/moe_relift/check_stream.py        # 22 条 0 FAIL
# E=4 设备验收（改后二进制）
cmake --build m15_layer_loop/build -j4 --target m15_layer_loop
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all     # 2068+290/0（纯跑，无 dump）
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt chain   # 910+52/0（纯跑）
M15_DUMP=1 M15_STEPS=3 ./m15_layer_loop/build/m15_layer_loop \
    m15_layer_loop/weights_manifest.txt all                                       # 2068+291/0（带 dump：多一条条件 guard）
# 静态判据的常备负向对照（12 条破坏对照必须全红 + 1 条健全性正对照必须绿；只落 /tmp 副本）
bash m15_layer_loop/moe_relift/m95_negctl.sh
# E=512 独立锤点
cmake -B m15_layer_loop/moe_relift/m95_e512/build -S m15_layer_loop/moe_relift/m95_e512 -DCMAKE_BUILD_TYPE=Release
cmake --build m15_layer_loop/moe_relift/m95_e512/build -j4
M95_OUT=/tmp/m95_e512_real ./m15_layer_loop/moe_relift/m95_e512/build/m95_e512 \
    --m 33 --seed 7 --w m22_router512/data/router_weight.bin
/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/moe_relift/m95_e512/check_ref.py \
    /tmp/m95_e512_real --w m22_router512/data/router_weight.bin --x-seed 7   # RESULT: OK
```

**并发背景如实记录**：跑设备时 NPU 0 上有**别的 mission 的进程**
（`probe_mask_lane`（M94）；某一轮还见到另一个 `m15_layer_loop` 进程）。本 mission 的 kernel
是单 AIV 块、2ms 级，未做独占；`npu-smi info` 原样收在 `run_commands.log` §7。

## 8. 显式未完成项（本 mission **没有**交付的东西）

1. **`#1/#2` 规模常量与权重装载（归 MoE-A）**：仓库档仍是 `NUM_EXPERTS = 4` / `TOPK_MAX = 4`。
   `NUM_EXPERTS 4→512`、`TOPK_MAX 4→10`、`MOE_W_STRIDE` / host arena（`m15_layer_resources.h`、
   `m15_layer_loop.asc`）、`slice_layer_manifest.py` / `weights_manifest.txt` 的 manifest 切片
   —— 全部未动，依赖 M88（attn 段）与 M93（manifest/tools）。本 mission 只在**测试档**里把
   E/topk 换到 512/10（`moe_relift/m95_e512/` 的构建期替换），**没有**让融合 kernel 在 E=512 下
   可运行（那需要上面那批 + UB 峰值重算）。
2. **E=512 的整层/端到端验收**：只做了 router 段（人类原话「只测一层，不需要把所有权重都加载了」
   的形态）。整层 E=512 的对拍仍依赖 MoE-A 落地后的 m17/m21 参考链。
3. **性能**：E=4 缩形档现在要跑 16 块 + 15 次归并（改前 1 块、0 归并）。本 mission 按人类口径
   「先打通功能」不优化（例如按 `ceil(E/32)` 缩块数、或在 `E ≤ 32` 时短路归并树）。
4. **`m17_moe_real` 侧未做任何改动**（只读引用）；本 mission 没有用它当锤点，理由见
   `m95_e512/m95_e512.asc` 头注（它需要 1.24 GiB checkpoint、且 router 与本段不同源）。
5. **`sgate`（共享门）不在锤点判据覆盖内**：`m95_e512` 只跑 router + 共享门裸点积，共享门权重
   传 0，判据不覆盖它（S2 的 sgate 语义未变，仍由 E=4 的验收链覆盖）。
6. **`m17_moe_real` 的 `RT_IDS` 行距疑点（M94 在裁）**：本段改前那处 `CreateMask<int32_t, ALL>()`
   配 64-lane 行距的写法**已随本次重写消失**（新代码用**显式 64 lane** `UpdateMask<int32_t>(RT_WROW)`，
   恰一行、不跨行）；本 mission 没有裁决 ALL mask 的宽度，也没有改 `m17_moe_real`。

---

## M194 漂移标注（2026-10-05）：本目录日志里的 `m15_moe_layer.h` / `m15_moe_resources.h` 指纹已陈旧

本节由 mission **M194**（agent-stalepin）追加。落点选在本 README，**两份日志正文一字未动** ——
`m95_r3_fixes.log` 与 `m95_e512/run_commands.log` 的内容与 main 逐位相同，可核：
`git diff main...HEAD -- m15_layer_loop/moe_relift/m95_r3_fixes.log m15_layer_loop/moe_relift/m95_e512/run_commands.log` 为空。
落点安全的依据：全仓 7 条会写 `m15_layer_loop/*.h` 的生成器里，与本目录相关的只有 `m15_layer_loop/lift_moe_segment.py`，
其写目标是自身目录下的 `m15_moe_layer.h`（`DST`），**不写 `moe_relift/**`**；其余 6 条经逐脚本核对也都没有指向本目录的写路径，
故本 README 不会被任何生成器重写。

下列记录值是**写日志当时的快照**，一律保留原值，只在此标注其所属 rev、失效 rev 与当前值。

**`m95_r3_fixes.log:65`**（pin 目标 `m15_layer_loop/m15_moe_layer.h`）
- 记录值（M95 #4 时）：
  `0af84032dc194e8a0350caf822e91ae005833dae8df52b0ce98ea65c577b2a64`
- 该值最后所属 rev：M95 #4 `9935c77`（合并 `40dd6dedfa`，2026-09-27）
- 失效 rev：M105 `186d97c`（2026-09-27）
- 当前实际值（M181 r4 `a0996eb` 起）：
  `36d00d187a6c6e500cb269b7275e1e84bc4cd08e82ec9c569f947cb1d1f02c62`

**`m95_e512/run_commands.log:10`、`:11`**（pin 目标 `m15_layer_loop/m15_moe_layer.h` 及其被测副本）
- `:10` 是源，`:11` 是 `m95_e512/build/` 下的被测副本（build 产物，未入库，被 `.gitignore` 忽略）。
- 记录值同为（源与副本当时逐字节相同）：
  `0af84032dc194e8a0350caf822e91ae005833dae8df52b0ce98ea65c577b2a64`
- 源侧最后所属 rev：M95 #4 `9935c77`；失效 rev：M105 `186d97c`；当前实际值同 `m15_moe_layer.h`（见上）。
- 副本侧无法离线复算（build 目录被忽略，不入库）。

**`m95_e512/run_commands.log:12`、`:13`**（pin 目标 `m15_layer_loop/m15_moe_resources.h` 及其档位改写副本）
- `:12` 是源，`:13` 是 `m95_e512/build/` 下的副本（只改 `NUM_EXPERTS`/`TOPK_MAX` 两处，build 产物，未入库）。
- `:12` 记录值（M95 #4 时）：
  `971dd3b02342fcdcbf47b4815bd1e14ec6bcfe982c2b5ac0b7629a8cd5b41d5b`
- 该值最后所属 rev：M95 #4 `9935c77`；失效 rev：M95 r2 `8337542`；当前实际值：
  `ba39888b4ca4b1e75f622a91c816f63616e9e25c666002b0072f8dc1a648dbf7`
- 说明：`ba39888b` 自 M95 合并 `40dd6dedfa` 起在 main 上；`971dd3b0` 只存在于 M95 分支，从未出现在 main。
- `:13` 副本记录值：
  `02553fbe3f9a738046473c001e75ec9749c8114b31a0d599be85210eae8b7605`
  无法离线复算（build 目录被忽略）。

两点更正与说明：
- `m15_moe_layer.h` 的陈旧起点是 **M105（`186d97c`）**，而不是 M181 r4：`a0996eb` 是**当前值**的引入 commit，
  不是记录值失效的 commit（本表以逐 rev 的 `git show <rev>:<path> | sha256sum` 为准）。
- 一键离线复算：`python3 m15_layer_loop/moe_relift/m194_check_source_pins.py`
