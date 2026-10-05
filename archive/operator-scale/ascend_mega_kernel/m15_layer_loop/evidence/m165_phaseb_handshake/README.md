# M164 —— 相位 B 的 H2→B AIV→AIC 跨核握手（M152 机理 M1 的修法与设备证据）

> 分支 `feat/m165-phase-b-h2-to-b-aiv-aic-handshake`，工作树 `wt-164`，基线 `0cd7fe5`（main）。
> **结论（窄）**：M152 的主候选机理 **M1 被证实，并已修好**。在 H2 → 相位 B 的边界补一条
> 「**AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）**」的握手后，`m1_p12` / `m1_p15` 的
> **clean（arm0）** 由 FAIL 转 PASS，logits 量级由 ~1e8（`0xCDCD` 毒值量级）回到 O(1)，
> 且**跨 3 次进程重复跑逐字节相同**（修前同配置 3 次是 6.47e8 / 6.79e8 / 1.33e9 的抖动）。
> 「修前 arm0 红、修后 arm0 绿」是**单变量**对照：两个二进制只差本 mission 新增的 62 行
> （`handshake.patch`），其余源码一字未动。
>
> **不预判因果**：本档的基线 `0cd7fe5` 上，M163 的 epilog 挂载（`m164-epilog-mount-into-prefill-path`）
> **未合入**；epilog 合入后 H2 的数值输入会变（`hcAttnOut` 从宿主合成面变真产出）⇒ 本节读数需在
> 合入后的 tip 上重跑一次（复算命令见 §7）。本 mission 只主张"握手补上后，相位 B 的 AIC router 不再
> 读到未写完的 `pfHcBlk1`"，不主张 H2 的数值来源已经齐备。

## 1. 机理链（回源，带 文件:行）

行号以本分支 tip 为准（同 tip 的 commit 见 §8 的 `binary_sha256.txt`）；括号里给内容锚点，便于行号漂移后重定位。

| # | 环节 | 位置 | 事实 |
| - | ---- | ---- | ---- |
| 1 | 相位边界只同步 AIV | `m15_layer_kernel.h:925-927`（锚点 `M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>`，注释「边界：H2 → 相位 B」） | H2→B 的边界调用包在 `if ASCEND_IS_AIV` 内 |
| 2 | 该边界是 AIV-only mode-0 | `m15_layer_kernel.h:415-420`（锚点 `inline void M15L_PhaseBoundaryAiv`） | set+wait 同号、`CC_MODE0`，参与方只有 AIV；AIC 不进入 |
| 3 | 相位 B 的层输入 = H2 产出 | `m15_layer_kernel.h:813`（锚点 `moeIn = (h2On && A.pfHcBlk1 != nullptr)`） | H2 开时 `moeIn` = `A.pfHcBlk1` |
| 4 | router 消费时机 | `m15_moe_prefill.h:1273` `ProcessAic` 开头；`m15_moe_prefill.h:1288` `router.RunTile(nb)`；初始化 `m15_moe_prefill.h:1198` `router.Init(a.xLayer, …)` | AIC 一进相位 B 就在 S2 用 `xLayer`（即 `pfHcBlk1`）做 `Mmad`；此前无任何跨核 wait |
| 5 | `pfHcBlk1` 的生产者（AIV） | `m15_hc_prefill.h:441` `RelocatePass`；`:384` `RelocateTile`；`:400` `DmaBytes(p.blk + …)` | 由**全体 AIV**在 `HcPF::Body` 的 `if ASCEND_IS_AIV` 里以 MTE2→MTE3 写 GM；写的是**行条带**（`for j = bid; j < nPair; j += nAiv`） |
| 6 | 依赖面 | 上面 4+5 | 每个 AIC 读**整块** x（全部 m 行），行由**全体 AIV**分写 ⇒ **all-AIV → 每个 AIC**，是跨类型 **all-to-all** |

`AIC 侧不需要相位边界同步` 这条老口径（`m15_layer_kernel.h:58-59`）只对**段内自带**的 mode-2 交接成立；
MoE 段自带的 AIV→AIC 交接从 S5 起，而 **router 的输入是段输入**，不经过它 —— 这就是 M1。

## 2. 修法

新增 `M15L_H2ToBHandshake<M2_FLAG, AIC_BAR_FLAG>()`（`m15_layer_kernel.h:406-419` 定义、
`m15_layer_kernel.h:821` 在 `M15L_PrefillPhaseB` 进块循环**之前**调用一次）：

```
AIV: CrossCoreSetFlag <mode 2, PIPE_MTE3>(6)                       // MTE3 drain 后发信号
AIC: CrossCoreWaitFlag<mode 2, PIPE_S >(6)   // 每个 AIC 等它配对的 2 个 AIV：2 set ↔ 1 wait
     CrossCoreSetFlag <mode 0, PIPE_MTE2>(7)   // 全体 AIC barrier：保证所有 AIC 都收到了各自配对的信号
     CrossCoreWaitFlag<mode 0, PIPE_S >(7)
```

**mode 选择的论证**（按人类口径「mode 2 = 单个 AIC 与它那 2 个 AIV；all-to-all 用 mode 0」，
以 `docs/05 §2` 逐字为准）：

- 依赖面是 **all-AIV → 每个 AIC**（§1 第 6 条）⇒ 不是"单个 AIC 与它配对 2 个 AIV"的点对点 ⇒
  不能只用 mode 2。`docs/05 §2` 逐字给的标准组合是「**AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）**」；
  mode 0 **仅同类型**，跨类型 all-to-all 必须靠这个组合。
- **配对数**：mode 2 段严格 `2 AIV set ↔ 1 AIC wait`（`m15_moe_prefill.h:1293` 的 AIC set→AIV wait
  是反向同款）。既有正确形态对照：`m15_attn_prolog_probe.h` 的 `AP_AIC_M0_OUT`（AIC 全体 mode-0）
  → `AP_A2V_GEMM`（mode2），登记见 `m15_layer_resources.h:669-673`（M124 PLE ③ 的同款组合）。
- **位置**：在块循环外一次。整个 `pfHcBlk1` 已在 `M15L_HcPrefillBoundary(1u, A)` 里写全，相位边界
  `FLAG_HC2_BOUND_AIV`（set 挂 MTE3）已把本核写排空，故 AIV 的 set 一定排在写之后。
- 只用 `CrossCoreSet/WaitFlag`（"set cross core" 系列），**不使用 set_flag/wait_flag 系列**；
  核内生命周期不涉及（本函数不含 BufferID）。

**flagId**：新用 **6（mode 2）/ 7（AIC mode0）**。`m15_layer_resources.h` 不在本 mission 的 scope，
故这两个号作为本头局部常量定义（`m15_layer_kernel.h:397-398`）+ 相邻性 `static_assert`（`:399-404`），
**权威登记作为 follow-up 报塔**（塔已收到 `flagid-request`）。相邻性：AIV mode2 上 H2 的 10/11
（`m15_hc_layer.h:506-507`）→ **6** → MoE 的 0（`m15_moe_prefill.h:1227`）；AIC mode0 上 H2 的 0/1/2/3
（`m15_hc_layer.h:530-547`）→ **7** → MoE 的 8（`m15_moe_prefill.h:1291`）。空闲性：prefill 入口
（`KIND_GDN`，H1/A/H2/B 全开）里 AIV 占 `{0-5,8-15}`、AIC 占 `{0-5,8-11}` ⇒ 两核同时空闲的只有 `{6,7}`
（6/7 现仅被 decode GDN 用，`m15_gdn_resources.h:156-157`）。

## 3. 设备档与判据

档 = M140/M152 的同一套相位掩码（位 = H1:1 / A:2 / H2:4 / B:8）：`p8`（只 B）、`p12`（H2+B）、
`p15`（全四相位）；m=1 与 m=4097。每档各自进一次 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`
（快照落在该档日志里）、`timeout` 在锁内、一次进锁一条命令。

**判据** = `m26_moe_prefill/check_ref.py <dumpdir>/m26_Pf.gdn`（独立 numpy 参考；本 mission 不复用
in-kernel 的自判，而是像 M140 一样在 dump 上跑 donor 判据）。三档主判（m=1）：

| 档 | 修前（无握手，`/tmp/m15_prefix`） | 修后（带握手，`/tmp/m15_postfix`） |
| -- | -- | -- |
| `m1_p8`（只 B，无 AIV 生产者） | clean **PASS** 6/6 | clean **PASS** 6/6 |
| `m1_p12`（H2+B） | clean **FAIL** 0/4（`maxAbs=6.4704e8`） | clean **PASS** 5/5（`maxAbs=2.9017e0`） |
| `m1_p15`（全相位） | clean **FAIL** 0/5（`maxAbs=6.8160e8`） | clean **PASS** 6/6（`maxAbs=2.0385e0`） |
| `m4097_p15` | clean PASS 6/6 | clean PASS 6/6 |

逐档日志（含锁内 `npu-smi`、完整输出、`exit=`）：`logs/prefix_*.log`、`logs/post_*.log`。
全量判据输出：`logs/judge_all.txt`（144 行）。`m1_p8` 在修前修后**逐字节相同**（`maxAbs=4.1255e0`，
sha256 前缀 `5ca3b41e…`）⇒ 无 AIV 生产者的档不受影响，修法只动了"H2 先跑过"的那两档。

## 4. 稳定性（主判据的收口）

`logs/stability.txt`（clean arm `m26_logits.bin`）：

| 组 | 读数 |
| -- | ---- |
| 修前 p12 ×3 | `6.4704e8` / `6.7885e8` / `1.3283e9`；`max|diff|` 5.49e8 / 1.15e9（**同配置重复跑就不稳**） |
| 修后 p12 ×3 | `2.9017e0` ×3，sha256 前缀 `1d8238b90fab482f` ×3，`max|diff|=0` |
| 修后 p15 ×3 | `2.0385e0` ×3，sha256 前缀 `30aa62314522971d` ×3，`max|diff|=0` |

⇒ 修后不再有 2.58e8–4.31e8 量级的 run 间抖动；这正是 M152 记的"p12 同配置不可复现"被消掉。

## 5. 形状对照（为什么必须带 mode 0 barrier）

只留 mode 2、**去掉全体 AIC mode-0 barrier** 的变体（构建后源码即还原，`binary_sha256.txt` 给
该变体二进制的 sha256）：

| 变体 | `m1_p12` clean | `m1_p15` clean |
| ---- | ---- | ---- |
| 无握手（修前） | FAIL（6.47e8 起，抖动） | FAIL（6.82e8） |
| **只 mode 2** | **FAIL**（`maxAbs=2.9002e8`，0/4） | **FAIL**（`maxAbs=2.9002e8`，0/5） |
| mode 2 + 全体 AIC mode 0（修后） | **PASS** | **PASS** |

⇒ 每个 AIC 只等自己配对的 2 个 AIV **不够**（它读的是全体 AIV 写的整块 x）；`docs/05 §2` 的
「mode2 → 全体 AIC mode 0」组合是这里成立的最小形态。日志：`logs/mutant_m2only_*.log`，
判定在 `logs/judge_all.txt` 的 `mutant_m2only` 段。

## 6. 边界与既有红项（不预判因果）

- `m1_p12` 的 **in-kernel** 判据在修前修后同为 `fails=3`（`logs/prefix_m1_p12.log` 与
  `logs/post_m1_p12_rep1.log` 尾行同为 `FAILURES PRESENT（checks=23, guards=9, fails=3）`）。
  这 3 条落在 `m15_layer_loop.asc:4246-4251` 的 H2 分支：判 H2 的 arena1（H'）非毒值，
  而 H2 的输入 `A.pfHcArena0` 来自 H1 —— p12 档 H1 未开，arena0 本就是毒值。它与本修法无关
  （修前修后逐项相同），本 mission **不**声称已解释或修它。
- `m1_p8` / `m1_p15` / `m4097_p15` 的 in-kernel 判据分别 `ALL PASS（checks=17）` /
  `ALL PASS（checks=44）` / `ALL PASS（checks=44）`。
- **不把"不挂了"当"数值对了"**：本 mission 的判据是 m26 的**数值**对拍（J1/J1b/J1c/J2/J3/J4），
  且用同一二进制上的独立重复跑与"只 mode 2"变体收口；不是"能跑完"。
- **M140 那次实验的差异**（`m140_prefill_full_layer/README.md` §8 第 4 条 (b)）：它复用了
  `PFR_FLAG_M2_RING[0]`（=0）而 MoE 段自己也在 0 上 set/wait（`m15_moe_prefill.h:1293` / `:1317`），
  且**缺** all-AIC barrier ⇒ 与本次"6/7 独立号 + 全组合"不是同一形态。

## 7. 复算

```bash
# 离线：在已生成的 dump 上重跑判据（失败会以非 0 退出传播）
bash m15_layer_loop/evidence/m165_phaseb_handshake/reproduce.sh

# 设备：重建当前树（修后），重跑 p8 / p12×3 / p15×3 / m4097_p15，再跑判据
M164_DEVICE=1 bash m15_layer_loop/evidence/m165_phaseb_handshake/reproduce.sh

# 修前对照：用 handshake.patch 反向应用 → 重建 → 跑 p8/p12×3/p15/m4097，再还原
M164_PREFIX=1 bash m15_layer_loop/evidence/m165_phaseb_handshake/reproduce.sh
```

`reproduce.sh` 的默认（离线）路径只消费 dump；dump 大面不入库（`.gitignore`）。设备路径会把
`dumps_prefix/`、`dumps_post/`、`dumps_mutant_m2only/` 生成到本目录，判据读它们。

## 8. 入库清单

| 文件 | 内容 |
| ---- | ---- |
| `README.md` | 本文件 |
| `handshake.patch` | 本 mission 的源码改动（`git diff`，62 行），也是修前对照的反向应用对象 |
| `reproduce.sh` | 复算脚本（可传播失败） |
| `logs/prefix_*.log` | 修前逐档（p8 / p12 ×3 / p15 / m4097_p15）：锁内 `npu-smi` + 逐字命令 + 完整输出 + `exit=` |
| `logs/post_*.log` | 修后逐档（p8 / p12 ×3 / p15 ×3 / m4097_p15） |
| `logs/mutant_m2only_*.log` | 「只 mode 2」形状对照（p12 / p15） |
| `logs/judge_all.txt` | 全部 dump tag 的 `check_ref.py` 判定汇总 |
| `logs/stability.txt` | clean arm 的 `maxAbs` / sha256 / 跨进程 `max|diff|` |
| `binary_sha256.txt` | 三个二进制的 sha256（修前 / 修后 / 只-mode2 变体） |

## 9. 纪律自查

- 禁用词规范六条：本目录的**陈述文本**（`README.md`）中未使用（自查命令 = `reproduce.sh` 末段；
  按塔口径，只声明"在本 pattern 与本扫描范围内未出现"，不在此处重复列出六条字面）。
- 设备纪律：每档各自进锁、锁内 `npu-smi`、`timeout` 在锁内、一次进锁一条命令；未出现"等锁未取得读数"的情形。
- 只改了 scope 内文件：`m15_layer_kernel.h`（+62 行）与 `m15_layer_loop/evidence/m165_phaseb_handshake/**`；
  `m15_moe_prefill.h` 本 mission **未改**（机理链只读到它）。
