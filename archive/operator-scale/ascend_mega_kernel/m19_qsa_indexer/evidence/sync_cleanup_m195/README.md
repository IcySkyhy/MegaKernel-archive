# M195 证据：m19 QSA indexer 同步卫生清理（冗余 `set_flag/wait_flag` 对）

工作分支 `feat/m196-m19-sync-hygiene-redundant-event-pa`（worktree `wt-195`）。只改 `m19_qsa_indexer/**` 与
`docs/**`；未改任何已提交读数的数值，未加探针，`_cut` 锚点未碰。

## 1. 改了什么（逐对）

`m19_qsa_indexer.asc` 改前有 **10 对**文本 `SetFlag/WaitFlag`（另有 CrossCore 系列，属核间同步，不动）。
逐对判定与处置见 `m19_qsa_indexer/README.md` §5.5 的表；要点：

- **删 7 对冗余**：`QHeadNormRope`/`KPath`/`DumpLogits`/`EmitSub` 的 `V_MTE3`（各与同 id 的
  `BufRel<PIPE_V>` → `BufAcq<PIPE_MTE3>` 重复）、`CountTokens`/`ScanStage1` 的 `MTE2_V`（各与同 id 的
  `BufRel<PIPE_MTE2>` → `BufAcq<PIPE_V>` 重复）、AIC 入口的 `FIX_S`（barrier set 与 mode2 set 同挂 `PIPE_FIX`；
  AIC 不访问 UB，无「标量写 UB」形态）。
- **改 2 处**：`WriteToken`/`DumpStats` 的 `S_MTE3`（「标量写 UB → MTE3 搬」且**无** BufferID 覆盖）改为
  `BufAcq/BufRel<PIPE_S>` 阻塞释放 + 对侧 `BufAcq<PIPE_MTE3>`（`docs/05 §6.1 ⓔ`）。
- **留 1 对**：`VSync()` 的 `V_S`（标量读 V 刚算出的值 = 有值依赖；scalar 侧无 BufferID acquire，无其它覆盖）。

改后文件中 `SetFlag`/`WaitFlag`（非 CrossCore）只剩 `VSync()` 的 `V_S` 一对。

## 2. 设备读数（Ascend950PR，NPU 0）

纪律：每个设备档 `flock -w 300 /tmp/npu0.lock`，进锁先 `npu-smi`（快照落该档日志），`timeout` 在锁内。
等锁未取得读数即记「未取得读数」。

| 档 | 日志 | 读数 |
|---|---|---|
| 改前 默认 4 档 | `device_before_default4.log` | launch 全 OK；自检 PASS；count **2048/2050/1024/2048** |
| 改前 判据 | `check_ref_before.log` / `check_select_before.log` | `check_ref` ALL PASS；`check_select` ALL PASS |
| 改后 默认 4 档 | `device_after_default4.log` | 同上（计数逐档相同） |
| 改后 判据 | `check_ref_after.log` / `check_select_after.log` | `check_ref` ALL PASS；`check_select` ALL PASS |
| 改后 全矩阵 | `device_after_fullmatrix.log` | `run_select_matrix.sh`：30 组矩阵 + 跨核一致（6 组 + 2048/512 的 1↔2/4/8/56）+ 9 次确定性 |
| `V_S` 消融 | `device_ablate_novsync.log` | **变红**：自检 FAIL，tokens **1266/1278/1024/0**（对照改后 2048/2050/1024/2048）⇒ `V_S` 承载 |

**dump 逐字节对比**（`dump_compare.txt`）：改前/改后 `*_stats.bin` 之外全部 dump 逐字节相同。
`*_stats.bin` 仅字段 23–31 有别 —— 该字段代码里标注「保留」、是 `UB_TRACE` 前 128B 内的未初始化残值；
host 判定只读字段 7/8（`handoffOK`/`tokenSeen`），**字段 0–22 逐位相同**。
另：改后二进制的 `*_stats.bin`（含保留字段 23–31）在矩阵 ④ 的 **9 次独立进程里逐位一致**
（保留字段 payload 与字段 0–22 的哈希各只有 1 种）⇒ 差异来自「改前/改后两个二进制之间」的编译期 UB 落位，
不是同一二进制内的运行抖动，且不触任何判据字段。

## 3. 判据会咬（负对照，离线）

- `negctl_check_select.log`：把 `A_close_2048_out.bin` 的首个 token 改成 12345 ⇒ `check_select.py` 报
  「选中 513 个 block（oracle 512）；差集 1；非 4 元组 block 2」并 `exit 1`。
- `negctl_check_ref.log`：翻转 `A_close_2048_qk.bin` 首 bf16 的一个 bit ⇒ `check_ref.py` 报「proj 最大位差 1」并 `exit 1`。

（另：`docs/scan_doc_refs.py` 与 `docs/scan_quote_refs.py` 的读数与 M195 base 对比：doc_refs 的
`doc refs`/`section refs`/`path refs` 分母与 `out-of-range` 不变；quote_refs 的 finding 集合与 base 逐行相同。
`docs/05` 的例外子条**并进原 bullet**（不新增行），故未移动后续行号、未波及其它文件的 `docs/05:NNN` 引用。）

## 4. 复现

`dump_compare.txt` 的**比对内容**来自设备实测 dump（改前 / 改后两个二进制各跑一轮），二进制 dump 未入仓 ⇒ 这一部分离线不可重放；但报告的**生成器与期望值**已入仓（`compare_dumps.py`），离线可校验报告本身。两部分分开如下。

### 4.1 纯离线（不需设备）

```bash
# 一次跑齐：改后事件清点 / 文档裁定落库 / 禁用词 / 引用扫描 / 现有守卫 / dump_compare 校验
bash m19_qsa_indexer/evidence/sync_cleanup_m195/reproduce.sh
```

`dump_compare.txt` 的离线校验也可单独跑（内置期望、逐字节比对、失败非零退出）：

```bash
python3 m19_qsa_indexer/evidence/sync_cleanup_m195/compare_dumps.py
# OK   已入仓报告与内置期望逐字节一致（74 行 / 64 文件 / DIFF=4）；rc=0
```

它断言：① 报告与脚本内置的 64 个文件哈希表逐字节一致；② 4 个 DIFF 全落在 `*_stats.bin`；③ `all non-stats dumps bit-identical: True`；④ 每档 `*_stats.bin` 的差异字段恰为 23..31、字段 0..22 逐位相同。任一条不成立即非零退出。

### 4.2 需设备（NPU 0）

要重放**这份比对本身**（实测 dump 的字节差异），需两个二进制各跑一轮：

1. 当前分支（改后）编并 dump：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m19_qsa_indexer/build -S m19_qsa_indexer -DCMAKE_BUILD_TYPE=Release
cmake --build m19_qsa_indexer/build -j4
M19_OUT=/tmp/m195_after ./m19_qsa_indexer/build/m19_qsa_indexer
```

（设备纪律同 `reproduce.sh`：每档 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`、`timeout` 在锁内。）

2. 切到 M195 父提交（`3da656d^`）编出改前二进制，同法 dump 到 `/tmp/m195_base`。

3. 重新生成并比对：

```bash
python3 m19_qsa_indexer/evidence/sync_cleanup_m195/compare_dumps.py \
    --before /tmp/m195_base --after /tmp/m195_after --out /tmp/dump_compare.regen.txt
```

脚本逐行把重生成报告与内置期望比对（哈希 + stats 形态），任一不符即非零退出。`--out` 写出的报告抬头一律用仓内报告的固定标签（`/tmp/m195_base` / `/tmp/m195_after`），与你实际用的 dump 目录名无关 ⇒ 只要实测 dump 与 M195 时点一致，重生成文件就与仓内 `dump_compare.txt` `cmp` 相等（目录可任意命名）。

```bash
# 设备轨迹另可复算（需要 NPU 0；纪律同上）
M195_DEVICE=1 bash m19_qsa_indexer/evidence/sync_cleanup_m195/reproduce.sh
```
