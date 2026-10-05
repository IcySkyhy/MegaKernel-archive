# m9_gdn_prolog 证据（M147：m>1 / MTP / prefill prolog 段体）

被测：`gdn_prolog_mt_kernel`（类 `MtGdnProlog`，`m9_gdn_prolog.asc`）。
判据：`check_ref.py`（m=1 路 + m>1 路 + 反向对照）。
复算：`bash m9_gdn_prolog/reproduce.sh`；逐字命令见 `commands.txt`。

## 设备纪律

每档各自进一次 `flock -w 300 /tmp/npu0.lock`；进锁先 `npu-smi info`（快照落 `logs/*_npu_smi.txt`）；
`timeout 280` 放在锁内。本次 9 次进锁全部取得读数（无「未取得读数」档）。

## 逐档读数（logs/check_default.log 同源）

被测二进制 = 本 commit 构建的 `build/m9_gdn_prolog`（构建产物不入库；复算脚本重建）。

| 档（tag） | m | blk | qScaleOn | mut | 日志 | 判定 |
|---|---|---|---|---|---|---|
| m1_all | 1 | 56/28/16/8/4 | — | 0 | `logs/m1_all_run.log` + `logs/check_default.log` | 原 m=1 五档 PASS（未回归） |
| mt0 | 1 | 56 | 1 | 0 | `logs/mt0_m1_run.log` | state 位精确 + q/k/v/g/β PASS（与 m9 语义交叉复现） |
| mt1 | 4 | 56 | 0 | 0 | `logs/mt1_m4_run.log` | PASS |
| mt2 | 65 | 56 | 0 | 0 | `logs/mt2_m65_run.log` | PASS |
| mt3 | 257 | 56 | 0 | 0 | `logs/mt3_m257_run.log` | PASS |
| mt4 | 65 | 4 | 0 | 0 | `logs/mt4_m65_b4_run.log` | PASS |
| mt5 | 65 | 16 | 0 | 0 | `logs/mt5_m65_b16_run.log` | PASS |
| mt6 | 65 | 56 | 0 | 1（不交接 state） | `logs/mt6_m65_mut1_run.log` | 反向对照：state + q/k/v 变红（`check_mut1.log`） |
| mt7 | 257 | 56 | 0 | 2（不写 pad） | `logs/mt7_m257_mut2_run.log` | 反向对照：qpad/kpad 变红（`check_mut2.log`） |

判据完整逐字输出：`logs/check_default.log`（rc=0）、`logs/check_mut1.log`（变红）、`logs/check_mut2.log`（变红）。

## 判据口径

- T1 逐位（容差 0）：conv_state_out 的 bf16 位型、q/k 尾部 `[m, m+64)` 行、g/β 行内 `[m, tp)` 列。
- T3（`|got-exp| <= 1e-5·|exp| + 1e-6`）：q/k/v/g/β 有效区。
- 分类口径出处：`docs/22-prefill-prolog-epilog-wiring.md:333-344`。
- 反向对照机制：host 在设备输出面上先埋 `POISON = 1234.5f`，并可选注入 `mut=1`（不交接 state）、
  `mut=2`（不写 q/k pad）；`check_ref.py --mutant k` 期望这些档变红 —— 实测两档都变红。

## 文件

| 文件 | 内容 |
|---|---|
| `commands.txt` | 逐字命令（列 + 逐档 flock + 判据） |
| `logs/*_run.log` | 各档 host 运行日志（`[M9MT] ... launch/readback OK`） |
| `logs/*_npu_smi.txt` | 各档进锁时的 `npu-smi info` 快照 |
| `logs/check_default.log` | 默认判据完整输出（m=1 五档 + m>1 六档） |
| `logs/check_mut1.log` / `check_mut2.log` | 两个反向对照的完整输出（如预期变红） |
| `logs/baseline_m1_run.log` / `baseline_m1_npu_smi.txt` | 改动前的 m=1 基线读数（对照用） |
