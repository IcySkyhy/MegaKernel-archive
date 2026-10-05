# M179 —— 文档陈旧文本全仓扫描（旧 KV 几何 / 旧 PLE 判定项计数）

> 本文件是 mission M179（`feat/m179-doc-residue-resync-after-criterion`）任务 5 的落盘清单：
> 把「旧 KV 几何」与「旧 PLE 判定项计数（10）」两类**同族陈旧文本**全仓扫一遍，逐条给 `文件:行`。
> 纪律：**只改 scope 内的**；`*.log` / 探针转录等历史读数**不改、只标注**；负向对照或迁移记实里引旧值**不是**陈旧文本。
>
> 可复算的两条扫描命令：
>
> ```
> LC_ALL=C grep -rn "4,210,688\|50,528,256\|48\.19\|各 16×256\|页 16,384 B\|token 步长 512\|4,194,304" \
>   --include=*.md --include=*.h --include=*.asc --include=*.py --include=*.sh --include=*.txt .
> grep -rn "判定项 10 条\|10 条 Pw\|10/10" --include=*.md --include=*.log --include=*.txt .
> ```

## 1. 本 mission 已改（scope 内）

| 文件:行 | 旧文本 | 改为 |
|---|---|---|
| `m15_layer_loop/m15_loop_layout.h:32` | `[257 页/层, 2, 16, 256]`、`4,210,688 B/层（×12 = 48.19 MiB）` | `[257 页/层, 2, 16, 512]`、`8,421,376 B/层（= 257 页 × 32,768 B；×12 = 101,056,512 B = 96.38 MiB）`（对齐权威 `m15_attn_kv.h`） |
| `m15_layer_loop/evidence/attn_cache/GEOMETRY.md:116,134` | `KV_PLANE_BYTES = 101,056,512 B（96.40 MiB）` | `96.38 MiB`（101,056,512 / 1,048,576 = 96.375） |
| `m15_layer_loop/evidence/attn_cache/README.md:122` | 同上 | 同上 |
| `m15_layer_loop/evidence/attn_cache/GEOMETRY.md:161-165`、`README.md:126-128` | 「README 仍旧」注记（`归 M97/塔收口`、`.asc:564`） | 指向真正 owner：`.asc:808` 归 **M175**（仍未改）；README 已由 **M178 合入** 收口，注记随之同步 |
| `m15_layer_loop/README.md:1186`（M82-5） | `runs=kv` 判据读数 `56 条判定项 + 130 条 guard`（M82 历史值，未标时点） | 标为「M82 时点」并补**现 tip 真数** `83 条判定项 + 142 条 guard`（来源 `evidence/m178_kv_geom/logs/m178_runs_kv.log`，`checks=215 / guards=191`） |
| `m15_layer_loop/evidence/ple_wire/WITNESS.md` | 判据 10 条 / `10/10` / `5 / 10` / `2 / 10` | 当前 tip **12 条**（+`Pw.kv.T3`/`Pw.kv.nonvac`）；A/B 记 `12 / 0`；§2 加计数口径注 |
| `m15_layer_loop/evidence/ple_wire/m124/M124_CUBE_MMAD.md:26` | `10/10 Pw PASS` | `12/12 Pw PASS` |
| `m15_layer_loop/evidence/ple_wire/m124/M124_CUBE_MMAD.md:420` | `一字未动 ⇒ 层路径仍然是 10 条 Pw 判定项` | 既有 10 条 + 新增 2 条 = 最终 **12 条** |
| `m15_layer_loop/evidence/ple_wire/M111_LANDING_PATH.md`（抬头） | —（补注） | 计数口径注：本文件记 M111 时点 10 条；M124 后 12 条 |
| `m18_gdn_prefill/README.md:411` | 引 `check_phaseA_nan.py:59` 称「同类，未改…已由塔分给 M162」 | 改为现状：M162（`e088afb`）已把默认参考改成指数差（`ref_head_dt`），旧式 `eg*ig` 只在 `naive=True` 分支 |

> 时点说明：M179 开始时 `m15_layer_loop/README.md` 不在 scope；运行中 **M178 合入 main**（`4e626bb`），
> 塔把该文件释放给本 mission。本 mission 因此**同步了 main**（merge commit `bda4f26`）后再改 M82-5 的计数；
> 同时把「README 仍旧」注记改为「M178 已合入收口」（M178 也把 `m15_attn_kv.h:222` 的 96.40 同步为 96.38）。

## 2. out-of-scope 陈旧文本（follow-up，报塔）

| 文件:行 | 类别 | 现状 / 归属 |
|---|---|---|
| `m15_layer_loop/m15_layer_loop.asc:808` | 旧 KV 几何注释 `50,528,256 B（257 页/层 × 16,384 B）` | 陈旧；归 **M175**（`.asc` owner，本 mission 落笔时仍未改） |
| `m15_layer_loop/evidence/prefill_contract/new_plewire_{body,full,off}.log:239-242` | 旧 PLE 计数（`判定项 10 条`） | 历史转录，不改 |

## 3. 历史设备日志（不改，仅标注）

- pre-M98 的 H_Alloc 行（`主 KV 50.53 MB（12 层 × 4.211 MB/层 = 257 页/层 × 16384 B）`）在 committed
  `*.log` 里共 **37 个文件**（`m15_layer_loop/evidence/**` 含 `moe_race/`、`m15_layer_loop/moe_relift/**`、
  `tools/weights/evidence/**`；命令见抬头）。数值仍在，按纪律不改。
- pre-M124 的 `判定项 10 条` 出现在 `m15_layer_loop/evidence/ple_wire/{A_body_stage15,B_full_stage31,
  C_wire_off,D_miswire_table,E_miswire_winbase,F2_real_vocab_body,F_real_vocab_full}.log`、
  `ple_wire/E_det_probe_summary.txt:37-42`、`ple_wire/m124/M124_wire_{A,A_final_bin,B,C,stage13_nokv}.log`
  等转录里；不改，改由 `WITNESS.md §2` 的计数口径注承担。

## 4. 引旧值但**不是**陈旧文本（负向对照 / 迁移记实）

- `m15_layer_loop/m15_attn_kv.h:93,329`、`m15_attn_kv_probe.h:10,83`、`m15_attn_cache.h:53`：
  **负向对照**（`KV_OLD_*` / `MODE_BROKEN_OLDGEOM`）显式描述旧几何，是设计的一部分。
- `docs/15-prefill-design.md:7,1411,1449`：M98 迁移记实（`16,384 → 32,768`）与审计 grep 命令。
- `m15_layer_loop/evidence/attn_cache/GEOMETRY.md:118,129,131`：分歧记实（r1 时点）。
- `m15_layer_loop/ple/REAL_TABLE.md:459,491`、`m15_layer_loop/ple/logs/hm_*.log` 的「判据合计 10 条」：
  **不是** Pw 计数，是 HM（host-mapped）judge 自己的 10 条（`H2/H3/H4/H5/Hd.*`），M124 未动。
