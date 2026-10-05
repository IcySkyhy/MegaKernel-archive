# M161 r1 整改记录（reviewer-m161 r1 = p2-1items / fix-then-merge）

新 tip 见 `git log`（本文件所在 commit，父 = r1 被审的 `f7f9096`）。**未改动**任何已提交读数/日志
（`evidence/{synth,real,real_miss,neg_table,neg_state}/**`、`evidence/reproduce_stdout.log`、
`evidence/sha256.txt` 保持原样）；本轮证据新增在 `evidence/r1/`。

## 必修 1（P2-1）：T3 判据式与 README 不一致、kv 档过紧

**选法**：采用建议 (a) —— **`B3.kv` 的界改用 `Σ|terms|`**（与 `m15_ple_check.py:492-503` 的
`cmp_t3_masked(extra_terms=Σ|terms|)` 同口径）；`gated/normed/out` 保留 `max(|out|,|ref|)` 代理，
README §5 已写明是代理及理由。代码在 `m29_ple_unit/check_ref.py`（`cmp_t3` 新增 `extra_terms`，
`B3.kv` 传 `kv_sumabs = |wcat| @ |emb|`），判据行现在自带 `grid=Sum|terms|(...)` / `grid=proxy=...`
标识，口径陈述与实现不再脱节。

**读数（同一批 `data_real`，判据重算；未经设备重跑）**：

| 项 | 代理式 `max(|out|,|ref|)`（改前） | Σ|terms|（改后） |
| --- | --- | --- |
| `B3.kv.s0` `max(d/bound)` | 0.974 | **0.033** |
| `B3.kv.s1` `max(d/bound)` | 0.973 | **0.410** |
| `Σ|terms|/max(|out|,|ref|)` 中位 / max | — | s0: **32 / 1.34e+06**；s1: **35.6 / 4.42e+05** |

即改前 kv 裕度 **贴着红界（0.974）**，改后回到 0.033/0.410。分布数字与复审独立复算
（中位≈33、max≈1.35e6）一致。改前方向保守（代理 ≤ 标准界，只会假红、不会漏真错），
故不推翻任何 PASS 结论。见 `evidence/r1/real/check.log` 的 `B3.kv` 行。

## 必修 2：多槽/跨分片路径的负向对照

新增 kernel 变异 **bit14（`M29_MUT=16384`）**：`SlotRowOf` 的槽内行偏移 +1 环绕
（`m29_ple_unit.asc` 的 `SlotRowOf`），在 `data_real` 上跑。`reproduce.sh` 的负向档从 2 条增到
**3 条**（新增 `neg_slot_real`，env 串 `M29_MUT=16384`）。

读数（`evidence/r1/neg_slot_real/check.log`）：

| 判据 | 结果 | 说明 |
| --- | --- | --- |
| `B2.emb.s0/s1` | **FAIL** 5116/5120、5117/5120 | 行映射错 ⇒ 行内容错（**该红**） |
| `B3.kv.s0/s1` | **FAIL** 25399/25600、25480/25600 | ② 错的连带下游（**该红**） |
| `B2.miss.s0/s1` | **PASS** 0 vs 0 | 覆盖**不变** ⇒ 红的不是「越窗」而是「取错行」 |
| `B1.ids.*` / `B4.*` / `B5.*` | PASS | 未受影响 |

`check_ref` rc=1（`reproduce.sh` 记 `dev rc=0 check rc=1 红档数=9`）。

逐字命令（也可由 `reproduce.sh` 的 `run_negative "neg_slot_real" "$HERE/data_real" "M29_MUT=16384"`
编排）：

```bash
OUT=/tmp/m29_repro_r1/neg_slot_real; mkdir -p $OUT
flock -w 300 /tmp/npu0.lock bash -c "
  cd '$OUT'; npu-smi info > npu_smi.txt 2>&1
  env M29_OUT='$OUT' M29_DATA='<wt>/m29_ple_unit/data_real' M29_MUT=16384 timeout 180 '<wt>/m29_ple_unit/build/m29_ple_unit'
" > $OUT/run.log 2>&1
/usr/local/python3.12.13/bin/python3 m29_ple_unit/check_ref.py $OUT <wt>/m29_ple_unit/data_real   # 期望 rc=1
```

## 必修 3：「独立重写」措辞写宽

`README.md §5` 与 `m29_common.py` 抬头已改为**同源转录（correlated transcription）**：逐字来自
`m15_layer_loop/m15_ple_check.py`，规则钉在仓外 `qwen4_exp/nvidia/ops/ple.py` 的三个 kernel 符号上；
并写明**风险**（同一转写错误两边一起错）与覆盖它的**独立证据**（真实档 `emb` 直读分片、
`docs/17` §1.2 非空洞见证、出处符号级对照）。

## 顺带（non-blocking）

- `H_PleWindowSetup` 行号：`README.md §8.2` 由 `:2643-2717` 改为 `:2696`（函数起点），
  单窗口赋值点 `:2762`。
- 死 env `M29_STEPS`：`m29_ple_unit.asc` 与 `CMakeLists.txt` 的用法注释已删该 env，并注明
  step 数取自数据目录 `unit_meta.txt` 的 `n_steps`。
- 跨 step 状态对 `out` 的不可观测性：`README.md §4` 加了写窄说明（2 step + lag 9/6/3/0 时
  step1 的 `out` 不依赖被搬运的状态行；状态由 `B5.state` 与 `neg_state` 见证）。

## 本轮设备纪律

`bash m29_ple_unit/reproduce.sh /tmp/m29_repro_r1`（**rc=0, ALL-AS-EXPECTED**）：6 档各自一次
`flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（落盘）、`timeout 180` 在锁内。逐档日志见
`evidence/r1/<档>/{run.log,check.log,npu_smi.txt}`，编排 stdout 见
`evidence/r1/reproduce_stdout.log`。
