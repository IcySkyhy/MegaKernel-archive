# M161 PLE 单元 —— 设备验证逐字命令与读数

设备槽纪律：每档各自进一次 `flock -w 300 /tmp/npu0.lock`；进锁先 `npu-smi`（快照落盘，见各档
`npu_smi.txt`）；`timeout 180` 放在锁内。全部命令由 `m29_ple_unit/reproduce.sh` 编排，也可逐条手跑。

## 0. 构建与数据

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m29_ple_unit/build -S m29_ple_unit -DCMAKE_BUILD_TYPE=Release
cmake --build m29_ple_unit/build -j4
/usr/local/python3.12.13/bin/python3 m29_ple_unit/gen_unit_data.py --mode synth --out m29_ple_unit/data_synth
/usr/local/python3.12.13/bin/python3 m29_ple_unit/gen_unit_data.py --mode real  --out m29_ple_unit/data_real
/usr/local/python3.12.13/bin/python3 m29_ple_unit/gen_unit_data.py --mode real --miss --out m29_ple_unit/data_real_miss
```

## 1. 正档：真实表多槽全覆盖（64 槽 / 54 分片）

```bash
OUT=/tmp/m29_repro/real; mkdir -p $OUT
flock -w 300 /tmp/npu0.lock bash -c "
  cd '$OUT'
  npu-smi info > npu_smi.txt 2>&1
  M29_OUT='$OUT' M29_DATA='<wt>/m29_ple_unit/data_real' timeout 180 '<wt>/m29_ple_unit/build/m29_ple_unit'
"
/usr/local/python3.12.13/bin/python3 m29_ple_unit/check_ref.py $OUT <wt>/m29_ple_unit/data_real
```

读数（`real/run.log` / `real/check.log`）：
- `multi-slot: n_slots=64 rows=256 win=81920 B | register rc=0 getDevPtr rc=0`
- `step 0: 2 tok | dev_oob=0 dev_miss=0`；`step 1: 2 tok | dev_oob=0 dev_miss=0`
- `B1.ids.*` 32/32、`B2.emb.*` 5120/5120（逐字节，与直读分片比）、`B3.kv.*` 25600/25600、
  `B4.gated/normed.*` 各 20480/20480、`B5.out.*` 20480/20480、`B5.state.*` 184320/184320 ⇒
  `VERDICT=OK steps=2 n_fail=0`（`real/check.log`）。

## 2. 越窗档：真实表只 stage 1 个分片

```bash
OUT=/tmp/m29_repro/real_miss; mkdir -p $OUT
flock -w 300 /tmp/npu0.lock bash -c "
  cd '$OUT'; npu-smi info > npu_smi.txt 2>&1
  M29_OUT='$OUT' M29_DATA='<wt>/m29_ple_unit/data_real_miss' timeout 180 '<wt>/m29_ple_unit/build/m29_ple_unit'
"
/usr/local/python3.12.13/bin/python3 m29_ple_unit/check_ref.py $OUT <wt>/m29_ple_unit/data_real_miss
```

读数（`real_miss/run.log` / `real_miss/check.log`）：
- `step 0: dev_miss=31`；`step 1: dev_miss=32`（设备侧越窗计数，与参考未覆盖数相等）
- covered 行的 `B2.emb.s0` 160/160 逐字节过；`B2.miss.*` 过 ⇒ `VERDICT=OK`
- 后级 T3 按设计跳过（部分覆盖时 kv/gate/conv 参考需全行定义）。

## 3. 负向对照（必须变红）

```bash
# 表基址偏 1 行
OUT=/tmp/m29_repro/neg_table; mkdir -p $OUT
flock -w 300 /tmp/npu0.lock bash -c "cd '$OUT'; npu-smi info > npu_smi.txt 2>&1; \
  env M29_OUT='$OUT' M29_DATA='<wt>/m29_ple_unit/data_synth' M29_NEG_TABLE=1 timeout 180 '<wt>/m29_ple_unit/build/m29_ple_unit'" > $OUT/run.log 2>&1
/usr/local/python3.12.13/bin/python3 m29_ple_unit/check_ref.py $OUT <wt>/m29_ple_unit/data_synth   # 期望 rc=1
# 不搬跨 step 状态
OUT=/tmp/m29_repro/neg_state; mkdir -p $OUT
flock -w 300 /tmp/npu0.lock bash -c "cd '$OUT'; npu-smi info > npu_smi.txt 2>&1; \
  env M29_OUT='$OUT' M29_DATA='<wt>/m29_ple_unit/data_synth' M29_NEG_STATE=1 timeout 180 '<wt>/m29_ple_unit/build/m29_ple_unit'" > $OUT/run.log 2>&1
/usr/local/python3.12.13/bin/python3 m29_ple_unit/check_ref.py $OUT <wt>/m29_ple_unit/data_synth   # 期望 rc=1
```

读数：
- `neg_table`：`B2.emb.s0` 5116/5120 不符、`B3.kv.s0` 25528/25600 不符、`B5.out.s1` 1/20480 不符
  ⇒ `VERDICT=FAILED n_fail=5`（`neg_table/check.log`）。
- `neg_state`：`B5.state.s1` 20480/184320 不符 ⇒ `VERDICT=FAILED n_fail=1`（`neg_state/check.log`）。

## 4. 一条命令复现

```bash
bash m29_ple_unit/reproduce.sh /tmp/m29_repro      # 退出码 0 = 全部符合预期
```

本批次读数：`synth OK / real OK / real_miss OK / neg_table 红 / neg_state 红` ⇒ `ALL-AS-EXPECTED`，rc=0。

## 5. 环境

- NPU：Ascend950PR，`npu-smi 25.7.rc1.10`，aic=28 / aiv=56（`real/run.log` 首行）。
- CANN 9.1.0，`source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
- checkpoint：`/workspace/Qwen3.8-Flash-Next-MXFP4`（真实 PLE 权重与 128 个 ngram 分片）。
