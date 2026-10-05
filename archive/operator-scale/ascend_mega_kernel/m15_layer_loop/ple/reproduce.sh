#!/usr/bin/env bash
# M171 reproduce —— 真实 ngram 表：多槽窗口池 + 跨分片窗口（设备档）
#
# 一键复算 4 档：
#   base  : 多槽池（槽 0 横跨 shard 0→1），ids 全部由池服务      → 期望 11/11 PASS
#   miss  : 同一池 + ids_hm_miss.bin（token0 的 16 个 id 在池外）→ 期望只 Hd.served 红
#   mut16 : M15_PLE_MUT=65536（bit16 把 rows_per_shard 改错）     → 期望 Hd.row_fail/H2.emb 等红
#   legacy: 单窗口（n_slots=0）M92 路径回归                        → 记录（与 main HEAD 二进制逐条一致）
#
# 设备纪律：每档各自 flock（-w 300）→ 进锁先 npu-smi 快照 → timeout 在锁内。
# 用法（仓库根目录）：bash m15_layer_loop/ple/reproduce.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PLE=m15_layer_loop/ple
BIN=m15_layer_loop/build/m15_ple
LOG=$PLE/logs
CK=$PLE/hm_multislot_check.py
PY=/usr/local/python3.12.13/bin/python3.12
mkdir -p "$LOG"
fail=0
note() { printf '\n===== %s =====\n' "$*"; }
bad()  { printf '[reproduce][FAIL] %s\n' "$*"; fail=1; }

source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true

note "build m15_ple + 见证 wire.h"
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build m15_layer_loop/build -j4 --target m15_ple >/dev/null 2>&1 || bad "构建失败"
"$PY" m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py --check || bad "wire.h 与 m15_ple.asc 漂移"

# 真实小张量（weights from checkpoint）：缺失才生成（幂等、与 M85 同源）
if [ ! -f "$PLE/data/w_key_proj.bin" ]; then
  note "gen_ple_data.py"
  "$PY" "$PLE/gen_ple_data.py" >/dev/null 2>&1 || bad "gen_ple_data 失败"
fi

# 每档：一次 flock 只跑一条命令（进锁先 npu-smi，timeout 在锁内）
run_case() { # <name> <outdir> <extra-env...>
  local name="$1" out="$2"; shift 2
  rm -rf "$out"; mkdir -p "$out"
  flock -w 300 /tmp/npu0.lock bash -c "
    npu-smi info > $LOG/multislot_${name}_npusmi.txt 2>&1
    timeout 300 env M15_PLE_HM=1 M15_PLE_OUT=$out $* $BIN
  " > "$LOG/multislot_${name}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] && ! grep -q "kernel-side OK" "$LOG/multislot_${name}.log"; then
    printf '[reproduce][warn] %s 设备档 rc=%d（见 %s；可能未取得读数）\n' "$name" "$rc" "$LOG/multislot_${name}.log"
  fi
}
ck_case() { "$PY" "$CK" "$1" --brief > "$LOG/multislot_${2}_check.log" 2>&1; }

# ---- base：多槽池（槽 0 跨界），ids 全由池服务 ----
note "emit multi-slot (slots=4 slot-mib=4 tokens=64)"
"$PY" "$PLE/real_table_probe.py" --emit --slots 4 --slot-mib 4 --tokens 64 > "$LOG/multislot_emit.log" 2>&1 || bad "emit 失败"
run_case base "$PLE/out_ms_base"
ck_case "$PLE/out_ms_base" base
grep -q "ALL PASS" "$LOG/multislot_base_check.log" || bad "base 判据非全绿"
grep -q "Hd.row_fail|T1-struct|1024|0|PASS" "$LOG/multislot_base_check.log" || bad "base Hd.row_fail 非 0"

# ---- miss：越窗入口（池服务不到的 16 个 id）----
note "NC1 miss（ids_hm_miss.bin）"
run_case miss "$PLE/out_ms_miss" "M15_PLE_HM_IDS=ids_hm_miss.bin"
ck_case "$PLE/out_ms_miss" miss
grep -q "Hd.served|T1-struct|1024|1|FAIL" "$LOG/multislot_miss_check.log" || bad "NC1 期望 Hd.served 红，未见"
n_fail_miss=$(grep -c "|FAIL" "$LOG/multislot_miss_check.log" || true)
[ "$n_fail_miss" = "1" ] || bad "NC1 期望只 1 条红，实际 $n_fail_miss"

# ---- mut16：rows_per_shard 改错（bit16）----
note "NC2 mut16（M15_PLE_MUT=65536，bit16 改错 rows_per_shard）"
run_case mut16 "$PLE/out_ms_mut16" "M15_PLE_MUT=65536"
ck_case "$PLE/out_ms_mut16" mut16
grep -q "Hd.row_fail|T1-struct|1024|1|FAIL" "$LOG/multislot_mut16_check.log" || bad "NC2 期望 Hd.row_fail 红，未见"
grep -q "H2.emb|T1|163840|.*|FAIL" "$LOG/multislot_mut16_check.log" || bad "NC2 期望 H2.emb 红，未见"

# ---- legacy：单窗口 M92 路径回归（n_slots=0）----
note "legacy 单窗口回归（M92 判据 m15_ple_check.py）"
"$PY" "$PLE/real_table_probe.py" --emit --shard 0 --tokens 64 > "$LOG/legacy_emit.log" 2>&1 || bad "legacy emit 失败"
run_case legacy "$PLE/out_leg"
"$PY" m15_layer_loop/m15_ple_check.py "$PLE/out_leg" > "$LOG/legacy_regress_check.log" 2>&1
printf '[reproduce] legacy 读数（预期与 main HEAD 二进制逐条一致；H4.normed 1 条红为 M124 后既存问题）：\n'
grep -E "RESULT|合计" "$LOG/legacy_regress_check.log" || true

note "结论"
if [ $fail -eq 0 ]; then echo "reproduce OK（base 全绿；NC1 只 Hd.served 红；NC2 多处红；legacy 已记录）"; else echo "reproduce 有 FAIL"; fi
exit $fail
