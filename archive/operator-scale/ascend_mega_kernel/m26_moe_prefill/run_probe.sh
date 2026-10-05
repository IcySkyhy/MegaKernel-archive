#!/usr/bin/env bash
# m26_moe_prefill 的一次性取证：构建 → **四档**设备（锁内，一次进锁一条短命令）→ 判据 → 7 个负向对照 → 归档
#
#   档 A（--stage 2）：router 段（AIC Mmad + AIV top-k），判 J1..J4
#   档 B（--stage 3）：**链路臂**（S1 + S2/S2b + **S3** + S4），判 J1..J9
#                       —— 这是唯一会执行 `IndexGenP`（AIV0 单核标量读全量 ids）的臂，
#                          也就是 r1 复审 P1-1 的竞态所在路径。
# 用法：bash m26_moe_prefill/run_probe.sh [m]
# 设备纪律：单进程、进锁后再 `npu-smi` 复查（塔裁：flock -w 900 /tmp/npu0.lock）
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
M="${1:-64}"
EV="$ROOT/m26_moe_prefill/evidence"
PY=/usr/local/python3.12.13/bin/python3
mkdir -p "$EV"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ---- 0. 运行时的**源码指纹**（判据日志里引用的读数必须能钉到这一份源码上）----
{
  echo "# m26_moe_prefill 设备档源码指纹（run_probe.sh 生成）"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null)"
  echo "cann=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | head -1)"
  echo "npu_smi=$(npu-smi info 2>/dev/null | sed -n '2p' | tr -s ' ' | tr -d '|' | sed 's/^ *//;s/ *$//')"
  echo "date=$(date -Iseconds)"
  for f in m15_layer_loop/m15_moe_prefill.h m15_layer_loop/m15_moe_prefill_res.h \
           m26_moe_prefill/m26_moe_prefill.asc m26_moe_prefill/check_ref.py; do
    printf "%s sha256=%s\n" "$f" "$(sha256sum "$f" | cut -d' ' -f1)"
  done
} > "$EV/source_sha256.txt"
cat "$EV/source_sha256.txt"

# ---- 1. 构建 ----
rm -rf "$ROOT/m26_moe_prefill/build"
cmake -B "$ROOT/m26_moe_prefill/build" -S "$ROOT/m26_moe_prefill" -DCMAKE_BUILD_TYPE=Release \
  > "$EV/build.log" 2>&1
cmake --build "$ROOT/m26_moe_prefill/build" -j4 >> "$EV/build.log" 2>&1
BRC=$?
echo "build rc=$BRC"
if [ "$BRC" != "0" ]; then echo "BUILD FAILED"; tail -20 "$EV/build.log"; exit 1; fi
"$ROOT/m26_moe_prefill/build/m26_consts" > "$EV/consts.txt" 2>&1

# ---- 2. 两个设备档（**一次进锁只放一条短命令**，塔裁：flock -w ... 在锁文件之前）----
: > "$EV/run.log"
# 四档：m=64 的 router/链路 + **m=1**（r1 复审另跑过的那一档，顺带验 `calcM` 抬到 ≥2 的 3510 quirk）
for spec in "2:router:64" "3:chain:64" "2:router_m1:1" "3:chain_m1:1"; do
  st="$(echo "$spec" | cut -d: -f1)"; tag="$(echo "$spec" | cut -d: -f2)"; mm="$(echo "$spec" | cut -d: -f3)"
  {
    echo "== run $tag: m=$mm --stage $st =="
    cd "$ROOT/m26_moe_prefill/build" && flock -w 180 /tmp/npu0.lock bash -c "
      echo '-- npu-smi (in-lock) --'; npu-smi info | sed -n '6,20p'
      timeout 170 ./m26_moe_prefill --manifest ../../m17_moe_real/m17_weight_manifest.txt \
          --m $mm --stage $st --out m26_out_$tag
      echo \"run-$tag-rc=\$?\"
    "
  } >> "$EV/run.log" 2>&1
done
echo "--- run.log ---"; cat "$EV/run.log"

# ---- 3. 判据（正对照：两档都必须绿）----
JRC=0
for tag in router chain router_m1 chain_m1; do
  "$PY" "$ROOT/m26_moe_prefill/check_ref.py" "$ROOT/m26_moe_prefill/build/m26_out_$tag" \
      > "$EV/check_ref_$tag.log" 2>&1
  rc=$?
  echo "--- check_ref [$tag] rc=$rc ---"
  cat "$EV/check_ref_$tag.log"
  if [ "$rc" != "0" ]; then JRC=1; fi
done

# ---- 4. 负向对照（把被测读数在**内存里**弄坏：判据必须变红）----
# V1..V4 污染 logits；V5/V6/V7 污染 ids/weights（r1 复审 nit：后两者原来没被证伪过）
: > "$EV/negctl.log"
NEG_OK=1
for v in 1 2 3 4 5 6 7; do
  "$PY" "$ROOT/m26_moe_prefill/check_ref.py" "$ROOT/m26_moe_prefill/build/m26_out_chain" \
      --negctl "$v" --quiet >> "$EV/negctl.log" 2>&1
  rc=$?
  echo "negctl V$v rc=$rc" | tee -a "$EV/negctl.log"
  if [ "$rc" = "0" ]; then NEG_OK=0; fi
done
echo "--- negctl.log ---"; cat "$EV/negctl.log"

# ---- 5. 结论 ----
if [ "$JRC" = "0" ] && [ "$NEG_OK" = "1" ]; then
  echo "EVIDENCE: PASS (4 device runs green, all 7 negative controls red)"
  exit 0
fi
echo "EVIDENCE: FAIL (judge rc=$JRC, negctl-all-red=$NEG_OK)"
exit 1
