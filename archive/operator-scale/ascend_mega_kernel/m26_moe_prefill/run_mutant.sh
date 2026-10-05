#!/usr/bin/env bash
# m26_moe_prefill 的**变异档**：把 r2 补在 `ig.Run()` **之前**的那道全体 AIV 屏障临时删掉，
# 重跑 `--stage 3` 链路臂 **N 次**，逐次判据 —— **期望 J5–J9 变红**。
#
# 为什么要有这个脚本（复审 r2 的"最要紧的限度"）："J5–J9 咬得住 P1-1"在 r2 时只是
# 机制论断 + G7 覆盖见证 + V5/V7 敏感度见证；本仓的口径要求判据能**证明它会红**。
# 为什么要跑 N 次：缺屏障造成的是一条**时序相关**的竞态 —— 单次不红**不能**推出"判据不敏感"，
# 只能如实记为"这 N 次没撞上"。跑 N 次是为了把这句话说得准（而不是把单次绿当结论）。
#
# 纪律：
#   · 变异只在**临时副本**上做，跑完**立刻**从备份恢复，并用 sha256 与 `git status` 核对树是干净的；
#   · 变异用**独立 build 目录**（/tmp/m26_mut），不碰 m26_moe_prefill/build；
#   · 设备：`flock -w 180` **一次有界等待** + **一次进锁一条短命令**（整体 `timeout 170`）+
#     锁内先 `npu-smi`；每档 dump 落在 /tmp，不覆盖归档的四个正常档。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
EV="$ROOT/m26_moe_prefill/evidence"
PY=/usr/local/python3.12.13/bin/python3
HDR="$ROOT/m15_layer_loop/m15_moe_prefill.h"
REL="m15_layer_loop/m15_moe_prefill.h"
MUT_MODE="${MUT_MODE:-no_barrier}"          # no_barrier | no_barrier_wide
OUT="$EV/mutant_no_pre_ig_barrier${MUT_MODE#no_barrier}.log"
N=3
BAK=$(mktemp /tmp/m118_hdr_XXXXXX.h)

source /usr/local/Ascend/ascend-toolkit/set_env.sh

cp "$HDR" "$BAK"
restore() { cp "$BAK" "$HDR"; }
trap 'restore; rm -f "$BAK"' EXIT INT TERM

{
  echo "# m26_moe_prefill 变异档（MUT_MODE=$MUT_MODE）：删掉 ig.Run() 之前的全体 AIV 屏障（= r2 的 P1-1 那一条），跑 $N 次"
  echo "#   no_barrier      = 只删屏障（观察真实时序）"
  echo "#   no_barrier_wide = 再让非 0 号 AIV 在其 topk 前自旋（**人为**拉宽竞态窗，只为定位"读不到陈旧值"是窗口问题还是判据问题）"
  echo "date=$(date -Iseconds)"
  echo "tip=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
  echo "source_before_mutant_sha256=$(sha256sum "$BAK" | cut -d' ' -f1)"

  # ---- 1. 变异（只删那一条屏障调用）----
  "$PY" - "$HDR" "$MUT_MODE" <<'PYEOF' || echo "MUTATION FAILED"
import sys
p, mode = sys.argv[1], sys.argv[2]
s = open(p).read()
old = "            BarrierAivStep<PIPE_S>(1);\n"
assert s.count(old) == 1, "anchor 出现 %d 次（期望 1）" % s.count(old)
s = s.replace(old, "            // [MUTANT] r2 的 ig.Run() 前屏障被删（仅本变异档）\n")
if mode == "no_barrier_wide":
    t = "            topk.Run(bid, nAiv);\n"
    assert s.count(t) == 1, "topk.Run 锚点出现 %d 次" % s.count(t)
    s = s.replace(t, "            if (bid != 0u) { for (uint32_t s_ = 0; s_ < 20000u; ++s_) { AscendC::PipeBarrier<PIPE_ALL>(); } }\n" + t)
open(p, "w").write(s)
PYEOF
  echo "== 变异前后**只差这一处**的证明（备份 vs 变异后，全量输出）=="
  diff -U2 "$BAK" "$HDR"
  echo "source_after_mutant_sha256=$(sha256sum "$HDR" | cut -d' ' -f1)"

  # ---- 2. 独立 build ----
  rm -rf /tmp/m26_mut
  cmake -B /tmp/m26_mut -S "$ROOT/m26_moe_prefill" -DCMAKE_BUILD_TYPE=Release > /tmp/m26_mut_build.log 2>&1
  cmake --build /tmp/m26_mut -j4 >> /tmp/m26_mut_build.log 2>&1
  echo "mutant build rc=$?"

  # ---- 3. N 次设备档（一次进锁；每次 timeout 60 —— 单档实际几秒）----
  # 内层命令写成**独立脚本**再由 flock 执行：避免在双引号里让外层 shell 提前展开内层变量
  # （第一版就是 `$i` 被外层展开 + `set -u` ⇒ "i: unbound variable"，脚本在设备档前就死了）。
  INNER=/tmp/m26_mut_inner.sh
  cat > "$INNER" <<EOS
#!/bin/bash
cd /tmp/m26_mut
echo '-- npu-smi (in-lock) --'; npu-smi info | sed -n '6,20p'
echo '-- device error report --'; npu-smi info -t health -i 0 2>&1 | head -5
for i in \$(seq 1 $N); do
  echo "-- mutant run \$i --"
  rm -rf /tmp/m26_mut/m26_out_mut\$i
  timeout 60 /tmp/m26_mut/m26_moe_prefill --manifest $ROOT/m17_moe_real/m17_weight_manifest.txt \
      --m 64 --stage 3 --out /tmp/m26_mut/m26_out_mut\$i 2>&1 | tail -3
  echo "mutant-run-\$i-rc=\$?"
done
EOS
  BIT=0
  flock -w 180 /tmp/npu0.lock bash "$INNER"
  echo "-- /tmp/m26_mut 的 dump 目录（判据前先看清楚谁在）--"
  ls -d /tmp/m26_mut/m26_out_mut* 2>/dev/null || echo "(none)"
  # 判据在锁外跑（纯 host）
  for i in $(seq 1 $N); do
    # **防"缺 dump 被当成判据变红"**：先确认 dump 真的存在（第一版就是因为 dump 落到别处
    # 而把 FileNotFoundError 的 rc=1 误读成 "MUTANT BITE" —— 这类假阳性必须由脚本挡住）
    if [ ! -f "/tmp/m26_mut/m26_out_mut$i/m26_meta.txt" ]; then
      echo "SCRIPT ERROR: /tmp/m26_mut/m26_out_mut$i/m26_meta.txt 不存在 ⇒ 本档不计入结论"
      continue
    fi
    echo "== mutant judge #$i（期望 rc=1）=="
    "$PY" "$ROOT/m26_moe_prefill/check_ref.py" /tmp/m26_mut/m26_out_mut$i --quiet
    rc=$?
    echo "mutant-judge-$i-rc=$rc"
    if [ "$rc" != "0" ]; then BIT=$((BIT + 1)); fi
  done

  # ---- 4. 恢复并核对 ----
  restore
  echo "== restore check =="
  echo "restored_sha256=$(sha256sum "$HDR" | cut -d' ' -f1)"
  echo "git_status_after_restore=$(git -C "$ROOT" status --porcelain -- "$REL" | tr '\n' ' ')"
  git -C "$ROOT" diff --stat -- "$REL" || true

  echo "== 结论 =="
  echo "mutant_judge_red_count=$BIT / $N"
  if [ "$BIT" -gt 0 ]; then
    echo "MUTANT BITE: CONFIRMED（$BIT/$N 次变红 ⇒ J5–J9 对这道屏障敏感）"
  else
    echo "MUTANT DID NOT BITE in $N runs（MUT_MODE=$MUT_MODE）：缺屏障的实现在这 $N 次里判据全绿。"
    echo "  这是一条**真读数**（不许含糊成"判据不敏感"）：本竞态是时序相关的，N 次没撞上"
    echo "  不等于判据不敏感 —— 判据对"读到错 ids"这件事的敏感性由负向对照 V5 直接见证"
    echo "  （V5 只改 ids 一行，J1/J1b/J5–J9 立刻变红，见 evidence/negctl.log）。"
  fi
} > "$OUT" 2>&1
cat "$OUT"
