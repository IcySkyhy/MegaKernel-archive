#!/usr/bin/env bash
# M155：m26 判据空档补强的**离线**复算 + 自检（**零设备**，只跑 numpy 判据与合成夹具）。
#
# 覆盖三件事：
#   A. 回归：对归档档（m1_p8/m1_p12/m1_p15/m4097_p15 的 clean/mut1/mut2）重跑判据，
#      断言**旧结论不变**（PASS/FAIL/SKIPPED 与归档一致）。m4097 的 *.bin 不入 git（只有 meta）
#      ⇒ 如实判成 SKIPPED(rc=2)，不假装比过。
#   B. 负向对照：(a) 的行内置换判据 J1c 必须能变红 —— V8 把置换**只加在 J1 跳过的行**上，
#      故这批档里**只有 J1c 能抓**（p12 档 n_safe=0）；对照组是 p8 档（n_safe=m，V8 空转）。
#   C. (b)/(c) 的合成夹具：逐块/首块覆盖报告 + 「算错 vs 读错输入」的 W1 判别。
#
# 退出码：0 = 全部断言通过；非 0 = 有断言不符（**可传播失败**，归档用）。
# 证据 → m26_moe_prefill/evidence/judge_gap.log
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
EV="$ROOT/m26_moe_prefill/evidence"
J="$ROOT/m26_moe_prefill/check_ref.py"
ARCH="${M140_ARCH:-$ROOT/m15_layer_loop/evidence/m140_prefill_full_layer}"
PY="${PY:-/usr/local/python3.12.13/bin/python3}"
[ -x "$PY" ] || PY=python3.12
OUT="$EV/judge_gap.log"
SC=$(mktemp -d /tmp/m155_judge_XXXXXX)
trap 'rm -rf "$SC"' EXIT INT TERM

FAILN=0

run() {  # run <期望 rc> <期望 grep（空=不比）> <判据参数...>
  local want_rc="$1" want="$2"; shift 2
  local out rc
  out=$("$PY" "$J" "$@" 2>&1); rc=$?
  printf '%s\n' "$out"
  echo "-- rc=$rc（期望 $want_rc）"
  if [ "$rc" != "$want_rc" ]; then echo "!! ASSERT rc 不符"; return 1; fi
  if [ -n "$want" ] && ! printf '%s\n' "$out" | grep -qF -- "$want"; then
    echo "!! ASSERT grep 未命中：$want"; return 1
  fi
  return 0
}

# ---- 把归档 dump 拷进 scratch（只读归档，不写证据目录外）----
for c in m1_p8 m1_p12 m1_p15 m4097_p15; do
  mkdir -p "$SC/arch/$c"
  cp -r "$ARCH/dumps_$c/m26_Pf.gdn"       "$SC/arch/$c/clean"
  cp -r "$ARCH/dumps_$c/m26_Pf.gdn_mut1"  "$SC/arch/$c/mut1"
  cp -r "$ARCH/dumps_$c/m26_Pf.gdn_mut2"  "$SC/arch/$c/mut2"
done
WPAD="$ARCH/dumps_m1_p15/m26_Pf.gdn/m26_wpad.bin"
find "$SC/arch" -type d \( -name clean -o -name mut1 -o -name mut2 \) | while read -r d; do
  [ -f "$d/m26_wpad.bin" ] || cp "$WPAD" "$d/m26_wpad.bin"
done

{
  echo "# M155：m26 判据空档补强 —— 离线复算 + 自检（零设备）"
  echo "date=$(date -Iseconds)"
  echo "head_at_generation=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)   # 生成本日志时的 HEAD；本文件 commit 后分支 tip 会前移（权威锚点是下面的 judge_sha256）"
  echo "python=$PY ($("$PY" -c 'import numpy,sys;print("numpy",numpy.__version__,"py",sys.version.split()[0])' 2>/dev/null))"
  echo "judge_sha256=$(sha256sum "$J" | cut -d' ' -f1)"
  echo "archive=$ARCH"
  echo
  echo "=================================================================="
  echo "== A. 回归矩阵（断言旧结论不变：0=PASS 1=FAIL 2=SKIPPED）=="
  echo "=================================================================="
  for c in m1_p8 m1_p12 m1_p15 m4097_p15; do
    for a in clean mut1 mut2; do
      case "$c/$a" in
        m1_p8/*)        exp=0 ;;
        m1_p12/clean)   exp=1 ;;
        m1_p12/*)       exp=0 ;;
        m1_p15/clean)   exp=1 ;;
        m1_p15/mut1)    exp=0 ;;
        m1_p15/mut2)    exp=1 ;;
        m4097_p15/*)    exp=2 ;;   # *.bin 不入 git，只有 meta ⇒ SKIPPED
      esac
      echo "---- $c/$a（期望 rc=$exp）----"
      run "$exp" "" "$SC/arch/$c/$a" || FAILN=$((FAILN + 1))
      echo
    done
  done

  echo "=================================================================="
  echo "== B. 负向对照（(a) J1c 必须能变红）=="
  echo "=================================================================="
  echo "---- B1 p12/mut1 --negctl 8：只在 J1 跳过的行置换 top-k 0/1 列 ⇒ **只有 J1c 能抓** ----"
  B1=$("$PY" "$J" "$SC/arch/m1_p12/mut1" --negctl 8 2>&1); B1RC=$?
  printf '%s\n' "$B1"; echo "-- rc=$B1RC（期望 1）"
  NF=$({ printf '%s\n' "$B1" | grep -c '^   \[FAIL\]' || true; })
  echo "-- [FAIL] 条目数=$NF（期望 1）"
  if [ "$B1RC" != "1" ] || [ "$NF" != "1" ] || ! printf '%s\n' "$B1" | grep -qF 'J1c topk_ids 行内顺序'; then
    echo "!! ASSERT B1 不符"; FAILN=$((FAILN + 1))
  else
    echo "SIGNATURE-B1 J1c_bite_only=1/1 m1_p12_mut1"
  fi
  echo
  echo "---- B2 p12/mut1 --negctl 7：**全行**置换 ⇒ J1c 也红（p12 档 J1 被跳过）----"
  run 1 "J1c topk_ids 行内顺序" "$SC/arch/m1_p12/mut1" --negctl 7 || FAILN=$((FAILN + 1))
  echo
  echo "---- B3 p8/mut1 --negctl 8：V8 的门在此档空转（该行 J1 会判）⇒ 期望仍 PASS ----"
  run 0 "RESULT: PASS" "$SC/arch/m1_p8/mut1" --negctl 8 || FAILN=$((FAILN + 1))
  echo
  echo "---- B4 p8/mut1 --negctl 7：J1 与 J1c 都该红 ----"
  B4=$("$PY" "$J" "$SC/arch/m1_p8/mut1" --negctl 7 2>&1); B4RC=$?
  printf '%s\n' "$B4"; echo "-- rc=$B4RC（期望 1）"
  if [ "$B4RC" != "1" ] || ! printf '%s\n' "$B4" | grep -qF '[FAIL] J1 topk_ids' \
     || ! printf '%s\n' "$B4" | grep -qF 'J1c topk_ids 行内顺序'; then
    echo "!! ASSERT B4 不符"; FAILN=$((FAILN + 1))
  else
    echo "SIGNATURE-B4 J1_and_J1c_bite=1/1 m1_p8_mut1"
  fi
  echo
  echo "---- B5 p12/mut2 --negctl 8（第二个 clean-ish 档，同样只有 J1c 能抓）----"
  B5=$("$PY" "$J" "$SC/arch/m1_p12/mut2" --negctl 8 2>&1); B5RC=$?
  printf '%s\n' "$B5"; echo "-- rc=$B5RC（期望 1）"
  NF5=$({ printf '%s\n' "$B5" | grep -c '^   \[FAIL\]' || true; })
  if [ "$B5RC" != "1" ] || [ "$NF5" != "1" ]; then echo "!! ASSERT B5 不符"; FAILN=$((FAILN + 1));
  else echo "SIGNATURE-B5 J1c_bite_only=1/1 m1_p12_mut2"; fi
  echo

  echo "=================================================================="
  echo "== C. 合成夹具（只涉判据脚本，不涉设备）=="
  echo "=================================================================="
  FXD="$SC/fx"
  "$PY" "$ROOT/m26_moe_prefill/tools/gen_judge_fixtures.py" "$FXD" || FAILN=$((FAILN + 1))
  echo
  echo "---- C1 逐块/首块覆盖：块 0、2 在档、块 1 缺 ----"
  run 0 "缺块 [1]" "$FXD/blocks" || FAILN=$((FAILN + 1))
  "$PY" "$J" "$FXD/blocks" 2>&1 | grep -qF '覆盖：首块(blk=0)' && \
    "$PY" "$J" "$FXD/blocks" 2>&1 | grep -qF '已覆盖' || { echo "!! ASSERT C1 首块覆盖报告缺"; FAILN=$((FAILN + 1)); }
  echo
  echo "---- C2 W1「读错输入」：logits 由消费快照算出、与回读面不同 ----"
  run 1 "**读错输入**" "$FXD/consumed_read" || FAILN=$((FAILN + 1))
  echo
  echo "---- C3 W1「算错」：logits 与快照、回读面都不一致 ----"
  run 1 "**算错**" "$FXD/consumed_calc" || FAILN=$((FAILN + 1))
  echo
  echo "---- C4 W1「未提供、不可区分」：无快照、无 checksum ----"
  run 1 "**未提供**" "$FXD/no_witness" || FAILN=$((FAILN + 1))
  echo
  echo "---- C5 尾块 MT=8 > m=4（tie-risk 被判行）：正常应 PASS 且 tie-risk 4/4 ----"
  C5=$("$PY" "$J" "$FXD/tail_tie" 2>&1); C5RC=$?
  printf '%s\n' "$C5"; echo "-- rc=$C5RC（期望 0）"
  if [ "$C5RC" != "0" ] || ! printf '%s\n' "$C5" | grep -qF '4 / 4'; then
    echo "!! ASSERT C5 不符"; FAILN=$((FAILN + 1))
  else
    echo "SIGNATURE-C5 tail_MT_gt_m_normal_pass=1/1"
  fi
  echo
  echo "---- C6 尾块 MT=8 > m=4 + --negctl 8：修复前抛 IndexError；现在**只有 J1c 变红**（pad 行不置换）----"
  C6=$("$PY" "$J" "$FXD/tail_tie" --negctl 8 2>&1); C6RC=$?
  printf '%s\n' "$C6"; echo "-- rc=$C6RC（期望 1）"
  NF6=$({ printf '%s\n' "$C6" | grep -c '^   \[FAIL\]' || true; })
  echo "-- [FAIL] 条目数=$NF6（期望 1）"
  if [ "$C6RC" != "1" ] || [ "$NF6" != "1" ] || ! printf '%s\n' "$C6" | grep -qF 'J1c topk_ids 行内顺序'; then
    echo "!! ASSERT C6 不符"; FAILN=$((FAILN + 1))
  else
    echo "SIGNATURE-C6 tail_MT_gt_m_J1c_bite_only=1/1"
  fi
  echo
  echo "---- C7 尾块 MT=8 > m=4（被判行全被 J1 判）+ --negctl 8：应报「负向对照未生效」而不是空转/崩 ----"
  C7=$("$PY" "$J" "$FXD/tail_plain" --negctl 8 2>&1); C7RC=$?
  printf '%s\n' "$C7"; echo "-- rc=$C7RC（期望 0）"
  if [ "$C7RC" != "0" ] || ! printf '%s\n' "$C7" | grep -qF '负向对照未生效'; then
    echo "!! ASSERT C7 不符"; FAILN=$((FAILN + 1))
  else
    echo "SIGNATURE-C7 tail_MT_gt_m_V8_inapplicable_reported=1/1"
  fi
  echo

  echo "=================================================================="
  if [ "$FAILN" = "0" ]; then
    echo "SELFTEST: OK（全部断言通过）"
  else
    echo "SELFTEST: FAIL（$FAILN 条断言不符）"
  fi
} > "$OUT" 2>&1
cat "$OUT"
[ "$FAILN" = "0" ] || exit 1
exit 0
