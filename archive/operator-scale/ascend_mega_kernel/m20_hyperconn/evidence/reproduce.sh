#!/usr/bin/env bash
# M174 —— h2.blk 判据门限口径修正（M20 独立校验门限）的复现脚本。
# 零设备；三部分：
#   ① 孤立负向对照：新子条件「良态 ulp>2 占比 ≤1e-3」有牙（零 dump）
#   ② 回归：M140/M169 台架（check_full_layer.py，未改调用）在**入库的** m1 p15 dump 上跑新口径
#   ③ h2.blk 的 m4097 复算（需 M169 的 m4097 dump，不随仓库入库；由 M169 reproduce.sh 在设备上再生成）
#      —— 用 M174_DUMP_ROOT=<M169 evidence 目录> 指定；未指定则显式 SKIPPED（不静默算过）。
#
# 用法：bash m20_hyperconn/evidence/reproduce.sh
#       M174_DUMP_ROOT=<.../m15_layer_loop/evidence/m169_whole_layer_4phase> bash m20_hyperconn/evidence/reproduce.sh
# 退出码：0 = 全部读数与期望一致；1 = 有读数与期望不符（传播失败）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${M174_PY:-/usr/local/python3.12.13/bin/python3}"
rc=0

echo "== ① 孤立负向对照（零 dump）：新子条件 over2 把判据打红，另两条子条件均 PASS =="
echo "\$ $PY m20_hyperconn/evidence/negctl_over2.py"
"$PY" "$HERE/negctl_over2.py"
[ $? -eq 0 ] || { echo "!! 期望 rc=0（三条断言成立）"; rc=1; }

echo
echo "== ② 回归：M140/M169 台架调用不变 + 新口径（入库 m1 p15 dump）=="
M1DUMP="$REPO/m15_layer_loop/evidence/m140_prefill_full_layer/dumps_m1_p15"
echo "\$ $PY m15_layer_loop/evidence/m140_prefill_full_layer/check_full_layer.py $M1DUMP Pf.gdn 1 15"
"$PY" "$REPO/m15_layer_loop/evidence/m140_prefill_full_layer/check_full_layer.py" "$M1DUMP" Pf.gdn 1 15
[ $? -eq 0 ] || { echo "!! 期望 rc=0（m1 各相位全绿）"; rc=1; }

echo
echo "== ③ h2.blk 的 m4097 复算（新/旧两种口径并列；dump 不入库）=="
DUMP_ROOT="${M174_DUMP_ROOT:-}"
if [ -z "$DUMP_ROOT" ]; then
  echo "SKIPPED：未设 M174_DUMP_ROOT ⇒ m4097 dump 不在手边（不随仓库入库，"
  echo "         由 m15_layer_loop/evidence/m169_whole_layer_4phase/reproduce.sh 在设备上再生成）"
else
  # 期望 rc：nolinks = 0（新口径转绿）；links = 1（**保留的**良态逐位率 0.98995 < 0.99 仍红）
  for pair in "nolinks 0" "links 1"; do
    t="${pair%% *}"; want="${pair##* }"
    echo "\$ $PY m20_hyperconn/evidence/h2_blk_recalib.py $DUMP_ROOT/dumps_m4097_p15_$t Pf.gdn 4097"
    "$PY" "$HERE/h2_blk_recalib.py" "$DUMP_ROOT/dumps_m4097_p15_$t" Pf.gdn 4097
    got=$?
    echo "   rc=$got（期望 $want）"
    [ "$got" -eq "$want" ] || { echo "!! rc 与期望不符（$t）"; rc=1; }
  done
  echo "-- 负向对照：向 h2.blk 注入一个 64 行块的缺陷，新口径仍须红（期望 rc=1）--"
  for inj in scale zero shift; do
    echo "\$ $PY m20_hyperconn/evidence/h2_blk_recalib.py $DUMP_ROOT/dumps_m4097_p15_nolinks Pf.gdn 4097 --inject $inj"
    "$PY" "$HERE/h2_blk_recalib.py" "$DUMP_ROOT/dumps_m4097_p15_nolinks" Pf.gdn 4097 --inject "$inj" >/dev/null
    got=$?
    echo "   rc=$got（期望 1）"
    [ "$got" -eq 1 ] || { echo "!! 注入 $inj 后新口径未打红"; rc=1; }
  done
fi

echo
[ "$rc" -eq 0 ] && echo "RESULT: OK" || echo "RESULT: FAILED"
exit "$rc"
