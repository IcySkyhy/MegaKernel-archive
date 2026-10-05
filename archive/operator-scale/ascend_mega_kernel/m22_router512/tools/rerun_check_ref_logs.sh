#!/usr/bin/env bash
# 重生成 evidence/check_ref_mode{2,4}.log（M56）。
#
# WHY：M46（e330796）修好 `tools/golden/moe_block_ref.py::router_topk` 的设备 FTZ 建模后，
#      报告项 R0 的含义变了 —— 修前它量「未建模 FTZ 的 golden vs 本参考(已 FTZ)」，m=4097 档
#      读到 125 槽 / 20 行；修后 golden 自己也建模 FTZ，R0 变成「两套 FTZ 实现是否一致」，
#      全档 0 槽 / 0 行。归档日志是 M46 之前的快照，故按其归档口径重跑刷新。
#
# 用法：  m22_router512/tools/rerun_check_ref_logs.sh        # 从任意 cwd 运行
# 退出码：0 = 两棵树 20 档全部 rc=0（与日志内每档的 `RESULT: OK` 一致）；1 = 有档非 0
#
# 前置：dumps 在 evidence/mode{2,4}/<case>/（仓内已归档）；解释器需 numpy。
# 注意：`real_m4097` 的 x.bin / router_logits.bin 未入库 ⇒ 走 --x-seed 重建并强制核对 --x-sha256。
set -u
cd "$(dirname "$0")/.." || exit 2
PY=${PY:-/usr/local/python3.12.13/bin/python3}
[ -x "$PY" ] || { echo "缺少解释器 $PY（numpy 只在该解释器下；可 PY=... 覆盖）" >&2; exit 2; }

W=data/router_weight.bin
X=$(awk '/\/real_m4097\/x.bin/{print $1; exit}' evidence/dump_sha256.txt)
if [ -z "$X" ]; then
  echo "evidence/dump_sha256.txt 里找不到 real_m4097/x.bin 的 sha256（M22 的归档契约）" >&2
  exit 2
fi

rc=0
for mode in mode2 mode4; do
  log="evidence/check_ref_${mode}.log"
  : > "$log"
  for c in real_m1 real_m2 real_m16 real_m33 real_m64; do
    echo "===== check_ref  evidence/$mode/$c  --w $W" >> "$log"
    "$PY" check_ref.py "evidence/$mode/$c" --w "$W" >> "$log" 2>&1 || rc=1
  done
  # real_m4097 紧跟 real_m64（与归档日志的档序一致，便于逐档 diff）
  echo "===== check_ref  evidence/$mode/real_m4097  --w $W --x-seed 5 --x-sha256 $X --no-logits" >> "$log"
  "$PY" check_ref.py "evidence/$mode/real_m4097" --w "$W" --x-seed 5 --x-sha256 "$X" --no-logits \
      >> "$log" 2>&1 || rc=1
  for c in nu_m1 nu_m7 nu_m33 nu_m64; do
    echo "===== check_ref  evidence/$mode/$c  --w onehot" >> "$log"
    "$PY" check_ref.py "evidence/$mode/$c" --w onehot >> "$log" 2>&1 || rc=1
  done
  n_ok=$(grep -c "RESULT: OK" "$log")
  n_r0=$(grep -c "R0 golden(已建模 FTZ) 与本参考的 ids 差.*：0 槽 / 0 行" "$log")
  echo "wrote $log：RESULT: OK $n_ok/10 档；R0=0/0 的档 $n_r0/10（R0 非零 = 两侧 FTZ 建模不一致）"
done
exit $rc
