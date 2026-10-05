#!/usr/bin/env bash
# M164 —— 相位 B 的 H2→B AIV→AIC 跨核握手：证据复算脚本
#
#   默认（离线）          : 在已有 dump 上重跑 m26 判据；**修后的 clean arm 若 FAIL 则以非 0 退出**（传播失败）
#   M164_DEVICE=1         : 重建当前树（修后）+ 跑 p8 / p12×3 / p15×3 / m4097_p15 + 判据
#   M164_PREFIX=1         : 修前对照 —— 反向应用 handshake.patch（去掉握手）→ 重建 → 跑 p8 / p12×3 / p15 / m4097
#                           → 判据 → 还原源码（trap 保证还原）
#
# 设备纪律（塔）：每档各自进一次 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（快照落该档日志）、
# `timeout` 在锁内、一次进锁一条短命令。等锁未取得 ⇒ 该档记「未取得读数」并跳过（不静默）。
#
# 相位掩码：位 = H1:1 / A:2 / H2:4 / B:8（p8=8 只 B，p12=12 H2+B，p15=15 全四相位）
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-/usr/local/python3.12.13/bin/python3}"
BIN="$REPO/m15_layer_loop/build/m15_layer_loop"
MAN="$REPO/m15_layer_loop/weights_manifest.txt"
LOCK=/tmp/npu0.lock
PATCH="$HERE/handshake.patch"

build() {
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cmake -B "$REPO/m15_layer_loop/build" -S "$REPO/m15_layer_loop" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build "$REPO/m15_layer_loop/build" -j4
}

# 一档设备运行：$1=掩码 $2=m $3=档名 $4=timeout(s) $5=dump 组名(prefix/post/mutant_m2only)
run_one() {
  local mask="$1" m="$2" name="$3" to="$4" grp="$5"
  local dir="$HERE/dumps_${grp}/${name}"
  local log="$HERE/logs/${grp}_${name}.log"
  mkdir -p "$dir" "$HERE/logs"
  flock -w 300 "$LOCK" bash -c "
    mkdir -p '$dir'
    { echo '=== lock acquired '\$(date -Is)' mask=$mask m=$m name=$name bin=$grp ==='
      echo '--- npu-smi (lock entry) ---'
      npu-smi info
      timeout $to env M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
        M15_PREFILL_PHASES=$mask M15_PREFILL_M=$m M15_PREFILL_DUMPDIR='$dir' \
        '$BIN' '$MAN' prefill
      echo \"exit=\$?\"
    } > '$log' 2>&1
  " || { echo "[m164] 未取得读数：锁未拿到（$grp/$name）"; return 1; }
  echo "[m164] 设备档完成：$dir"
}

# 判一档：$1=m26 dump 目录。回 PASS/FAIL；rc=0/1
verdict() {
  local out
  out="$("$PY" "$REPO/m26_moe_prefill/check_ref.py" "$1" 2>&1)" || true
  echo "$out" | grep -E '判定项 .*：' | tail -1
  echo "$out" | grep -E 'RESULT:' | tail -1
  echo "$out" | grep -q 'RESULT: PASS'
}

run_device_group() {   # $1 = dump 组名；跑该组的档（p8/p12×3/p15×3/m4097）
  local grp="$1"
  run_one 8  1    m1_p8_rep1   300 "$grp"
  run_one 12 1    m1_p12_rep1  300 "$grp"
  run_one 12 1    m1_p12_rep2  300 "$grp"
  run_one 12 1    m1_p12_rep3  300 "$grp"
  run_one 15 1    m1_p15_rep1  300 "$grp"
  run_one 15 1    m1_p15_rep2  300 "$grp"
  run_one 15 1    m1_p15_rep3  300 "$grp"
  run_one 15 4097 m4097_p15    500 "$grp"
}

# 判据：对 dumps_$grp 下的每个 clean arm（m26_Pf.gdn）跑 m26 判据
judge_group() {   # $1 = dump 组名；$2 = "require"（clean FAIL 时返回 1）或 "info"
  local grp="$1" mode="$2" rc=0
  for d in "$HERE/dumps_${grp}"/*/; do
    [ -f "$d/m26_Pf.gdn/m26_meta.txt" ] || continue
    echo "===== ${grp} $(basename "$d") / Pf.gdn (clean=arm0) ====="
    if verdict "$d/m26_Pf.gdn"; then :; else rc=1; fi
  done
  if [ "$mode" = require ] && [ "$rc" -ne 0 ]; then
    echo "[m164] 判据失败：${grp} 的 clean arm 存在 FAIL"
  fi
  return "$rc"
}

# ---------------- 入口 ----------------
if [ "${M164_PREFIX:-0}" = "1" ]; then
  restore() { ( cd "$REPO" && git checkout -- m15_layer_loop/m15_layer_kernel.h ); }
  trap restore EXIT
  echo "[m164] 修前对照：反向应用 handshake.patch（去掉握手）"
  ( cd "$REPO" && git apply -R "$PATCH" ) || { echo "[m164] 反向应用失败（树不是 handshake 后的状态？）"; exit 2; }
  build || { echo "[m164] 构建失败"; exit 2; }
  run_device_group prefix
  judge_group prefix info || true
  restore; trap - EXIT
  echo "[m164] 已还原源码；重建修后二进制"
  build || true
  exit 0
fi

if [ "${M164_DEVICE:-0}" = "1" ]; then
  build || { echo "[m164] 构建失败"; exit 2; }
  run_device_group post
fi

# 离线（或设备跑完后）：判修后组的 clean arm，FAIL 则非 0 退出（传播失败）
if [ -d "$HERE/dumps_post" ]; then
  judge_group post require || exit 1
else
  echo "[m164] 无 dumps_post ⇒ 离线跳过（要重跑请 M164_DEVICE=1）"
fi

# 若存在修前/形状对照组，只作信息输出（它们**预期** clean FAIL）
[ -d "$HERE/dumps_prefix" ]        && { echo; echo "== 修前对照（预期 clean FAIL）=="; judge_group prefix info || true; }
[ -d "$HERE/dumps_mutant_m2only" ] && { echo; echo "== 只-mode2 形状对照（预期 clean FAIL）=="; judge_group mutant_m2only info || true; }

# 禁用词自查：六条禁用词在本目录**陈述文本**（README.md）中的出现次数
# （按塔口径只声明"该 pattern 与这个范围内"；扫描目标是本目录的文档文本）
echo
echo "== 禁用词自查（扫描范围：$HERE/README.md）=="
for w in '已全部' '无残留' '0 命中' '零命中' '绝不' '完全正确'; do
  n=$(grep -F -- "$w" "$HERE/README.md" 2>/dev/null | wc -l)
  echo "  pattern [$w] 在 README.md 范围内 $n 行"
done
exit 0
