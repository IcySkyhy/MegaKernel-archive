#!/usr/bin/env bash
# probe_v_align/run_ildl.sh —— `probe_v_ildl`（Interleave/DeInterleave 语义）逐用例运行 + 归档
# 用法（本目录下）：bash run_ildl.sh
# 产物：evidence/logs/ildl_<case>.log、evidence/dumps/il_<case>.bin、evidence/check_ref_ildl.log

set -u
cd "$(dirname "$0")"

ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
DUMPS="$EV/dumps"
BUILD="$ROOT/build"

mkdir -p "$LOGS" "$DUMPS"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

if [ ! -x "$BUILD/probe_v_ildl" ]; then
    cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_configure.log" 2>&1
    cmake --build "$BUILD" --target probe_v_ildl -j8 > "$LOGS/build_ildl.log" 2>&1 || {
        echo "构建失败，见 $LOGS/build_ildl.log"; exit 1; }
fi

: > "$LOGS/ildl_matrix.txt"
for case in $("$BUILD/probe_v_ildl" list); do
    log="$LOGS/ildl_${case}.log"
    timeout 180 "$BUILD/probe_v_ildl" run "$case" "$DUMPS" > "$log" 2>&1
    rc=$?
    outcome="$(grep -m1 '^OUTCOME:' "$log" | awk '{print $2}')"
    [ -n "$outcome" ] || outcome="NO-OUTPUT"
    aclerr="$(grep -m1 '^LAUNCH aclError=' "$log" | sed 's/.*aclError=//; s/ .*//')"
    printf '%-8s case=%-16s rc=%s aclError=%s\n' "$outcome" "$case" "$rc" "${aclerr:-NA}" | tee -a "$LOGS/ildl_matrix.txt"
done

echo "== 独立复核（check_ref_ildl.py）=="
"$PY" "$ROOT/check_ref_ildl.py" "$ROOT" > "$EV/check_ref_ildl.log" 2>&1
echo "check_ref_ildl rc=$?  → $EV/check_ref_ildl.log"
