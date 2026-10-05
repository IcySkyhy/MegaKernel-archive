#!/usr/bin/env bash
# run_probe.sh —— M134 探针证据生成（设备槽纪律：每档各自进锁 + 进锁先 npu-smi + 锁内 timeout）
#
# 用法（本目录下）：
#   bash run_probe.sh                         # 默认变体集 × REPS 次（每档各自进一次锁）
#   REPS=3 N=16 bash run_probe.sh             # 指定重复数与 N
#   VARIANTS="base c_mode4free" bash run_probe.sh
#   TO=20 bash run_probe.sh                   # 单次运行超时（秒）
#
# 纪律（mission 判据与纪律）：
#   · 每档各自 `flock -w 300 /tmp/npu0.lock`（未取得锁 ⇒ 记「未取得读数」，不记「未复现」）
#   · 进锁先 `npu-smi`；每条变体每次运行都 `timeout`（在锁内）
#   · exit=124 记成「HANG（未取得内容读数）」，不写成「没复现」
# 产物（进 git）：evidence/logs/<variant>.log、evidence/logs/matrix.txt、evidence/commands.txt
set -u

cd "$(dirname "$0")"
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
OUT="$ROOT/out_tmp"
BIN="$ROOT/build/probe_crosscore_tail"
REPS="${REPS:-3}"
N="${N:-16}"
TO="${TO:-20}"
LOCKWAIT="${LOCKWAIT:-300}"
TRACE="${TRACE:-0}"     # 1 ⇒ 给内核加 --trace（逐轮打印）；档名加 TAG 以免覆盖非 trace 档
TAG="${TAG:-}"          # 追加到 <变体>_N<N><TAG>.log / matrix_N<N><TAG>.txt
VARIANTS="${VARIANTS:-base a_nowait b_via_gm c_rotate c_wait1 c_broadcast d_no_acquire dbg_rdyonly dbg_nosync c_mode4free c_m2setonly neg_nofree neg_wrongval c_preset c_m2sameid c_m2mte3 dbg_nodata dbg_nobar c_mode4full c_m2canon}"
TRACEARG=""
[ "$TRACE" = "1" ] && TRACEARG="--trace"

mkdir -p "$LOGS" "$OUT"

if [ ! -x "$BIN" ]; then
    echo "== 构建 =="
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release > /tmp/pct.cmake 2>&1 || { cat /tmp/pct.cmake; exit 1; }
    cmake --build build -j4 > /tmp/pct.build 2>&1 || { tail -30 /tmp/pct.build; exit 1; }
fi

bash tools/gen_commands.sh

: > "$LOGS/matrix_N${N}${TAG}.txt"
echo "variant N=$N reps=$REPS timeout=${TO}s" >> "$LOGS/matrix_N${N}${TAG}.txt"

# 每档各自进锁（失败 ⇒ 记录「未取得读数」）
for v in $VARIANTS; do
    log="$LOGS/${v}_N${N}${TAG}.log"
    : > "$log"
    # 用 flock 包一个子 shell；子 shell 内进锁先 npu-smi，然后逐次 timeout
    lock_ok=0
    flock -w "$LOCKWAIT" /tmp/npu0.lock bash -c '
        set -u
        BIN="$1"; LOG="$2"; OUT="$3"; V="$4"; N="$5"; REPS="$6"; TO="$7"; TRACEARG="$8"
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        {
          echo "=== lock acquired $(date -Is) variant=$V N=$N reps=$REPS timeout=${TO}s ==="
          echo "--- npu-smi (lock entry) ---"
          npu-smi info 2>&1 | sed -n "1,12p"
        } >> "$LOG"
        n0=0; hang=0; n1=0; n2=0
        for i in $(seq 1 "$REPS"); do
            echo "--- run $i (timeout ${TO}s) ---" >> "$LOG"
            timeout "$TO" stdbuf -oL "$BIN" "$V" "$N" "$i" "$OUT" $TRACEARG >> "$LOG" 2>&1
            rc=$?
            echo "exit=$rc" >> "$LOG"
            case "$rc" in
              0) n0=$((n0+1));;
              1) n1=$((n1+1));;
              2) n2=$((n2+1));;
              124) hang=$((hang+1)); echo "SUMMARY variant=$V run=$i HANG=1 rc=124（锁内超时未返回：记「未取得内容读数」，不记「没复现」）" >> "$LOG";;
            esac
        done
        echo "AGG variant=$V N=$N rc0=$n0 rc1=$n1 rc2=$n2 hang=$hang reps=$REPS" >> "$LOG"
    ' _ "$BIN" "$log" "$OUT" "$v" "$N" "$REPS" "$TO" "$TRACEARG"
    if [ $? -eq 0 ]; then lock_ok=1; fi
    if [ "$lock_ok" -ne 1 ]; then
        echo "AGG variant=$v N=$N LOCKFAIL=1（${LOCKWAIT}s 内未取得设备锁：记「未取得读数」）" >> "$log"
    fi
    agg=$(grep -m1 '^AGG ' "$log" | sed 's/^AGG //')
    printf '%-14s %s\n' "$v" "${agg:-未取得读数（锁未取得）}" | tee -a "$LOGS/matrix_N${N}${TAG}.txt"
done

echo "完成。证据目录：$EV"
