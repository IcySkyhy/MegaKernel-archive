#!/usr/bin/env bash
# run_probe.sh —— M141 探针证据生成（设备槽纪律：每档各自进锁 + 进锁先 npu-smi + 锁内 timeout）
#
# 纪律（mission 判据与纪律，另参 probe_ub2l1/evidence/slot.sh:1-63）：
#   · **每档各自 `flock -w 300 /tmp/npu0.lock`**（未取得锁 ⇒ 记「未取得读数」，不记「未复现」）
#   · 进锁先 `npu-smi`，快照原样落盘（每档 log + 汇总 evidence/npu_smi_snapshots.txt）
#   · 每条变体每次运行都 `timeout`，且 **timeout 放在锁内**
#   · exit=124 记成「HANG（未取得内容读数）」，不写成「没复现」
#   · 锁内只放设备那一段；编译在锁外
#
# 用法（本目录下）：
#   bash run_probe.sh                     # 默认变体集 × REPS 次（每档各自进一次锁）
#   REPS=1 TO=60 bash run_probe.sh
#   VARIANTS="fixp_l0c2gm ub2gm_api" bash run_probe.sh
#
# 产物（进 git）：evidence/logs/<variant>.log、evidence/logs/matrix.txt、
#                 evidence/logs/commands.txt、evidence/npu_smi_snapshots.txt
set -u

cd "$(dirname "$0")"
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
OUT="$EV/dumps"
BIN="$ROOT/build/probe_aic_gm_dma"
REPS="${REPS:-3}"
TO="${TO:-60}"
LOCKWAIT="${LOCKWAIT:-300}"
SNAP="$EV/npu_smi_snapshots.txt"
VARIANTS="${VARIANTS:-fixp_l0c2gm ub2gm_api ub2gm_capi aiv_ub2gm neg_fixp_short aic_scalar_ub}"

mkdir -p "$LOGS" "$OUT"

if [ ! -x "$BIN" ]; then
    echo "== 构建 =="
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release > /tmp/pagd.cmake 2>&1 || { cat /tmp/pagd.cmake; exit 1; }
    cmake --build build -j4 > /tmp/pagd.build 2>&1 || { tail -30 /tmp/pagd.build; exit 1; }
fi

{
    echo "# probe_aic_gm_dma 证据生成记录（M141）"
    echo "date            : $(date -Is)"
    echo "host            : $(uname -srm)"
    echo "reps per variant: $REPS ; per-run timeout=${TO}s ; lock wait=${LOCKWAIT}s"
    echo "variants        : $VARIANTS"
    echo
    echo "## 复现命令（本目录执行；每档各自进锁）"
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"
    echo "REPS=$REPS VARIANTS=\"$VARIANTS\" bash run_probe.sh"
    echo
    echo "## 源码 sha256"
    sha256sum probe_aic_gm_dma.asc CMakeLists.txt run_probe.sh
} > "$LOGS/commands.txt"

: > "$LOGS/matrix.txt"
echo "variant reps=$REPS timeout=${TO}s" >> "$LOGS/matrix.txt"
: > "$LOGS/check_ref.log"
: > "$SNAP"

# 每档各自进锁（失败 ⇒ 记录「未取得读数」）
for v in $VARIANTS; do
    log="$LOGS/${v}.log"
    : > "$log"
    lock_rc=0
    flock -w "$LOCKWAIT" /tmp/npu0.lock bash -c '
        set -u
        BIN="$1"; LOG="$2"; OUT="$3"; SNAP="$4"; V="$5"; REPS="$6"; TO="$7"
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        {
          echo "=== lock acquired $(date -Is) variant=$V reps=$REPS timeout=${TO}s ==="
          echo "--- npu-smi (lock entry) ---"
          npu-smi info 2>&1
        } >> "$LOG"
        { echo "=== $(date -Is) variant=$V lock acquired ==="; npu-smi info 2>&1; } >> "$SNAP"
        n0=0; n1=0; n2=0; hang=0
        for i in $(seq 1 "$REPS"); do
            echo "--- run $i (timeout ${TO}s) ---" >> "$LOG"
            timeout "$TO" stdbuf -oL "$BIN" "$V" "$i" "$OUT" >> "$LOG" 2>&1
            rc=$?
            echo "exit=$rc" >> "$LOG"
            case "$rc" in
              0) n0=$((n0+1));;
              1) n1=$((n1+1));;
              2) n2=$((n2+1));;
              124) hang=$((hang+1)); echo "SUMMARY variant=$V run=$i HANG=1 rc=124（锁内超时未返回：记「未取得内容读数」，不记「没复现」）" >> "$LOG";;
            esac
        done
        echo "AGG variant=$V rc0=$n0 rc1=$n1 rc2=$n2 hang=$hang reps=$REPS" >> "$LOG"
    ' _ "$BIN" "$log" "$OUT" "$SNAP" "$v" "$REPS" "$TO"
    lock_rc=$?
    if [ "$lock_rc" -ne 0 ]; then
        echo "AGG variant=$v LOCKFAIL=1（${LOCKWAIT}s 内未取得设备锁：记「未取得读数」）" >> "$log"
    fi
    agg=$(grep -m1 '^AGG ' "$log" | sed 's/^AGG //')
    printf '%-16s %s\n' "$v" "${agg:-未取得读数（锁未取得）}" | tee -a "$LOGS/matrix.txt"
    # 独立宿主侧复核（零设备）：对每份落盘 dump 用 check_ref.py 重算（与 C++ 判据互为独立实现）
    for b in "$OUT"/out_${v}_run*.bin; do
        [ -f "$b" ] || continue
        python3 "$ROOT/check_ref.py" "$v" "$b" 2>&1 | tee -a "$LOGS/check_ref.log"
    done
done

echo "完成。证据目录：$EV"
