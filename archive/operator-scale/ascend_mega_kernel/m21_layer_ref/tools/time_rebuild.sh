#!/usr/bin/env bash
# Measure the wall time of a full reference rebuild, N times, and report the
# spread -- NOT a single number.
#
#   bash tools/time_rebuild.sh [N] [dumps|with-evidence|quick]
#   writes evidence/rebuild_timing[_with_evidence|_quick].txt
#     dumps          time the 6 run_reference.py calls only   (default)
#     with-evidence  also time `tools/make_evidence.sh`
#     quick          only the m=1 tags
#
# Why a range and not one number: this box is shared with 10+ concurrent agent
# workers, so host wall clock is not a stable quantity. The project rule
# broadcast 2026-09-26 ("host 墙钟计时在共享卡上不构成证据") says a single number is
# not acceptable evidence and any timing claim must come with its spread. Run
# this and quote min/max, or label the number "qualitative only".
#
# Exit codes: 0 = measured and reported; 2 = could not measure anything.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/workspace/venvs/baseline/bin/python3
N="${1:-5}"
MODE="${2:-dumps}"
case "$MODE" in
  with-evidence) OUT="evidence/rebuild_timing_with_evidence.txt" ;;
  quick)         OUT="evidence/rebuild_timing_quick.txt" ;;
  *)             OUT="evidence/rebuild_timing.txt" ;;
esac
mkdir -p evidence

TAGS=(
  "--layer 0 --m 1 --warmup 8 --input embed --pending none --tag layer0_decode_m1"
  "--layer 0 --m 1 --warmup 8 --input embed --pending synth --tag layer0_decode_m1_pending"
  "--layer 3 --m 1 --warmup 2100 --input synthetic --pending synth --tag layer3_decode_m1"
)
ALL_TAGS=(
  "${TAGS[@]}"
  "--layer 0 --m 64 --input embed --pending synth --tag layer0_chunk_m64"
  "--layer 3 --m 64 --input synthetic --pending none --tag layer3_chunk_m64"
  "--layer 3 --m 64 --input synthetic --seed 1 --pending none --tag layer3_chunk_m64_seed1"
)
if [ "$MODE" != "quick" ]; then TAGS=("${ALL_TAGS[@]}"); fi

{
  echo "# rebuild wall time, mode=$MODE, $N consecutive runs, M39_THREADS=${M39_THREADS:-8}"
  echo "# host wall clock on a SHARED box -> read as a range, not a number"
  echo "# (project rule 2026-09-26: single-number host timings are not evidence)"
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)  load=$(cut -d' ' -f1-3 /proc/loadavg)"
  echo
} > "$OUT"

count=0
for i in $(seq 1 "$N"); do
  start=$(date +%s.%N)
  for spec in "${TAGS[@]}"; do
    # shellcheck disable=SC2086
    M39_THREADS="${M39_THREADS:-8}" $PY run_reference.py $spec >/dev/null 2>&1 || {
      echo "run $i FAILED on: $spec" | tee -a "$OUT"; exit 2; }
  done
  end=$(date +%s.%N)
  secs=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.1f", b-a}')
  line="run $i: ${secs} s"
  if [ "$MODE" = "with-evidence" ]; then
    estart=$(date +%s.%N)
    bash tools/make_evidence.sh >/dev/null 2>&1 || {
      echo "evidence step FAILED on run $i" | tee -a "$OUT"; exit 2; }
    eend=$(date +%s.%N)
    esecs=$(awk -v a="$estart" -v b="$eend" 'BEGIN{printf "%.1f", b-a}')
    line="$line  (+ make_evidence.sh ${esecs} s -> total $(awk -v x="$secs" -v y="$esecs" 'BEGIN{printf "%.1f", x+y}') s)"
  fi
  echo "$line" | tee -a "$OUT"
  count=$((count + 1))
done

$PY - "$OUT" <<'EOF' | tee -a "$OUT"
import re, sys, statistics
p = sys.argv[1]
txt = open(p).read()
dumps = [float(m) for m in re.findall(r"^run \d+: ([\d.]+) s", txt, re.M)]
totals = [float(m) for m in re.findall(r"-> total ([\d.]+) s", txt)]
if not dumps:
    print("RESULT: SKIPPED (0 runs measured)")
    sys.exit(2)
def line(name, vals):
    return (f"{name}: min {min(vals):.1f} s / max {max(vals):.1f} s / "
            f"median {statistics.median(vals):.1f} s / spread {max(vals)/min(vals):.2f}x")
print(f"RESULT: OK ({len(dumps)}/{len(dumps)} runs measured)")
print(line("dumps only", dumps))
if totals:
    print(line("dumps + evidence", totals))
EOF
