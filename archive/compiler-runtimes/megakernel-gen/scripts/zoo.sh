#!/bin/bash
# Compile, gate and benchmark every model in the zoo, one shot each.
# Usage: scripts/zoo.sh [model_dir ...]
set -u
cd "$(dirname "$0")/.."
source env.sh
MK="${MKC_BIN:-${CARGO_TARGET_DIR}/release/mkc}"
V=$MKGEN_PY
OUT=$MKGEN_WORK/out
LOG=$MKGEN_WORK/logs
RES="${MKGEN_RESULTS:-$MKGEN_WORK/results.tsv}"
CTX=${CTX:-1024}
DECODE=${DECODE:-128}
GATE=${GATE:-8}

[ -f "$RES" ] || printf "model\tarch\tparams\tquant\tcompile_s\tdecode_ms\ttok_s\tbw_util\troofline_ms\tfloor_ms\tpredicted_ms\tgate\tguard\n" > "$RES"

for M in "$@"; do
  NAME=$(basename "$M")
  echo "================ $NAME"
  t0=$(date +%s.%N)
  if ! "$MK" build "$M" -o "$OUT/$NAME" --ctx "$CTX" ${VERIFY:+--verify} > "$LOG/$NAME.compile.log" 2>&1; then
    echo "COMPILE FAILED"; tail -5 "$LOG/$NAME.compile.log"; continue
  fi
  t1=$(date +%s.%N)
  # a weight-only-quantised checkpoint is gated against the SAME weights
  # dequantised: the stock reference also quantises activations, and that error
  # swamps the comparison (see docs/RESULTS.md)
  DQ=""
  grep -qE '"quant_method": *"(fp8|compressed-tensors|awq|gptq)"' "$M/config.json" 2>/dev/null && DQ="--ref-dequant"
  # Delete the previous result FIRST.  A run that dies -- the reference model
  # reaching for the network on an offline node, say -- would otherwise leave
  # the old file in place and the table would report a stale number next to a
  # fresh compile time, which is worse than reporting nothing.
  rm -f "$OUT/$NAME/result.json"
  $V harness/run.py "$OUT/$NAME" --model "$M" --gate "$GATE" --prefill "$CTX" $DQ \
      --decode "$DECODE" --guard --weight-sample 24 --ref-device "${REF:-cuda:1}" \
      --json "$OUT/$NAME/result.json" \
      > "$LOG/$NAME.run.log" 2>&1
  [ -f "$OUT/$NAME/result.json" ] || { echo "RUN FAILED"; tail -3 "$LOG/$NAME.run.log"; }
  grep -E "decode_ms_per_tok|tok_per_s|bw_util_pct|GATE|roofline_ms|predicted_ms" "$LOG/$NAME.run.log"
  $V - "$OUT/$NAME" "$NAME" "$(echo "$t1-$t0"|bc)" "$RES" <<'PY'
import json,sys,os
out,name,ct,res = sys.argv[1:5]
cfg=json.load(open(os.path.join(out,"build.json")))
try: r=json.load(open(os.path.join(out,"result.json")))
except Exception: r={}          # no result file = the run died; the row says so
g=r.get("gate",{}); gd=r.get("guard",{})
# roofline_ms charges every byte the machine's streaming ceiling; floor_ms
# charges each byte the measured ceiling of the gemv core that reads it, which
# is the floor a KERNEL can reach -- a 4-bit matrix cannot stream at the dense
# rate however the schedule is arranged.
row=[name,cfg["arch"],f"{cfg['bytes_per_token']/1e6:.0f}MB/tok",cfg["q_ffn"],
     f"{float(ct):.2f}",
     f"{r.get('decode_ms_per_tok',float('nan')):.4f}",
     f"{1000/r['decode_ms_per_tok']:.1f}" if r.get("decode_ms_per_tok") else "-",
     f"{r.get('bw_util_pct',float('nan')):.1f}",
     f"{cfg['roofline_ms']:.4f}",f"{cfg.get('floor_ms',float('nan')):.4f}",
     f"{cfg['predicted_ms']:.4f}",
     "PASS" if g.get("passed") else "FAIL",
     "PASS" if gd.get("passed") else ("FAIL" if gd else "-")]
open(res,"a").write("\t".join(row)+"\n")
print("\t".join(row))
PY
done
