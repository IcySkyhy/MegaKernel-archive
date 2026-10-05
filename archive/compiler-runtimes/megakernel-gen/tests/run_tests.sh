#!/bin/bash
# The compiler's own test suite.  Three layers:
#   1. the runtime primitives, exhaustively where the domain allows it
#   2. every supported model compiles and builds
#   3. unsupported models are REFUSED, by name, rather than miscompiled
set -u
cd "$(dirname "$0")/.."
source env.sh
MK="${CARGO_TARGET_DIR}/release/mkc"
M=$MKGEN_MODELS
B=$MKGEN_WORK/build
T=$MKGEN_WORK/test-out
fail=0

echo "== [1] runtime primitives"
nvcc -O2 -arch=sm_90a -std=c++17 -Iruntime/include -o "$B/test_prims" tests/test_prims.cu || exit 1
"$B/test_prims" || fail=$((fail+1))

echo
echo "== [2] every supported model compiles and builds"
for D in "$M"/Qwen3-0.6B "$M"/Qwen3-1.7B "$M"/TinyLlama-1.1B-Chat-v1.0 \
         "$M"/SmolLM2-1.7B-Instruct "$M"/Qwen2.5-1.5B-Instruct \
         "$M"/Phi-3-mini-4k-instruct "$M"/gpt-oss-20b "$M"/Qwen3-8B \
         "$M"/Qwen3-30B-A3B "$M"/gemma-2-2b-it "$M"/Qwen1.5-MoE-A2.7B \
         "$M"/Qwen3-0.6B-FP8 "$M"/Qwen3-1.7B-FP8-dynamic \
         "$M"/Qwen2.5-1.5B-Instruct-AWQ \
         "$MKGEN_MODEL_120B"; do
  [ -d "$D" ] || { printf "[skip] %s\n" "$(basename "$D")"; continue; }
  printf "%-28s " "$(basename "$D")"
  if "$MK" compile "$D" -o "$T" --ctx 1024 >/dev/null 2>&1 \
     && (cd "$T" && make clean >/dev/null 2>&1; make -j4 >/dev/null 2>&1); then
    echo "[ ok ]"
  else
    echo "[FAIL]"; fail=$((fail+1))
  fi
done

echo
echo "== [2b] the register search: every candidate compiles, one is chosen"
# `compile` only emits; this is the path that enumerates occupancies, compiles
# them all in parallel and reads the assembler's register count back.
printf "%-28s " "build search"
if "$MK" build "$M"/Qwen3-0.6B -o "$T/search" --ctx 1024 > "$B/search.log" 2>&1 \
   && grep -q "blk/SM" "$B/search.log"; then
  echo "[ ok ] $(grep -c 'blk/SM' "$B/search.log") schedules placed"
else
  echo "[FAIL]"; tail -3 "$B/search.log"; fail=$((fail+1))
fi

echo
echo "== [2b2] the same inputs produce byte-identical output"
# The README says compilation is a pure function of the checkpoint and the
# machine file.  That is a testable claim, so test it.
printf "%-28s " "reproducibility"
rm -rf "$T/rep_a" "$T/rep_b"
"$MK" compile "$M"/gpt-oss-20b -o "$T/rep_a" --ctx 1024 >/dev/null 2>&1
"$MK" compile "$M"/gpt-oss-20b -o "$T/rep_b" --ctx 1024 >/dev/null 2>&1
if diff -r -q "$T/rep_a" "$T/rep_b" >/dev/null 2>&1; then
  echo "[ ok ] two compiles, byte-identical"
else
  echo "[FAIL]"; diff -r -q "$T/rep_a" "$T/rep_b" | head -3; fail=$((fail+1))
fi
rm -rf "$T/rep_a" "$T/rep_b"

echo
echo "== [2c] no template hole survives into the generated source"
# A `@NAME@` that no filler ever replaces compiles fine as long as it lands in a
# comment, and is a silent wrong answer if it lands anywhere else.
printf "%-28s " "template holes"
if grep -rl '@[A-Z0-9_]\+@' "$T"/*.cu "$T"/*.cuh "$T"/*.h >/dev/null 2>&1; then
  echo "[FAIL]"; grep -rn '@[A-Z0-9_]\+@' "$T"/*.cu "$T"/*.cuh "$T"/*.h | head -3
  fail=$((fail+1))
else
  echo "[ ok ]"
fi

echo
echo "== [3] unsupported models are refused, not miscompiled"
# Real checkpoints, not synthetic configs: the format is detected from the
# tensors a checkpoint actually ships, so a mislabelled config proves nothing.
mk_expect() {   # dir, "ok" | expected refusal substring
  [ -d "$1" ] || { printf "%-28s [skip]\n" "$(basename "$1")"; return; }
  printf "%-28s " "$(basename "$1")"
  out=$("$MK" compile "$1" -o "$T" --ctx 1024 2>&1)
  if [ "$2" = "ok" ]; then
    if echo "$out" | grep -q "^mkc: error"; then echo "[FAIL] refused: $(echo "$out"|tail -1)"; fail=$((fail+1))
    elif (cd "$T" && make clean >/dev/null 2>&1; make -j4 >/dev/null 2>&1); then echo "[ ok ] compiles"
    else echo "[FAIL] build"; fail=$((fail+1)); fi
  else
    if echo "$out" | grep -q "$2"; then echo "[ ok ] refused"
    else echo "[FAIL] not refused"; fail=$((fail+1)); fi
  fi
}
mk_expect "$M/Qwen2.5-1.5B-Instruct-AWQ" ok              # int4 AWQ, repacked on load
# NB: a synthetic quantization_config proves nothing -- the format is detected
# from the tensors a checkpoint ships, so only real checkpoints test it.
mk_reject() {   # name, python edit of config.json, expected substring
  local d="$M/_reject_$1"; rm -rf "$d"; mkdir -p "$d"
  ln -sf "$M"/Qwen3-0.6B/*.safetensors "$d"/
  python3 -c "
import json; c=json.load(open('$M/Qwen3-0.6B/config.json')); $2
json.dump(c,open('$d/config.json','w'))"
  printf "%-28s " "$1"
  if "$MK" compile "$d" -o "$T" --ctx 1024 2>&1 | grep -q "$3"; then echo "[ ok ] refused"
  else echo "[FAIL] not refused"; fail=$((fail+1)); fi
  rm -rf "$d"
}
mk_reject head_dim80 "c['head_dim']=80"                  "multiple of 32"
mk_reject odd_gqa    "c['num_key_value_heads']=5"        "do not divide"

echo
[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "$fail TEST GROUP(S) FAILED"
exit $fail
