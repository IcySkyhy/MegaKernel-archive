#!/bin/bash
#SBATCH --job-name=mkgen-occ
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
# Calibrate the ONE rule the register search cannot derive: how much register
# spilling is worth how much occupancy.  For each model, every (block size,
# blocks per SM) is compiled and timed on synthetic weights, with the
# assembler's register count and spill next to the measurement.  The output is
# a table that says what the planner's spill policy should be -- it is not a
# search: nothing here feeds back into a build.
set -u
cd "${MKGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source env.sh
export CUDA_VISIBLE_DEVICES=0 MKGEN_GPU_AUTO=0
# Freeze the compiler for the life of the job: the shared target directory is
# rebuilt by whoever is working, and a benchmark whose rows came from two
# different binaries is not a benchmark.
MK=$MKGEN_WORK/build/mkc-$SLURM_JOB_ID
cp "${CARGO_TARGET_DIR}/release/mkc" "$MK"
export MKC_BIN="$MK"
M=$MKGEN_MODELS
O=$MKGEN_WORK/test-out/occ
TSV=$MKGEN_WORK/occupancy.tsv

echo "== node $(hostname)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
printf "model\tnt\tbps\tregs\tspill_B\tms\tpredicted_ms\n" > "$TSV"
for NAME in Qwen3-1.7B Qwen3-8B gpt-oss-20b Qwen3-30B-A3B; do
  D="$M/$NAME"; [ -d "$D" ] || continue
  for NT in 256 512 1024; do
    for BPS in 1 2 3 4 5 6 8; do
      rm -rf "$O"
      MKC_NT=$NT MKC_BPS=$BPS "$MK" compile "$D" -o "$O" --ctx 1024 > "$O.log" 2>&1 || continue
      PRED=$(grep -oP '(?<=predicted )[0-9.]+(?= ms/token)' "$O.log" | head -1)
      (cd "$O" && make -j8 mkbench >/dev/null 2>&1) || continue
      REG=$(grep -oP '(?<=Used )\d+(?= registers)' "$O/ptxas.log" | sort -n | tail -1)
      SP=$(grep -oP '\d+(?= bytes spill stores)' "$O/ptxas.log" | sort -n | tail -1)
      MS=$(cd "$O" && timeout 300 ./mkbench 1024 64 2>/dev/null)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$NT" "$BPS" "${REG:-0}" "${SP:-0}" "${MS:-nan}" "${PRED:-nan}" >> "$TSV"
    done
  done
  echo "-- $NAME"; grep "^$NAME" "$TSV" | column -t
done
echo "== done"; column -t < "$TSV"
